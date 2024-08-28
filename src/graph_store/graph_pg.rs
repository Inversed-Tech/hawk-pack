use sqlx::postgres::PgPoolOptions;
use sqlx::postgres::PgRow;
use sqlx::Row;
use std::marker::PhantomData;

use crate::{
    hnsw_db::{FurthestQueue, FurthestQueueV},
    GraphStore, VectorStore,
};

use super::EntryPoint;

const DB_URL: &str = "postgres://postgres:postgres@localhost/postgres";

const POOL_SIZE: u32 = 5;

const CREATE_TABLE_LINKS: &str = "
CREATE TABLE IF NOT EXISTS hawk_graph_links (
    source_ref text NOT NULL,
    layer integer NOT NULL,
    links jsonb NOT NULL,
    CONSTRAINT hawk_graph_pkey PRIMARY KEY (source_ref, layer)
)";

const CREATE_TABLE_ENTRY: &str = "
CREATE TABLE IF NOT EXISTS hawk_graph_entry (
    entry_point jsonb,
    id integer NOT NULL,
    CONSTRAINT hawk_graph_entry_pkey PRIMARY KEY (id)
)
";

pub struct GraphPg<V: VectorStore> {
    pool: sqlx::PgPool,
    phantom: PhantomData<V>,
}

impl<V: VectorStore> GraphPg<V> {
    #[allow(dead_code)]
    pub async fn new() -> Result<Self, sqlx::Error> {
        let pool = PgPoolOptions::new()
            .max_connections(POOL_SIZE)
            .connect(DB_URL)
            .await?;

        sqlx::query(CREATE_TABLE_LINKS).execute(&pool).await?;
        sqlx::query(CREATE_TABLE_ENTRY).execute(&pool).await?;

        Ok(GraphPg {
            pool,
            phantom: PhantomData,
        })
    }
}

impl<V: VectorStore> GraphStore<V> for GraphPg<V> {
    async fn get_entry_point(&self) -> Option<EntryPoint<V::VectorRef>> {
        sqlx::query(
            "
                SELECT entry_point FROM hawk_graph_entry WHERE id = 0
            ",
        )
        .fetch_optional(&self.pool)
        .await
        .expect("Failed to fetch entry point")
        .map(|row: PgRow| {
            let x: sqlx::types::Json<EntryPoint<V::VectorRef>> = row.get("entry_point");
            let y: EntryPoint<V::VectorRef> = x.as_ref().clone();
            y
        })
    }

    async fn set_entry_point(&mut self, entry_point: EntryPoint<V::VectorRef>) {
        sqlx::query(
            "
            INSERT INTO hawk_graph_entry (entry_point, id)
            VALUES ($1, 0) ON CONFLICT (id)
            DO UPDATE SET entry_point = EXCLUDED.entry_point
        ",
        )
        .bind(sqlx::types::Json(&entry_point))
        .execute(&self.pool)
        .await
        .expect("Failed to set entry point");
    }

    async fn get_links(
        &self,
        base: &<V as VectorStore>::VectorRef,
        lc: usize,
    ) -> FurthestQueueV<V> {
        let base_str = serde_json::to_string(base).unwrap();

        sqlx::query(
            "
            SELECT links FROM hawk_graph_links WHERE source_ref = $1 AND layer = $2
        ",
        )
        .bind(base_str)
        .bind(lc as i32)
        .fetch_optional(&self.pool)
        .await
        .expect("Failed to fetch links")
        .map(|row: PgRow| {
            let x: sqlx::types::Json<FurthestQueueV<V>> = row.get("links");
            x.as_ref().clone()
        })
        .unwrap_or_else(FurthestQueue::new)
    }

    async fn set_links(&mut self, base: V::VectorRef, links: FurthestQueueV<V>, lc: usize) {
        let base_str = serde_json::to_string(&base).unwrap();

        sqlx::query(
            "
            INSERT INTO hawk_graph_links (source_ref, layer, links)
            VALUES ($1, $2, $3) ON CONFLICT (source_ref, layer)
            DO UPDATE SET
            links = EXCLUDED.links
        ",
        )
        .bind(base_str)
        .bind(lc as i32)
        .bind(sqlx::types::Json(&links))
        .execute(&self.pool)
        .await
        .expect("Failed to set links");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examples::lazy_memory_store::LazyMemoryStore;
    use crate::hnsw_db::{FurthestQueue, HawkSearcher};
    use tokio;

    #[ignore]
    #[tokio::test]
    async fn test_db() {
        let mut graph = GraphPg::<LazyMemoryStore>::new().await.unwrap();
        let mut vector_store = LazyMemoryStore::new();

        sqlx::query("DELETE FROM hawk_graph_entry")
            .execute(&graph.pool)
            .await
            .unwrap();
        sqlx::query("DELETE FROM hawk_graph_links")
            .execute(&graph.pool)
            .await
            .unwrap();

        let vectors = {
            let mut v = vec![];
            for raw_query in 0..10 {
                let q = vector_store.prepare_query(raw_query);
                v.push(vector_store.insert(&q).await);
            }
            v
        };

        let distances = {
            let mut d = vec![];
            for v in vectors.iter() {
                d.push(vector_store.eval_distance(&vectors[0], v).await);
            }
            d
        };

        let ep = graph.get_entry_point().await;

        let ep2 = EntryPoint {
            vector_ref: vectors[0],
            layer_count: ep.map(|e| e.layer_count).unwrap_or_default() + 1,
        };

        graph.set_entry_point(ep2.clone()).await;

        let ep3 = graph.get_entry_point().await.unwrap();
        assert_eq!(ep2, ep3);

        for i in 1..4 {
            let mut links = FurthestQueue::new();

            for j in 4..7 {
                links.insert(&vector_store, vectors[j], distances[j]).await;
            }

            graph.set_links(vectors[i], links.clone(), 0).await;

            let links2 = graph.get_links(&vectors[i], 0).await;
            assert_eq!(*links, *links2);
        }
    }

    #[ignore]
    #[tokio::test]
    async fn test_hnsw_db() {
        let graph_store = GraphPg::new().await.unwrap();
        sqlx::query("DELETE FROM hawk_graph_entry")
            .execute(&graph_store.pool)
            .await
            .unwrap();
        sqlx::query("DELETE FROM hawk_graph_links")
            .execute(&graph_store.pool)
            .await
            .unwrap();

        let vector_store = LazyMemoryStore::new();
        let mut db = HawkSearcher::new(vector_store, graph_store);

        let queries = (0..10)
            .map(|raw_query| db.vector_store.prepare_query(raw_query))
            .collect::<Vec<_>>();

        // Insert the codes.
        for query in queries.iter() {
            let neighbors = db.search_to_insert(query).await;
            assert!(!db.is_match(&neighbors).await);
            // Insert the new vector into the store.
            let inserted = db.vector_store.insert(query).await;
            db.insert_from_search_results(inserted, neighbors).await;
        }

        // Search for the same codes and find matches.
        for query in queries.iter() {
            let neighbors = db.search_to_insert(query).await;
            assert!(db.is_match(&neighbors).await);
        }
    }
}
