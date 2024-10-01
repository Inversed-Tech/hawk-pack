use crate::{
    hnsw_db::{FurthestQueue, FurthestQueueV},
    DbStore, GraphStore, VectorStore,
};
use eyre::{eyre, Result};
use sqlx::postgres::PgRow;
use sqlx::Executor;
use sqlx::Row;
use sqlx::{migrate::Migrator, postgres::PgPoolOptions};
use std::{marker::PhantomData, path};

use super::{EntryPoint, GraphMem};

const MAX_CONNECTIONS: u32 = 5;

static MIGRATOR: Migrator = sqlx::migrate!("./migrations");

#[derive(Clone)]
pub struct GraphPg<V: VectorStore> {
    schema_name: String,
    pool: sqlx::PgPool,
    phantom: PhantomData<V>,
}

impl<V: VectorStore> DbStore for GraphPg<V> {
    async fn new(url: &str, schema_name: &str) -> Result<Self> {
        let connect_sql = sql_switch_schema(schema_name)?;

        let pool = PgPoolOptions::new()
            .max_connections(MAX_CONNECTIONS)
            .after_connect(move |conn, _meta| {
                // Switch to the given schema in every connection.
                let connect_sql = connect_sql.clone();
                Box::pin(async move {
                    conn.execute(connect_sql.as_ref()).await.inspect_err(|e| {
                        eprintln!("error in after_connect: {:?}", e);
                    })?;
                    Ok(())
                })
            })
            .connect(url)
            .await?;

        // Create the schema on the first startup.
        MIGRATOR.run(&pool).await?;

        Ok(GraphPg {
            schema_name: schema_name.to_owned(),
            pool,
            phantom: PhantomData,
        })
    }

    fn pool(&self) -> &sqlx::PgPool {
        &self.pool
    }

    fn schema_name(&self) -> String {
        self.schema_name.to_owned()
    }

    async fn copy_out(&self) -> Result<Vec<(String, String)>> {
        self.copy_out_with_filename(self.schema_name.clone()).await
    }

    async fn cleanup(&self) -> Result<()> {
        sqlx::query(&format!("DROP SCHEMA \"{}\" CASCADE", self.schema_name))
            .execute(&self.pool)
            .await?;
        Ok(())
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

impl<V: VectorStore> GraphPg<V> {
    pub async fn to_graph_mem<U>(&self) -> GraphMem<U>
    where
        U: VectorStore,
        U: VectorStore<DistanceRef = V::DistanceRef>,
        U: VectorStore<VectorRef = V::VectorRef>,
    {
        let mut graph_mem = GraphMem::new();

        let entry_point = self.get_entry_point().await;
        if entry_point.is_some() {
            graph_mem.set_entry_point(entry_point.unwrap()).await
        }

        let links = sqlx::query(
            "
            SELECT * FROM hawk_graph_links;
            ",
        )
        .fetch_all(self.pool())
        .await
        .expect("Failed to fetch hawk_graph_links")
        .into_iter()
        .map(|row: PgRow| {
            let node: String = row.get("source_ref");
            let links: sqlx::types::Json<FurthestQueueV<U>> = row.get("links");
            let layer: i32 = row.get("layer");

            let links = links.as_ref().clone();
            let layer = layer as usize;

            (node, links, layer)
        })
        .collect::<Vec<_>>();

        for (node, links, layer) in links.into_iter() {
            let vector_ref = serde_json::from_str(&node).expect("Could not deserialise vector ref");
            graph_mem.set_links(vector_ref, links, layer).await;
        }

        graph_mem
    }

    pub async fn copy_out_with_filename(&self, filename: String) -> Result<Vec<(String, String)>> {
        use futures::stream::TryStreamExt;
        use tokio::io::AsyncWriteExt;

        let tables = ["hawk_graph_entry", "hawk_graph_links"];
        let mut paths = vec![];

        for table_name in tables.iter() {
            let file_name = format!("{}_{}.csv", filename, table_name);
            let path = path::absolute(file_name.clone())?
                .as_os_str()
                .to_str()
                .unwrap()
                .to_owned();
            paths.push(path.clone());

            let mut file = tokio::fs::File::create(path).await?;
            let mut conn = self.pool.acquire().await?;

            let mut copy_stream = conn
                .copy_out_raw(&format!(
                    "COPY {} TO STDOUT (FORMAT CSV, HEADER)",
                    table_name
                ))
                .await?;

            while let Some(chunk) = copy_stream.try_next().await? {
                file.write_all(&chunk).await?;
            }
        }

        Ok(vec![
            (tables[0].to_owned(), paths[0].clone()),
            (tables[1].to_owned(), paths[1].clone()),
        ])
    }
}

fn sql_switch_schema(schema_name: &str) -> Result<String> {
    sanitize_identifier(schema_name)?;
    Ok(format!(
        "
        CREATE SCHEMA IF NOT EXISTS \"{}\";
        SET search_path TO \"{}\";
        ",
        schema_name, schema_name
    ))
}

fn sanitize_identifier(input: &str) -> Result<()> {
    if input.chars().all(|c| c.is_alphanumeric() || c == '_') {
        Ok(())
    } else {
        Err(eyre!("Invalid SQL identifier"))
    }
}

pub mod test_utils {
    use super::*;
    use std::{
        env,
        ops::{Deref, DerefMut},
    };
    const DOTENV_TEST: &str = ".env.test";
    const ENV_DB_URL: &str = "HAWK__DATABASE__URL";
    const SCHEMA_PREFIX: &str = "hawk_test";

    /// A test database. It creates a unique schema for each test. Call `cleanup` at the end of the test.
    ///
    /// Access the database with `&graph` or `graph.owned()`.
    pub struct TestGraphPg<V: VectorStore> {
        graph: GraphPg<V>,
        schema_name: String,
    }

    impl<V: VectorStore> TestGraphPg<V> {
        pub async fn new() -> Result<Self> {
            let schema_name = temporary_name();
            let graph = GraphPg::new(&test_db_url()?, &schema_name).await?;
            Ok(TestGraphPg { graph, schema_name })
        }

        pub fn owned(&self) -> GraphPg<V> {
            GraphPg {
                schema_name: self.schema_name.clone(),
                pool: self.graph.pool.clone(),
                phantom: PhantomData,
            }
        }
    }

    impl<V: VectorStore> Deref for TestGraphPg<V> {
        type Target = GraphPg<V>;
        fn deref(&self) -> &Self::Target {
            &self.graph
        }
    }

    impl<V: VectorStore> DerefMut for TestGraphPg<V> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.graph
        }
    }

    fn test_db_url() -> Result<String> {
        dotenvy::from_filename(DOTENV_TEST)?;
        Ok(env::var(ENV_DB_URL)?)
    }

    fn temporary_name() -> String {
        format!("{}_{}", SCHEMA_PREFIX, rand::random::<u32>())
    }
}

#[cfg(test)]
#[cfg(feature = "db_dependent")]
mod tests {
    use std::ops::DerefMut;

    use super::test_utils::TestGraphPg;
    use super::*;
    use crate::hnsw_db::{FurthestQueue, HawkSearcher};
    use crate::vector_store::lazy_memory_store::LazyMemoryStore;
    use aes_prng::AesRng;
    use rand::SeedableRng;
    use tokio;

    #[tokio::test]
    async fn test_db() {
        let mut graph = TestGraphPg::<LazyMemoryStore>::new().await.unwrap();
        let mut vector_store = LazyMemoryStore::new();

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

        graph.cleanup().await.unwrap();
    }

    #[tokio::test]
    async fn test_hnsw_db() {
        let mut graph = TestGraphPg::new().await.unwrap();
        let vector_store = &mut LazyMemoryStore::new();
        let rng = &mut AesRng::seed_from_u64(0_u64);
        let db = HawkSearcher::default();

        let queries = (0..10)
            .map(|raw_query| vector_store.prepare_query(raw_query))
            .collect::<Vec<_>>();

        // Insert the codes.
        for query in queries.iter() {
            let neighbors = db
                .search_to_insert(vector_store, graph.deref_mut(), query)
                .await;
            assert!(!db.is_match(vector_store, &neighbors).await);
            // Insert the new vector into the store.
            let inserted = vector_store.insert(query).await;
            db.insert_from_search_results(
                vector_store,
                graph.deref_mut(),
                rng,
                inserted,
                neighbors,
            )
            .await;
        }

        // Search for the same codes and find matches.
        for query in queries.iter() {
            let neighbors = db
                .search_to_insert(vector_store, graph.deref_mut(), query)
                .await;
            assert!(db.is_match(vector_store, &neighbors).await);
        }

        let graph_table_paths = graph.copy_out().await.unwrap();
        graph.cleanup().await.unwrap();

        // Test copy_in
        {
            let graph = TestGraphPg::new().await.unwrap();
            graph.copy_in(graph_table_paths).await.unwrap();

            let db = HawkSearcher::default();

            // Search for the same codes and find matches.
            for query in queries.iter() {
                let neighbors = db
                    .search_to_insert(&mut vector_store.clone(), &mut graph.owned(), query)
                    .await;
                assert!(db.is_match(vector_store, &neighbors).await);
            }
            graph.cleanup().await.unwrap();
        }
    }
}
