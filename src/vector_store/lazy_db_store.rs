use super::*;
use std::path;

use eyre::{eyre, Result};
use futures::stream::TryStreamExt;
use sqlx::postgres::PgRow;
use sqlx::{migrate::Migrator, postgres::PgPoolOptions};
use sqlx::{Executor, Row};
use tokio::io::AsyncWriteExt;

use crate::VectorStore;

const MAX_CONNECTIONS: u32 = 5;

static MIGRATOR: Migrator = sqlx::migrate!("./vector_migrations");

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Point {
    /// Whatever encoding of a vector.
    data: u64,
    /// Distinguish between queries that are pending, and those that were ultimately accepted into the vector store.
    is_persistent: bool,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct PointId(pub usize);

impl PointId {
    pub fn val(&self) -> usize {
        self.0
    }
}

/// Example implementation of a vector store - Lazy variant.
///
/// A distance is lazily represented in `DistanceRef` as a tuple of point IDs, and the actual distance is evaluated later in `less_than`.
#[derive(Clone, Debug)]
pub struct LazyDbStore {
    cache: Vec<Point>,
    schema_name: String,
    pool: sqlx::PgPool,
}

impl DbStore for LazyDbStore {
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

        Ok(LazyDbStore {
            cache: vec![],
            schema_name: schema_name.to_owned(),
            pool,
        })
    }

    fn pool(&self) -> &PgPool {
        &self.pool
    }

    fn schema_name(&self) -> String {
        self.schema_name.to_owned()
    }

    async fn copy_out(&self) -> Result<Vec<(String, String)>> {
        let table_name = "hawk_vectors";
        let file_name = format!("{}_vectors.csv", self.schema_name.clone());

        let path = path::absolute(file_name.clone())?
            .as_os_str()
            .to_str()
            .unwrap()
            .to_owned();

        let mut file = tokio::fs::File::create(path.clone()).await?;
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

        Ok(vec![(table_name.to_owned(), path)])
    }
}

impl LazyDbStore {
    async fn actually_evaluate_distance(&self, pair: &<Self as VectorStore>::DistanceRef) -> u32 {
        // Hamming distance
        let vector_0 = self.get_point(&pair.0).await.unwrap().data;
        let vector_1 = self.get_point(&pair.1).await.unwrap().data;
        (vector_0 ^ vector_1).count_ones()
    }

    async fn get_point(&self, point: &PointId) -> Option<Point> {
        if self.cache.len() > point.0 {
            Some(self.cache[point.0].clone())
        } else {
            sqlx::query(
                "
                        SELECT point FROM hawk_vectors WHERE id = $1
                    ",
            )
            .bind(point.0 as i32)
            .fetch_optional(&self.pool)
            .await
            .expect("Failed to fetch entry point")
            .map(|row: PgRow| {
                let x: sqlx::types::Json<Point> = row.get("point");
                x.as_ref().clone()
            })
        }
    }
}

impl VectorStore for LazyDbStore {
    type Data = u64;
    type QueryRef = PointId; // Vector ID, pending insertion.
    type VectorRef = PointId; // Vector ID, inserted.
    type DistanceRef = (PointId, PointId); // Lazy distance representation.

    async fn insert(&mut self, query: &Self::QueryRef) -> Self::VectorRef {
        let mut point = self.cache[query.0].clone();
        point.is_persistent = true;

        sqlx::query(
            "
            INSERT INTO hawk_vectors (id, point)
            VALUES ($1, $2)
        ",
        )
        .bind(query.0 as i32)
        .bind(sqlx::types::Json(point))
        .execute(&self.pool)
        .await
        .expect("Failed to set entry point");

        *query
    }

    fn prepare_query(&mut self, raw_query: u64) -> <Self as VectorStore>::QueryRef {
        self.cache.push(Point {
            data: raw_query,
            is_persistent: false,
        });

        let point_id = self.cache.len() - 1;
        PointId(point_id)
    }

    async fn eval_distance(
        &self,
        query: &Self::QueryRef,
        vector: &Self::VectorRef,
    ) -> Self::DistanceRef {
        // Do not compute the distance yet, just forward the IDs.
        (*query, *vector)
    }

    async fn is_match(&self, distance: &Self::DistanceRef) -> bool {
        self.actually_evaluate_distance(distance).await == 0
    }

    async fn less_than(
        &self,
        distance1: &Self::DistanceRef,
        distance2: &Self::DistanceRef,
    ) -> bool {
        self.actually_evaluate_distance(distance1).await
            < self.actually_evaluate_distance(distance2).await
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
    /// Access the database with `&vectors` or `vectors.owned()`.
    pub struct TestVectorPg {
        cache: Vec<Point>,
        vectors: LazyDbStore,
        schema_name: String,
    }

    impl TestVectorPg {
        pub async fn new() -> Result<Self> {
            let schema_name = temporary_name();
            let vectors = LazyDbStore::new(&test_db_url()?, &schema_name).await?;
            Ok(TestVectorPg {
                cache: vec![],
                vectors,
                schema_name,
            })
        }

        pub fn owned(&self) -> LazyDbStore {
            LazyDbStore {
                cache: self.cache.clone(),
                schema_name: self.schema_name.clone(),
                pool: self.vectors.pool.clone(),
            }
        }
    }

    impl Deref for TestVectorPg {
        type Target = LazyDbStore;
        fn deref(&self) -> &Self::Target {
            &self.vectors
        }
    }

    impl DerefMut for TestVectorPg {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.vectors
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
    use test_utils::TestVectorPg;

    use super::*;

    #[tokio::test]
    async fn test_eval_distance() {
        let mut store = TestVectorPg::new().await.unwrap();

        let query = store.prepare_query(11);
        let vector = store.insert(&query).await;
        let distance = store.eval_distance(&query, &vector).await;
        assert!(store.is_match(&distance).await);

        let other_query = store.prepare_query(22);
        let other_vector = store.insert(&other_query).await;
        let other_distance = store.eval_distance(&query, &other_vector).await;
        assert!(!store.is_match(&other_distance).await);
    }
}
