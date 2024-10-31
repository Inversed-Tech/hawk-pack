pub mod graph_store;
pub mod hnsw_db;

pub mod linear_db;
pub mod vector_store;

use std::fmt::Debug;
use std::hash::Hash;

use eyre::Result;
pub use graph_store::GraphStore;
use serde::Serialize;
use sqlx::PgPool;
pub use vector_store::VectorStore;

pub trait Ref:
    Clone + Debug + PartialEq + Eq + Hash + Sync + Serialize + for<'de> serde::Deserialize<'de>
{
}

impl<T> Ref for T where
    T: Clone + Debug + PartialEq + Eq + Hash + Sync + Serialize + for<'de> serde::Deserialize<'de>
{
}

#[allow(async_fn_in_trait)]
pub trait DbStore: Sized {
    async fn new(url: &str, schema_name: &str) -> Result<Self>;

    fn pool(&self) -> &PgPool;

    fn schema_name(&self) -> String;

    async fn copy_in(&self, table_paths: Vec<(String, String)>) -> Result<()> {
        for (table_name, path) in table_paths.iter() {
            let file = tokio::fs::File::open(path).await?;
            let mut conn = self.pool().acquire().await?;

            let mut copy_stream = conn
                .copy_in_raw(&format!(
                    "COPY {} FROM STDIN (FORMAT CSV, HEADER)",
                    table_name
                ))
                .await?;

            copy_stream.read_from(file).await?;
            copy_stream.finish().await?;
        }

        Ok(())
    }

    async fn copy_out(&self) -> Result<Vec<(String, String)>>;

    async fn cleanup(&self) -> Result<()> {
        sqlx::query(&format!("DROP SCHEMA \"{}\" CASCADE", self.schema_name()))
            .execute(self.pool())
            .await?;
        Ok(())
    }
}
