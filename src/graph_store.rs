use serde::{Deserialize, Serialize};

use crate::hnsw_db::FurthestQueueV;
use crate::VectorStore;
use std::fmt::Debug;

pub mod graph_mem;
mod graph_pg;

#[allow(async_fn_in_trait)]
pub trait GraphStore<V: VectorStore> {
    async fn get_entry_point(&self) -> Option<EntryPoint<V::VectorRef>>;

    async fn set_entry_point(&mut self, entry_point: EntryPoint<V::VectorRef>);

    async fn get_links(&self, base: &<V as VectorStore>::VectorRef, lc: usize)
        -> FurthestQueueV<V>;

    async fn set_links(&mut self, base: V::VectorRef, links: FurthestQueueV<V>, lc: usize);
}

#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntryPoint<VectorRef> {
    pub vector_ref: VectorRef,
    pub layer_count: usize,
}
