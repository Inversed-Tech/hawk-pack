use super::{EntryPoint, GraphStore};
use crate::{
    hnsw_db::{FurthestQueue, FurthestQueueV},
    VectorStore,
};
use std::collections::HashMap;

#[derive(Default, Clone)]
pub struct GraphMem<V: VectorStore> {
    pub entry_point: Option<EntryPoint<V::VectorRef>>,
    pub layers: Vec<Layer<V>>,
}

impl<V: VectorStore> GraphMem<V> {
    pub fn new() -> Self {
        GraphMem {
            entry_point: None,
            layers: vec![],
        }
    }
}

impl<V: VectorStore> GraphStore<V> for GraphMem<V> {
    async fn get_entry_point(&self) -> Option<EntryPoint<V::VectorRef>> {
        self.entry_point.clone()
    }

    async fn set_entry_point(&mut self, entry_point: EntryPoint<V::VectorRef>) {
        if let Some(previous) = self.entry_point.as_ref() {
            assert!(
                previous.layer_count < entry_point.layer_count,
                "A new entry point should be on a higher layer than before."
            );
        }

        while entry_point.layer_count > self.layers.len() {
            self.layers.push(Layer::new());
        }

        self.entry_point = Some(entry_point);
    }

    async fn get_links(
        &self,
        base: &<V as VectorStore>::VectorRef,
        lc: usize,
    ) -> FurthestQueueV<V> {
        let layer = &self.layers[lc];
        if let Some(links) = layer.get_links(base) {
            links.clone()
        } else {
            FurthestQueue::new()
        }
    }

    async fn set_links(&mut self, base: V::VectorRef, links: FurthestQueueV<V>, lc: usize) {
        let layer = &mut self.layers[lc];
        layer.set_links(base, links);
    }
}

#[derive(PartialEq, Eq, Default, Clone)]
pub struct Layer<V: VectorStore> {
    /// Map a base vector to its neighbors, including the distance base-neighbor.
    pub links: HashMap<V::VectorRef, FurthestQueueV<V>>,
}

impl<V: VectorStore> Layer<V> {
    fn new() -> Self {
        Layer {
            links: HashMap::new(),
        }
    }

    fn get_links(&self, from: &V::VectorRef) -> Option<&FurthestQueueV<V>> {
        self.links.get(from)
    }

    fn set_links(&mut self, from: V::VectorRef, links: FurthestQueueV<V>) {
        self.links.insert(from, links);
    }
}

// impl<U, T, W> From<GraphMem<U>> for GraphMem<T>
// where
//     U: VectorStore<VectorRef = W>,
//     T: VectorStore<VectorRef = W>,
// {
//     fn from(_value: GraphMem<U>) -> Self {
//         unimplemented!()
//     }
// }
