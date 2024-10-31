use serde::{Deserialize, Serialize};

use crate::VectorStore;

/// Example implementation of a vector store - Lazy variant.
///
/// A distance is lazily represented in `DistanceRef` as a tuple of point IDs, and the actual distance is evaluated later in `less_than`.
#[derive(Default, Clone, Debug)]
pub struct LazyMemoryStore {
    points: Vec<Point>,
}

#[derive(Clone, Debug)]
struct Point {
    /// Whatever encoding of a vector.
    data: u64,
    /// Distinguish between queries that are pending, and those that were ultimately accepted into the vector store.
    is_persistent: bool,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct PointId(usize);

impl LazyMemoryStore {
    pub fn new() -> Self {
        LazyMemoryStore { points: vec![] }
    }
}

impl LazyMemoryStore {
    pub fn prepare_query(&mut self, raw_query: u64) -> <Self as VectorStore>::QueryRef {
        self.points.push(Point {
            data: raw_query,
            is_persistent: false,
        });

        let point_id = self.points.len() - 1;
        PointId(point_id)
    }

    fn actually_evaluate_distance(&self, pair: &<Self as VectorStore>::DistanceRef) -> u32 {
        // Hamming distance
        let vector_0 = self.points[pair.0 .0].data;
        let vector_1 = self.points[pair.1 .0].data;
        (vector_0 ^ vector_1).count_ones()
    }
}

impl VectorStore for LazyMemoryStore {
    type QueryRef = PointId; // Vector ID, pending insertion.
    type VectorRef = PointId; // Vector ID, inserted.
    type DistanceRef = (PointId, PointId); // Lazy distance representation.

    async fn insert(&mut self, query: &Self::QueryRef) -> Self::VectorRef {
        // The query is now accepted in the store. It keeps the same ID.
        self.points[query.0].is_persistent = true;
        *query
    }

    async fn eval_distance(
        &self,
        query: &Self::QueryRef,
        vector: &Self::VectorRef,
    ) -> Self::DistanceRef {
        // Do not compute the distance yet, just forward the IDs.
        (*query, *vector)
    }

    async fn is_match(&mut self, distance: &Self::DistanceRef) -> bool {
        self.actually_evaluate_distance(distance) == 0
    }

    async fn less_than(
        &mut self,
        distance1: &Self::DistanceRef,
        distance2: &Self::DistanceRef,
    ) -> bool {
        self.actually_evaluate_distance(distance1) < self.actually_evaluate_distance(distance2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_eval_distance() {
        let mut store = LazyMemoryStore::new();

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
