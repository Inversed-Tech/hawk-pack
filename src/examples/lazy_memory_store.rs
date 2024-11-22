use serde::{Deserialize, Serialize};

use crate::VectorStore;

/// Example implementation of a vector store - Lazy variant.
///
/// A distance is lazily represented in `DistanceRef` as a tuple of point IDs, and the actual distance is evaluated later in `less_than`.
#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct LazyMemoryStore {
    points: Vec<Point>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Point {
    /// Index of the point in the store
    point_id: PointId,
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

impl LazyMemoryStore {
    pub fn new() -> Self {
        LazyMemoryStore { points: vec![] }
    }
}

impl LazyMemoryStore {
    pub fn get_point_position(&self, point_id: PointId) -> Option<usize> {
        self.points
            .iter()
            .position(|point| point.point_id == point_id)
    }

    pub fn prepare_query(&mut self, raw_query: u64) -> <Self as VectorStore>::QueryRef {
        let point_id = PointId(self.points.len());
        self.points.push(Point {
            point_id,
            data: raw_query,
            is_persistent: false,
        });

        point_id
    }

    fn actually_evaluate_distance(&self, pair: &<Self as VectorStore>::DistanceRef) -> u32 {
        let position_0 = self.get_point_position(pair.0).expect("Point not found");
        let position_1 = self.get_point_position(pair.1).expect("Point not found");
        // Hamming distance
        let vector_0 = self.points[position_0].data;
        let vector_1 = self.points[position_1].data;
        (vector_0 ^ vector_1).count_ones()
    }
}

impl VectorStore for LazyMemoryStore {
    type QueryRef = PointId; // Vector ID, pending insertion.
    type VectorRef = PointId; // Vector ID, inserted.
    type DistanceRef = (PointId, PointId); // Lazy distance representation.

    async fn num_entries(&self) -> usize {
        self.points.len()
    }

    async fn insert(&mut self, query: &Self::QueryRef) -> Self::VectorRef {
        let position = self
            .get_point_position(*query)
            .expect("query does not exist");
        // The query is now accepted in the store. It keeps the same ID.
        self.points[position].is_persistent = true;
        *query
    }

    async fn delete(&mut self, vector: &Self::VectorRef) {
        let position = self
            .get_point_position(*vector)
            .expect("vector does not exist");
        self.points.remove(position);
    }

    async fn eval_distance(
        &mut self,
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

    fn get_range(&self, range: std::ops::Range<usize>) -> Vec<PointId> {
        self.points[range].iter().map(|p| p.point_id).collect()
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
