use std::ops::Deref;

use serde::{Deserialize, Serialize};

use crate::VectorStore;

pub type FurthestQueueV<V> =
    FurthestQueue<<V as VectorStore>::VectorRef, <V as VectorStore>::DistanceRef>;
pub type NearestQueueV<V> =
    NearestQueue<<V as VectorStore>::VectorRef, <V as VectorStore>::DistanceRef>;

/// FurthestQueue is a list sorted in ascending order, with fast pop of the furthest element.
#[derive(Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FurthestQueue<Vector, Distance> {
    queue: Vec<(Vector, Distance)>,
}

impl<Vector: Clone, Distance: Clone> FurthestQueue<Vector, Distance> {
    pub fn new() -> Self {
        FurthestQueue { queue: vec![] }
    }

    pub fn from_ascending_vec(queue: Vec<(Vector, Distance)>) -> Self {
        FurthestQueue { queue }
    }

    /// Insert the element `to` with distance `dist` into the queue, maintaining the ascending order.
    ///
    /// Call the VectorStore to come up with the insertion index.
    pub async fn insert<V>(&mut self, store: &mut V, to: Vector, dist: Distance)
    where
        V: VectorStore<VectorRef = Vector, DistanceRef = Distance>,
    {
        let index_asc = search_sorted(
            store,
            &self
                .queue
                .iter()
                .map(|(_, dist)| dist.clone())
                .collect::<Vec<Distance>>(),
            &dist,
        )
        .await;
        self.queue.insert(index_asc, (to, dist));
    }

    pub fn get_nearest(&self) -> Option<&(Vector, Distance)> {
        self.queue.first()
    }

    pub fn get_furthest(&self) -> Option<&(Vector, Distance)> {
        self.queue.last()
    }

    pub fn pop_furthest(&mut self) -> Option<(Vector, Distance)> {
        self.queue.pop()
    }

    pub fn get_k_nearest(&self, k: usize) -> &[(Vector, Distance)] {
        &self.queue[..k]
    }

    pub fn trim_to_k_nearest(&mut self, k: usize) {
        self.queue.truncate(k);
    }

    // Assumes that distance map doesn't change the distance metric
    pub fn map<V, F1, F2>(
        self,
        vector_map: F1,
        distance_map: F2,
    ) -> FurthestQueue<V::VectorRef, V::DistanceRef>
    where
        V: VectorStore,
        F1: Fn(Vector) -> V::VectorRef,
        F2: Fn(Distance) -> V::DistanceRef,
    {
        let queue: Vec<(V::VectorRef, V::DistanceRef)> = self
            .queue
            .iter()
            .cloned()
            .map(|(v, d)| (vector_map(v), distance_map(d)))
            .collect();
        FurthestQueue::from_ascending_vec(queue)
    }

    pub fn as_vec_ref(&self) -> &[(Vector, Distance)] {
        &self.queue
    }
}

// Utility implementations.

impl<Vector, Distance> Deref for FurthestQueue<Vector, Distance> {
    type Target = [(Vector, Distance)];

    fn deref(&self) -> &Self::Target {
        &self.queue
    }
}

impl<Vector: Clone, Distance: Clone> Clone for FurthestQueue<Vector, Distance> {
    fn clone(&self) -> Self {
        FurthestQueue {
            queue: self.queue.clone(),
        }
    }
}

impl<Vector, Distance> From<FurthestQueue<Vector, Distance>> for Vec<(Vector, Distance)> {
    fn from(queue: FurthestQueue<Vector, Distance>) -> Self {
        queue.queue
    }
}

/// NearestQueue is a list sorted in descending order, with fast pop of the nearest element.
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NearestQueue<Vector, Distance> {
    queue: Vec<(Vector, Distance)>,
}

impl<Vector: Clone, Distance: Clone> NearestQueue<Vector, Distance> {
    pub fn from_furthest_queue(furthest_queue: &FurthestQueue<Vector, Distance>) -> Self {
        NearestQueue {
            queue: furthest_queue.iter().rev().cloned().collect(),
        }
    }

    /// Insert the element `to` with distance `dist` into the queue, maitaining the descending order.
    ///
    /// Call the VectorStore to come up with the insertion index.
    pub async fn insert<V>(&mut self, store: &mut V, to: Vector, dist: Distance)
    where
        V: VectorStore<VectorRef = Vector, DistanceRef = Distance>,
    {
        let index_asc = search_sorted(
            store,
            &self
                .queue
                .iter()
                .map(|(_, dist)| dist.clone())
                .rev() // switch to ascending order.
                .collect::<Vec<Distance>>(),
            &dist,
        )
        .await;
        let index_des = self.queue.len() - index_asc; // back to descending order.
        self.queue.insert(index_des, (to, dist));
    }

    #[allow(dead_code)]
    fn get_nearest(&self) -> Option<&(Vector, Distance)> {
        self.queue.last()
    }

    pub fn pop_nearest(&mut self) -> Option<(Vector, Distance)> {
        self.queue.pop()
    }
}

/// Find the insertion index for a target distance to maintain order in a list of ascending distances.
async fn search_sorted<V>(
    store: &mut V,
    distances: &[V::DistanceRef],
    target: &V::DistanceRef,
) -> usize
where
    V: VectorStore,
{
    let mut left = 0;
    let mut right = distances.len();

    while left < right {
        let mid = left + (right - left) / 2;

        match store.less_than(&distances[mid], target).await {
            true => left = mid + 1,
            false => right = mid,
        }
    }
    left
}

// Utility implementations.

impl<Vector, Distance> Deref for NearestQueue<Vector, Distance> {
    type Target = [(Vector, Distance)];

    fn deref(&self) -> &Self::Target {
        &self.queue
    }
}

impl<Vector: Clone, Distance: Clone> Clone for NearestQueue<Vector, Distance> {
    fn clone(&self) -> Self {
        NearestQueue {
            queue: self.queue.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector_store::lazy_memory_store::LazyMemoryStore;

    #[tokio::test]
    async fn test_furthest_queue() {
        let mut store = LazyMemoryStore::new();
        let query = store.prepare_query(1);
        let vector = store.insert(&query).await;
        let distance = store.eval_distance(&query, &vector).await;

        // Example usage for FurthestQueue
        let mut furthest_queue = FurthestQueue::new();
        furthest_queue.insert(&mut store, vector, distance).await;
        println!("{:?}", furthest_queue.get_furthest());
        println!("{:?}", furthest_queue.get_k_nearest(1));
        println!("{:?}", furthest_queue.pop_furthest());

        // Example usage for NearestQueue
        let mut nearest_queue = NearestQueue::from_furthest_queue(&furthest_queue);
        nearest_queue.insert(&mut store, vector, distance).await;
        println!("{:?}", nearest_queue.get_nearest());
        println!("{:?}", nearest_queue.pop_nearest());
    }
}
