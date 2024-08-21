use std::ops::Deref;

use serde::{Deserialize, Serialize};

use crate::VectorStore;

/// FurthestQueue is a list sorted in ascending order, with fast pop of the furthest element.
#[derive(Default, Debug, Serialize, Deserialize)]
pub struct FurthestQueue<V: VectorStore> {
    queue: Vec<(V::VectorRef, V::DistanceRef)>,
}

impl<V: VectorStore> FurthestQueue<V> {
    pub fn new() -> Self {
        FurthestQueue { queue: vec![] }
    }

    pub fn from_ascending_vec(queue: Vec<(V::VectorRef, V::DistanceRef)>) -> Self {
        FurthestQueue { queue }
    }

    /// Insert the element `to` with distance `dist` into the queue, maitaining the ascending order.
    ///
    /// Call the VectorStore to come up with the insertion index.
    pub async fn insert(&mut self, store: &mut V, to: V::VectorRef, dist: V::DistanceRef) {
        let index_asc = store
            .search_sorted(
                &self
                    .queue
                    .iter()
                    .map(|(_, dist)| dist.clone())
                    .collect::<Vec<_>>(),
                &dist,
            )
            .await;
        self.queue.insert(index_asc, (to, dist));
    }

    pub fn get_nearest(&self) -> Option<&(V::VectorRef, V::DistanceRef)> {
        self.queue.first()
    }

    pub fn get_furthest(&self) -> Option<&(V::VectorRef, V::DistanceRef)> {
        self.queue.last()
    }

    pub fn pop_furthest(&mut self) -> Option<(V::VectorRef, V::DistanceRef)> {
        self.queue.pop()
    }

    pub fn get_k_nearest(&self, k: usize) -> &[(V::VectorRef, V::DistanceRef)] {
        &self.queue[..k]
    }

    pub fn trim_to_k_nearest(&mut self, k: usize) {
        self.queue.truncate(k);
    }
}

// Utility implementations.

impl<V: VectorStore> Deref for FurthestQueue<V> {
    type Target = [(V::VectorRef, V::DistanceRef)];

    fn deref(&self) -> &Self::Target {
        &self.queue
    }
}

impl<V: VectorStore> Clone for FurthestQueue<V> {
    fn clone(&self) -> Self {
        FurthestQueue {
            queue: self.queue.clone(),
        }
    }
}

impl<V: VectorStore> From<FurthestQueue<V>> for Vec<(V::VectorRef, V::DistanceRef)> {
    fn from(queue: FurthestQueue<V>) -> Self {
        queue.queue
    }
}

/// NearestQueue is a list sorted in descending order, with fast pop of the nearest element.
#[derive(Debug, Serialize, Deserialize)]
pub struct NearestQueue<V: VectorStore> {
    queue: Vec<(V::VectorRef, V::DistanceRef)>,
}

impl<V: VectorStore> NearestQueue<V> {
    pub fn from_furthest_queue(furthest_queue: &FurthestQueue<V>) -> Self {
        NearestQueue {
            queue: furthest_queue.iter().rev().cloned().collect(),
        }
    }

    /// Insert the element `to` with distance `dist` into the queue, maitaining the descending order.
    ///
    /// Call the VectorStore to come up with the insertion index.
    pub async fn insert(&mut self, store: &mut V, to: V::VectorRef, dist: V::DistanceRef) {
        let index_asc = store
            .search_sorted(
                &self
                    .queue
                    .iter()
                    .map(|(_, dist)| dist.clone())
                    .rev() // switch to ascending order.
                    .collect::<Vec<_>>(),
                &dist,
            )
            .await;
        let index_des = self.queue.len() - index_asc; // back to descending order.
        self.queue.insert(index_des, (to, dist));
    }

    #[allow(dead_code)]
    fn get_nearest(&self) -> Option<&(V::VectorRef, V::DistanceRef)> {
        self.queue.last()
    }

    pub fn pop_nearest(&mut self) -> Option<(V::VectorRef, V::DistanceRef)> {
        self.queue.pop()
    }
}

// Utility implementations.

impl<V: VectorStore> Deref for NearestQueue<V> {
    type Target = [(V::VectorRef, V::DistanceRef)];

    fn deref(&self) -> &Self::Target {
        &self.queue
    }
}

impl<V: VectorStore> Clone for NearestQueue<V> {
    fn clone(&self) -> Self {
        NearestQueue {
            queue: self.queue.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examples::eager_memory_store::EagerMemoryStore;

    #[tokio::test]
    async fn test_furthest_queue() {
        let mut store = EagerMemoryStore::new();
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
