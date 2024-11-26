use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::hash::Hash;
use crate::data_structures::queue::FurthestQueueV;

pub trait Ref:
    Clone + Debug + PartialEq + Eq + Hash + Sync + Serialize + for<'de> serde::Deserialize<'de>
{
}

impl<T> Ref for T where
    T: Clone + Debug + PartialEq + Eq + Hash + Sync + Serialize + for<'de> serde::Deserialize<'de>
{
}

// The operations exposed by a vector store, sufficient for a search algorithm.
#[allow(async_fn_in_trait)]
pub trait VectorStore: Clone + Debug {
    /// Opaque reference to a query.
    ///
    /// Example: a preprocessed representation optimized for distance evaluations.
    type QueryRef: Ref;

    /// Opaque reference to a stored vector.
    ///
    /// Example: a vector ID.
    type VectorRef: Ref;

    /// Opaque reference to a distance metric.
    ///
    /// Example: an encrypted distance.
    type DistanceRef: Ref;

    /// Persist a query as a new vector in the store, and return a reference to it.
    async fn insert(&mut self, query: &Self::QueryRef) -> Self::VectorRef;

    /// Evaluate the distance between a query and a vector.
    async fn eval_distance(
        &mut self,
        query: &Self::QueryRef,
        vector: &Self::VectorRef,
    ) -> Self::DistanceRef;

    /// Check whether a distance is a match, meaning the query is considered equivalent to a previously inserted vector.
    async fn is_match(&mut self, distance: &Self::DistanceRef) -> bool;

    /// Compare two distances.
    async fn less_than(
        &mut self,
        distance1: &Self::DistanceRef,
        distance2: &Self::DistanceRef,
    ) -> bool;

    /// Find the insertion index for a target distance to maintain order in a list of ascending distances.
    async fn search_sorted(
        &mut self,
        distances: &[Self::DistanceRef],
        target: &Self::DistanceRef,
    ) -> usize {
        let mut left = 0;
        let mut right = distances.len();

        while left < right {
            let mid = left + (right - left) / 2;

            match self.less_than(&distances[mid], target).await {
                true => left = mid + 1,
                false => right = mid,
            }
        }
        left
    }

    // Batch variants.

    /// Persist a batch of queries as new vectors in the store, and return references to them.
    /// The default implementation is a loop over `insert`.
    /// Override for more efficient batch insertions.
    async fn insert_batch(&mut self, queries: &[Self::QueryRef]) -> Vec<Self::VectorRef> {
        let mut results = Vec::with_capacity(queries.len());
        for query in queries {
            results.push(self.insert(query).await);
        }
        results
    }

    /// Evaluate the distances between a query and a batch of vectors.
    /// The default implementation is a loop over `eval_distance`.
    /// Override for more efficient batch distance evaluations.
    async fn eval_distance_batch(
        &mut self,
        query: &Self::QueryRef,
        vectors: &[Self::VectorRef],
    ) -> Vec<Self::DistanceRef> {
        let mut results = Vec::with_capacity(vectors.len());
        for vector in vectors {
            results.push(self.eval_distance(query, vector).await);
        }
        results
    }

    /// Compare a distance with a batch of distances.
    /// The default implementation is a loop over `less_than`.
    /// Override for more efficient batch comparisons.
    async fn less_than_batch(
        &mut self,
        distance: &Self::DistanceRef,
        distances: &[Self::DistanceRef],
    ) -> Vec<bool> {
        let mut results = Vec::with_capacity(distances.len());
        for other_distance in distances {
            results.push(self.less_than(distance, other_distance).await);
        }
        results
    }
}

// TODO HawkSearcher trait

// TODO Comparisons trait

#[allow(async_fn_in_trait)]
pub trait GraphStore<V: VectorStore> {
    // Vertex type
    // Neighborhood type

    async fn get_entry_point(&self) -> Option<EntryPoint<V::VectorRef>>;

    async fn set_entry_point(&mut self, entry_point: EntryPoint<V::VectorRef>);

    // TODO make this an Option
    async fn get_links(&self, base: &<V as VectorStore>::VectorRef, lc: usize)
        -> FurthestQueueV<V>;

    // TODO make this a Result to handle if lc is invalid for given base?  (Does default insert unconditionally?)
    async fn set_links(&mut self, base: V::VectorRef, links: FurthestQueueV<V>, lc: usize);
}

#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntryPoint<VectorRef> {
    pub vector_ref: VectorRef,
    pub layer_count: usize,
}
