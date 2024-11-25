// Converted from Python to Rust.
use std::collections::HashSet;
mod queue;
pub use queue::{FurthestQueue, FurthestQueueV, NearestQueue, NearestQueueV};
use rand::RngCore;
use rand_distr::{Distribution, Geometric};
use serde::{Deserialize, Serialize};
pub mod coroutine;

use crate::{graph_store::EntryPoint, GraphStore, VectorStore};

// specify construction and search parameters by layer up to this value minus 1
// any higher layers will use the last set of parameters
const N_PARAM_LAYERS: usize = 5;

#[allow(non_snake_case)]
#[derive(PartialEq, Clone, Serialize, Deserialize)]
pub struct Params {
    pub M: [usize; N_PARAM_LAYERS], // number of neighbors for insertion
    pub M_max: [usize; N_PARAM_LAYERS], // maximum number of neighbors
    pub ef_constr_search: [usize; N_PARAM_LAYERS], // ef_constr for search layers
    pub ef_constr_insert: [usize; N_PARAM_LAYERS], // ef_constr for insertion layers
    pub ef_search: [usize; N_PARAM_LAYERS], // ef for search
    pub layer_probability: f64,     // p for geometric distribution of layer densities
}

#[allow(non_snake_case, clippy::too_many_arguments)]
impl Params {
    /// Construct a `Params` object corresponding to parameter configuration
    /// providing the functionality described in the original HNSW paper:
    /// - ef_construction exploration factor used for insertion layers
    /// - ef_search exploration factor used for layer 0 in search
    /// - Higher layers in both insertion and search use exploration factor 1,
    ///   representing simple greedy search
    /// - vertex degrees bounded by M_max = M in positive layer graphs
    /// - vertex degrees bounded by M_max0 = 2*M in layer 0 graph
    /// - m_L = 1 / ln(M) so that layer density decreases by a factor of M at
    ///   each successive hierarchical layer
    pub fn new(ef_construction: usize, ef_search: usize, M: usize) -> Self {
        let M_arr = [M; N_PARAM_LAYERS];
        let mut M_max_arr = [M; N_PARAM_LAYERS];
        M_max_arr[0] = 2 * M;
        let ef_constr_search_arr = [1usize; N_PARAM_LAYERS];
        let ef_constr_insert_arr = [ef_construction; N_PARAM_LAYERS];
        let mut ef_search_arr = [1usize; N_PARAM_LAYERS];
        ef_search_arr[0] = ef_search;
        let layer_probability = (M as f64).recip();

        Self {
            M: M_arr,
            M_max: M_max_arr,
            ef_constr_search: ef_constr_search_arr,
            ef_constr_insert: ef_constr_insert_arr,
            ef_search: ef_search_arr,
            layer_probability,
        }
    }

    /// Parameter configuration using fixed exploration factor for all layer
    /// search operations, both for insertion and for search.
    pub fn new_uniform(ef: usize, M: usize) -> Self {
        let M_arr = [M; N_PARAM_LAYERS];
        let mut M_max_arr = [M; N_PARAM_LAYERS];
        M_max_arr[0] = 2 * M;
        let ef_constr_search_arr = [ef; N_PARAM_LAYERS];
        let ef_constr_insert_arr = [ef; N_PARAM_LAYERS];
        let ef_search_arr = [ef; N_PARAM_LAYERS];
        let layer_probability = (M as f64).recip();

        Self {
            M: M_arr,
            M_max: M_max_arr,
            ef_constr_search: ef_constr_search_arr,
            ef_constr_insert: ef_constr_insert_arr,
            ef_search: ef_search_arr,
            layer_probability,
        }
    }

    /// Compute the parameter m_L associated with a geometric distribution
    /// parameter q describing the random layer of newly inserted graph nodes.
    ///
    /// E.g. for graph hierarchy where each layer has a factor of 32 fewer
    /// entries than the last, the `layer_probability` input is 1/32.
    pub fn m_L_from_layer_probability(layer_probability: f64) -> f64 {
        -layer_probability.ln().recip()
    }

    /// Compute the parameter q for the geometric distribution used to select
    /// the insertion layer for newly inserted graph nodes, from the parameter
    /// m_L of the original HNSW paper.
    pub fn layer_probability_from_m_L(m_L: f64) -> f64 {
        (-m_L.recip()).exp()
    }

    pub fn get_M(&self, lc: usize) -> usize {
        Self::get_val(&self.M, lc)
    }

    pub fn get_M_max(&self, lc: usize) -> usize {
        Self::get_val(&self.M_max, lc)
    }

    pub fn get_ef_constr_search(&self, lc: usize) -> usize {
        Self::get_val(&self.ef_constr_search, lc)
    }

    pub fn get_ef_constr_insert(&self, lc: usize) -> usize {
        Self::get_val(&self.ef_constr_insert, lc)
    }

    pub fn get_ef_search(&self, lc: usize) -> usize {
        Self::get_val(&self.ef_search, lc)
    }

    pub fn get_layer_probability(&self) -> f64 {
        self.layer_probability
    }

    pub fn get_m_L(&self) -> f64 {
        Self::m_L_from_layer_probability(self.layer_probability)
    }

    #[inline(always)]
    fn get_val(arr: &[usize; N_PARAM_LAYERS], lc: usize) -> usize {
        *arr.get(lc).unwrap_or_else(|| arr.last().unwrap())
    }
}

/// An implementation of the HNSW algorithm.
///
/// Operations on vectors are delegated to a VectorStore.
/// Operations on the graph are delegate to a GraphStore.
#[derive(Clone, Serialize, Deserialize)]
pub struct HawkSearcher {
    pub params: Params,
}

// TODO remove default value; this varies too much between applications
// to make sense to specify something "obvious"
impl Default for HawkSearcher {
    fn default() -> Self {
        HawkSearcher {
            params: Params::new(64, 32, 32),
        }
    }
}

#[allow(non_snake_case)]
impl HawkSearcher {
    async fn connect_bidir<V: VectorStore, G: GraphStore<V>>(
        &self,
        vector_store: &mut V,
        graph_store: &mut G,
        q: &V::VectorRef,
        mut neighbors: FurthestQueueV<V>,
        lc: usize,
    ) {
        let M = self.params.get_M(lc);
        let max_links = self.params.get_M_max(lc);

        neighbors.trim_to_k_nearest(M);

        // Connect all n -> q.
        for (n, nq) in neighbors.iter() {
            let mut links = graph_store.get_links(n, lc).await;
            links.insert(vector_store, q.clone(), nq.clone()).await;
            links.trim_to_k_nearest(max_links);
            graph_store.set_links(n.clone(), links, lc).await;
        }

        // Connect q -> all n.
        graph_store.set_links(q.clone(), neighbors, lc).await;
    }

    fn select_layer(&self, rng: &mut impl RngCore) -> usize {
        let p_geom = 1f64 - self.params.get_layer_probability();
        let geom_distr = Geometric::new(p_geom).unwrap();

        geom_distr.sample(rng) as usize
    }

    #[allow(non_snake_case)]
    async fn search_init<V: VectorStore, G: GraphStore<V>>(
        &self,
        vector_store: &mut V,
        graph_store: &mut G,
        query: &V::QueryRef,
    ) -> (FurthestQueueV<V>, usize) {
        if let Some(entry_point) = graph_store.get_entry_point().await {
            let entry_vector = entry_point.vector_ref;
            let distance = vector_store.eval_distance(query, &entry_vector).await;

            let mut W = FurthestQueueV::<V>::new();
            W.insert(vector_store, entry_vector, distance).await;

            (W, entry_point.layer_count)
        } else {
            (FurthestQueue::new(), 0)
        }
    }

    /// Mutate W into the ef nearest neighbors of q_vec in the given layer.
    #[allow(non_snake_case)]
    async fn search_layer<V: VectorStore, G: GraphStore<V>>(
        &self,
        vector_store: &mut V,
        graph_store: &mut G,
        q: &V::QueryRef,
        W: &mut FurthestQueueV<V>,
        ef: usize,
        lc: usize,
    ) {
        // v: The set of already visited vectors.
        let mut v = HashSet::<V::VectorRef>::from_iter(W.iter().map(|(e, _eq)| e.clone()));

        // C: The set of vectors to visit, ordered by increasing distance to the query.
        let mut C = NearestQueue::from_furthest_queue(W);

        // fq: The current furthest distance in W.
        let (_, mut fq) = W.get_furthest().expect("W cannot be empty").clone();

        while C.len() > 0 {
            let (c, cq) = C.pop_nearest().expect("C cannot be empty").clone();

            // If the nearest distance to C is greater than the furthest distance in W, then we can stop.
            if vector_store.less_than(&fq, &cq).await {
                break;
            }

            // Visit all neighbors of c.
            let c_links = graph_store.get_links(&c, lc).await;

            // Evaluate the distances of the neighbors to the query, as a batch.
            let c_links = {
                let e_batch = c_links
                    .iter()
                    .map(|(e, _ec)| e.clone())
                    .filter(|e| {
                        // Visit any node at most once.
                        v.insert(e.clone())
                    })
                    .collect::<Vec<_>>();

                let distances = vector_store.eval_distance_batch(q, &e_batch).await;

                e_batch
                    .into_iter()
                    .zip(distances.into_iter())
                    .collect::<Vec<_>>()
            };

            for (e, eq) in c_links.into_iter() {
                if W.len() == ef {
                    // When W is full, we decide whether to replace the furthest element.
                    if vector_store.less_than(&eq, &fq).await {
                        // Make room for the new better candidate…
                        W.pop_furthest();
                    } else {
                        // …or ignore the candidate and do not continue on this path.
                        continue;
                    }
                }

                // Track the new candidate in C so we will continue this path later.
                C.insert(vector_store, e.clone(), eq.clone()).await;

                // Track the new candidate as a potential k-nearest.
                W.insert(vector_store, e, eq).await;

                // fq stays the furthest distance in W.
                (_, fq) = W.get_furthest().expect("W cannot be empty").clone();
            }
        }
    }

    #[allow(non_snake_case)]
    pub async fn search_to_insert<V: VectorStore, G: GraphStore<V>>(
        &self,
        vector_store: &mut V,
        graph_store: &mut G,
        query: &V::QueryRef,
    ) -> Vec<FurthestQueueV<V>> {
        let mut links = vec![];

        let (mut W, layer_count) = self.search_init(vector_store, graph_store, query).await;

        // From the top layer down to layer 0.
        for lc in (0..layer_count).rev() {
            // TODO pass insertion layer of query so different ef values can be
            // used for layers in which the query is and is not being inserted
            let ef = self.params.get_ef_constr_insert(lc);
            self.search_layer(vector_store, graph_store, query, &mut W, ef, lc)
                .await;

            links.push(W.clone());
        }

        links.reverse(); // We inserted top-down, so reverse to match the layer indices (bottom=0).
        links
    }

    pub async fn insert_from_search_results<V: VectorStore, G: GraphStore<V>>(
        &self,
        vector_store: &mut V,
        graph_store: &mut G,
        rng: &mut impl RngCore,
        inserted_vector: V::VectorRef,
        links: Vec<FurthestQueueV<V>>,
    ) {
        let layer_count = links.len();

        // Choose a maximum layer for the new vector. It may be greater than the current number of layers.
        let l = self.select_layer(rng);

        // Connect the new vector to its neighbors in each layer.
        for (lc, layer_links) in links.into_iter().enumerate().take(l + 1) {
            self.connect_bidir(vector_store, graph_store, &inserted_vector, layer_links, lc)
                .await;
        }

        // If the new vector goes into a layer higher than ever seen before, then it becomes the new entry point of the graph.
        if l >= layer_count {
            graph_store
                .set_entry_point(EntryPoint {
                    vector_ref: inserted_vector,
                    layer_count: l + 1,
                })
                .await;
        }
    }

    pub async fn is_match<V: VectorStore>(
        &self,
        vector_store: &mut V,
        neighbors: &[FurthestQueueV<V>],
    ) -> bool {
        match neighbors
            .first()
            .and_then(|bottom_layer| bottom_layer.get_nearest())
        {
            None => false, // Empty database.
            Some((_, smallest_distance)) => vector_store.is_match(smallest_distance).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examples::lazy_memory_store::LazyMemoryStore;
    use crate::graph_store::graph_mem::GraphMem;
    use aes_prng::AesRng;
    use rand::SeedableRng;
    use tokio;

    #[tokio::test]
    async fn test_hnsw_db() {
        let vector_store = &mut LazyMemoryStore::new();
        let graph_store = &mut GraphMem::new();
        let rng = &mut AesRng::seed_from_u64(0_u64);
        let db = HawkSearcher::default();

        let queries = (0..100)
            .map(|raw_query| vector_store.prepare_query(raw_query))
            .collect::<Vec<_>>();

        // Insert the codes.
        for query in queries.iter() {
            let neighbors = db.search_to_insert(vector_store, graph_store, query).await;
            assert!(!db.is_match(vector_store, &neighbors).await);
            // Insert the new vector into the store.
            let inserted = vector_store.insert(query).await;
            db.insert_from_search_results(vector_store, graph_store, rng, inserted, neighbors)
                .await;
        }

        // Search for the same codes and find matches.
        for query in queries.iter() {
            let neighbors = db.search_to_insert(vector_store, graph_store, query).await;
            assert!(db.is_match(vector_store, &neighbors).await);
        }
    }
}
