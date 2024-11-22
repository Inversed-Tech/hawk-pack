use aes_prng::AesRng;
use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BenchmarkId;
use criterion::Criterion;
use hawk_pack::examples::lazy_memory_store::LazyMemoryStore;
use hawk_pack::graph_store::graph_mem::GraphMem;
use hawk_pack::hnsw_db::HawkSearcher;
use hawk_pack::linear_db::LinearDb;
use hawk_pack::VectorStore;
use rand::SeedableRng;

fn hnsw_db(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw");
    for database_size in [1000, 10000, 100000] {
        let vector_store = &mut LazyMemoryStore::new();
        let graph_store = &mut GraphMem::new();
        let rng = &mut AesRng::seed_from_u64(0_u64);
        let initial_db = &HawkSearcher::default();

        let queries = (0..database_size)
            .map(|raw_query| vector_store.prepare_query(raw_query))
            .collect::<Vec<_>>();

        let runtime = tokio::runtime::Runtime::new().unwrap();
        // Insert the codes.

        runtime.block_on(async {
            for query in queries.iter() {
                let (neighbors, buffers) = initial_db
                    .search_to_insert(vector_store, graph_store, query)
                    .await;
                assert!(!initial_db.is_match(vector_store, &neighbors).await);
                // Insert the new vector into the store.
                let inserted = vector_store.insert(query).await;
                initial_db
                    .insert_from_search_results(
                        vector_store,
                        graph_store,
                        rng,
                        inserted,
                        neighbors,
                        buffers,
                    )
                    .await;
            }
        });
        group.bench_function(BenchmarkId::new("hnsw-insertions", database_size), |b| {
            b.iter_batched_ref(
                || (vector_store.clone(), graph_store.clone(), rng.clone()),
                |(vector_store, graph_store, rng)| {
                    runtime.block_on(async move {
                        let raw_query = database_size;
                        let query = vector_store.prepare_query(raw_query);
                        let (neighbors, buffers) = initial_db
                            .search_to_insert(vector_store, graph_store, &query)
                            .await;
                        let inserted = vector_store.insert(&query).await;
                        initial_db
                            .insert_from_search_results(
                                vector_store,
                                graph_store,
                                rng,
                                inserted,
                                neighbors,
                                buffers,
                            )
                            .await;
                    });
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }
}

fn linear(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear");
    for database_size in [1000, 10000, 100000] {
        let vector_store = LazyMemoryStore::new();
        let mut initial_db = LinearDb::new(vector_store);

        let queries = (0..database_size)
            .map(|raw_query| initial_db.store.prepare_query(raw_query))
            .collect::<Vec<_>>();

        let runtime = tokio::runtime::Runtime::new().unwrap();
        // Insert the codes.

        let full_db = runtime.block_on(async move {
            for query in queries.iter() {
                initial_db.insert(query).await;
            }
            initial_db
        });
        group.bench_function(BenchmarkId::new("hnsw-insertions", database_size), |b| {
            b.iter_batched_ref(
                || full_db.clone(),
                |my_db| {
                    runtime.block_on(async move {
                        let raw_query = database_size;
                        let query = my_db.store.prepare_query(raw_query);
                        let _inserted = my_db.store.insert(&query).await;
                    });
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }
}

criterion_group!(hnsw, hnsw_db, linear);
criterion_main!(hnsw);
