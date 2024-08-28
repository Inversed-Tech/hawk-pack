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
        let vector_store = LazyMemoryStore::new();
        let graph_store = GraphMem::new();
        let mut rng = AesRng::seed_from_u64(0_u64);
        let mut initial_db = HawkSearcher::new(vector_store, graph_store, &mut rng);

        let queries = (0..database_size)
            .map(|raw_query| initial_db.vector_store.prepare_query(raw_query))
            .collect::<Vec<_>>();

        let runtime = tokio::runtime::Runtime::new().unwrap();
        // Insert the codes.

        let full_db = runtime.block_on(async move {
            for query in queries.iter() {
                let neighbors = initial_db.search_to_insert(query).await;
                assert!(!initial_db.is_match(&neighbors).await);
                // Insert the new vector into the store.
                let inserted = initial_db.vector_store.insert(query).await;
                initial_db
                    .insert_from_search_results(inserted, neighbors)
                    .await;
            }
            initial_db
        });
        group.bench_function(BenchmarkId::new("hnsw-insertions", database_size), |b| {
            b.iter_batched_ref(
                || full_db.clone(),
                |my_db| {
                    runtime.block_on(async move {
                        let raw_query = database_size;
                        let query = my_db.vector_store.prepare_query(raw_query);
                        let neighbors = my_db.search_to_insert(&query).await;
                        let inserted = my_db.vector_store.insert(&query).await;
                        my_db.insert_from_search_results(inserted, neighbors).await;
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
