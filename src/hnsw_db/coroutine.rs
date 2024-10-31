use crate::{
    graph_store::EntryPoint,
    hnsw_db::{FurthestQueue, HawkSearcher},
    GraphStore, Ref, VectorStore,
};
use std::fmt::Debug;
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;

// TODO: handle channel errors.

/// Coroutine-based implementation of HNSW.
///
/// In this implementation, the HNSW algorithm is split into multiple asynchronous
/// operations that are executed in a coroutine-like fashion. The `HawkStream` struct
/// holds the coroutine state and interactions.
///
/// To perform a search, create a `HawkRun` with a query and call `next` until it
/// returns `None`. Each call to `next` will return an `Op` that represents the
/// operation that the search routine is waiting for. The caller is responsible for
/// executing the operation and sending the result back to the routine.
pub fn search_to_insert_stream<Q, V, D>(query: Q) -> HawkStream<Q, V, D>
where
    Q: Ref + Send + 'static,
    V: Ref + Send + 'static,
    D: Ref + Send + 'static,
{
    let (tx, rx) = mpsc::channel(1);
    tokio::spawn(async move {
        let hawk = HawkSearcher::default();
        let vector_store = &mut OpsCollector { ops: tx.clone() };
        let graph_store = &mut OpsCollector { ops: tx.clone() };
        let result = hawk
            .search_to_insert(vector_store, graph_store, &query)
            .await;
        tx.send(Op::SearchResult { query, result }).await.unwrap();
    });
    ReceiverStream::new(rx)
}

pub type HawkStream<Q, V, D> = ReceiverStream<Op<Q, V, D>>;

#[derive(Debug)]
pub enum Op<Query, Vector, Distance> {
    // VectorStore operations.
    EvalDistanceBatch {
        query: Query,
        vectors: Vec<Vector>,
        reply: oneshot::Sender<Vec<Distance>>,
    },
    LessThanBatch {
        distance: Distance,
        distances: Vec<Distance>,
        reply: oneshot::Sender<Vec<bool>>,
    },

    // GraphStore operations.
    GetEntryPoint {
        reply: oneshot::Sender<Option<EntryPoint<Vector>>>,
    },
    SetEntryPoint {
        entry_point: EntryPoint<Vector>,
    },
    GetLinks {
        base: Vector,
        lc: usize,
        reply: oneshot::Sender<FurthestQueue<Vector, Distance>>,
    },
    SetLinks {
        base: Vector,
        links: FurthestQueue<Vector, Distance>,
        lc: usize,
    },

    // Result.
    SearchResult {
        query: Query,
        result: Vec<FurthestQueue<Vector, Distance>>,
    },
}

#[derive(Clone, Debug)]
struct OpsCollector<Q, V, D> {
    ops: mpsc::Sender<Op<Q, V, D>>,
}

impl<Q: Ref, V: Ref, D: Ref> VectorStore for OpsCollector<Q, V, D> {
    type QueryRef = Q;
    type VectorRef = V;
    type DistanceRef = D;

    async fn insert(&mut self, _query: &Self::QueryRef) -> Self::VectorRef {
        todo!()
    }

    async fn eval_distance(
        &self,
        query: &Self::QueryRef,
        vector: &Self::VectorRef,
    ) -> Self::DistanceRef {
        self.eval_distance_batch(query, &[vector.clone()])
            .await
            .pop()
            .unwrap()
    }

    async fn eval_distance_batch(
        &self,
        query: &Self::QueryRef,
        vectors: &[Self::VectorRef],
    ) -> Vec<Self::DistanceRef> {
        let (reply, get_reply) = oneshot::channel();

        let op = Op::EvalDistanceBatch {
            query: query.clone(),
            vectors: vectors.to_vec(),
            reply,
        };

        self.ops.send(op).await.unwrap();
        get_reply.await.unwrap()
    }

    async fn less_than(
        &mut self,
        distance1: &Self::DistanceRef,
        distance2: &Self::DistanceRef,
    ) -> bool {
        self.less_than_batch(distance1, &[distance2.clone()])
            .await
            .pop()
            .unwrap()
    }

    async fn less_than_batch(
        &mut self,
        distance: &Self::DistanceRef,
        distances: &[Self::DistanceRef],
    ) -> Vec<bool> {
        let (reply, get_reply) = oneshot::channel();

        let op = Op::LessThanBatch {
            distance: distance.clone(),
            distances: distances.to_vec(),
            reply,
        };

        self.ops.send(op).await.unwrap();
        get_reply.await.unwrap()
    }

    async fn is_match(&mut self, _distance: &Self::DistanceRef) -> bool {
        todo!()
    }
}

impl<Q: Ref, V: Ref, D: Ref> GraphStore<OpsCollector<Q, V, D>> for OpsCollector<Q, V, D> {
    async fn get_entry_point(&self) -> Option<EntryPoint<V>> {
        let (reply, get_reply) = oneshot::channel();

        let op = Op::GetEntryPoint { reply };

        self.ops.send(op).await.unwrap();
        get_reply.await.unwrap()
    }

    async fn set_entry_point(&mut self, entry_point: EntryPoint<V>) {
        let op = Op::SetEntryPoint { entry_point };
        self.ops.send(op).await.unwrap();
    }

    async fn get_links(&self, base: &V, lc: usize) -> FurthestQueue<V, D> {
        let (reply, get_reply) = oneshot::channel();

        let op = Op::GetLinks {
            base: base.clone(),
            lc,
            reply,
        };

        self.ops.send(op).await.unwrap();
        get_reply.await.unwrap()
    }

    async fn set_links(&mut self, base: V, links: FurthestQueue<V, D>, lc: usize) {
        let op = Op::SetLinks { base, links, lc };
        self.ops.send(op).await.unwrap();
    }
}

#[cfg(test)]
mod tests {
    use std::vec;

    use super::Op::*;
    use super::*;
    use futures::stream::StreamExt;

    type Q = i64;
    type V = i64;
    type D = i64;

    #[tokio::test]
    async fn test_search_to_insert_stream_empty() {
        let mut stream = search_to_insert_stream::<Q, V, D>(0);

        let op = stream.next().await.unwrap();
        match op {
            GetEntryPoint { reply } => {
                reply.send(None).unwrap();
            }
            _ => panic!("Expected GetEntryPoint, got {:?}", op),
        }

        let op = stream.next().await.unwrap();
        match op {
            Op::SearchResult { query, result } => {
                assert_eq!(query, 0);
                assert!(result.is_empty());
            }
            _ => panic!("Expected SearchResult, got {:?}", op),
        }
    }

    #[tokio::test]
    async fn test_search_to_insert_stream() {
        let some_vec = 0;
        let some_query = 1;
        let some_distance = 10;
        let ep = EntryPoint {
            vector_ref: some_vec,
            layer_count: 1,
        };

        let mut stream = search_to_insert_stream::<Q, V, D>(some_query);

        let op = stream.next().await.unwrap();
        match op {
            GetEntryPoint { reply } => {
                reply.send(Some(ep)).unwrap();
            }
            _ => panic!("Expected GetEntryPoint, got {:?}", op),
        }

        let op = stream.next().await.unwrap();
        match op {
            EvalDistanceBatch {
                query,
                vectors,
                reply,
            } => {
                assert_eq!(query, some_query);
                assert_eq!(vectors, vec![some_vec]);
                reply.send(vec![some_distance]).unwrap();
            }
            _ => panic!("Expected EvalDistanceBatch, got {:?}", op),
        }

        let op = stream.next().await.unwrap();
        match op {
            LessThanBatch {
                distance,
                distances,
                reply,
            } => {
                assert_eq!(distance, some_distance);
                assert_eq!(distances, vec![some_distance]);
                reply.send(vec![true]).unwrap();
            }
            _ => panic!("Expected LessThanBatch, got {:?}", op),
        }

        let op = stream.next().await.unwrap();
        match op {
            Op::SearchResult { query, result } => {
                assert_eq!(query, some_query);
                assert_eq!(
                    result,
                    vec![FurthestQueue::from_ascending_vec(vec![(
                        some_vec,
                        some_distance
                    )])]
                );
            }
            _ => panic!("Expected SearchResult, got {:?}", op),
        }
    }
}
