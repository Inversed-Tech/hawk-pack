use crate::VectorStore;

#[derive(Clone)]
pub struct LinearDb<V: VectorStore> {
    pub store: V,
    vectors: Vec<V::VectorRef>,
}

impl<V: VectorStore> LinearDb<V> {
    pub fn new(store: V) -> Self {
        LinearDb {
            store,
            vectors: vec![],
        }
    }

    pub async fn insert(&mut self, query: &V::QueryRef) -> bool {
        if self.exists(query).await {
            return false;
        }

        let vector = self.store.insert(query).await;
        self.vectors.push(vector);
        true
    }

    async fn exists(&mut self, query: &V::QueryRef) -> bool {
        for vector in &self.vectors {
            let distance = self.store.eval_distance(query, vector).await;
            if self.store.is_match(&distance).await {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examples::lazy_memory_store::LazyMemoryStore;

    #[tokio::test]
    async fn test_linear_db() {
        let store = LazyMemoryStore::new();
        let mut db = LinearDb::new(store);

        let query = db.store.prepare_query(123);

        assert!(db.insert(&query).await);
        assert!(!db.insert(&query).await);
    }
}
