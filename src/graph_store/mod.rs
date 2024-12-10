pub mod graph_mem;
pub use graph_mem::GraphMem;

// Postgres database graph store implementation
#[cfg(feature = "db_dependent")]
pub mod graph_pg;
#[cfg(feature = "db_dependent")]
pub use graph_pg::{test_utils::TestGraphPg, GraphPg};

use serde::{Deserialize, Serialize};

/// Utility struct, not exposed externally because it doesn't have enough use
/// semantically to add an extra abstraction to the public interface.
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct EntryPoint<VectorRef> {
    pub point: VectorRef,
    pub layer: usize,
}
