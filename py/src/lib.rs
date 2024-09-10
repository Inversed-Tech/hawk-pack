use pyo3::prelude::*;
use std::sync::Mutex;

#[pyclass]
struct MyRustLibrary {
    state: Mutex<i32>,
    // db_connection: Option<DatabaseConnection>, // Add your database connection type here
}

#[pymethods]
impl MyRustLibrary {
    #[new]
    fn new() -> Self {
        MyRustLibrary {
            state: Mutex::new(0),
            // db_connection: None, // Initialize your database connection here
        }
    }

    fn set_state(&self, value: i32) {
        let mut state = self.state.lock().unwrap();
        *state = value;
    }

    fn get_state(&self) -> i32 {
        let state = self.state.lock().unwrap();
        *state
    }

    /* fn connect_to_db(&self, connection_str: &str) {
        // Initialize and store a database connection
    } */
}

#[pymodule]
fn hawk_pack_py(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<MyRustLibrary>()?;
    Ok(())
}
