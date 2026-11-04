pub mod csv_loader;
pub mod decision_tree;

use decision_tree::DecisionTree;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::time::Instant;

/// Benchmark only the model operations, after PyO3 has converted the inputs.
#[pyfunction]
fn benchmark_rust(
    py: Python<'_>,
    x_train: Vec<Vec<f64>>,
    y_train: Vec<usize>,
    x_test: Vec<Vec<f64>>,
    max_depth: usize,
) -> PyResult<(Vec<usize>, f64, f64)> {
    if x_train.is_empty() || x_train.len() != y_train.len() {
        return Err(PyValueError::new_err(
            "Training features and labels must have the same nonzero length",
        ));
    }
    let n_features = x_train[0].len();
    if n_features == 0
        || x_train
            .iter()
            .chain(&x_test)
            .any(|row| row.len() != n_features || row.iter().any(|value| !value.is_finite()))
    {
        return Err(PyValueError::new_err(
            "Feature matrices must have the same positive width and contain only finite values",
        ));
    }

    Ok(py.detach(move || {
        let mut tree = DecisionTree::new(max_depth);
        let start = Instant::now();
        tree.fit(&x_train, &y_train);
        let train_seconds = start.elapsed().as_secs_f64();

        let start = Instant::now();
        let predictions = tree.predict(&x_test);
        let predict_seconds = start.elapsed().as_secs_f64();

        (predictions, train_seconds, predict_seconds)
    }))
}

#[pymodule]
fn decision_tree_project(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(benchmark_rust, module)?)
}
