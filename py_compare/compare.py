import argparse
from importlib.machinery import ExtensionFileLoader
from importlib.util import module_from_spec, spec_from_file_location
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if not __package__:
    sys.path.insert(0, str(PROJECT_ROOT))

from py_data.preprocess_adult import preprocess_adult
from py_data.preprocess_covtype import preprocess_covtype
from py_data.preprocess_kddcup99 import preprocess_kddcup99

DATASETS = {
    "covtype": preprocess_covtype,
    "adult": preprocess_adult,
    "kddcup99": preprocess_kddcup99,
}

def load_rust_extension():
    """Build and load the PyO3 extension with the current Python interpreter."""
    module_name = "decision_tree_project"
    if module_name in sys.modules:
        return sys.modules[module_name]

    env = os.environ.copy()
    env["PYO3_PYTHON"] = sys.executable
    env["PYO3_BUILD_EXTENSION_MODULE"] = "1"
    subprocess.run(
        [
            "cargo", "build", "--release", "--lib",
            "--manifest-path", str(PROJECT_ROOT / "Cargo.toml"),
            "--target-dir", str(PROJECT_ROOT / "target"),
        ],
        check=True,
        env=env,
    )

    if sys.platform == "win32":
        filename = f"{module_name}.dll"
    elif sys.platform == "darwin":
        filename = f"lib{module_name}.dylib"
    else:
        filename = f"lib{module_name}.so"
    library_path = PROJECT_ROOT / "target" / "release" / filename
    loader = ExtensionFileLoader(module_name, str(library_path))
    spec = spec_from_file_location(module_name, library_path, loader=loader)
    module = module_from_spec(spec)
    loader.exec_module(module)
    sys.modules[module_name] = module
    return module

def compare_dataset(preprocess, rust_module, max_depth=10, test_size=0.2):
    x_train, x_test, y_train, y_test = preprocess(test_size=test_size)
    # sklearn uses float32 internally; give both models those same feature values.
    x_train = np.ascontiguousarray(x_train, dtype=np.float32)
    x_test = np.ascontiguousarray(x_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)

    # PyO3 extracts NumPy arrays directly, avoiding a large intermediate .tolist().
    predictions, train_seconds, predict_seconds = rust_module.benchmark_rust(
        x_train, y_train, x_test, max_depth,
    )
    rust_result = (
        "Rust", train_seconds, predict_seconds, accuracy_score(y_test, predictions),
    )

    model = DecisionTreeClassifier(
        criterion="gini", splitter="best", max_depth=max_depth, random_state=42,
    )
    start = perf_counter()
    model.fit(x_train, y_train)
    train_seconds = perf_counter() - start

    start = perf_counter()
    predictions = model.predict(x_test)
    predict_seconds = perf_counter() - start
    python_result = (
        "Python/sklearn", train_seconds, predict_seconds,
        accuracy_score(y_test, predictions),
    )
    return rust_result, python_result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    args = parser.parse_args()
    if args.max_depth < 1:
        parser.error("--max-depth must be at least 1")
    if not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1")

    rust_module = load_rust_extension()
    print(f"max_depth={args.max_depth}, test_size={args.test_size}", flush=True)
    print("Times include only model fit/predict; exclude preprocessing and PyO3 conversion.")
    print(f"{'Dataset':<12} {'Model':<16} {'Train (s)':>12} {'Predict (s)':>12} {'Accuracy':>10}")
    for name in args.datasets:
        # print(f"Preparing {name}...", flush=True)
        for model, train_seconds, predict_seconds, accuracy in compare_dataset(
            DATASETS[name], rust_module, args.max_depth, args.test_size,
        ):
            print(
                f"{name:<12} {model:<16} {train_seconds:>12.6f} "
                f"{predict_seconds:>12.6f} {accuracy:>10.4%}",
                flush=True,
            )

if __name__ == "__main__":
    main()
