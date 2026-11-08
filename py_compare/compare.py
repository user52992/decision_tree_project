import argparse
from importlib.machinery import ExtensionFileLoader
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import NamedTuple
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

MIN_DEPTH = 5
DEPTH_STEP = 3
BENCHMARK_REPEATS = 3
RESULTS_PATH = PROJECT_ROOT / "data" / "benchmark" / "results.json"

class BenchmarkResult(NamedTuple):
    train_seconds: float
    test_seconds: float
    train_accuracy: float
    test_accuracy: float

def load_rust_extension():
    """Build and load the PyO3 extension with the current Python interpreter."""
    module_name = "decision_tree_project"
    if module_name in sys.modules:
        return sys.modules[module_name]

    env = os.environ.copy()
    env["PYO3_PYTHON"] = sys.executable
    env["PYO3_BUILD_EXTENSION_MODULE"] = "1"
    # Keep CLI builds separate from DLLs loaded by older notebook sessions.
    target_dir = PROJECT_ROOT / "target" / "py_compare"
    subprocess.run(
        [
            "cargo", "build", "--release", "--lib",
            "--manifest-path", str(PROJECT_ROOT / "Cargo.toml"),
            "--target-dir", str(target_dir),
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
    library_path = target_dir / "release" / filename
    loader = ExtensionFileLoader(module_name, str(library_path))
    spec = spec_from_file_location(module_name, library_path, loader=loader)
    module = module_from_spec(spec)
    loader.exec_module(module)
    sys.modules[module_name] = module
    return module

def compare_dataset(preprocess, rust_module, max_depth=50, test_size=0.2):
    """Use one data split for all depths and average three fresh models per depth."""
    x_train, x_test, y_train, y_test = preprocess(test_size=test_size)
    # sklearn uses float32 internally; give both models those same feature values.
    x_train = np.ascontiguousarray(x_train, dtype=np.float32)
    x_test = np.ascontiguousarray(x_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)

    results = []
    for depth in range(MIN_DEPTH, max_depth + 1, DEPTH_STEP):
        rust_runs = []
        sklearn_runs = []
        for _ in range(BENCHMARK_REPEATS):
            # PyO3 conversion and training-set prediction are outside the timers.
            train_predictions, test_predictions, train_seconds, test_seconds = (
                rust_module.benchmark_rust(x_train, y_train, x_test, depth)
            )
            rust_runs.append(BenchmarkResult(
                train_seconds, test_seconds,
                accuracy_score(y_train, train_predictions),
                accuracy_score(y_test, test_predictions),
            ))

            model = DecisionTreeClassifier(
                criterion="gini", splitter="best", max_depth=depth, random_state=42,
            )
            start = perf_counter()
            model.fit(x_train, y_train)
            train_seconds = perf_counter() - start

            start = perf_counter()
            test_predictions = model.predict(x_test)
            test_seconds = perf_counter() - start

            train_predictions = model.predict(x_train)
            sklearn_runs.append(BenchmarkResult(
                train_seconds, test_seconds,
                accuracy_score(y_train, train_predictions),
                accuracy_score(y_test, test_predictions),
            ))

        results.append((
            depth,
            BenchmarkResult(*np.mean(rust_runs, axis=0)),
            BenchmarkResult(*np.mean(sklearn_runs, axis=0)),
        ))
    return results


def print_dataset_tables(name, results):
    """Print separate timing and accuracy tables with one row per depth."""
    print(f"\nDataset: {name} (mean of {BENCHMARK_REPEATS} runs per model/depth)")
    print("Training / test time (seconds)")
    print(
        f"{'Max depth':>9} {'sklearn train':>18} {'sklearn test':>18} "
        f"{'Rust train':>18} {'Rust test':>18}"
    )
    for depth, rust, sklearn in results:
        print(
            f"{depth:>9} {sklearn.train_seconds:>18.6f} {sklearn.test_seconds:>18.6f} "
            f"{rust.train_seconds:>18.6f} {rust.test_seconds:>18.6f}"
        )

    print("\nTraining / test accuracy (%)")
    print(
        f"{'Max depth':>9} {'sklearn train':>18} {'sklearn test':>18} "
        f"{'Rust train':>18} {'Rust test':>18}"
    )
    for depth, rust, sklearn in results:
        print(
            f"{depth:>9} {sklearn.train_accuracy:>18.4%} {sklearn.test_accuracy:>18.4%} "
            f"{rust.train_accuracy:>18.4%} {rust.test_accuracy:>18.4%}"
        )
    sys.stdout.flush()


def save_benchmark_results(results_by_dataset, output_path=RESULTS_PATH, test_size=0.2):
    """Save the same averaged results used by the tables for the plotting module."""
    payload = {
        "repeats": BENCHMARK_REPEATS,
        "test_size": test_size,
        "datasets": {
            name: [
                {
                    "max_depth": depth,
                    "rust": {key: float(value) for key, value in rust._asdict().items()},
                    "sklearn": {key: float(value) for key, value in sklearn._asdict().items()},
                }
                for depth, rust, sklearn in results
            ]
            for name, results in results_by_dataset.items()
        },
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-depth", type=int, default=50,
        help="Upper depth bound; sweep starts at 5 with step 3 (default: 5, 8, ..., 50).",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument(
        "--output", type=Path, default=RESULTS_PATH,
        help="Save averaged benchmark results as JSON for py_compare.plot.",
    )
    args = parser.parse_args()
    if args.max_depth < MIN_DEPTH:
        parser.error(f"--max-depth must be at least {MIN_DEPTH}")
    if not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1")

    rust_module = load_rust_extension()
    depths = list(range(MIN_DEPTH, args.max_depth + 1, DEPTH_STEP))
    print(
        f"max_depths={depths}, repeats={BENCHMARK_REPEATS}, test_size={args.test_size}",
        flush=True,
    )
    print(
        "Times include only fit and test-set predict; exclude preprocessing, "
        "PyO3 conversion, training-set prediction, and accuracy calculation.",
        flush=True,
    )
    results_by_dataset = {}
    for name in args.datasets:
        print(f"\nBenchmarking {name}...", flush=True)
        results = compare_dataset(
            DATASETS[name], rust_module, args.max_depth, args.test_size,
        )
        results_by_dataset[name] = results
        print_dataset_tables(name, results)
    save_benchmark_results(results_by_dataset, args.output, args.test_size)
    print(f"\nSaved benchmark results to {args.output}", flush=True)

if __name__ == "__main__":
    main()
