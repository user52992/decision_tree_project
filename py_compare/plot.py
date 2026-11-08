"""Plot saved decision-tree benchmark results without running the benchmark."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

DEFAULT_RESULTS_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "benchmark" / "results.json"
)


def plot_benchmark_results(results_path=DEFAULT_RESULTS_PATH):
    """Read saved means, show one sklearn/Rust figure per dataset, and return them."""
    results_path = Path(results_path)
    try:
        with results_path.open(encoding="utf-8") as results_file:
            benchmark = json.load(results_file)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Benchmark results not found: {results_path}. "
            "Run `python -m py_compare.compare` from the project root first."
        ) from error

    figures = []
    for dataset_name, dataset_results in benchmark["datasets"].items():
        if not dataset_results:
            raise ValueError(f"No benchmark rows found for dataset '{dataset_name}'.")

        depths = [row["max_depth"] for row in dataset_results]
        all_accuracies = [
            result[metric]
            for row in dataset_results
            for result in (row["rust"], row["sklearn"])
            for metric in ("train_accuracy", "test_accuracy")
        ]
        accuracy_padding = max(
            (max(all_accuracies) - min(all_accuracies)) * 0.1, 0.005,
        )
        accuracy_limits = (
            max(0, min(all_accuracies) - accuracy_padding),
            min(1, max(all_accuracies) + accuracy_padding),
        )
        time_upper = max(
            result[metric]
            for row in dataset_results
            for result in (row["rust"], row["sklearn"])
            for metric in ("train_seconds", "test_seconds")
        )
        time_upper = max(time_upper * 1.1, 1e-9)

        legend_handles = None
        fig, time_axes = plt.subplots(
            1, 2, figsize=(15, 5), sharey=True, layout="constrained",
        )
        for time_ax, model_name, model_key in zip(
            time_axes, ("sklearn", "Rust"), ("sklearn", "rust"),
        ):
            model_results = [row[model_key] for row in dataset_results]
            accuracy_ax = time_ax.twinx()
            time_ax.plot(
                depths, [result["train_seconds"] for result in model_results],
                color="tab:blue", marker="o", label="Training time",
            )
            time_ax.plot(
                depths, [result["test_seconds"] for result in model_results],
                color="tab:orange", marker="s", label="Test time",
            )
            accuracy_ax.plot(
                depths, [result["train_accuracy"] for result in model_results],
                color="tab:green", marker="^", linestyle="--", label="Training accuracy",
            )
            accuracy_ax.plot(
                depths, [result["test_accuracy"] for result in model_results],
                color="tab:red", marker="D", linestyle="--", label="Test accuracy",
            )
            time_ax.set(title=model_name, xlabel="max_depth", ylabel="Time (s)")
            time_ax.set_xticks(depths)
            time_ax.set_ylim(0, time_upper)
            time_ax.grid(alpha=0.25)
            accuracy_ax.set_ylabel("Accuracy (%)")
            accuracy_ax.set_ylim(*accuracy_limits)
            accuracy_ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=2))

            if legend_handles is None:
                legend_handles = time_ax.get_lines() + accuracy_ax.get_lines()

        fig.legend(
            legend_handles, [line.get_label() for line in legend_handles],
            loc="outside lower center", ncols=4, frameon=False,
        )
        fig.suptitle(
            f"{dataset_name} - mean of {benchmark['repeats']} runs per depth", fontsize=15,
        )
        figures.append(fig)
        plt.show()

    return figures

if __name__ == "__main__":
    plot_benchmark_results()
