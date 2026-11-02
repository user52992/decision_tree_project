from pathlib import Path
from py_data.loaders import (
    load_covtype,
    load_adult,
    load_kddcup99,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "csv"

# This script is only used to generate CSV files for human inspection
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {
        "covtype": load_covtype,
        "adult": load_adult,
        "kddcup99": load_kddcup99,
    }

    for name, loader in datasets.items():
        print(f"Exporting {name}...")

        X, y = loader()

        df = X.copy()
        df["target"] = y

        output_path = OUTPUT_DIR / f"{name}.csv"
        df.to_csv(output_path, index=False)

        print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
