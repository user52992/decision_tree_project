from py_data.loaders import (
    load_covtype,
    load_adult,
    load_kddcup99,
)

def main():
    datasets = {
        "Covertype": load_covtype,
        "Adult": load_adult,
        "KDDCup99": load_kddcup99,
    }

    for name, loader in datasets.items():
        print(f"Loading {name}...")

        X, y = loader()

        print(f"{name}: {X.shape[0]} samples, {X.shape[1]} features")

if __name__ == "__main__":
    main()
