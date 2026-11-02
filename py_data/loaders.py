from pathlib import Path
from sklearn.datasets import (
    fetch_covtype,
    fetch_openml,
    fetch_kddcup99,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_HOME = PROJECT_ROOT / "data" / "raw"


def load_covtype():
    return fetch_covtype(
        data_home=DATA_HOME,
        return_X_y=True,
        as_frame=True,
    )


def load_adult():
    return fetch_openml(
        "adult",
        version=2,
        data_home=DATA_HOME,
        return_X_y=True,
        as_frame=True,
    )


def load_kddcup99():
    return fetch_kddcup99(
        data_home=DATA_HOME,
        return_X_y=True,
        as_frame=True,
    )
