from py_data.loaders import load_covtype
from sklearn.model_selection import train_test_split
from py_data.utils import export_to_csv

def preprocess_covtype(test_size=0.2):
    X, y = load_covtype()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    export_to_csv(X_train, "covtype_X_train")
    export_to_csv(X_test, "covtype_X_test")
    export_to_csv(y_train, "covtype_y_train")
    export_to_csv(y_test, "covtype_y_test")

    return X_train, X_test, y_train, y_test
