import pandas as pd
from py_data.loaders import load_kddcup99
from sklearn.model_selection import train_test_split
from py_data.utils import export_to_csv

def decode_bytes(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value

def preprocess_kddcup99(test_size=0.2):
    X, y = load_kddcup99()
    X = X.copy()

    # 1. Decode categorical features
    categorical_cols = ["protocol_type", "service", "flag"]

    for col in categorical_cols:
        X[col] = X[col].map(decode_bytes)

    # 2. Ensure other features are numeric
    for col in X.columns:
        if col not in categorical_cols:
            X[col] = pd.to_numeric(X[col])

    # 3. Decode and encode target labels
    y = y.map(decode_bytes)

    y_encoded, class_names = pd.factorize(y, sort=True)
    y = pd.Series(y_encoded, index=y.index, name="target")

    if (y == -1).any():
        raise ValueError("Missing labels in KDDCup99")

    # 4. Split training and test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    # 5. One-Hot Encoding
    X_train = pd.get_dummies(
        X_train,
        columns=categorical_cols,
        dtype="uint8",
    )

    X_test = pd.get_dummies(
        X_test,
        columns=categorical_cols,
        dtype="uint8",
    )

    # 6. Ensure matching feature columns
    X_test = X_test.reindex(
        columns=X_train.columns,
        fill_value=0,
    )

    # 7. Validate the preprocessed data
    assert not X_train.isna().any().any()
    assert not X_test.isna().any().any()
    assert X_train.columns.equals(X_test.columns)

    # 8. Export processed data to CSV
    export_to_csv(X_train, "kddcup99_X_train")
    export_to_csv(X_test, "kddcup99_X_test")
    export_to_csv(y_train, "kddcup99_y_train")
    export_to_csv(y_test, "kddcup99_y_test")

    # 9. Save the mapping between class IDs and original labels
    label_mapping = pd.DataFrame({
        "class_id": range(len(class_names)),
        "original_label": class_names,
    })
    export_to_csv(label_mapping, "kddcup99_labels")

    return X_train, X_test, y_train, y_test
