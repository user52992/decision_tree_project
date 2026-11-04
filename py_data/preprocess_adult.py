import pandas as pd
from py_data.loaders import load_adult
from sklearn.model_selection import train_test_split
from py_data.utils import export_to_csv

def preprocess_adult(test_size=0.2):
    X, y = load_adult()

    # 1. Encode labels: <=50K -> 0, >50K -> 1
    y = y.astype("string").str.strip().map({
        "<=50K": 0,
        ">50K": 1,
    })

    if y.isna().any():
        raise ValueError("Unknown or missing Adult labels")

    y = y.astype(int)

    # 2. Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    # 3. Identify numerical and categorical columns
    numerical_cols = X_train.select_dtypes(
        include="number"
    ).columns

    categorical_cols = X_train.select_dtypes(
        exclude="number"
    ).columns

    # 4. Fill missing numerical values
    for col in numerical_cols:
        median = X_train[col].median()

        X_train[col] = X_train[col].fillna(median)
        X_test[col] = X_test[col].fillna(median)

    # 5. Fill missing categorical values
    for col in categorical_cols:
        X_train[col] = X_train[col].astype("object").fillna("Missing")
        X_test[col] = X_test[col].astype("object").fillna("Missing")

    # 6. One-Hot Encoding
    X_train = pd.get_dummies(X_train, dtype=int)
    X_test = pd.get_dummies(X_test, dtype=int)

    # 7. Ensure training and test columns match
    X_test = X_test.reindex(
        columns=X_train.columns,
        fill_value=0,
    )

    # 8. Validate the preprocessed data
    assert not X_train.isna().any().any()
    assert not X_test.isna().any().any()

    # 9. Export processed data to CSV
    export_to_csv(X_train, "adult_X_train")
    export_to_csv(X_test, "adult_X_test")
    export_to_csv(y_train, "adult_y_train")
    export_to_csv(y_test, "adult_y_test")

    return X_train, X_test, y_train, y_test
