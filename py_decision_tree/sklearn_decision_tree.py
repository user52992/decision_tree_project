from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from py_data.preprocess_covtype import preprocess_covtype
from py_data.preprocess_adult import preprocess_adult
from py_data.preprocess_kddcup99 import preprocess_kddcup99

def main():
    X_train, X_test, y_train, y_test = preprocess_kddcup99(0.2)

    model = DecisionTreeClassifier(random_state=42)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
