# README

## Activate the python virtual environment

```
.venv\Scripts\activate
```

## Download the datasets

```
python -m py_data.download_data
```

## Export to CSV

```
python -m py_data.export_csv
```

## Run the scikit-learn decision tree

```
python -m py_decision_tree.sklearn_decision_tree
```

## Compare decision trees

```
python -m py_compare.compare
```

For each dataset, benchmark sklearn and Rust decision trees at maximum depths 5, 8, ..., 50 (upper bound 50, step 3). Each model/depth is run three times on the same train/test split. The output includes two tables per dataset: mean training/test time in seconds, and mean training/test accuracy in percent. Timing excludes preprocessing, PyO3 conversion, training-set prediction, and accuracy calculation. Use `--datasets adult` to select a dataset, `--max-depth` to change the sweep's upper bound, or `--test-size` to change the test split fraction.

