use decision_tree_project::csv_loader::{read_features, read_labels};
use decision_tree_project::decision_tree::{DecisionTree, accuracy};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let x_train = read_features("covtype_X_train")?;
    let y_train = read_labels("covtype_y_train")?;

    let x_test = read_features("covtype_X_test")?;
    let y_test = read_labels("covtype_y_test")?;

    println!("Samples: {}", x_train.len());
    println!("Features: {}", x_train.first().map_or(0, Vec::len));
    println!("Labels: {}", y_train.len());

    let mut tree = DecisionTree::new(10);
    tree.fit(&x_train, &y_train);

    let predictions = tree.predict(&x_test);

    let accuracy = accuracy(&y_test, &predictions);

    println!("Accuracy: {:.4}", accuracy);

    Ok(())
}
