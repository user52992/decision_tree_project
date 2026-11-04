use decision_tree_project::csv_loader::{read_features, read_labels};
use decision_tree_project::decision_tree::{DecisionTree, accuracy};
use std::{env, error::Error};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    // Choose "covtype", "adult", or "kddcup99".
    let dataset = args.get(1).ok_or("Missed dataset name")?;

    let x_train = read_features(&format!("{dataset}_X_train"))?;
    let y_train = read_labels(&format!("{dataset}_y_train"))?;
    let x_test = read_features(&format!("{dataset}_X_test"))?;
    let y_test = read_labels(&format!("{dataset}_y_test"))?;

    let mut tree = DecisionTree::new(10);
    tree.fit(&x_train, &y_train);

    let predictions = tree.predict(&x_test);
    let accuracy = accuracy(&y_test, &predictions);
    println!("Accuracy: {:.4}", accuracy);

    Ok(())
}
