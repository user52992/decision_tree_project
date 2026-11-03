use decision_tree_project::csv_loader::{read_features, read_labels};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let X_train = read_features("covtype_X_train")?;
    let y_train = read_labels("covtype_y_train")?;

    println!("Samples: {}", X_train.len());
    println!("Features: {}", X_train.first().map_or(0, Vec::len));
    println!("Labels: {}", y_train.len());

    Ok(())
}
