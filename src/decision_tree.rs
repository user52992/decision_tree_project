use std::collections::HashMap;

pub enum Node {
    Leaf {
        prediction: usize,
    },

    Split {
        feature_index: usize,
        threshold: f64,
        left: Box<Node>,
        right: Box<Node>,
    },
}

pub struct DecisionTree {
    root: Option<Node>,
    max_depth: usize,
}

impl DecisionTree {
    pub fn new(max_depth: usize) -> Self {
        Self {
            root: None,
            max_depth,
        }
    }

    pub fn fit(&mut self, x: &[Vec<f64>], y: &[usize]) {
        assert!(!x.is_empty(), "Training data must not be empty");
        assert_eq!(
            x.len(),
            y.len(),
            "Features and labels must have the same length"
        );
        assert!(
            x.iter().all(|row| row.len() == x[0].len()),
            "All feature rows must have the same length"
        );
        assert!(
            x.iter().flatten().all(|value| value.is_finite()),
            "Feature values must be finite"
        );

        let rows: Vec<&[f64]> = x.iter().map(Vec::as_slice).collect();
        self.root = Some(self.build_tree(&rows, y, 0));
    }

    fn build_tree(&self, x: &[&[f64]], y: &[usize], depth: usize) -> Node {
        let counts = Self::class_counts(y);
        let prediction = Self::majority_class(y, &counts);

        if counts.len() == 1 || depth >= self.max_depth {
            return Node::Leaf { prediction };
        }

        let Some((feature_index, threshold)) = Self::find_best_split(x, y, &counts) else {
            return Node::Leaf { prediction };
        };

        let mut left_x = Vec::new();
        let mut left_y = Vec::new();
        let mut right_x = Vec::new();
        let mut right_y = Vec::new();

        for (&row, &label) in x.iter().zip(y) {
            if row[feature_index] <= threshold {
                left_x.push(row);
                left_y.push(label);
            } else {
                right_x.push(row);
                right_y.push(label);
            }
        }

        let left = Box::new(self.build_tree(&left_x, &left_y, depth + 1));
        let right = Box::new(self.build_tree(&right_x, &right_y, depth + 1));

        Node::Split {
            feature_index,
            threshold,
            left,
            right,
        }
    }

    fn class_counts(y: &[usize]) -> HashMap<usize, usize> {
        let mut counts = HashMap::new();

        for &label in y {
            let count = counts.entry(label).or_insert(0usize);
            *count += 1;
        }

        counts
    }

    fn find_best_split(
        x: &[&[f64]],
        y: &[usize],
        counts: &HashMap<usize, usize>,
    ) -> Option<(usize, f64)> {
        let n_features = x[0].len();
        let n = y.len() as f64;
        let squared_counts: f64 = counts.values().map(|&count| (count as f64).powi(2)).sum();
        let mut best_gini = 1.0 - squared_counts / (n * n);
        let mut best_split = None;

        for feature_index in 0..n_features {
            let mut values: Vec<(f64, usize)> = x
                .iter()
                .zip(y)
                .map(|(row, &label)| (row[feature_index], label))
                .collect();
            values.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
            if values[0].0 == values[values.len() - 1].0 {
                continue;
            }

            let mut left_counts = HashMap::new();
            let mut left_squared_counts = 0.0;
            let mut right_squared_counts = squared_counts;

            // Move samples once per feature instead of rescanning each threshold.
            for (index, pair) in values.windows(2).enumerate() {
                let (value, label) = pair[0];
                let next_value = pair[1].0;

                let left_count = left_counts.entry(label).or_insert(0usize);
                let right_count = counts[&label] - *left_count;
                left_squared_counts += 2.0 * *left_count as f64 + 1.0;
                right_squared_counts += 1.0 - 2.0 * right_count as f64;
                *left_count += 1;

                if value == next_value {
                    continue;
                }

                let left_n = (index + 1) as f64;
                let right_n = n - left_n;
                let weighted_gini =
                    1.0 - (left_squared_counts / left_n + right_squared_counts / right_n) / n;

                if weighted_gini < best_gini {
                    best_gini = weighted_gini;
                    let midpoint = value.midpoint(next_value);
                    // Rounding to the upper value would leave the right child empty.
                    let threshold = if midpoint < next_value {
                        midpoint
                    } else {
                        value
                    };
                    best_split = Some((feature_index, threshold));
                }
            }
        }
        best_split
    }

    fn majority_class(y: &[usize], counts: &HashMap<usize, usize>) -> usize {
        let mut prediction = y[0];
        let mut max_count = 0;

        for &label in y {
            if counts[&label] > max_count {
                max_count = counts[&label];
                prediction = label;
            }
        }

        prediction
    }

    fn predict_one(node: &Node, row: &[f64]) -> usize {
        match node {
            Node::Leaf { prediction } => *prediction,

            Node::Split {
                feature_index,
                threshold,
                left,
                right,
            } => {
                if row[*feature_index] <= *threshold {
                    Self::predict_one(left, row)
                } else {
                    Self::predict_one(right, row)
                }
            }
        }
    }

    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<usize> {
        let root = self
            .root
            .as_ref()
            .expect("Decision tree has not been fitted");

        x.iter().map(|row| Self::predict_one(root, row)).collect()
    }
}

pub fn accuracy(y_true: &[usize], y_pred: &[usize]) -> f64 {
    assert_eq!(
        y_true.len(),
        y_pred.len(),
        "Labels and predictions must have the same length"
    );
    assert!(!y_true.is_empty(), "Accuracy is undefined for empty inputs");

    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(true_label, pred_label)| true_label == pred_label)
        .count();

    correct as f64 / y_true.len() as f64
}
