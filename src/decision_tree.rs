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
    root: Option<Box<Node>>,
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
        assert!(!x.is_empty());
        assert_eq!(x.len(), y.len());

        let root = self.build_tree(x, y, 0);
        self.root = Some(Box::new(root));
    }

    fn build_tree(&self, x: &[Vec<f64>], y: &[usize], depth: usize) -> Node {
        let prediction = Self::majority_class(y);

        if Self::gini(y) == 0.0 || depth >= self.max_depth {
            return Node::Leaf {
                prediction: prediction,
            };
        }

        let Some((feature_index, threshold)) = Self::find_best_split(x, y) else {
            return Node::Leaf { prediction };
        };

        let mut left_x = Vec::new();
        let mut left_y = Vec::new();
        let mut right_x = Vec::new();
        let mut right_y = Vec::new();

        for (row, &label) in x.iter().zip(y.iter()) {
            if row[feature_index] <= threshold {
                left_x.push(row.clone());
                left_y.push(label);
            } else {
                right_x.push(row.clone());
                right_y.push(label)
            }
        }

        if left_y.is_empty() || right_y.is_empty() {
            return Node::Leaf { prediction };
        }

        let n = y.len() as f64;

        let weighted_gini = left_y.len() as f64 / n * Self::gini(&left_y)
            + right_y.len() as f64 / n * Self::gini(&right_y);

        if weighted_gini >= Self::gini(y) {
            return Node::Leaf { prediction };
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

    fn gini(y: &[usize]) -> f64 {
        if y.is_empty() {
            return 0.0;
        }

        let mut counts = HashMap::new();

        for &label in y {
            let count = counts.entry(label).or_insert(0usize);
            *count += 1;
        }

        let n = y.len() as f64;
        let mut sum = 0.0;

        for &count in counts.values() {
            let p = count as f64 / n;
            sum += p * p;
        }

        1.0 - sum
    }

    fn find_best_split(x: &[Vec<f64>], y: &[usize]) -> Option<(usize, f64)> {
        let n_features = x[0].len();

        let mut best_gini = f64::INFINITY;
        let mut best_split = None;

        for feature_index in 0..n_features {
            let mut values: Vec<f64> = x.iter().map(|row| row[feature_index]).collect();

            values.sort_by(|a, b| a.total_cmp(b));
            values.dedup();

            for pair in values.windows(2) {
                let threshold: f64 = (pair[1] + pair[0]) / 2.0;

                let mut left_y = Vec::new();
                let mut right_y = Vec::new();

                for (row, &label) in x.iter().zip(y.iter()) {
                    if row[feature_index] <= threshold {
                        left_y.push(label);
                    } else {
                        right_y.push(label);
                    }
                }

                if left_y.is_empty() || right_y.is_empty() {
                    continue;
                }

                let n = y.len() as f64;

                let weighted_gini = left_y.len() as f64 / n * Self::gini(&left_y)
                    + right_y.len() as f64 / n * Self::gini(&right_y);

                if weighted_gini < best_gini {
                    best_gini = weighted_gini;
                    best_split = Some((feature_index, threshold));
                }
            }
        }
        best_split
    }

    fn majority_class(y: &[usize]) -> usize {
        let mut counts = HashMap::new();

        for &label in y {
            let count = counts.entry(label).or_insert(0usize);
            *count += 1;
        }

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
    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(true_label, pred_label)| true_label == pred_label)
        .count();

    correct as f64 / y_true.len() as f64
}
