use crate::chatbot::WeightedFeature;
use crate::phinetwork::{PhiNetwork, TrainingExample};

use super::curve_plot::add_curve_points;
use super::EncodedChatExample;

#[derive(Debug)]
pub(crate) struct DenseCurveClassifier {
    network: PhiNetwork,
    input_size: usize,
    max_degree: usize,
}

impl DenseCurveClassifier {
    pub(super) fn new(input_size: usize, learning_rate: f64, max_degree: usize) -> Self {
        Self {
            network: PhiNetwork::new(input_size, learning_rate),
            input_size,
            max_degree,
        }
    }

    pub(super) fn train(
        &mut self,
        examples: &[EncodedChatExample],
        positive_response: &str,
        epochs: usize,
        epsilon: f64,
    ) {
        let training_data = examples
            .iter()
            .map(|example| TrainingExample {
                inputs: dense_features(&example.features, self.input_size),
                target: if example.response == positive_response {
                    1.0
                } else {
                    0.0
                },
            })
            .collect::<Vec<_>>();

        self.network
            .train_until_quiet(&training_data, epsilon, epochs, self.max_degree);
    }

    pub(super) fn predict(&self, features: &[WeightedFeature], input_size: usize) -> f64 {
        self.network.predict(&dense_features(features, input_size))
    }

    pub(super) fn aggregate_curve_points(&self) -> Option<Vec<f64>> {
        let mut aggregate = None::<Vec<f64>>;

        for term_index in 0..self.network.term_count() {
            if let Some(points) = self.network.curve_points(term_index) {
                add_curve_points(&mut aggregate, points);
            }
        }

        aggregate
    }
}

fn dense_features(features: &[WeightedFeature], input_size: usize) -> Vec<f64> {
    let mut dense = vec![0.0; input_size];

    for feature in features {
        if let Some(slot) = dense.get_mut(feature.index) {
            *slot = feature.value.clamp(0.0, 1.0);
        }
    }

    dense
}
