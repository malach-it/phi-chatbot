mod curve_plot;
mod dense_curve;
mod sparse_phi;

use crate::chatbot::{ChatModelMode, WeightedFeature};

use self::dense_curve::DenseCurveClassifier;
use self::sparse_phi::SparsePhiClassifier;

pub(crate) use self::curve_plot::{add_curve_points, draw_curve};
pub(crate) use self::sparse_phi::SparsePhiKind;

#[derive(Debug)]
pub(crate) struct EncodedChatExample {
    pub(crate) features: Vec<WeightedFeature>,
    pub(crate) response: String,
}

#[derive(Debug)]
pub(crate) enum ChatClassifier {
    DenseCurve(DenseCurveClassifier),
    Sparse(SparsePhiClassifier),
}

impl ChatClassifier {
    pub(crate) fn train(
        mode: ChatModelMode,
        examples: &[EncodedChatExample],
        positive_response: &str,
        epochs: usize,
        epsilon: f64,
        input_size: usize,
        max_degree: usize,
    ) -> Self {
        match mode {
            ChatModelMode::DenseCurve => {
                let mut classifier = DenseCurveClassifier::new(input_size, 0.08, max_degree);
                classifier.train(examples, positive_response, epochs, epsilon);
                Self::DenseCurve(classifier)
            }
            ChatModelMode::SparseScalar => {
                let mut classifier =
                    SparsePhiClassifier::new(0.08, max_degree, SparsePhiKind::Scalar);
                classifier.train(examples, positive_response, epochs, epsilon);
                Self::Sparse(classifier)
            }
            ChatModelMode::SparseCurve => {
                let mut classifier =
                    SparsePhiClassifier::new(0.08, max_degree, SparsePhiKind::Curve);
                classifier.train(examples, positive_response, epochs, epsilon);
                Self::Sparse(classifier)
            }
        }
    }

    pub(crate) fn predict(&self, features: &[WeightedFeature], input_size: usize) -> f64 {
        match self {
            Self::DenseCurve(classifier) => classifier.predict(features, input_size),
            Self::Sparse(classifier) => classifier.predict(features),
        }
    }

    pub(crate) fn aggregate_curve_points(&self) -> Option<Vec<f64>> {
        match self {
            Self::DenseCurve(classifier) => classifier.aggregate_curve_points(),
            Self::Sparse(classifier) => classifier.aggregate_curve_points(),
        }
    }
}
