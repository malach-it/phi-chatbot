mod curve_plot;
mod dense_curve;
mod sparse_phi;

use crate::chatbot::{ChatModelMode, WeightedFeature};

use self::dense_curve::DenseCurveClassifier;
use self::sparse_phi::SparsePhiClassifier;

pub(crate) use self::curve_plot::{add_curve_points, draw_curve};
pub(crate) use self::sparse_phi::{
    control_points_from_piecewise_linear, control_points_from_polynomial, ensure_sparse_state,
    parse_float_list, parse_global_phi_expression, parse_index_list, parse_merged_phi_expression,
    remap_sparse_snapshot_states, SparsePhiKind, SparsePhiSnapshotState,
};

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

    pub(crate) fn from_sparse_snapshot(
        kind: SparsePhiKind,
        max_degree: usize,
        state: SparsePhiSnapshotState,
    ) -> Self {
        Self::Sparse(SparsePhiClassifier::from_snapshot(kind, max_degree, state))
    }

    pub(crate) fn write_phi_snapshot(
        &self,
        response_index: usize,
        output: &mut String,
    ) -> Option<()> {
        match self {
            Self::DenseCurve(_) => None,
            Self::Sparse(classifier) => {
                classifier.write_phi_snapshot(response_index, output);
                Some(())
            }
        }
    }

    pub(crate) fn merged_phi_expression(&self) -> Option<String> {
        match self {
            Self::DenseCurve(_) => None,
            Self::Sparse(classifier) => Some(classifier.merged_phi_expression()),
        }
    }

    pub(crate) fn aggregate_curve_points(&self) -> Option<Vec<f64>> {
        match self {
            Self::DenseCurve(classifier) => classifier.aggregate_curve_points(),
            Self::Sparse(classifier) => classifier.aggregate_curve_points(),
        }
    }
}
