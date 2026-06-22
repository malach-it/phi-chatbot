use std::collections::HashMap;

use crate::chatbot::{ChatModelMode, WeightedFeature};

use super::curve_plot::add_curve_points;
use super::EncodedChatExample;

#[derive(Clone, Copy, Debug)]
pub(crate) enum SparsePhiKind {
    Scalar,
    Curve,
}

impl SparsePhiKind {
    pub(crate) fn for_mode(mode: ChatModelMode) -> Option<Self> {
        match mode {
            ChatModelMode::DenseCurve => None,
            ChatModelMode::SparseScalar => Some(Self::Scalar),
            ChatModelMode::SparseCurve => Some(Self::Curve),
        }
    }
}

#[derive(Debug)]
pub(crate) struct SparsePhiClassifier {
    weights: HashMap<Vec<usize>, f64>,
    curves: HashMap<Vec<usize>, SparsePhiCurve>,
    learning_rate: f64,
    max_degree: usize,
    kind: SparsePhiKind,
}

impl SparsePhiClassifier {
    pub(super) fn new(learning_rate: f64, max_degree: usize, kind: SparsePhiKind) -> Self {
        Self {
            weights: HashMap::new(),
            curves: HashMap::new(),
            learning_rate,
            max_degree,
            kind,
        }
    }

    pub(super) fn from_snapshot(
        kind: SparsePhiKind,
        max_degree: usize,
        state: SparsePhiSnapshotState,
    ) -> Self {
        Self {
            weights: state.weights.into_iter().collect(),
            curves: state
                .curves
                .into_iter()
                .map(|(key, control_points)| (key, SparsePhiCurve { control_points }))
                .collect(),
            learning_rate: 0.08,
            max_degree,
            kind,
        }
    }

    pub(super) fn train(
        &mut self,
        examples: &[EncodedChatExample],
        positive_response: &str,
        epochs: usize,
        epsilon: f64,
    ) {
        let cached_examples = examples
            .iter()
            .map(|example| CachedSparseExample {
                target: if example.response == positive_response {
                    1.0
                } else {
                    0.0
                },
                active_terms: active_terms(&example.features, self.max_degree),
            })
            .collect::<Vec<_>>();

        for epoch in 0..epochs {
            let mut max_error: f64 = 0.0;

            for example in &cached_examples {
                let prediction = self.predict_from_terms(&example.active_terms);
                let error = example.target - prediction;
                max_error = max_error.max(error.abs());

                for term in &example.active_terms {
                    match self.kind {
                        SparsePhiKind::Scalar => {
                            *self.weights.entry(term.key.clone()).or_insert(0.0) +=
                                self.learning_rate * error * term.value;
                        }
                        SparsePhiKind::Curve => {
                            self.curves
                                .entry(term.key.clone())
                                .or_insert_with(SparsePhiCurve::new)
                                .train(term.value, error, self.learning_rate);
                        }
                    }
                }
            }

            if epoch > 0 && max_error <= epsilon {
                break;
            }
        }
    }

    pub(super) fn predict(&self, features: &[WeightedFeature]) -> f64 {
        let terms = active_terms(features, self.max_degree);
        self.predict_from_terms(&terms)
    }

    fn predict_from_terms(&self, terms: &[SparseTerm]) -> f64 {
        terms
            .iter()
            .map(|term| match self.kind {
                SparsePhiKind::Scalar => {
                    self.weights.get(&term.key).copied().unwrap_or(0.0) * term.value
                }
                SparsePhiKind::Curve => self
                    .curves
                    .get(&term.key)
                    .map(|curve| curve.value(term.value))
                    .unwrap_or(0.0),
            })
            .sum()
    }

    pub(super) fn write_phi_snapshot(&self, response_index: usize, output: &mut String) {
        let mut weights = self.weights.iter().collect::<Vec<_>>();
        weights.sort_by(|(left_key, _), (right_key, _)| left_key.cmp(right_key));

        for (term_key, value) in weights {
            output.push_str(&format!(
                "weight\t{response_index}\t{}\t{value:.12}\n",
                format_index_list(term_key)
            ));
        }

        let mut curves = self.curves.iter().collect::<Vec<_>>();
        curves.sort_by(|(left_key, _), (right_key, _)| left_key.cmp(right_key));

        if !curves.is_empty() {
            output.push_str(&format!(
                "phi\t{response_index}\tsum\t{}\n",
                merged_phi_expression(&curves)
            ));
        }
    }

    pub(super) fn merged_phi_expression(&self) -> String {
        let mut curves = self.curves.iter().collect::<Vec<_>>();
        curves.sort_by(|(left_key, _), (right_key, _)| left_key.cmp(right_key));

        merged_phi_expression(&curves)
    }

    pub(super) fn aggregate_curve_points(&self) -> Option<Vec<f64>> {
        match self.kind {
            SparsePhiKind::Scalar => {
                let weight_sum = self.weights.values().sum::<f64>();
                Some(
                    (0..8)
                        .map(|index| {
                            let x = index as f64 / 7.0;
                            weight_sum * x
                        })
                        .collect(),
                )
            }
            SparsePhiKind::Curve => {
                let mut aggregate = None::<Vec<f64>>;

                for curve in self.curves.values() {
                    add_curve_points(&mut aggregate, &curve.control_points);
                }

                aggregate
            }
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct SparsePhiSnapshotState {
    pub(crate) weights: Vec<(Vec<usize>, f64)>,
    pub(crate) curves: Vec<(Vec<usize>, Vec<f64>)>,
}

pub(crate) fn ensure_sparse_state(states: &mut Vec<SparsePhiSnapshotState>, index: usize) {
    while states.len() <= index {
        states.push(SparsePhiSnapshotState::default());
    }
}

pub(crate) fn remap_sparse_snapshot_states(
    mut states: Vec<SparsePhiSnapshotState>,
    snapshot_vocabulary: &[String],
    snapshot_responses: &[String],
    current_vocabulary: &[String],
    current_responses: &[String],
) -> Option<Vec<SparsePhiSnapshotState>> {
    let mut remapped_states = (0..current_responses.len())
        .map(|_| SparsePhiSnapshotState::default())
        .collect::<Vec<_>>();

    for response in snapshot_responses {
        current_responses.binary_search(response).ok()?;
    }

    for snapshot_response_index in 0..snapshot_responses.len() {
        let current_response_index = current_responses
            .binary_search(&snapshot_responses[snapshot_response_index])
            .ok()?;

        if snapshot_response_index >= states.len() {
            continue;
        }

        let state = std::mem::take(&mut states[snapshot_response_index]);

        for (term_key, value) in state.weights {
            remapped_states[current_response_index].weights.push((
                remap_term_key(&term_key, snapshot_vocabulary, current_vocabulary)?,
                value,
            ));
        }

        for (term_key, points) in state.curves {
            remapped_states[current_response_index].curves.push((
                remap_term_key(&term_key, snapshot_vocabulary, current_vocabulary)?,
                points,
            ));
        }
    }

    Some(remapped_states)
}

fn remap_term_key(
    term_key: &[usize],
    snapshot_vocabulary: &[String],
    current_vocabulary: &[String],
) -> Option<Vec<usize>> {
    term_key
        .iter()
        .map(|index| {
            snapshot_vocabulary
                .get(*index)
                .and_then(|feature| current_vocabulary.binary_search(feature).ok())
        })
        .collect()
}

#[derive(Debug)]
struct SparsePhiCurve {
    control_points: Vec<f64>,
}

impl SparsePhiCurve {
    fn new() -> Self {
        Self {
            control_points: vec![0.0; 8],
        }
    }

    fn value(&self, input: f64) -> f64 {
        let input = input.clamp(0.0, 1.0);
        let scaled = input * (self.control_points.len() - 1) as f64;
        let lower_index = scaled.floor() as usize;
        let upper_index = (lower_index + 1).min(self.control_points.len() - 1);
        let upper_weight = scaled - lower_index as f64;
        let lower_weight = 1.0 - upper_weight;

        input
            * (self.control_points[lower_index] * lower_weight
                + self.control_points[upper_index] * upper_weight)
    }

    fn train(&mut self, input: f64, error: f64, learning_rate: f64) {
        let input = input.clamp(0.0, 1.0);
        let scaled = input * (self.control_points.len() - 1) as f64;
        let lower_index = scaled.floor() as usize;
        let upper_index = (lower_index + 1).min(self.control_points.len() - 1);
        let upper_weight = scaled - lower_index as f64;
        let lower_weight = 1.0 - upper_weight;

        self.control_points[lower_index] += learning_rate * error * input * lower_weight;
        self.control_points[upper_index] += learning_rate * error * input * upper_weight;
    }

    fn polynomial_formula(&self) -> String {
        polynomial_formula(&self.control_points)
    }
}

#[derive(Debug)]
struct CachedSparseExample {
    target: f64,
    active_terms: Vec<SparseTerm>,
}

#[derive(Clone, Debug)]
struct SparseTerm {
    key: Vec<usize>,
    value: f64,
}

fn active_terms(features: &[WeightedFeature], max_degree: usize) -> Vec<SparseTerm> {
    let mut terms = Vec::new();
    let mut current = Vec::<WeightedFeature>::new();

    for degree in 1..=max_degree.min(features.len()) {
        add_active_term_combinations(features, degree, 0, &mut current, &mut terms);
    }

    terms
}

fn add_active_term_combinations(
    features: &[WeightedFeature],
    degree: usize,
    start: usize,
    current: &mut Vec<WeightedFeature>,
    terms: &mut Vec<SparseTerm>,
) {
    if current.len() == degree {
        terms.push(SparseTerm {
            key: current.iter().map(|feature| feature.index).collect(),
            value: current.iter().map(|feature| feature.value).product(),
        });
        return;
    }

    for index in start..features.len() {
        current.push(features[index].clone());
        add_active_term_combinations(features, degree, index + 1, current, terms);
        current.pop();
    }
}

fn format_index_list(values: &[usize]) -> String {
    values
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

pub(crate) fn parse_index_list(encoded: &str) -> Option<Vec<usize>> {
    if encoded.is_empty() {
        return Some(Vec::new());
    }

    encoded
        .split(',')
        .map(|value| value.parse::<usize>().ok())
        .collect()
}

pub(crate) fn parse_float_list(encoded: &str) -> Option<Vec<f64>> {
    if encoded.is_empty() {
        return Some(Vec::new());
    }

    encoded
        .split(',')
        .map(|value| value.parse::<f64>().ok())
        .collect()
}

fn merged_phi_expression(curves: &[(&Vec<usize>, &SparsePhiCurve)]) -> String {
    curves
        .iter()
        .map(|(term_key, curve)| {
            format!(
                "term[{}]{{{}}}",
                format_index_list(term_key),
                curve.polynomial_formula()
            )
        })
        .collect::<Vec<_>>()
        .join(";")
}

pub(crate) fn parse_merged_phi_expression(encoded: &str) -> Option<Vec<(Vec<usize>, Vec<f64>)>> {
    if encoded.is_empty() {
        return Some(Vec::new());
    }

    encoded
        .split(';')
        .filter(|component| !component.is_empty())
        .map(parse_merged_phi_component)
        .collect()
}

type ResponseCurves = Vec<(usize, Vec<(Vec<usize>, Vec<f64>)>)>;

pub(crate) fn parse_global_phi_expression(encoded: &str) -> Option<ResponseCurves> {
    if encoded.is_empty() {
        return Some(Vec::new());
    }

    encoded
        .split("||")
        .filter(|component| !component.is_empty())
        .map(parse_global_phi_component)
        .collect()
}

fn parse_global_phi_component(encoded: &str) -> Option<(usize, Vec<(Vec<usize>, Vec<f64>)>)> {
    let rest = encoded.strip_prefix("response[")?;
    let (response_index, expression) = rest.split_once("]{")?;
    let expression = expression.strip_suffix('}')?;

    Some((
        response_index.parse::<usize>().ok()?,
        parse_merged_phi_expression(expression)?,
    ))
}

fn parse_merged_phi_component(encoded: &str) -> Option<(Vec<usize>, Vec<f64>)> {
    let rest = encoded.strip_prefix("term[")?;
    let (term_key, formula) = rest.split_once("]{")?;
    let formula = formula.strip_suffix('}')?;

    Some((
        parse_index_list(term_key)?,
        control_points_from_polynomial(formula)?,
    ))
}

fn polynomial_formula(points: &[f64]) -> String {
    let coefficients = interpolating_polynomial_coefficients(points).unwrap_or_default();

    if coefficients.is_empty() {
        return "y=0.000000000000*x^0".to_string();
    }

    format!(
        "y={}",
        coefficients
            .iter()
            .enumerate()
            .map(|(power, coefficient)| format!("{coefficient:.12}*x^{power}"))
            .collect::<Vec<_>>()
            .join(" + ")
    )
}

pub(crate) fn control_points_from_polynomial(encoded: &str) -> Option<Vec<f64>> {
    let coefficients = polynomial_coefficients_from_formula(encoded)?;

    if coefficients.is_empty() {
        return None;
    }

    let last_index = coefficients.len().saturating_sub(1).max(1);
    Some(
        (0..coefficients.len())
            .map(|index| {
                let x = index as f64 / last_index as f64;
                evaluate_polynomial(&coefficients, x)
            })
            .collect(),
    )
}

fn polynomial_coefficients_from_formula(encoded: &str) -> Option<Vec<f64>> {
    let formula = encoded.strip_prefix("y=")?;
    let mut coefficients = Vec::<f64>::new();

    for term in formula.split(" + ") {
        let (coefficient, power) = term.split_once("*x^")?;
        let coefficient = coefficient.parse::<f64>().ok()?;
        let power = power.parse::<usize>().ok()?;

        if coefficients.len() <= power {
            coefficients.resize(power + 1, 0.0);
        }

        coefficients[power] = coefficient;
    }

    Some(coefficients)
}

fn interpolating_polynomial_coefficients(points: &[f64]) -> Option<Vec<f64>> {
    if points.is_empty() {
        return None;
    }

    if points.len() == 1 {
        return Some(vec![points[0]]);
    }

    let size = points.len();
    let last_index = size - 1;
    let mut matrix = vec![vec![0.0; size + 1]; size];

    for row in 0..size {
        let x = row as f64 / last_index as f64;
        let mut power = 1.0;

        for column in 0..size {
            matrix[row][column] = power;
            power *= x;
        }

        matrix[row][size] = points[row];
    }

    for pivot_column in 0..size {
        let pivot_row = (pivot_column..size).max_by(|left, right| {
            matrix[*left][pivot_column]
                .abs()
                .total_cmp(&matrix[*right][pivot_column].abs())
        })?;

        if matrix[pivot_row][pivot_column].abs() < 1e-12 {
            return None;
        }

        matrix.swap(pivot_column, pivot_row);

        let pivot = matrix[pivot_column][pivot_column];
        for column in pivot_column..=size {
            matrix[pivot_column][column] /= pivot;
        }

        for row in 0..size {
            if row == pivot_column {
                continue;
            }

            let factor = matrix[row][pivot_column];
            for column in pivot_column..=size {
                matrix[row][column] -= factor * matrix[pivot_column][column];
            }
        }
    }

    Some((0..size).map(|row| matrix[row][size]).collect())
}

fn evaluate_polynomial(coefficients: &[f64], x: f64) -> f64 {
    coefficients
        .iter()
        .rev()
        .fold(0.0, |value, coefficient| value * x + coefficient)
}

#[cfg_attr(not(test), allow(dead_code))]
fn piecewise_linear_formula(points: &[f64]) -> String {
    if points.len() < 2 {
        return points
            .first()
            .map(|value| {
                format!("x=[0.000000000000,1.000000000000]:y=0.000000000000*x{value:+.12}")
            })
            .unwrap_or_default();
    }

    let last_index = points.len() - 1;
    (0..last_index)
        .map(|index| {
            let x0 = index as f64 / last_index as f64;
            let x1 = (index + 1) as f64 / last_index as f64;
            let y0 = points[index];
            let y1 = points[index + 1];
            let slope = (y1 - y0) / (x1 - x0);
            let intercept = y0 - slope * x0;

            format!("x=[{x0:.12},{x1:.12}]:y={slope:.12}*x{intercept:+.12}")
        })
        .collect::<Vec<_>>()
        .join(";")
}

pub(crate) fn control_points_from_piecewise_linear(encoded: &str) -> Option<Vec<f64>> {
    let segments = encoded
        .split(';')
        .filter(|segment| !segment.is_empty())
        .map(parse_piecewise_linear_segment)
        .collect::<Option<Vec<_>>>()?;

    if segments.is_empty() {
        return None;
    }

    let mut points = Vec::with_capacity(segments.len() + 1);

    for (index, (x0, x1, slope, intercept)) in segments.into_iter().enumerate() {
        if index == 0 {
            points.push(slope * x0 + intercept);
        }

        points.push(slope * x1 + intercept);
    }

    Some(points)
}

fn parse_piecewise_linear_segment(segment: &str) -> Option<(f64, f64, f64, f64)> {
    let rest = segment.strip_prefix("x=[")?;
    let (range, formula) = rest.split_once("]:y=")?;
    let (x0, x1) = range.split_once(',')?;
    let (slope, intercept) = formula.split_once("*x")?;

    Some((
        x0.parse::<f64>().ok()?,
        x1.parse::<f64>().ok()?,
        slope.parse::<f64>().ok()?,
        intercept.parse::<f64>().ok()?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polynomial_formula_round_trips_control_points() {
        let points = vec![0.0, 0.25, -0.5, 1.0];
        let formula = polynomial_formula(&points);
        let decoded = control_points_from_polynomial(&formula).expect("decoded polynomial");

        assert_eq!(decoded.len(), points.len());
        for (left, right) in decoded.iter().zip(points) {
            assert!((left - right).abs() < 1e-9);
        }
    }

    #[test]
    fn piecewise_linear_formula_round_trips_control_points() {
        let points = vec![0.0, 0.25, -0.5, 1.0];
        let formula = piecewise_linear_formula(&points);
        let decoded = control_points_from_piecewise_linear(&formula).expect("decoded formula");

        assert_eq!(decoded.len(), points.len());
        for (left, right) in decoded.iter().zip(points) {
            assert!((left - right).abs() < 1e-9);
        }
    }

    #[test]
    fn merged_phi_expression_round_trips_term_curves() {
        let left_curve = SparsePhiCurve {
            control_points: vec![0.0, 0.25, -0.5, 1.0],
        };
        let right_curve = SparsePhiCurve {
            control_points: vec![1.0, 0.5, 0.25, 0.0],
        };
        let left_key = vec![1];
        let right_key = vec![2, 3];
        let curves = vec![(&left_key, &left_curve), (&right_key, &right_curve)];

        let encoded = merged_phi_expression(&curves);
        let decoded = parse_merged_phi_expression(&encoded).expect("decoded expression");

        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].0, left_key);
        assert_eq!(decoded[1].0, right_key);

        for (left, right) in decoded[0].1.iter().zip(&left_curve.control_points) {
            assert!((left - right).abs() < 1e-9);
        }

        for (left, right) in decoded[1].1.iter().zip(&right_curve.control_points) {
            assert!((left - right).abs() < 1e-9);
        }
    }

    #[test]
    fn global_phi_expression_round_trips_response_curves() {
        let curve = SparsePhiCurve {
            control_points: vec![0.0, 0.5, 1.0],
        };
        let key = vec![4, 5];
        let curves = vec![(&key, &curve)];
        let expression = format!("response[2]{{{}}}", merged_phi_expression(&curves));
        let decoded = parse_global_phi_expression(&expression).expect("decoded expression");

        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].0, 2);
        assert_eq!(decoded[0].1[0].0, key);

        for (left, right) in decoded[0].1[0].1.iter().zip(&curve.control_points) {
            assert!((left - right).abs() < 1e-9);
        }
    }
}
