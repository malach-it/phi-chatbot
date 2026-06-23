#[derive(Debug)]
pub struct PhiNetwork {
    input_size: usize,
    terms: Vec<PhiTerm>,
    curve: PhiCurve,
    learning_rate: f64,
}

const DEFAULT_CURVE_KNOTS: usize = 8;

impl PhiNetwork {
    pub fn new(input_size: usize, learning_rate: f64) -> Self {
        Self {
            input_size,
            terms: Vec::new(),
            curve: PhiCurve::new(DEFAULT_CURVE_KNOTS),
            learning_rate,
        }
    }

    pub fn predict(&self, inputs: &[f64]) -> f64 {
        let phi_inputs = self.cached_phi_inputs(inputs);
        self.predict_from_cached_phi_inputs(&phi_inputs)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn train_until(
        &mut self,
        training_data: &[TrainingExample],
        epsilon: f64,
        max_epochs_per_degree: usize,
    ) -> bool {
        self.train_until_with_options(
            training_data,
            epsilon,
            max_epochs_per_degree,
            self.input_size,
            true,
        )
    }

    pub fn train_until_quiet(
        &mut self,
        training_data: &[TrainingExample],
        epsilon: f64,
        max_epochs_per_degree: usize,
        max_degree: usize,
    ) -> bool {
        self.train_until_with_options(
            training_data,
            epsilon,
            max_epochs_per_degree,
            max_degree,
            false,
        )
    }

    fn train_until_with_options(
        &mut self,
        training_data: &[TrainingExample],
        epsilon: f64,
        max_epochs_per_degree: usize,
        max_degree: usize,
        verbose: bool,
    ) -> bool {
        for degree in 1..=max_degree.min(self.input_size) {
            self.add_terms_for_degree(degree);
            let cached_training_data = self.cache_training_data(training_data);

            if verbose {
                println!("added degree {degree} terms: {}", self.formula());
                println!(
                    "cached {} phi curve inputs per example",
                    cached_training_data
                        .first()
                        .map(|example| example.phi_inputs.len())
                        .unwrap_or(0)
                );
            }

            for epoch in 1..=max_epochs_per_degree {
                for example in &cached_training_data {
                    self.train_one_cached(&example.phi_inputs, example.target);
                }

                let max_error = self.max_error_cached(&cached_training_data);
                if verbose && (epoch == 1 || epoch % 5_000 == 0 || max_error <= epsilon) {
                    println!("  degree {degree}, epoch {epoch}: max error {max_error:.6}");
                }

                if max_error <= epsilon {
                    return true;
                }
            }
        }

        false
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn term_count(&self) -> usize {
        self.terms.len()
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn term_indices(&self, index: usize) -> Option<&[usize]> {
        self.terms
            .get(index)
            .map(|term| term.input_indices.as_slice())
    }

    pub fn curve_points(&self, index: usize) -> Option<&[f64]> {
        self.terms
            .get(index)
            .map(|_| self.curve.control_points.as_slice())
    }

    fn train_one_cached(&mut self, phi_inputs: &[CachedPhiInput], target: f64) {
        let prediction = self.predict_from_cached_phi_inputs(phi_inputs);
        let error = target - prediction;

        for phi_input in phi_inputs {
            self.curve.train(phi_input, error, self.learning_rate);
        }
    }

    fn max_error_cached(&self, training_data: &[CachedTrainingExample]) -> f64 {
        training_data
            .iter()
            .map(|example| {
                (example.target - self.predict_from_cached_phi_inputs(&example.phi_inputs)).abs()
            })
            .fold(0.0, f64::max)
    }

    fn add_terms_for_degree(&mut self, degree: usize) {
        let mut indices = Vec::new();
        self.add_combinations(degree, 0, &mut indices);
    }

    fn add_combinations(&mut self, degree: usize, start: usize, indices: &mut Vec<usize>) {
        if indices.len() == degree {
            self.terms.push(PhiTerm {
                input_indices: indices.clone(),
            });
            return;
        }

        for index in start..self.input_size {
            indices.push(index);
            self.add_combinations(degree, index + 1, indices);
            indices.pop();
        }
    }

    fn formula(&self) -> String {
        self.terms
            .iter()
            .map(PhiTerm::name)
            .collect::<Vec<_>>()
            .join(" + ")
    }

    fn cache_training_data(&self, training_data: &[TrainingExample]) -> Vec<CachedTrainingExample> {
        training_data
            .iter()
            .map(|example| CachedTrainingExample {
                target: example.target,
                phi_inputs: self.cached_phi_inputs(&example.inputs),
            })
            .collect()
    }

    fn cached_phi_inputs(&self, inputs: &[f64]) -> Vec<CachedPhiInput> {
        self.terms
            .iter()
            .map(|term| self.curve.cache_input(term.value(inputs)))
            .collect()
    }

    fn predict_from_cached_phi_inputs(&self, phi_inputs: &[CachedPhiInput]) -> f64 {
        phi_inputs
            .iter()
            .map(|phi_input| self.curve.value_cached(phi_input))
            .sum()
    }
}

#[derive(Debug)]
struct PhiCurve {
    control_points: Vec<f64>,
}

impl PhiCurve {
    fn new(knots: usize) -> Self {
        assert!(knots >= 2, "a curve needs at least two knots");

        Self {
            control_points: vec![0.0; knots],
        }
    }

    fn cache_input(&self, input: f64) -> CachedPhiInput {
        let input = input.clamp(0.0, 1.0);
        let scaled = input * (self.control_points.len() - 1) as f64;
        let lower_index = scaled.floor() as usize;
        let upper_index = (lower_index + 1).min(self.control_points.len() - 1);
        let upper_weight = scaled - lower_index as f64;
        let lower_weight = 1.0 - upper_weight;

        CachedPhiInput {
            gate: input,
            lower_index,
            upper_index,
            lower_weight,
            upper_weight,
        }
    }

    fn value_cached(&self, input: &CachedPhiInput) -> f64 {
        let interpolated = self.control_points[input.lower_index] * input.lower_weight
            + self.control_points[input.upper_index] * input.upper_weight;

        input.gate * interpolated
    }

    fn train(&mut self, input: &CachedPhiInput, error: f64, learning_rate: f64) {
        let lower_gradient = input.gate * input.lower_weight;
        let upper_gradient = input.gate * input.upper_weight;

        self.control_points[input.lower_index] += learning_rate * error * lower_gradient;
        self.control_points[input.upper_index] += learning_rate * error * upper_gradient;
    }
}

#[derive(Debug)]
struct CachedPhiInput {
    gate: f64,
    lower_index: usize,
    upper_index: usize,
    lower_weight: f64,
    upper_weight: f64,
}

#[derive(Debug)]
struct PhiTerm {
    input_indices: Vec<usize>,
}

impl PhiTerm {
    fn value(&self, inputs: &[f64]) -> f64 {
        self.input_indices
            .iter()
            .map(|index| inputs[*index])
            .product()
    }

    fn name(&self) -> String {
        let variables = self
            .input_indices
            .iter()
            .map(|index| variable_name(*index))
            .collect::<String>();

        format!("phi({variables})")
    }
}

#[derive(Debug)]
pub struct TrainingExample {
    pub inputs: Vec<f64>,
    pub target: f64,
}

#[derive(Debug)]
struct CachedTrainingExample {
    target: f64,
    phi_inputs: Vec<CachedPhiInput>,
}

fn variable_name(index: usize) -> String {
    if index < 26 {
        ((b'a' + index as u8) as char).to_string()
    } else {
        format!("x{index}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xor_data() -> Vec<TrainingExample> {
        parity_data(2)
    }

    fn parity3_data() -> Vec<TrainingExample> {
        parity_data(3)
    }

    fn parity_data(input_size: usize) -> Vec<TrainingExample> {
        let example_count = 1_usize << input_size;

        (0..example_count)
            .map(|bits| {
                let inputs = (0..input_size)
                    .map(|index| ((bits >> index) & 1) as f64)
                    .collect::<Vec<_>>();
                let ones = inputs.iter().filter(|input| **input == 1.0).count();
                let target = (ones % 2) as f64;

                TrainingExample { inputs, target }
            })
            .collect()
    }

    fn square_curve_data() -> Vec<TrainingExample> {
        let steps = 7;

        (0..=steps)
            .map(|step| {
                let x = step as f64 / steps as f64;

                TrainingExample {
                    inputs: vec![x],
                    target: x * x,
                }
            })
            .collect()
    }

    #[test]
    fn shared_phi_adds_pair_term_but_cannot_represent_xor() {
        let training_data = xor_data();
        let mut network = PhiNetwork::new(2, 0.1);

        assert!(!network.train_until(&training_data, 0.01, 10_000));
        assert_eq!(network.term_count(), 3);
        assert_eq!(network.term_indices(2), Some(&[0, 1][..]));
        assert_eq!(network.curve_points(0), network.curve_points(1));
        assert_eq!(network.curve_points(1), network.curve_points(2));

        let one_active = network.predict(&[1.0, 0.0]);
        let two_active = network.predict(&[1.0, 1.0]);

        assert!((two_active - (3.0 * one_active)).abs() <= 0.001);
    }

    #[test]
    fn shared_phi_adds_three_input_terms() {
        let training_data = parity3_data();
        let mut network = PhiNetwork::new(3, 0.1);

        assert!(!network.train_until(&training_data, 0.001, 0));
        assert_eq!(network.term_count(), 7);
        assert_eq!(network.term_indices(6), Some(&[0, 1, 2][..]));
        assert_eq!(network.curve_points(0), network.curve_points(6));
    }

    #[test]
    fn shared_phi_adds_four_input_terms() {
        let training_data = parity_data(4);
        let mut network = PhiNetwork::new(4, 0.05);

        assert!(!network.train_until(&training_data, 0.0005, 0));
        assert_eq!(network.term_count(), 15);
        assert_eq!(network.term_indices(14), Some(&[0, 1, 2, 3][..]));
        assert_eq!(network.curve_points(0), network.curve_points(14));
    }

    #[test]
    fn learns_continuous_square_curve() {
        let training_data = square_curve_data();
        let mut network = PhiNetwork::new(1, 0.5);

        assert!(network.train_until(&training_data, 0.001, 100_000));

        for example in training_data {
            let prediction = network.predict(&example.inputs);
            assert!(
                (example.target - prediction).abs() <= 0.001,
                "expected {:?} to be near {}, got {prediction}",
                example.inputs,
                example.target
            );
        }
    }
}
