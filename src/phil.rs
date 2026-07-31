use crate::chatbot::PhiOutput;

const LONG_RESPONSE_MIN_CHARACTERS: usize = 11;

/// Learns a binary response-length target for each raw phi response class.
#[derive(Debug)]
pub(crate) struct Phil {
    class_targets: Vec<usize>,
}

impl Phil {
    pub(crate) fn train(response_texts: &[String]) -> Option<Self> {
        if response_texts.is_empty() {
            return None;
        }

        Some(Self {
            class_targets: response_texts
                .iter()
                .map(|response| {
                    usize::from(response.chars().count() >= LONG_RESPONSE_MIN_CHARACTERS)
                })
                .collect(),
        })
    }

    pub(crate) fn from_class_targets(class_targets: Vec<usize>) -> Option<Self> {
        (!class_targets.is_empty() && class_targets.iter().all(|target| *target <= 1))
            .then_some(Self { class_targets })
    }

    pub(crate) fn class_targets(&self) -> &[usize] {
        &self.class_targets
    }

    /// Applies phil to phi's raw class output without inspecting any text.
    pub(crate) fn apply(&self, phi_output: PhiOutput) -> Option<PhilTransform> {
        let output = *self.class_targets.get(phi_output.response_index)?;

        Some(PhilTransform {
            response_index: phi_output.response_index,
            phi_score: phi_output.score,
            output,
            classification: if output == 1 { "long" } else { "short" },
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PhilTransform {
    pub(crate) response_index: usize,
    pub(crate) phi_score: f64,
    pub(crate) output: usize,
    pub(crate) classification: &'static str,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learns_response_lengths_for_raw_phi_classes() {
        let phil = Phil::train(&["Goodbye.".to_string(), "hello world".to_string()])
            .expect("response classes");

        assert_eq!(
            phil.apply(PhiOutput {
                response_index: 0,
                score: 0.95,
            })
            .expect("short class")
            .output,
            0
        );
        assert_eq!(
            phil.apply(PhiOutput {
                response_index: 1,
                score: 0.95,
            })
            .expect("long class")
            .output,
            1
        );
    }

    #[test]
    fn inference_only_needs_raw_phi_output() {
        let phil = Phil::from_class_targets(vec![1]).expect("stored function");
        let result = phil
            .apply(PhiOutput {
                response_index: 0,
                score: 0.75,
            })
            .expect("known class");

        assert_eq!(result.response_index, 0);
        assert_eq!(result.phi_score, 0.75);
        assert_eq!(result.output, 1);
        assert_eq!(result.classification, "long");
    }

    #[test]
    fn rejects_invalid_stored_targets() {
        assert!(Phil::from_class_targets(vec![]).is_none());
        assert!(Phil::from_class_targets(vec![2]).is_none());
    }
}
