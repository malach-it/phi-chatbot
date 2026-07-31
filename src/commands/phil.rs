use crate::chatbot::ChatBot;

pub(crate) fn run(bot: &ChatBot, message: &str) {
    println!("{}", report(bot, message));
}

fn report(bot: &ChatBot, message: &str) -> String {
    let Some(phi_output) = bot.phi_output(message) else {
        return "phi: no prediction\nphil o phi: unavailable".to_string();
    };
    let response = bot
        .responses()
        .get(phi_output.response_index)
        .map(String::as_str)
        .unwrap_or("<unknown response>");
    let Some(result) = bot.apply_phil(phi_output) else {
        return format!(
            "phi(message): {response}\nphi output: class {} ({:.4})\nphil o phi: untrained",
            phi_output.response_index, phi_output.score,
        );
    };

    format!(
        "phi(message): {response}\nphi output: class {} ({:.4})\nphil o phi: class {} -> {} ({})",
        result.response_index,
        result.phi_score,
        result.response_index,
        result.output,
        result.classification
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_phi_prediction_then_phil_transformation() {
        let mut bot = ChatBot::new();
        bot.add_example("greeting", "hello world");
        bot.train(1_000, 0.01);

        let output = report(&bot, "greeting");

        assert!(output.starts_with("phi(message): hello world\nphi output: class 0 ("));
        assert!(output.contains("phil o phi: class 0 -> 1"));
        assert!(output.ends_with("(long)"));
    }

    #[test]
    fn reports_when_phi_has_no_prediction() {
        assert_eq!(
            report(&ChatBot::new(), "anything"),
            "phi: no prediction\nphil o phi: unavailable"
        );
    }
}
