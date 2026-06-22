use crate::chatbot::{format_context_features, ChatBot};

pub(crate) fn run(bot: &ChatBot) {
    for (index, example) in bot.examples().iter().enumerate() {
        if !example.context_features().is_empty() {
            println!(
                "  {}. [{}] {} => {}",
                index + 1,
                format_context_features(example.context_features()),
                example.message(),
                example.response()
            );
        } else {
            println!(
                "  {}. {} => {}",
                index + 1,
                example.message(),
                example.response()
            );
        }
    }
}
