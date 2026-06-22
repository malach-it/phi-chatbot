use std::io;

use crate::chatbot::{save_phi_memory, ChatBot, DEFAULT_TRAIN_EPOCHS, DEFAULT_TRAIN_EPSILON};

pub(crate) fn run(bot: &mut ChatBot, rest: &str) -> io::Result<()> {
    let mut parts = rest.split_whitespace();
    let epochs = parts
        .next()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_TRAIN_EPOCHS);
    let epsilon = parts
        .next()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(DEFAULT_TRAIN_EPSILON);

    bot.train(epochs, epsilon);
    save_phi_memory(bot)?;
    println!(
        "trained {} examples into {} responses with {} word features",
        bot.example_count(),
        bot.responses().len(),
        bot.vocabulary().len()
    );

    Ok(())
}
