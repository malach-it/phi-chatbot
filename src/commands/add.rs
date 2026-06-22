use std::io;

use crate::chatbot::{append_example_to_file, ChatBot, SessionContext, MEMORY_PATH};

pub(crate) fn run(
    bot: &mut ChatBot,
    session_context: &SessionContext,
    rest: &str,
) -> io::Result<()> {
    if let Some((message, response)) = rest.split_once("=>") {
        let context_features = session_context.context_features();
        if bot.add_example_with_context_if_missing(message, response, context_features.clone()) {
            append_example_to_file(
                MEMORY_PATH,
                message.trim(),
                response.trim(),
                &context_features,
            )?;
            println!("added and remembered example. Run `train` to update the model.");
        } else {
            println!("that example already exists");
        }
    } else {
        println!("expected: add <message> => <reply>");
    }

    Ok(())
}
