use std::io;

use crate::chatbot::{append_example_to_file, ChatBot, MEMORY_PATH};

pub(crate) fn run(bot: &mut ChatBot, rest: &str) -> io::Result<()> {
    if let Some((message, response)) = rest.split_once("=>") {
        if bot.add_example_if_missing(message, response) {
            append_example_to_file(MEMORY_PATH, message.trim(), response.trim(), &[])?;
            println!("added and remembered example. Run `train` to update the model.");
        } else {
            println!("that example already exists");
        }
    } else {
        println!("expected: add <message> => <reply>");
    }

    Ok(())
}
