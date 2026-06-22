use std::io;

use crate::chatbot::{answer_or_learn, ChatBot, SessionContext};

pub(crate) fn run(
    bot: &mut ChatBot,
    session_context: &mut SessionContext,
    message: &str,
) -> io::Result<()> {
    answer_or_learn(bot, session_context, message)
}
