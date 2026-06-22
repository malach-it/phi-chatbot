use std::io;

use crate::chatbot::{ChatBot, SessionContext};

pub(crate) mod add;
pub(crate) mod ask;
pub(crate) mod clear_context;
pub(crate) mod curve;
pub(crate) mod examples;
pub(crate) mod help;
pub(crate) mod responses;
pub(crate) mod tokens;
pub(crate) mod train;
pub(crate) mod vocab;

pub(crate) enum CommandAction {
    Continue,
    Quit,
}

pub(crate) fn dispatch(
    line: &str,
    bot: &mut ChatBot,
    session_context: &mut SessionContext,
) -> io::Result<CommandAction> {
    if line == "quit" || line == "exit" {
        Ok(CommandAction::Quit)
    } else if line == "help" {
        help::run();
        Ok(CommandAction::Continue)
    } else if line == "clear context" {
        clear_context::run(session_context);
        Ok(CommandAction::Continue)
    } else if let Some(rest) = line.strip_prefix("add ") {
        add::run(bot, rest)?;
        Ok(CommandAction::Continue)
    } else if let Some(rest) = line.strip_prefix("train") {
        train::run(bot, rest)?;
        Ok(CommandAction::Continue)
    } else if let Some(message) = line.strip_prefix("ask ") {
        ask::run(bot, session_context, message)?;
        Ok(CommandAction::Continue)
    } else if line == "examples" {
        examples::run(bot);
        Ok(CommandAction::Continue)
    } else if line == "responses" {
        responses::run(bot);
        Ok(CommandAction::Continue)
    } else if line == "curve" {
        curve::run(bot);
        Ok(CommandAction::Continue)
    } else if let Some(message) = line.strip_prefix("tokens ") {
        tokens::run(message);
        Ok(CommandAction::Continue)
    } else if line == "vocab" {
        vocab::run(bot);
        Ok(CommandAction::Continue)
    } else {
        ask::run(bot, session_context, line)?;
        Ok(CommandAction::Continue)
    }
}
