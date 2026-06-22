use crate::chatbot::ChatBot;

const DEFAULT_SUGGESTION_LIMIT: usize = 5;

pub(crate) fn run(bot: &ChatBot, rest: &str) {
    let (limit, message) = parse_suggest_args(rest);

    if message.trim().is_empty() {
        println!("expected: suggest [limit] <message>");
        return;
    }

    let suggestions = bot.suggest(message, limit);
    if suggestions.is_empty() {
        println!("no suggestions found");
        return;
    }

    for (index, suggestion) in suggestions.iter().enumerate() {
        println!(
            "  {}. {} ({:.3}, matched: {})",
            index + 1,
            suggestion.response,
            suggestion.score.clamp(0.0, 1.0),
            suggestion.matched_example
        );
    }
}

fn parse_suggest_args(rest: &str) -> (usize, &str) {
    let rest = rest.trim();
    let Some((first, message)) = rest.split_once(char::is_whitespace) else {
        return (DEFAULT_SUGGESTION_LIMIT, rest);
    };

    first
        .parse::<usize>()
        .ok()
        .filter(|limit| *limit > 0)
        .map(|limit| (limit, message.trim()))
        .unwrap_or((DEFAULT_SUGGESTION_LIMIT, rest))
}
