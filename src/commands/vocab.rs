use crate::chatbot::ChatBot;

pub(crate) fn run(bot: &ChatBot) {
    println!("{}", bot.vocabulary().join(", "));
}
