use crate::chatbot::ChatBot;

pub(crate) fn run(bot: &ChatBot) {
    for response in bot.responses() {
        println!("  {response}");
    }
}
