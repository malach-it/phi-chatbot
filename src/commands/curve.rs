use crate::chatbot::ChatBot;

pub(crate) fn run(bot: &ChatBot) {
    match bot.curve_report() {
        Some(report) => print!("{report}"),
        None => println!("curve is available only in dense curve or sparse curve mode"),
    }
}
