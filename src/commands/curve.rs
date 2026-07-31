use std::io;

use crate::chatbot::ChatBot;

pub(crate) fn run(bot: &ChatBot) -> io::Result<()> {
    match bot.curve_report() {
        Some(report) => print!("{report}"),
        None => println!("curve is available only in dense curve or sparse curve mode"),
    }

    if let Some(report) = crate::age::curve_report()? {
        print!("{report}");
    }

    Ok(())
}
