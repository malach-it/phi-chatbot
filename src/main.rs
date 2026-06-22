mod chatbot;
mod classifiers;
mod commands;
mod phinetwork;

fn main() -> std::io::Result<()> {
    chatbot::run_chatbot_cli()
}
