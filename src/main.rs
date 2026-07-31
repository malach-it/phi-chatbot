mod age;
mod chatbot;
mod classifiers;
mod commands;
mod phi_key;
mod phil;
mod phinetwork;

fn main() -> std::io::Result<()> {
    chatbot::run_chatbot_cli()
}
