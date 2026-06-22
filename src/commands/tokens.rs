use crate::chatbot::tokenize;

pub(crate) fn run(message: &str) {
    println!("{}", tokenize(message).join(", "));
}
