pub(crate) fn run() {
    println!("commands:");
    println!("  add <message> => <reply>     add one training example");
    println!("                               includes current context and saves it");
    println!("  train [epochs] [epsilon]     rebuild and train the chatbot");
    println!("  ask <message>                ask the trained chatbot");
    println!("                               low-confidence answers prompt for training");
    println!("  examples                     list training examples");
    println!("  responses                    list learned response classes");
    println!("  clear context                forget accumulated session phi terms");
    println!("  curve                        draw learned phi curve");
    println!("  tokens <message>             show word tokens for a message");
    println!("  vocab                        list bag-of-words features");
    println!("  help                         show this help");
    println!("  quit                         exit");
    println!("plain text without a command is treated like ask <message>");
}
