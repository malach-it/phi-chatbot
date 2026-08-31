pub(crate) fn run() {
    println!("commands:");
    println!("  add <message> => <reply>     add one training example");
    println!("                               includes current context and saves it");
    println!("  train [epochs] [epsilon]     rebuild and train the chatbot");
    println!("  train age <tsv> [epochs] [epsilon]");
    println!("                               train name -> age -> over-18 curves");
    println!("  ask <message>                ask the trained chatbot");
    println!("                               low-confidence answers prompt for training");
    println!("  suggest [limit] <message>    list likely replies from remembered examples");
    println!("  examples                     list training examples");
    println!("  responses                    list learned response classes");
    println!("  clear context                forget accumulated session phi terms");
    println!("  curve                        draw chatbot, phil, and stored age curves");
    println!("  keypair [shares]             print encoded phi and encrypted phin shares");
    println!("  phil <message>               classify raw phi output as short or long");
    println!("  over18 <name>                apply stored age curves and return true or false");
    println!("  tokens <message>             show word tokens for a message");
    println!("  vocab                        list bag-of-words features");
    println!("  help                         show this help");
    println!("  quit                         exit");
    println!("plain text without a command is treated like ask <message>");
}
