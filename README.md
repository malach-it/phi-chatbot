# PhiNetwork Chatbot

A small Rust experiment around a PhiNetwork-style learner and a command-line chatbot.

The model learns response classes from text examples. It tokenizes input words, builds phi terms from active features, and trains either dense learned curves or sparse phi components. The chatbot can remember new examples between sessions and can draw the learned overall phi curve as ASCII.

## Run

```bash
cargo run
```

At startup, choose a model:

```text
1. dense curve   - smoother, original curve PhiNetwork, poor scaling
2. sparse scalar - scalable, more exact lexical matching
3. sparse curve  - scalable sparse terms with learned curves
```

Press Enter to use the default sparse curve mode.

## CLI Commands

```text
add <message> => <reply>     add one training example and save it
train [epochs] [epsilon]     rebuild and train the chatbot
ask <message>                ask the trained chatbot
examples                     list training examples
responses                    list learned response classes
clear context                forget accumulated session phi terms
curve                        draw the learned overall phi curve
tokens <message>             show word tokens for a message
vocab                        list bag-of-words features
help                         show command help
quit                         exit
```

Plain text without a command is treated like `ask <message>`.

If the chatbot is not confident enough, it asks for the right response and trains from that answer.
When an answer is confident, recursion scores each candidate with `phi(a) + phi(phi(a)) + phi(a * phi(a))`: the prompt, the candidate result as input, and the prompt with candidate-result interaction context. It prints each confident intermediate result until the next score drops below the confidence threshold, repeats, or reaches the recursion cap.

## Example Session

```text
> add rust ownership borrowing => Rust ownership controls memory without a garbage collector.
added and remembered example. Run `train` to update the model.
> train
trained 22 examples into 10 responses with 34 word features
> ask explain rust borrowing
Rust ownership controls memory without a garbage collector. (0.742)
> curve
phi_all
    ...
```

## Persistence

The chatbot stores learned data under `data/`:

```text
data/chatbot_memory.tsv       remembered training examples
data/chatbot_phi_all.tsv      learned phi state
```

Memory examples are loaded on startup. Sparse phi state is also loaded on startup when compatible with the selected mode.

## Project Layout

```text
src/main.rs                    entrypoint
src/chatbot.rs                 chatbot/session/persistence orchestration
src/phinetwork.rs              dense PhiNetwork core
src/commands/                  one module per CLI command
src/classifiers/               classifier implementations
src/classifiers/dense_curve.rs dense curve classifier
src/classifiers/sparse_phi.rs  sparse scalar and sparse curve classifier
src/classifiers/curve_plot.rs  ASCII curve plotting helpers
```

## Development

```bash
cargo fmt --check
cargo test
```

## License

MIT. See [LICENSE](LICENSE).
