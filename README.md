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
train age <tsv> [epochs] [epsilon]
                              train name -> age -> over-18 curves
ask <message>                ask the trained chatbot
suggest [limit] <message>    list likely replies from remembered examples
examples                     list training examples
responses                    list learned response classes
clear context                forget accumulated session phi terms
curve                        draw chatbot, phil, and stored age curves
keypair [shares]             print and plot encoded phi and encrypted phin shares
phil <message>               apply phil o phi and print both transformations
over18 <name>                apply stored age curves and return true or false
tokens <message>             show word tokens for a message
vocab                        list bag-of-words features
help                         show command help
quit                         exit
```

Plain text without a command is treated like `ask <message>`.

`train age` accepts `name<TAB>age` rows. Each distinct exact age is assigned an
opaque random vocabulary token for that training run. `phi2(name)` learns the
token, and `phi1` is trained on raw tokens toward `0` for under 18 and `1` for
18 or older. The age-to-token dictionary is discarded rather than persisted;
incremental examples receive fresh token aliases. Names are deterministically
encoded before training. The resulting curves and SHA-256 name fingerprints
are written to `data/phi_age.tsv`; source names, ages, and the random vocabulary
are not written literally to that model file. This reduces direct age recovery
but is not a cryptographic privacy mechanism or a predictor for unseen people.

An example dataset is available at `examples/name_age.tsv`:

```bash
cargo run
# Then enter: train age examples/name_age.tsv
# Then query: over18 Alice
```

`over18` loads the stored curves and prints only the composed boolean result; it
does not print the intermediate random token produced by `phi2`. If a name is
unknown, it asks for an age, incrementally updates both curves, and saves them.
The model stores SHA-256 name fingerprints to recognize learned names, but name
fingerprints can be dictionary-guessed and should not be treated as anonymous.

`phil` performs `phil o phi`. It learns a character-length target for each raw
phi response class: responses with up to 10 Unicode characters have target `0`
(`short`), while responses with 11 or more have target `1` (`long`). At
inference, phi produces a class index and score, and `phil` maps only that raw
class output to `0` or `1`; it reads neither text nor vocabulary. The explicit
form `phil o phi <message>` is also accepted.

The command prints phi's selected textual response for presentation, followed
by phi's raw class output and `phil`'s binary output and classification. The
displayed response text is not passed into `phil` during inference. The trained
`phil` class function is stored and loaded alongside phi in
`data/chatbot_phi_all.tsv`.

The `curve` command displays only the labeled ASCII graphs for `phi_all` and
`phil`; it omits the underlying point and class listings. After `train age`, it
also displays graph-only plots for `phi2` (encoded name to a random age token)
and `phi1` (random token to the over-18 result).

If the chatbot is not confident enough, it asks for the right response and remembers that answer. Run `train` to rebuild the learned phi state from remembered examples.
Phi predictions require a confidence score of at least `0.50` to be accepted as output.
When an answer is confident, recursion feeds that answer back as the next prompt and prints each confident memorized transition. It stops when the next answer is low confidence, repeats, lacks an exact remembered transition from the previous answer, or reaches the recursion cap.

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

Memory examples are loaded on startup. Sparse phi state is also loaded on startup when compatible with the selected mode. The phi state also stores an encoded `phi_all` export line for inspection.

The `keypair [shares]` command is inspired by BLS12-381 key material and curve points, but it is an application-specific encoding of `phi_all` points. It prints and plots the direct encoded `phi_all` points, then `n` encrypted phin shares whose component formulas show how each share is composed, for example `phi(a) + phi(ab) + phi(ac)` when `n = 3`. It also plots each public share's assigned phi point slots. The encrypted phin shares are masked additive shares: all `n` unmasked shares combine to recover encoded `phi_all`. It should not be treated as a standard BLS signature key, wallet key, or general-purpose identity secret.

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
