use std::collections::{BTreeSet, HashMap};
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::Path;

use crate::classifiers::{add_curve_points, draw_curve, ChatClassifier, EncodedChatExample};
use crate::commands::{self, CommandAction};

pub(crate) const DEFAULT_TRAIN_EPOCHS: usize = 2_000;
pub(crate) const DEFAULT_TRAIN_EPSILON: f64 = 0.02;
const UNKNOWN_CONFIDENCE_THRESHOLD: f64 = 0.50;
const MAX_RECURSIVE_RESULT_DEPTH: usize = 8;
const SESSION_CONTEXT_DECAY: f64 = 0.65;
const SESSION_CONTEXT_INPUT_WEIGHT: f64 = 0.25;
const SESSION_CONTEXT_RESULT_WEIGHT: f64 = 0.15;
const SESSION_CONTEXT_MIN_WEIGHT: f64 = 0.01;
const CONTEXT_FEATURE_BOOST: f64 = 1.5;
const CONTEXT_MEMORY_BONUS: f64 = 0.10;
pub(crate) const MEMORY_PATH: &str = "data/chatbot_memory.tsv";

#[derive(Debug)]
pub struct ChatBot {
    examples: Vec<ChatExample>,
    vocabulary: Vec<String>,
    responses: Vec<String>,
    classifiers: Vec<ChatClassifier>,
    max_degree: usize,
    mode: ChatModelMode,
}

impl ChatBot {
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn new() -> Self {
        Self::new_with_mode(ChatModelMode::SparseCurve)
    }

    pub fn new_with_mode(mode: ChatModelMode) -> Self {
        Self {
            examples: Vec::new(),
            vocabulary: Vec::new(),
            responses: Vec::new(),
            classifiers: Vec::new(),
            max_degree: 2,
            mode,
        }
    }

    pub fn add_example(&mut self, message: &str, response: &str) {
        self.add_example_with_context(message, response, Vec::new());
    }

    fn add_example_with_context(
        &mut self,
        message: &str,
        response: &str,
        context_features: Vec<ContextFeature>,
    ) {
        self.examples.push(ChatExample {
            message: message.trim().to_string(),
            response: response.trim().to_string(),
            context_features,
        });
    }

    pub(crate) fn add_example_if_missing(&mut self, message: &str, response: &str) -> bool {
        self.add_example_with_context_if_missing(message, response, Vec::new())
    }

    fn add_example_with_context_if_missing(
        &mut self,
        message: &str,
        response: &str,
        context_features: Vec<ContextFeature>,
    ) -> bool {
        let message = message.trim();
        let response = response.trim();
        let context_features = normalized_context_features(context_features);

        if self.examples.iter().any(|example| {
            example.message == message
                && example.response == response
                && features_equal(&example.context_features, &context_features)
        }) {
            return false;
        }

        self.add_example_with_context(message, response, context_features);
        true
    }

    pub fn train(&mut self, epochs: usize, epsilon: f64) {
        self.rebuild_vocabulary();
        self.rebuild_responses();

        let encoded_examples = self
            .examples
            .iter()
            .map(|example| EncodedChatExample {
                features: self.features_with_context(&example.message, &example.context_features),
                response: example.response.clone(),
            })
            .collect::<Vec<_>>();

        self.classifiers = self
            .responses
            .iter()
            .map(|response| {
                ChatClassifier::train(
                    self.mode,
                    &encoded_examples,
                    response,
                    epochs,
                    epsilon,
                    self.vocabulary.len(),
                    self.max_degree,
                )
            })
            .collect();
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn reply(&self, message: &str) -> Option<ChatPrediction> {
        self.reply_with_last_result(message, None)
    }

    pub fn reply_with_last_result(
        &self,
        message: &str,
        last_result: Option<&ChatPrediction>,
    ) -> Option<ChatPrediction> {
        let mut context = SessionContext::default();

        if let Some(last_result) = last_result {
            context.add_text(&last_result.response, SESSION_CONTEXT_RESULT_WEIGHT);
        }

        self.reply_with_session_context(message, &context)
    }

    pub fn reply_with_session_context(
        &self,
        message: &str,
        session_context: &SessionContext,
    ) -> Option<ChatPrediction> {
        self.reply_with_session_context_base(message, session_context)
    }

    fn reply_with_session_context_base(
        &self,
        message: &str,
        session_context: &SessionContext,
    ) -> Option<ChatPrediction> {
        if self.classifiers.is_empty() {
            return None;
        }

        let context_features = session_context.context_features();
        let features = self.features_with_context(message, &context_features);

        self.classifiers
            .iter()
            .zip(&self.responses)
            .map(|(classifier, response)| ChatPrediction {
                response: response.clone(),
                score: classifier.predict(&features, self.vocabulary.len())
                    + self.context_memory_bonus(response, &features),
            })
            .max_by(|left, right| left.score.total_cmp(&right.score))
    }

    fn reply_with_recursive_formula(
        &self,
        message: &str,
        session_context: &SessionContext,
    ) -> Option<ChatPrediction> {
        if self.classifiers.is_empty() {
            return None;
        }

        let base_context_features = session_context.context_features();
        let base_features = self.features_with_context(message, &base_context_features);

        self.classifiers
            .iter()
            .zip(&self.responses)
            .map(|(classifier, response)| {
                let mut interaction_context = session_context.clone();
                interaction_context.add_text(response, SESSION_CONTEXT_RESULT_WEIGHT);
                let interaction_features =
                    self.features_with_context(message, &interaction_context.context_features());
                let result_features = self.features_with_context(response, &base_context_features);

                ChatPrediction {
                    response: response.clone(),
                    score: classifier.predict(&base_features, self.vocabulary.len())
                        + classifier.predict(&result_features, self.vocabulary.len())
                        + classifier.predict(&interaction_features, self.vocabulary.len())
                        + self.context_memory_bonus(response, &interaction_features),
                }
            })
            .max_by(|left, right| left.score.total_cmp(&right.score))
    }

    fn context_memory_bonus(&self, response: &str, features: &[WeightedFeature]) -> f64 {
        self.examples
            .iter()
            .filter(|example| example.response == response)
            .map(|example| self.example_context_match(example, features) * CONTEXT_MEMORY_BONUS)
            .fold(0.0, f64::max)
    }

    fn example_context_match(&self, example: &ChatExample, features: &[WeightedFeature]) -> f64 {
        let message_terms = self
            .message_features(&example.message)
            .into_iter()
            .map(|feature| feature.index)
            .collect::<BTreeSet<_>>();

        example
            .context_features
            .iter()
            .filter_map(|context_feature| {
                let index = self.vocabulary.binary_search(&context_feature.name).ok()?;
                if message_terms.contains(&index) {
                    return None;
                }

                features
                    .iter()
                    .find(|feature| feature.index == index)
                    .map(|feature| {
                        feature.value
                            * (context_feature.value * CONTEXT_FEATURE_BOOST).clamp(0.0, 1.0)
                    })
            })
            .sum()
    }

    pub fn example_count(&self) -> usize {
        self.examples.len()
    }

    pub fn vocabulary(&self) -> &[String] {
        &self.vocabulary
    }

    pub fn responses(&self) -> &[String] {
        &self.responses
    }

    pub fn examples(&self) -> &[ChatExample] {
        &self.examples
    }

    pub fn mode(&self) -> ChatModelMode {
        self.mode
    }

    fn dense_curve_report(&self) -> Option<String> {
        if self.mode != ChatModelMode::DenseCurve {
            return None;
        }

        let mut global_curve = None::<Vec<f64>>;

        for classifier in &self.classifiers {
            let Some(points) = classifier.aggregate_curve_points() else {
                continue;
            };

            add_curve_points(&mut global_curve, &points);
        }

        let mut output = String::new();

        if let Some(points) = global_curve {
            output.push_str("phi_all\n");
            output.push_str(&draw_curve(&points, 48));
            output.push('\n');
        }

        if output.is_empty() {
            output.push_str("no dense phi curves are learned yet\n");
        }

        Some(output)
    }

    pub(crate) fn curve_report(&self) -> Option<String> {
        match self.mode {
            ChatModelMode::DenseCurve => self.dense_curve_report(),
            ChatModelMode::SparseCurve => Some(self.sparse_phi_curve_report()),
            ChatModelMode::SparseScalar => Some(self.sparse_phi_curve_report()),
        }
    }

    fn sparse_phi_curve_report(&self) -> String {
        let mut global_curve = None::<Vec<f64>>;

        for classifier in &self.classifiers {
            let Some(points) = classifier.aggregate_curve_points() else {
                continue;
            };

            add_curve_points(&mut global_curve, &points);
        }

        let mut output = String::new();

        if let Some(points) = global_curve {
            output.push_str("phi_all\n");
            output.push_str(&draw_curve(&points, 48));
            output.push('\n');
        }

        if output.is_empty() {
            output.push_str("no phi terms are learned yet\n");
        }

        output
    }

    fn rebuild_vocabulary(&mut self) {
        let mut words = BTreeSet::new();

        for example in &self.examples {
            for token in tokenize(&example.message) {
                words.insert(format!("msg:{token}"));
            }

            for feature in &example.context_features {
                words.insert(feature.name.clone());
            }
        }

        self.vocabulary = words.into_iter().collect();
    }

    fn rebuild_responses(&mut self) {
        let mut responses = BTreeSet::new();

        for example in &self.examples {
            responses.insert(example.response.clone());
        }

        self.responses = responses.into_iter().collect();
    }

    fn features_with_context(
        &self,
        message: &str,
        context_features: &[ContextFeature],
    ) -> Vec<WeightedFeature> {
        let mut features = self.message_features(message);
        features.extend(self.weighted_context_features(context_features));
        normalized_features(features)
    }

    fn weighted_context_features(
        &self,
        context_features: &[ContextFeature],
    ) -> Vec<WeightedFeature> {
        context_features
            .iter()
            .filter_map(|context_feature| {
                self.vocabulary
                    .binary_search(&context_feature.name)
                    .ok()
                    .map(|index| WeightedFeature {
                        index,
                        value: (context_feature.value * CONTEXT_FEATURE_BOOST).min(1.0),
                    })
            })
            .collect()
    }

    fn message_features(&self, message: &str) -> Vec<WeightedFeature> {
        let tokens = tokenize(message).into_iter().collect::<BTreeSet<_>>();

        self.vocabulary
            .iter()
            .enumerate()
            .filter_map(|(index, word)| {
                if let Some(token) = word.strip_prefix("msg:") {
                    tokens
                        .contains(token)
                        .then_some(WeightedFeature { index, value: 1.0 })
                } else {
                    None
                }
            })
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct WeightedFeature {
    pub(crate) index: usize,
    pub(crate) value: f64,
}

#[derive(Clone, Debug)]
pub struct ContextFeature {
    name: String,
    value: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChatModelMode {
    DenseCurve,
    SparseScalar,
    SparseCurve,
}

impl ChatModelMode {
    fn name(self) -> &'static str {
        match self {
            Self::DenseCurve => "dense curve",
            Self::SparseScalar => "sparse scalar",
            Self::SparseCurve => "sparse curve",
        }
    }
}

fn normalized_context_features(features: Vec<ContextFeature>) -> Vec<ContextFeature> {
    let mut weights = HashMap::<String, f64>::new();

    for feature in features {
        *weights.entry(feature.name).or_insert(0.0) += feature.value;
    }

    let mut features = weights
        .into_iter()
        .filter(|(_, value)| value.abs() >= SESSION_CONTEXT_MIN_WEIGHT)
        .map(|(name, value)| ContextFeature { name, value })
        .collect::<Vec<_>>();
    features.sort_by(|left, right| left.name.cmp(&right.name));
    features
}

fn normalized_features(features: Vec<WeightedFeature>) -> Vec<WeightedFeature> {
    let mut weights = HashMap::<usize, f64>::new();

    for feature in features {
        *weights.entry(feature.index).or_insert(0.0) += feature.value;
    }

    let mut features = weights
        .into_iter()
        .filter(|(_, value)| value.abs() >= SESSION_CONTEXT_MIN_WEIGHT)
        .map(|(index, value)| WeightedFeature {
            index,
            value: value.min(1.0),
        })
        .collect::<Vec<_>>();
    features.sort_by_key(|feature| feature.index);
    features
}

fn features_equal(left: &[ContextFeature], right: &[ContextFeature]) -> bool {
    if left.len() != right.len() {
        return false;
    }

    left.iter()
        .zip(right)
        .all(|(left, right)| left.name == right.name && (left.value - right.value).abs() < 1e-9)
}

#[derive(Debug)]
pub struct ChatExample {
    message: String,
    response: String,
    context_features: Vec<ContextFeature>,
}

impl ChatExample {
    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn response(&self) -> &str {
        &self.response
    }

    pub fn context_features(&self) -> &[ContextFeature] {
        &self.context_features
    }
}

#[derive(Clone, Debug)]
pub struct ChatPrediction {
    pub response: String,
    pub score: f64,
}

#[derive(Clone, Debug, Default)]
pub struct SessionContext {
    features: HashMap<String, f64>,
}

impl SessionContext {
    fn context_features(&self) -> Vec<ContextFeature> {
        normalized_context_features(
            self.features
                .iter()
                .map(|(name, value)| ContextFeature {
                    name: name.clone(),
                    value: *value,
                })
                .collect(),
        )
    }

    fn decay(&mut self) {
        self.features
            .values_mut()
            .for_each(|weight| *weight *= SESSION_CONTEXT_DECAY);
        self.features
            .retain(|_, weight| weight.abs() >= SESSION_CONTEXT_MIN_WEIGHT);
    }

    fn add_text(&mut self, text: &str, weight: f64) {
        for token in tokenize(text) {
            *self.features.entry(format!("msg:{token}")).or_insert(0.0) += weight;
        }
    }

    fn record_turn(&mut self, message: &str, response: Option<&str>) {
        self.decay();
        self.add_text(message, SESSION_CONTEXT_INPUT_WEIGHT);

        if let Some(response) = response {
            self.add_text(response, SESSION_CONTEXT_RESULT_WEIGHT);
        }
    }

    pub(crate) fn clear(&mut self) {
        self.features.clear();
    }
}

pub(crate) fn tokenize(message: &str) -> Vec<String> {
    message
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .filter(|token| !token.is_empty())
        .map(str::to_string)
        .collect()
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn seed_chatbot() -> ChatBot {
    seed_chatbot_with_mode(ChatModelMode::SparseCurve)
}

pub fn seed_chatbot_with_mode(mode: ChatModelMode) -> ChatBot {
    let mut bot = ChatBot::new_with_mode(mode);

    bot.add_example("hello", "Hello. What would you like to work on?");
    bot.add_example("hi there", "Hello. What would you like to work on?");
    bot.add_example("good morning", "Hello. What would you like to work on?");
    bot.add_example("bye", "Goodbye.");
    bot.add_example("goodbye", "Goodbye.");
    bot.add_example("see you later", "Goodbye.");
    bot.add_example("what is your name", "I am a small PhiNetwork chatbot.");
    bot.add_example("who are you", "I am a small PhiNetwork chatbot.");
    bot.add_example(
        "help me",
        "Use add, train, ask, examples, responses, vocab, or quit.",
    );
    bot.add_example(
        "what can you do",
        "Use add, train, ask, examples, responses, vocab, or quit.",
    );
    bot.train(DEFAULT_TRAIN_EPOCHS, DEFAULT_TRAIN_EPSILON);

    bot
}

fn add_examples_from_tsv(bot: &mut ChatBot, seed_data: &str) -> LoadResult {
    let mut result = LoadResult::default();

    for line in seed_data.lines() {
        let line = line.trim();

        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        if let Some((message, rest)) = line.split_once('\t') {
            result.total += 1;
            let message = unescape_tsv(message);
            let (response, context_features) = rest
                .split_once('\t')
                .map(|(response, encoded_context)| {
                    (
                        unescape_tsv(response),
                        parse_context_features(&unescape_tsv(encoded_context)),
                    )
                })
                .unwrap_or_else(|| (unescape_tsv(rest), Vec::new()));

            if bot.add_example_with_context_if_missing(&message, &response, context_features) {
                result.added += 1;
            } else {
                result.skipped += 1;
            }
        }
    }

    result
}

fn load_examples_from_file(bot: &mut ChatBot, path: impl AsRef<Path>) -> io::Result<LoadResult> {
    let path = path.as_ref();

    if !path.exists() {
        return Ok(LoadResult::default());
    }

    let contents = fs::read_to_string(path)?;
    Ok(add_examples_from_tsv(bot, &contents))
}

pub(crate) fn append_example_to_file(
    path: impl AsRef<Path>,
    message: &str,
    response: &str,
    context_features: &[ContextFeature],
) -> io::Result<()> {
    let path = path.as_ref();

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    if context_features.is_empty() {
        writeln!(file, "{}\t{}", escape_tsv(message), escape_tsv(response))
    } else {
        writeln!(
            file,
            "{}\t{}\t{}",
            escape_tsv(message),
            escape_tsv(response),
            escape_tsv(&format_context_features(context_features))
        )
    }
}

pub(crate) fn format_context_features(features: &[ContextFeature]) -> String {
    normalized_context_features(features.to_vec())
        .iter()
        .map(|feature| format!("{}={:.6}", feature.name, feature.value))
        .collect::<Vec<_>>()
        .join(";")
}

fn parse_context_features(encoded: &str) -> Vec<ContextFeature> {
    if encoded.contains('=') {
        return normalized_context_features(
            encoded
                .split(';')
                .filter_map(|part| {
                    let (name, value) = part.split_once('=')?;
                    Some(ContextFeature {
                        name: name.to_string(),
                        value: value.parse::<f64>().ok()?,
                    })
                })
                .collect(),
        );
    }

    // Backward compatibility for older memory rows that stored raw previous text.
    normalized_context_features(
        tokenize(encoded)
            .into_iter()
            .map(|token| ContextFeature {
                name: format!("msg:{token}"),
                value: SESSION_CONTEXT_RESULT_WEIGHT,
            })
            .collect(),
    )
}

fn escape_tsv(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('\t', "\\t")
        .replace('\n', "\\n")
}

fn unescape_tsv(value: &str) -> String {
    let mut output = String::new();
    let mut chars = value.chars();

    while let Some(ch) = chars.next() {
        if ch != '\\' {
            output.push(ch);
            continue;
        }

        match chars.next() {
            Some('t') => output.push('\t'),
            Some('n') => output.push('\n'),
            Some('\\') => output.push('\\'),
            Some(other) => {
                output.push('\\');
                output.push(other);
            }
            None => output.push('\\'),
        }
    }

    output
}

#[derive(Debug, Default, PartialEq, Eq)]
struct LoadResult {
    total: usize,
    added: usize,
    skipped: usize,
}

pub fn run_chatbot_cli() -> io::Result<()> {
    let mode = prompt_model_mode()?;
    let mut bot = seed_chatbot_with_mode(mode);
    let mut session_context = SessionContext::default();
    let memory_load = load_examples_from_file(&mut bot, MEMORY_PATH)?;

    bot.train(DEFAULT_TRAIN_EPOCHS, DEFAULT_TRAIN_EPSILON);

    println!("PhiNetwork chatbot");
    println!("model: {}", bot.mode().name());
    println!(
        "seeded with {} examples, {} words, {} responses",
        bot.example_count(),
        bot.vocabulary().len(),
        bot.responses().len()
    );
    println!(
        "loaded {} remembered examples from {MEMORY_PATH}",
        memory_load.added
    );
    if bot.mode() == ChatModelMode::DenseCurve {
        println!("dense curve mode retrains from examples at startup");
    }
    commands::help::run();

    loop {
        print!("> ");
        io::stdout().flush()?;

        let mut line = String::new();
        if io::stdin().read_line(&mut line)? == 0 {
            break;
        }

        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        if matches!(
            commands::dispatch(line, &mut bot, &mut session_context)?,
            CommandAction::Quit
        ) {
            break;
        }
    }

    Ok(())
}

fn prompt_model_mode() -> io::Result<ChatModelMode> {
    println!("Choose chatbot model:");
    println!("  1. dense curve   - smoother, original curve PhiNetwork, poor scaling");
    println!("  2. sparse scalar - scalable, more exact lexical matching");
    println!("  3. sparse curve  - scalable sparse terms with learned curves");
    print!("model [3]: ");
    io::stdout().flush()?;

    let mut line = String::new();
    io::stdin().read_line(&mut line)?;

    Ok(match line.trim() {
        "1" | "dense" | "dense curve" => ChatModelMode::DenseCurve,
        "2" | "sparse scalar" | "scalar" => ChatModelMode::SparseScalar,
        _ => ChatModelMode::SparseCurve,
    })
}

pub(crate) fn answer_or_learn(
    bot: &mut ChatBot,
    session_context: &mut SessionContext,
    message: &str,
) -> io::Result<()> {
    let context_features = session_context.context_features();
    let prediction_chain = recursive_prediction_chain(bot, session_context, message);

    if !prediction_chain.confident.is_empty() {
        for prediction in &prediction_chain.confident {
            print_prediction(prediction);
        }

        if let Some(prediction) = &prediction_chain.terminal {
            print_below_threshold_prediction(prediction);
        }

        record_prediction_chain(session_context, message, &prediction_chain.confident);
        return Ok(());
    }

    if let Some(prediction) = prediction_chain.terminal {
        let confidence = prediction.score.clamp(0.0, 1.0);
        println!(
            "I am not confident. Best guess was: {} ({confidence:.3})",
            prediction.response
        );
    } else {
        println!("I do not have a trained response yet.");
    }

    print!("Teach me the right response, or press Enter to skip: ");
    io::stdout().flush()?;

    let mut response = String::new();
    if io::stdin().read_line(&mut response)? == 0 {
        return Ok(());
    }

    let response = response.trim();
    if response.is_empty() {
        println!("skipped");
        session_context.record_turn(message, None);
        return Ok(());
    }

    if bot.add_example_with_context_if_missing(message, response, context_features.clone()) {
        append_example_to_file(MEMORY_PATH, message, response, &context_features)?;
        bot.train(DEFAULT_TRAIN_EPOCHS, DEFAULT_TRAIN_EPSILON);
        session_context.record_turn(message, Some(response));
        println!(
            "learned. trained {} examples into {} responses with {} word features",
            bot.example_count(),
            bot.responses().len(),
            bot.vocabulary().len()
        );
    } else {
        println!("that example already exists");
    }

    Ok(())
}

#[derive(Debug)]
struct PredictionChain {
    confident: Vec<ChatPrediction>,
    terminal: Option<ChatPrediction>,
}

fn recursive_prediction_chain(
    bot: &ChatBot,
    session_context: &SessionContext,
    message: &str,
) -> PredictionChain {
    let mut context = session_context.clone();
    let mut input = message.to_string();
    let mut seen_responses = BTreeSet::new();
    let mut confident = Vec::new();

    for _ in 0..MAX_RECURSIVE_RESULT_DEPTH {
        let Some(prediction) = bot.reply_with_recursive_formula(&input, &context) else {
            return PredictionChain {
                confident,
                terminal: None,
            };
        };

        if needs_training(Some(&prediction)) {
            return PredictionChain {
                confident,
                terminal: Some(prediction),
            };
        }

        let response = prediction.response.clone();
        context.record_turn(&input, Some(&response));
        confident.push(prediction);

        if !seen_responses.insert(response.clone()) {
            return PredictionChain {
                confident,
                terminal: None,
            };
        }

        input = response;
    }

    PredictionChain {
        confident,
        terminal: None,
    }
}

fn record_prediction_chain(
    session_context: &mut SessionContext,
    message: &str,
    predictions: &[ChatPrediction],
) {
    let mut input = message.to_string();

    for prediction in predictions {
        session_context.record_turn(&input, Some(&prediction.response));
        input = prediction.response.clone();
    }
}

fn needs_training(prediction: Option<&ChatPrediction>) -> bool {
    prediction
        .map(|prediction| prediction.score.clamp(0.0, 1.0) < UNKNOWN_CONFIDENCE_THRESHOLD)
        .unwrap_or(true)
}

fn print_prediction(prediction: &ChatPrediction) {
    let confidence = prediction.score.clamp(0.0, 1.0);
    println!("{} ({confidence:.3})", prediction.response);
}

fn print_below_threshold_prediction(prediction: &ChatPrediction) {
    let confidence = prediction.score.clamp(0.0, 1.0);
    println!(
        "stopped below threshold: {} ({confidence:.3})",
        prediction.response
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeded_bot_answers_greeting() {
        let bot = seed_chatbot();
        let prediction = bot.reply("hello there").expect("prediction");

        assert_eq!(
            prediction.response,
            "Hello. What would you like to work on?"
        );
    }

    #[test]
    fn can_train_custom_response() {
        let mut bot = ChatBot::new();
        bot.add_example("rust ownership borrow checker", "Rust answer");
        bot.add_example("pizza tomato cheese", "Pizza answer");
        bot.train(1_000, 0.01);

        let prediction = bot.reply("borrow checker").expect("prediction");
        assert_eq!(prediction.response, "Rust answer");
    }

    #[test]
    fn tokenizes_words_case_and_punctuation_insensitively() {
        assert_eq!(
            tokenize("Hello, Rust-world! 2026"),
            vec!["hello", "rust", "world", "2026"]
        );
    }

    #[test]
    fn draws_ascii_curve_points() {
        let drawing = draw_curve(&[0.0, 0.5, 1.0], 8);

        assert!(drawing.contains("x=0.00"));
        assert!(drawing.contains("x=1.00"));
        assert!(drawing.contains('*'));
        assert!(drawing.contains('+'));
    }

    #[test]
    fn sparse_curve_report_draws_global_phi() {
        let mut bot = ChatBot::new_with_mode(ChatModelMode::SparseCurve);
        bot.add_example("rust borrow checker", "Rust answer");
        bot.add_example("pizza tomato basil", "Pizza answer");
        bot.train(1_000, 0.01);

        let report = bot.curve_report().expect("curve report");

        assert!(report.contains("phi_all"));
        assert!(!report.contains("response phi:"));
        assert!(report.contains('*'));
    }

    #[test]
    fn dense_curve_report_draws_global_phi() {
        let mut bot = ChatBot::new_with_mode(ChatModelMode::DenseCurve);
        bot.add_example("rust borrow checker", "Rust answer");
        bot.add_example("pizza tomato basil", "Pizza answer");
        bot.train(1_000, 0.01);

        let report = bot.curve_report().expect("curve report");

        assert!(report.contains("phi_all"));
        assert!(!report.contains("response:"));
        assert!(!report.contains("phi(msg:"));
        assert!(report.contains('*'));
    }

    #[test]
    fn sparse_scalar_curve_report_draws_global_phi() {
        let mut bot = ChatBot::new_with_mode(ChatModelMode::SparseScalar);
        bot.add_example("rust borrow checker", "Rust answer");
        bot.add_example("pizza tomato basil", "Pizza answer");
        bot.train(1_000, 0.01);

        let report = bot.curve_report().expect("curve report");

        assert!(report.contains("phi_all"));
        assert!(report.contains('*'));
    }

    #[test]
    fn all_model_modes_can_train_basic_response() {
        for mode in [
            ChatModelMode::DenseCurve,
            ChatModelMode::SparseScalar,
            ChatModelMode::SparseCurve,
        ] {
            let mut bot = ChatBot::new_with_mode(mode);
            bot.add_example("rust ownership borrow checker", "Rust answer");
            bot.add_example("pizza tomato cheese", "Pizza answer");
            bot.train(1_000, 0.01);

            let prediction = bot.reply("borrow checker").expect("prediction");
            assert_eq!(prediction.response, "Rust answer", "mode {mode:?}");
        }
    }

    #[test]
    fn unknown_messages_need_training() {
        let bot = seed_chatbot();
        let prediction = bot.reply("zebra nebula capacitor");

        assert!(needs_training(prediction.as_ref()));
    }

    #[test]
    fn last_result_adds_bounded_response_context() {
        let bot = seed_chatbot();
        let previous = ChatPrediction {
            response: "Goodbye.".to_string(),
            score: 1.0,
        };

        let without_context = bot.reply("zebra nebula capacitor").expect("prediction");
        let with_context = bot
            .reply_with_last_result("zebra nebula capacitor", Some(&previous))
            .expect("prediction");

        assert!(with_context.score >= without_context.score);
        assert!(needs_training(Some(&with_context)));
    }

    #[test]
    fn learning_can_depend_on_previous_result_context() {
        let mut bot = ChatBot::new();
        bot.add_example_with_context(
            "continue",
            "Borrowing answer",
            vec![ContextFeature {
                name: "msg:rust".to_string(),
                value: SESSION_CONTEXT_RESULT_WEIGHT,
            }],
        );
        bot.add_example_with_context(
            "continue",
            "Cheese answer",
            vec![ContextFeature {
                name: "msg:pizza".to_string(),
                value: SESSION_CONTEXT_RESULT_WEIGHT,
            }],
        );
        bot.train(2_000, 0.01);

        let mut rust_context = SessionContext::default();
        rust_context.add_text("rust", SESSION_CONTEXT_RESULT_WEIGHT);
        let mut pizza_context = SessionContext::default();
        pizza_context.add_text("pizza", SESSION_CONTEXT_RESULT_WEIGHT);

        let rust_prediction = bot
            .reply_with_session_context("continue", &rust_context)
            .expect("rust context prediction");
        let pizza_prediction = bot
            .reply_with_session_context("continue", &pizza_context)
            .expect("pizza context prediction");

        assert_eq!(rust_prediction.response, "Borrowing answer");
        assert_eq!(pizza_prediction.response, "Cheese answer");
    }

    #[test]
    fn confident_result_is_recursively_scored_as_next_input() {
        let mut bot = ChatBot::new();
        bot.add_example("start", "middle");
        bot.add_example("middle", "end");
        bot.add_example("pizza tomato", "pizza answer");
        bot.train(2_000, 0.01);

        let chain = recursive_prediction_chain(&bot, &SessionContext::default(), "start");

        assert!(chain.confident.len() >= 2);
        assert_eq!(chain.confident[0].response, "middle");
        assert_eq!(chain.confident[1].response, "end");
    }

    #[test]
    fn recursive_formula_scores_result_and_prompt_result_interaction() {
        let mut bot = ChatBot::new();
        bot.add_example("rust borrow checker", "Rust answer");
        bot.add_example("Rust answer", "Rust follow up");
        bot.add_example_with_context(
            "rust borrow checker",
            "Rust follow up",
            vec![ContextFeature {
                name: "msg:rust".to_string(),
                value: SESSION_CONTEXT_RESULT_WEIGHT,
            }],
        );
        bot.add_example("pizza tomato basil", "Pizza answer");
        bot.train(2_000, 0.01);

        let base = bot
            .reply_with_session_context_base("rust borrow checker", &SessionContext::default())
            .expect("base prediction");
        let recursive = bot
            .reply_with_recursive_formula("rust borrow checker", &SessionContext::default())
            .expect("recursive prediction");

        assert!(recursive.score >= base.score);
    }

    #[test]
    fn persists_and_reloads_memory_examples() {
        let path = std::env::temp_dir().join("phinetwork_chatbot_memory_test.tsv");
        let _ = fs::remove_file(&path);

        append_example_to_file(
            &path,
            "remember espresso",
            "Coffee mode",
            &[ContextFeature {
                name: "msg:previous".to_string(),
                value: 0.15,
            }],
        )
        .unwrap();

        let mut bot = ChatBot::new();
        let result = load_examples_from_file(&mut bot, &path).unwrap();

        assert_eq!(
            result,
            LoadResult {
                total: 1,
                added: 1,
                skipped: 0,
            }
        );
        assert_eq!(bot.examples()[0].message(), "remember espresso");
        assert_eq!(bot.examples()[0].response(), "Coffee mode");
        assert_eq!(
            format_context_features(bot.examples()[0].context_features()),
            "msg:previous=0.150000"
        );

        let _ = fs::remove_file(&path);
    }
}
