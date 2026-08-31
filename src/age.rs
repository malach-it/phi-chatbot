use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
#[cfg(not(test))]
use std::fs::File;
#[cfg(not(test))]
use std::io::Read;
use std::io::{self, Write};
#[cfg(not(test))]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(not(test))]
use std::time::{SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

use crate::classifiers::draw_curve_graph;
use crate::phinetwork::{PhiNetwork, TrainingExample};

const MAX_AGE: usize = 120;
const ADULT_AGE: usize = 18;
const DEFAULT_EPOCHS: usize = 2_000;
const DEFAULT_EPSILON: f64 = 0.02;
const MODEL_PATH: &str = "data/phi_age.tsv";

pub(crate) fn run(rest: &str) -> io::Result<()> {
    let Some(options) = TrainOptions::parse(rest) else {
        println!("usage: train age <name-age.tsv> [epochs] [epsilon]");
        return Ok(());
    };
    let source = fs::read_to_string(&options.path)?;
    let samples = match parse_samples(&source) {
        Ok(samples) => samples,
        Err(error) => {
            println!("could not train age curves: {error}");
            return Ok(());
        }
    };
    let curves = AgeCurves::train(&samples, options.epochs, options.epsilon);

    fs::create_dir_all("data")?;
    fs::write(MODEL_PATH, curves.snapshot())?;

    println!(
        "trained phi1 o phi2 on {} names and stored curves plus name fingerprints in {MODEL_PATH}",
        samples.len()
    );
    println!(
        "training accuracy: {:.1}%",
        curves.accuracy(&samples) * 100.0
    );

    Ok(())
}

pub(crate) fn run_over18(name: &str) -> io::Result<()> {
    let name = name.trim();
    if name.is_empty() {
        println!("usage: over18 <name>");
        return Ok(());
    }

    let snapshot = match fs::read_to_string(MODEL_PATH) {
        Ok(snapshot) => snapshot,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            println!("age curves are not trained; run: train age <name-age.tsv>");
            return Ok(());
        }
        Err(error) => return Err(error),
    };
    let Some(mut curves) = AgeCurves::from_snapshot(&snapshot) else {
        println!("could not load age curves from {MODEL_PATH}; retrain them with train age");
        return Ok(());
    };

    if !curves.knows(name) {
        print!("I do not know {name}. Enter age, or press Enter to skip: ");
        io::stdout().flush()?;

        let mut age = String::new();
        if io::stdin().read_line(&mut age)? == 0 || age.trim().is_empty() {
            println!("skipped");
            return Ok(());
        }
        let Ok(age) = age.trim().parse::<usize>() else {
            println!("age must be an integer between 0 and {MAX_AGE}");
            return Ok(());
        };
        if age > MAX_AGE {
            println!("age must be an integer between 0 and {MAX_AGE}");
            return Ok(());
        }

        curves.learn(name, age, DEFAULT_EPOCHS, DEFAULT_EPSILON);
        fs::write(MODEL_PATH, curves.snapshot())?;
        println!("learned {name} and updated the stored age curves");
    }

    println!(
        "over18({name}): {}",
        curves.predicts_adult(encode_name(name))
    );
    Ok(())
}

pub(crate) fn curve_report() -> io::Result<Option<String>> {
    let snapshot = match fs::read_to_string(MODEL_PATH) {
        Ok(snapshot) => snapshot,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };

    Ok(AgeCurves::from_snapshot(&snapshot).map(|curves| curves.graph_report()))
}

#[derive(Debug)]
struct TrainOptions {
    path: String,
    epochs: usize,
    epsilon: f64,
}

impl TrainOptions {
    fn parse(rest: &str) -> Option<Self> {
        let mut parts = rest.split_whitespace();
        let path = parts.next()?.to_string();
        let epochs = parts
            .next()
            .map(str::parse::<usize>)
            .transpose()
            .ok()?
            .unwrap_or(DEFAULT_EPOCHS);
        let epsilon = parts
            .next()
            .map(str::parse::<f64>)
            .transpose()
            .ok()?
            .unwrap_or(DEFAULT_EPSILON);

        (parts.next().is_none() && epochs > 0 && epsilon.is_finite() && epsilon > 0.0).then_some(
            Self {
                path,
                epochs,
                epsilon,
            },
        )
    }
}

#[derive(Clone, Debug)]
struct AgeSample {
    name_input: f64,
    name_fingerprint: String,
    age: usize,
}

#[derive(Debug)]
struct AgeCurves {
    phi2: PhiNetwork,
    phi1: PhiNetwork,
    known_names: BTreeSet<String>,
}

impl AgeCurves {
    fn train(samples: &[AgeSample], epochs: usize, epsilon: f64) -> Self {
        let curve_knots = (samples.len() * 32).clamp(64, 16_384);
        let vocabulary_seed = random_seed();
        let age_vocabulary = samples
            .iter()
            .map(|sample| sample.age)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .map(|age| (age, random_age_token(&vocabulary_seed, age)))
            .collect::<BTreeMap<_, _>>();
        let phi2_data = samples
            .iter()
            .map(|sample| TrainingExample {
                inputs: vec![sample.name_input],
                target: age_vocabulary[&sample.age],
            })
            .collect::<Vec<_>>();
        let mut phi2 = PhiNetwork::new_with_curve_knots(1, 0.1, curve_knots);
        phi2.train_until_quiet(&phi2_data, epsilon, epochs, 1);

        let phi1_data = samples
            .iter()
            .map(|sample| TrainingExample {
                inputs: vec![phi2.predict(&[sample.name_input])],
                target: adult_target(sample.age),
            })
            .collect::<Vec<_>>();
        let mut phi1 = PhiNetwork::new_with_curve_knots(1, 0.1, 64);
        phi1.train_until_quiet(&phi1_data, epsilon, epochs, 1);

        Self {
            phi2,
            phi1,
            known_names: samples
                .iter()
                .map(|sample| sample.name_fingerprint.clone())
                .collect(),
        }
    }

    fn predicts_adult(&self, name_input: f64) -> bool {
        let age_token = self.phi2.predict(&[name_input]);
        self.phi1.predict(&[age_token]) >= 0.5
    }

    fn knows(&self, name: &str) -> bool {
        self.known_names.contains(&fingerprint_name(name))
    }

    fn learn(&mut self, name: &str, age: usize, epochs: usize, epsilon: f64) {
        let name_input = encode_name(name);
        let age_token = random_age_token(&random_seed(), age);
        self.phi2.train_existing_until_quiet(
            &[TrainingExample {
                inputs: vec![name_input],
                target: age_token,
            }],
            epsilon,
            epochs,
        );
        let learned_token = self.phi2.predict(&[name_input]);
        self.phi1.train_existing_until_quiet(
            &[TrainingExample {
                inputs: vec![learned_token],
                target: adult_target(age),
            }],
            epsilon,
            epochs,
        );
        self.known_names.insert(fingerprint_name(name));
    }

    fn from_snapshot(snapshot: &str) -> Option<Self> {
        let mut phi2_points = None;
        let mut phi1_points = None;
        let mut known_names = BTreeSet::new();
        for line in snapshot.lines() {
            let fields = line.split('\t').collect::<Vec<_>>();
            match fields.as_slice() {
                ["max_age", value] if *value == MAX_AGE.to_string() => {}
                ["adult_age", value] if *value == ADULT_AGE.to_string() => {}
                ["phi2", "curve", points] => phi2_points = parse_points(points),
                ["phi1", "curve", points] => phi1_points = parse_points(points),
                ["known", fingerprint] if valid_fingerprint(fingerprint) => {
                    known_names.insert((*fingerprint).to_string());
                }
                _ => return None,
            }
        }

        if known_names.is_empty() {
            return None;
        }

        Some(Self {
            phi2: PhiNetwork::from_single_curve_points(phi2_points?)?,
            phi1: PhiNetwork::from_single_curve_points(phi1_points?)?,
            known_names,
        })
    }

    fn accuracy(&self, samples: &[AgeSample]) -> f64 {
        let correct = samples
            .iter()
            .filter(|sample| self.predicts_adult(sample.name_input) == (sample.age >= ADULT_AGE))
            .count();

        correct as f64 / samples.len() as f64
    }

    fn snapshot(&self) -> String {
        let phi2 = format_points(self.phi2.curve_points(0).unwrap_or_default());
        let phi1 = format_points(self.phi1.curve_points(0).unwrap_or_default());
        let known_names = self
            .known_names
            .iter()
            .map(|fingerprint| format!("known\t{fingerprint}\n"))
            .collect::<String>();

        format!(
            "max_age\t{MAX_AGE}\nadult_age\t{ADULT_AGE}\nphi2\tcurve\t{phi2}\nphi1\tcurve\t{phi1}\n{known_names}"
        )
    }

    fn graph_report(&self) -> String {
        let mut output = String::new();

        output.push_str("phi2 (encoded name -> random age token)\n");
        output.push_str(&draw_curve_graph(
            self.phi2.curve_points(0).unwrap_or_default(),
            48,
        ));
        output.push('\n');
        output.push_str("phi1 (random age token -> over18)\n");
        output.push_str(&draw_curve_graph(
            self.phi1.curve_points(0).unwrap_or_default(),
            48,
        ));
        output.push('\n');

        output
    }
}

fn parse_samples(source: &str) -> Result<Vec<AgeSample>, AgeDataError> {
    let mut ages_by_name = BTreeMap::<String, usize>::new();

    for (index, raw_line) in source.lines().enumerate() {
        let line_number = index + 1;
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }

        let Some((name, age)) = line.split_once('\t') else {
            return Err(AgeDataError::new(line_number, "expected name<TAB>age"));
        };
        let name = name.trim();
        let age = age
            .trim()
            .parse::<usize>()
            .map_err(|_| AgeDataError::new(line_number, "age must be an integer"))?;

        if name.is_empty() {
            return Err(AgeDataError::new(line_number, "name cannot be empty"));
        }
        if age > MAX_AGE {
            return Err(AgeDataError::new(
                line_number,
                &format!("age must be between 0 and {MAX_AGE}"),
            ));
        }
        if let Some(previous_age) = ages_by_name.insert(name.to_string(), age) {
            if previous_age != age {
                return Err(AgeDataError::new(
                    line_number,
                    "the same name has conflicting ages",
                ));
            }
        }
    }

    if ages_by_name.is_empty() {
        return Err(AgeDataError::new(0, "the dataset is empty"));
    }

    Ok(ages_by_name
        .into_iter()
        .map(|(name, age)| AgeSample {
            name_input: encode_name(&name),
            name_fingerprint: fingerprint_name(&name),
            age,
        })
        .collect())
}

fn encode_name(name: &str) -> f64 {
    let digest = name_digest(name);
    let mut bytes = [0_u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    let unit = u64::from_be_bytes(bytes) as f64 / u64::MAX as f64;

    0.05 + 0.95 * unit
}

fn fingerprint_name(name: &str) -> String {
    name_digest(name)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn name_digest(name: &str) -> [u8; 32] {
    Sha256::digest(name.trim().to_lowercase().as_bytes()).into()
}

fn valid_fingerprint(fingerprint: &str) -> bool {
    fingerprint.len() == 64
        && fingerprint
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn adult_target(age: usize) -> f64 {
    usize::from(age >= ADULT_AGE) as f64
}

fn random_age_token(seed: &[u8; 32], age: usize) -> f64 {
    let mut hasher = Sha256::new();
    hasher.update(b"phi-age-vocabulary-token-v1");
    hasher.update(seed);
    hasher.update(age.to_be_bytes());
    let digest = hasher.finalize();
    let mut bytes = [0_u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    let unit = u64::from_be_bytes(bytes) as f64 / u64::MAX as f64;

    0.05 + 0.90 * unit
}

#[cfg(test)]
fn random_seed() -> [u8; 32] {
    [0x5a; 32]
}

#[cfg(not(test))]
fn random_seed() -> [u8; 32] {
    let mut seed = [0_u8; 32];
    if File::open("/dev/urandom")
        .and_then(|mut file| file.read_exact(&mut seed))
        .is_ok()
    {
        return seed;
    }

    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let mut hasher = Sha256::new();
    hasher.update(b"phi-age-vocabulary-fallback-seed-v1");
    hasher.update(std::process::id().to_be_bytes());
    hasher.update(COUNTER.fetch_add(1, Ordering::Relaxed).to_be_bytes());
    hasher.update(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
            .to_be_bytes(),
    );
    hasher.finalize().into()
}

fn format_points(points: &[f64]) -> String {
    points
        .iter()
        .map(|point| point.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn parse_points(encoded: &str) -> Option<Vec<f64>> {
    let points = encoded
        .split(',')
        .map(|point| point.parse::<f64>().ok())
        .collect::<Option<Vec<_>>>()?;

    (points.len() >= 2 && points.iter().all(|point| point.is_finite())).then_some(points)
}

#[derive(Debug)]
struct AgeDataError {
    line: usize,
    message: String,
}

impl AgeDataError {
    fn new(line: usize, message: &str) -> Self {
        Self {
            line,
            message: message.to_string(),
        }
    }
}

impl fmt::Display for AgeDataError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.line == 0 {
            formatter.write_str(&self.message)
        } else {
            write!(formatter, "line {}: {}", self.line, self.message)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_name_and_age_rows() {
        let samples = parse_samples("Alice\t17\nBob\t18\n").expect("valid dataset");

        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].age, 17);
        assert_eq!(samples[1].age, 18);
        assert!(samples.iter().all(|sample| sample.name_input >= 0.05));
        assert!(samples
            .iter()
            .all(|sample| valid_fingerprint(&sample.name_fingerprint)));
    }

    #[test]
    fn rejects_conflicting_ages_for_one_name() {
        let error = parse_samples("Alice\t17\nAlice\t18\n").expect_err("conflict");

        assert_eq!(
            error.to_string(),
            "line 2: the same name has conflicting ages"
        );
    }

    #[test]
    fn trains_and_snapshots_only_curves() {
        let samples = parse_samples("Alice\t12\nBob\t35\n").expect("valid dataset");
        let curves = AgeCurves::train(&samples, 2_000, 0.02);
        let snapshot = curves.snapshot();

        assert_eq!(curves.accuracy(&samples), 1.0);
        assert!(snapshot.contains("phi2\tcurve\t"));
        assert!(snapshot.contains("phi1\tcurve\t"));
        assert!(!snapshot.contains("Alice"));
        assert!(!snapshot.contains("Bob"));
        assert!(!snapshot.contains("\t12\n"));
        assert!(!snapshot.contains("\t35\n"));
        assert!(snapshot.contains("known\t"));
    }

    #[test]
    fn stored_curves_can_answer_over18_without_source_rows() {
        let samples = parse_samples("Alice\t12\nBob\t35\n").expect("valid dataset");
        let trained = AgeCurves::train(&samples, 2_000, 0.02);
        let loaded = AgeCurves::from_snapshot(&trained.snapshot()).expect("stored curves");

        assert!(!loaded.predicts_adult(encode_name("Alice")));
        assert!(loaded.predicts_adult(encode_name("Bob")));
        assert!(loaded.knows("Alice"));
        assert!(!loaded.knows("Charlie"));
    }

    #[test]
    fn exact_ages_receive_distinct_random_vocabulary_tokens() {
        let seed = random_seed();

        let age_12 = random_age_token(&seed, 12);
        let age_35 = random_age_token(&seed, 35);

        assert!((0.05..=0.95).contains(&age_12));
        assert!((0.05..=0.95).contains(&age_35));
        assert_ne!(age_12, age_35);
    }

    #[test]
    fn phi2_learns_random_tokens_instead_of_normalized_age() {
        let samples = parse_samples("Alice\t12\nBob\t35\n").expect("valid dataset");
        let curves = AgeCurves::train(&samples, 2_000, 0.02);

        let alice_token = curves.phi2.predict(&[encode_name("Alice")]);
        let bob_token = curves.phi2.predict(&[encode_name("Bob")]);
        let seed = random_seed();

        assert!((alice_token - random_age_token(&seed, 12)).abs() <= 0.02);
        assert!((bob_token - random_age_token(&seed, 35)).abs() <= 0.02);
    }

    #[test]
    fn incrementally_learns_an_unknown_name() {
        let samples = parse_samples("Alice\t12\nBob\t35\n").expect("valid dataset");
        let mut curves = AgeCurves::train(&samples, 2_000, 0.02);

        curves.learn("Charlie", 24, 2_000, 0.02);

        assert!(curves.knows("Charlie"));
        assert!(curves.predicts_adult(encode_name("Charlie")));
        assert!(!curves.predicts_adult(encode_name("Alice")));
        assert!(curves.predicts_adult(encode_name("Bob")));
    }

    #[test]
    fn age_curve_report_contains_only_graphs() {
        let samples = parse_samples("Alice\t12\nBob\t35\n").expect("valid dataset");
        let curves = AgeCurves::train(&samples, 2_000, 0.02);
        let report = curves.graph_report();

        assert!(report.contains("phi2 (encoded name -> random age token)"));
        assert!(report.contains("phi1 (random age token -> over18)"));
        assert!(report.contains('*'));
        assert!(!report.contains("point x="));
        assert!(!report.contains("Alice"));
        assert!(!report.contains("Bob"));
    }
}
