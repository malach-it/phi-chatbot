use crate::chatbot::ChatBot;
use crate::phi_key::PhiKeyError;

pub(crate) fn run(bot: &ChatBot, rest: &str) {
    let Some(share_count) = parse_share_count(rest) else {
        println!("usage: keypair [shares]");
        return;
    };

    let key_pair = match bot.phi_all_key_pair(share_count) {
        Ok(key_pair) => key_pair,
        Err(error) => {
            println!("{}", key_error_message(error));
            return;
        }
    };

    if key_pair.shares.is_empty() {
        println!("could not encode phi_all key shares; no shares were produced");
        return;
    };

    println!("phi_all bls12-381 inspired key components");
    println!("source: deterministic phi_all curve points");
    println!("components: {}", key_pair.shares.len());
    println!("fingerprint: {}", key_pair.fingerprint_hex);

    let encoded_phi_points = match key_pair.encoded_phi_points() {
        Ok(encoded_phi_points) => encoded_phi_points,
        Err(error) => {
            println!("{}", key_error_message(error));
            return;
        }
    };

    println!("encoded phi:");
    for (point_index, point) in encoded_phi_points {
        println!("  {point_index}: {point}");
    }

    let encrypted_phin_shares = match key_pair.encrypted_phin_shares() {
        Ok(encrypted_phin_shares) => encrypted_phin_shares,
        Err(error) => {
            println!("{}", key_error_message(error));
            return;
        }
    };

    println!("encrypted phin shares:");
    for (index, formula, points) in encrypted_phin_shares {
        println!("  {index}: {formula}");
        for (point_index, point) in points {
            println!("    {point_index}: {point}");
        }
    }

    println!("component share formulas:");
    for (index, terms) in key_pair.component_terms() {
        println!("  {index}: {terms}");
    }

    println!(
        "note: BLS-inspired deterministic model key; not a standard BLS signature key, wallet, or identity secret"
    );
}

fn parse_share_count(rest: &str) -> Option<usize> {
    let rest = rest.trim();

    if rest.is_empty() {
        return Some(1);
    }

    let mut parts = rest.split_whitespace();
    let share_count = parts.next()?.parse::<usize>().ok()?;

    (share_count > 0 && parts.next().is_none()).then_some(share_count)
}

fn key_error_message(error: PhiKeyError) -> String {
    match error {
        PhiKeyError::EmptyPhiAll => {
            "could not encode phi_all key shares; phi_all has no curve points".to_string()
        }
        PhiKeyError::InvalidShareCount => {
            "could not encode phi_all key shares; share count must be greater than zero".to_string()
        }
        PhiKeyError::PointOutOfRange { point, limit } => format!(
            "could not encode phi_all key shares; phi_all point {point:.6} is outside +/-{limit:.1}"
        ),
        PhiKeyError::InvalidShare => {
            "could not encode phi_all key shares; generated an invalid share".to_string()
        }
    }
}
