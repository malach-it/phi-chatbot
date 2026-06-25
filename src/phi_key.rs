use std::collections::HashMap;
use std::sync::OnceLock;

use bls12_381::{G1Affine, G1Projective, Scalar};
use sha2::{Digest, Sha512};

const DOMAIN_SEPARATOR: &[u8] = b"phi-chatbot:phi-all-points:bls12-381-keypair:v1";
const POINT_MASK_DOMAIN_SEPARATOR: &[u8] = b"phi-chatbot:phi-all-point-mask:v1";
const POINT_SCALE: f64 = 10.0;
const POINT_LIMIT: i64 = 50_000_000;
const POINT_SCALAR_OFFSET: i64 = POINT_LIMIT + 3;
const NAN_POINT: i64 = POINT_LIMIT + 1;
const POSITIVE_INFINITY_POINT: i64 = POINT_LIMIT + 2;
const NEGATIVE_INFINITY_POINT: i64 = -POINT_LIMIT - 2;
#[cfg(test)]
const LOOKUP_POINT_LIMIT: i64 = 20;
#[cfg(not(test))]
const LOOKUP_POINT_LIMIT: i64 = POINT_LIMIT;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PhiKeyPair {
    pub(crate) shares: Vec<PhiKeyShare>,
    pub(crate) fingerprint_hex: String,
    recovered_phi_points: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PhiKeyShare {
    pub(crate) index: usize,
    pub(crate) share_hex: String,
    pub(crate) public_phi_points_hex: Vec<(usize, String)>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum PhiKeyError {
    EmptyPhiAll,
    InvalidShareCount,
    PointOutOfRange { point: f64, limit: f64 },
    InvalidShare,
}

impl PhiKeyPair {
    pub(crate) fn from_phi_all_points(
        points: &[f64],
        share_count: usize,
    ) -> Result<Self, PhiKeyError> {
        if share_count == 0 {
            return Err(PhiKeyError::InvalidShareCount);
        }

        if points.is_empty() {
            return Err(PhiKeyError::EmptyPhiAll);
        }

        let encoded_points = encode_phi_all_points(points);
        let shares = encode_phi_shares(points, &encoded_points, share_count)?;
        let recovered_phi_points = points
            .iter()
            .map(|point| quantize_point(*point).map(dequantize_point))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            shares,
            fingerprint_hex: hex_encode(&Sha512::digest(&encoded_points)[..16]),
            recovered_phi_points,
        })
    }

    pub(crate) fn encrypted_phin_shares(
        &self,
    ) -> Result<Vec<(usize, String, Vec<(usize, String)>)>, PhiKeyError> {
        self.component_terms()
            .into_iter()
            .map(|(share_index, terms)| {
                let points = self
                    .recovered_phi_points
                    .iter()
                    .enumerate()
                    .map(|(point_index, point)| {
                        encode_phin_share_point(*point, share_index, self.shares.len(), point_index)
                            .map(|encoded| (point_index, encoded))
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                Ok((share_index, terms, points))
            })
            .collect()
    }

    pub(crate) fn encoded_phi_points(&self) -> Result<Vec<(usize, String)>, PhiKeyError> {
        encoded_phi_points_from_points(&self.recovered_phi_points)
    }

    pub(crate) fn component_terms(&self) -> Vec<(usize, String)> {
        let share_count = self.shares.len();

        self.shares
            .iter()
            .map(|share| {
                let index = share.index;
                let singleton = format!("phi({})", share_label(index));
                let pairs = (0..share_count)
                    .filter(|other| *other != index)
                    .map(|other| format!("phi({})", pair_label(index, other)))
                    .collect::<Vec<_>>();
                let terms = std::iter::once(singleton)
                    .chain(pairs)
                    .collect::<Vec<_>>()
                    .join(" + ");

                (index, terms)
            })
            .collect()
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn recover_phi_all_points(&self) -> Option<Vec<f64>> {
        let point_count = self
            .shares
            .iter()
            .flat_map(|share| {
                share
                    .public_phi_points_hex
                    .iter()
                    .map(|(point_index, _)| *point_index)
            })
            .max()
            .map(|index| index + 1)?;

        let mut recovered = vec![None; point_count];

        for share in &self.shares {
            let share_scalar = scalar_from_hex(&share.share_hex)?;

            for (point_index, encoded_point) in &share.public_phi_points_hex {
                let point = g1_from_hex(encoded_point)?;
                let mask = point_mask(share_scalar, *point_index);
                let unmasked = G1Projective::from(point) - (G1Projective::generator() * mask);
                recovered[*point_index] = recover_quantized_point(&G1Affine::from(unmasked));
            }
        }

        recovered.into_iter().collect()
    }
}

pub(crate) fn encoded_phi_points_from_points(
    points: &[f64],
) -> Result<Vec<(usize, String)>, PhiKeyError> {
    points
        .iter()
        .enumerate()
        .map(|(point_index, point)| {
            encode_phi_point(*point, point_index).map(|encoded| (point_index, encoded))
        })
        .collect()
}

fn encode_phi_point(point: f64, point_index: usize) -> Result<String, PhiKeyError> {
    let quantized_point = quantize_point(point)?;
    let encoded_scalar =
        quantized_point_scalar(quantized_point) + point_mask(phi_secret(), point_index);
    let encoded_point = G1Affine::from(G1Projective::generator() * encoded_scalar);

    Ok(hex_encode(&encoded_point.to_compressed()))
}

fn phi_secret() -> Scalar {
    let mut hasher = Sha512::new();
    hasher.update(DOMAIN_SEPARATOR);
    hasher.update(b":encoded-phi");

    Scalar::from_bytes_wide(&hasher.finalize().into())
}

fn encode_phin_share_point(
    point: f64,
    share_index: usize,
    share_count: usize,
    point_index: usize,
) -> Result<String, PhiKeyError> {
    let quantized_point = quantize_point(point)?;
    let share_value = phin_share_value(quantized_point, share_index, share_count, point_index);
    let share_scalar = phin_share_secret(share_index);
    let encoded_scalar = share_value + point_mask(share_scalar, point_index);
    let encoded_point = G1Affine::from(G1Projective::generator() * encoded_scalar);

    Ok(hex_encode(&encoded_point.to_compressed()))
}

fn phin_share_value(
    quantized_point: i64,
    share_index: usize,
    share_count: usize,
    point_index: usize,
) -> Scalar {
    if share_count == 1 {
        return quantized_point_scalar(quantized_point);
    }

    if share_index + 1 < share_count {
        return phin_split_component(quantized_point, share_index, point_index);
    }

    let mut value = quantized_point_scalar(quantized_point);
    for index in 0..share_count - 1 {
        value -= phin_split_component(quantized_point, index, point_index);
    }

    value
}

fn phin_split_component(quantized_point: i64, share_index: usize, point_index: usize) -> Scalar {
    let mut hasher = Sha512::new();
    hasher.update(DOMAIN_SEPARATOR);
    hasher.update(b":phin-split");
    hasher.update(quantized_point.to_le_bytes());
    hasher.update((share_index as u64).to_le_bytes());
    hasher.update((point_index as u64).to_le_bytes());

    Scalar::from_bytes_wide(&hasher.finalize().into())
}

fn phin_share_secret(share_index: usize) -> Scalar {
    let mut hasher = Sha512::new();
    hasher.update(DOMAIN_SEPARATOR);
    hasher.update(b":phin-share");
    hasher.update((share_index as u64).to_le_bytes());

    Scalar::from_bytes_wide(&hasher.finalize().into())
}

fn share_label(index: usize) -> String {
    if index < 26 {
        ((b'a' + index as u8) as char).to_string()
    } else {
        format!("s{index}")
    }
}

fn pair_label(left: usize, right: usize) -> String {
    let first = left.min(right);
    let second = left.max(right);

    format!("{}{}", share_label(first), share_label(second))
}

fn encode_phi_shares(
    points: &[f64],
    encoded_points: &[u8],
    share_count: usize,
) -> Result<Vec<PhiKeyShare>, PhiKeyError> {
    let mut shares = (0..share_count)
        .map(|index| {
            let share_scalar = share_secret(encoded_points, index);

            PhiKeyShare {
                index,
                share_hex: hex_encode(&share_scalar.to_bytes()),
                public_phi_points_hex: Vec::new(),
            }
        })
        .collect::<Vec<_>>();

    for (point_index, point) in points.iter().enumerate() {
        let share_index = point_index % share_count;
        let share_scalar =
            scalar_from_hex(&shares[share_index].share_hex).ok_or(PhiKeyError::InvalidShare)?;
        let encoded_point = encode_public_phi_point(*point, share_scalar, point_index)?;

        shares[share_index]
            .public_phi_points_hex
            .push((point_index, encoded_point));
    }

    Ok(shares)
}

fn encode_public_phi_point(
    point: f64,
    share_scalar: Scalar,
    point_index: usize,
) -> Result<String, PhiKeyError> {
    let quantized_point = quantize_point(point)?;
    let encoded_scalar =
        quantized_point_scalar(quantized_point) + point_mask(share_scalar, point_index);
    let encoded_point = G1Affine::from(G1Projective::generator() * encoded_scalar);

    Ok(hex_encode(&encoded_point.to_compressed()))
}

fn share_secret(encoded_points: &[u8], share_index: usize) -> Scalar {
    let mut hasher = Sha512::new();
    hasher.update(DOMAIN_SEPARATOR);
    hasher.update([0]);
    hasher.update(encoded_points);
    hasher.update((share_index as u64).to_le_bytes());

    Scalar::from_bytes_wide(&hasher.finalize().into())
}

fn quantize_point(point: f64) -> Result<i64, PhiKeyError> {
    if point.is_nan() {
        return Ok(NAN_POINT);
    }

    if point == f64::INFINITY {
        return Ok(POSITIVE_INFINITY_POINT);
    }

    if point == f64::NEG_INFINITY {
        return Ok(NEGATIVE_INFINITY_POINT);
    }

    let quantized = (canonical_point(point) * POINT_SCALE).round() as i64;
    if quantized.abs() > POINT_LIMIT {
        return Err(PhiKeyError::PointOutOfRange {
            point,
            limit: POINT_LIMIT as f64 / POINT_SCALE,
        });
    }

    Ok(quantized)
}

fn dequantize_point(quantized: i64) -> f64 {
    match quantized {
        NAN_POINT => f64::NAN,
        POSITIVE_INFINITY_POINT => f64::INFINITY,
        NEGATIVE_INFINITY_POINT => f64::NEG_INFINITY,
        _ => quantized as f64 / POINT_SCALE,
    }
}

fn quantized_point_scalar(quantized: i64) -> Scalar {
    Scalar::from((quantized + POINT_SCALAR_OFFSET) as u64)
}

fn recover_quantized_point(point: &G1Affine) -> Option<f64> {
    quantized_point_lookup()
        .get(&point.to_compressed())
        .copied()
        .map(dequantize_point)
}

fn quantized_point_lookup() -> &'static HashMap<[u8; 48], i64> {
    static LOOKUP: OnceLock<HashMap<[u8; 48], i64>> = OnceLock::new();

    LOOKUP.get_or_init(|| {
        (-LOOKUP_POINT_LIMIT..=LOOKUP_POINT_LIMIT)
            .chain([NAN_POINT, POSITIVE_INFINITY_POINT, NEGATIVE_INFINITY_POINT])
            .map(|quantized| {
                let candidate =
                    G1Affine::from(G1Projective::generator() * quantized_point_scalar(quantized));
                (candidate.to_compressed(), quantized)
            })
            .collect()
    })
}

fn point_mask(secret_key: Scalar, index: usize) -> Scalar {
    let mut hasher = Sha512::new();
    hasher.update(POINT_MASK_DOMAIN_SEPARATOR);
    hasher.update([0]);
    hasher.update(secret_key.to_bytes());
    hasher.update((index as u64).to_le_bytes());

    Scalar::from_bytes_wide(&hasher.finalize().into())
}

fn encode_phi_all_points(points: &[f64]) -> Vec<u8> {
    let mut output = Vec::with_capacity(8 + (points.len() * 8));
    output.extend_from_slice(&(points.len() as u64).to_le_bytes());

    for point in points {
        output.extend_from_slice(&canonical_point(*point).to_le_bytes());
    }

    output
}

fn canonical_point(point: f64) -> f64 {
    if point == 0.0 {
        0.0
    } else if point.is_nan() {
        f64::NAN
    } else {
        point
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";

    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn scalar_from_hex(encoded: &str) -> Option<Scalar> {
    let bytes = hex_decode_array::<32>(encoded)?;
    Option::from(Scalar::from_bytes(&bytes))
}

fn g1_from_hex(encoded: &str) -> Option<G1Affine> {
    let bytes = hex_decode_array::<48>(encoded)?;
    Option::from(G1Affine::from_compressed(&bytes))
}

fn hex_decode_array<const N: usize>(encoded: &str) -> Option<[u8; N]> {
    if encoded.len() != N * 2 {
        return None;
    }

    let mut bytes = [0u8; N];
    for (index, byte) in bytes.iter_mut().enumerate() {
        let high = hex_value(encoded.as_bytes()[index * 2])?;
        let low = hex_value(encoded.as_bytes()[index * 2 + 1])?;
        *byte = (high << 4) | low;
    }

    Some(bytes)
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_pair_is_deterministic_for_same_phi_points() {
        let points = [0.0, 0.25, -0.5, 1.0];

        assert_eq!(
            PhiKeyPair::from_phi_all_points(&points, 3),
            PhiKeyPair::from_phi_all_points(&points, 3)
        );
    }

    #[test]
    fn key_pair_changes_when_phi_points_change() {
        let left = PhiKeyPair::from_phi_all_points(&[0.0, 0.25, 0.5, 1.0], 2).unwrap();
        let right = PhiKeyPair::from_phi_all_points(&[0.0, 0.25, 0.75, 1.0], 2).unwrap();

        assert_ne!(left.shares, right.shares);
    }

    #[test]
    fn key_pair_uses_bls12_381_serialized_share_lengths() {
        let key_pair = PhiKeyPair::from_phi_all_points(&[0.0, 0.5, 1.0], 2).unwrap();

        assert_eq!(key_pair.shares.len(), 2);
        assert!(key_pair
            .shares
            .iter()
            .all(|share| share.share_hex.len() == 64));
        assert_eq!(
            key_pair
                .shares
                .iter()
                .map(|share| share.public_phi_points_hex.len())
                .sum::<usize>(),
            3
        );
        assert!(key_pair.shares.iter().all(|share| share
            .public_phi_points_hex
            .iter()
            .all(|(_, point)| point.len() == 96)));
        assert_eq!(key_pair.fingerprint_hex.len(), 32);
    }

    #[test]
    fn negative_zero_and_positive_zero_encode_the_same() {
        assert_eq!(
            PhiKeyPair::from_phi_all_points(&[-0.0, 1.0], 2),
            PhiKeyPair::from_phi_all_points(&[0.0, 1.0], 2)
        );
    }

    #[test]
    fn shares_recover_phi_points_from_public_encoded_points() {
        let points = [-1.2, 0.0, 0.1, 1.5];
        let key_pair = PhiKeyPair::from_phi_all_points(&points, 3).unwrap();
        let recovered = key_pair.recover_phi_all_points().unwrap();

        assert_eq!(recovered, points);
    }

    #[test]
    fn encodes_non_finite_phi_points_as_reserved_values() {
        let key_pair =
            PhiKeyPair::from_phi_all_points(&[f64::NAN, f64::INFINITY, f64::NEG_INFINITY], 2)
                .unwrap();

        assert_eq!(key_pair.shares.len(), 2);
        assert_eq!(
            key_pair
                .shares
                .iter()
                .map(|share| share.public_phi_points_hex.len())
                .sum::<usize>(),
            3
        );
    }

    #[test]
    fn each_phi_point_belongs_to_one_share() {
        let points = [-1.2, 0.0, 0.1, 1.5];
        let key_pair = PhiKeyPair::from_phi_all_points(&points, 3).unwrap();
        let mut assigned_indices = key_pair
            .shares
            .iter()
            .flat_map(|share| {
                share
                    .public_phi_points_hex
                    .iter()
                    .map(|(point_index, _)| *point_index)
            })
            .collect::<Vec<_>>();
        assigned_indices.sort_unstable();

        assert_eq!(assigned_indices, vec![0, 1, 2, 3]);
    }

    #[test]
    fn encrypted_phin_shares_use_component_formulas() {
        let key_pair = PhiKeyPair::from_phi_all_points(&[0.0, 0.1], 3).unwrap();
        let shares = key_pair.encrypted_phin_shares().unwrap();

        assert_eq!(shares.len(), 3);
        assert_eq!(shares[0].1, "phi(a) + phi(ab) + phi(ac)");
        assert_eq!(shares[1].1, "phi(b) + phi(ab) + phi(bc)");
        assert_eq!(shares[2].1, "phi(c) + phi(ac) + phi(bc)");
        assert!(shares
            .iter()
            .all(|(_, _, points)| points.len() == 2
                && points.iter().all(|(_, point)| point.len() == 96)));
    }

    #[test]
    fn all_encrypted_phin_shares_combine_to_phi() {
        let key_pair = PhiKeyPair::from_phi_all_points(&[0.0, 0.1], 3).unwrap();
        let shares = key_pair.encrypted_phin_shares().unwrap();

        for point_index in 0..2 {
            let mut combined = G1Projective::identity();

            for (share_index, _, points) in &shares {
                let encoded_point = points
                    .iter()
                    .find(|(index, _)| *index == point_index)
                    .map(|(_, point)| point)
                    .expect("share point");
                let point = g1_from_hex(encoded_point).expect("valid g1");
                let mask = point_mask(phin_share_secret(*share_index), point_index);

                combined += G1Projective::from(point) - (G1Projective::generator() * mask);
            }

            let quantized = quantize_point(key_pair.recovered_phi_points[point_index]).unwrap();
            let expected =
                G1Affine::from(G1Projective::generator() * quantized_point_scalar(quantized));

            assert_eq!(
                G1Affine::from(combined).to_compressed(),
                expected.to_compressed()
            );
        }
    }

    #[test]
    fn encoded_phi_points_include_all_phi_points() {
        let key_pair = PhiKeyPair::from_phi_all_points(&[0.0, 0.1, f64::NAN], 2).unwrap();
        let encoded_phi_points = key_pair.encoded_phi_points().unwrap();

        assert_eq!(encoded_phi_points.len(), 3);
        assert_eq!(
            encoded_phi_points
                .iter()
                .map(|(point_index, _)| *point_index)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert!(encoded_phi_points
            .iter()
            .all(|(_, point)| point.len() == 96));
    }

    #[test]
    fn rejects_points_outside_recoverable_range() {
        assert!(matches!(
            PhiKeyPair::from_phi_all_points(&[5_000_001.0], 1),
            Err(PhiKeyError::PointOutOfRange { .. })
        ));
    }

    #[test]
    fn rejects_zero_shares() {
        assert!(matches!(
            PhiKeyPair::from_phi_all_points(&[0.0], 0),
            Err(PhiKeyError::InvalidShareCount)
        ));
    }
}
