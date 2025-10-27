#![no_std]
//! # DTMF Table
//!
//! A zero-heap, `no_std`, const-first implementation of the standard DTMF keypad
//! frequencies with ergonomic runtime helpers for real-world audio decoding.
//!
//! ## Features
//! - Type-safe closed enum for DTMF keys — invalid keys are unrepresentable.
//! - Fully `const` forward and reverse mappings (key ↔ frequencies).
//! - Runtime helpers for tolerance-based reverse lookup and nearest snapping.
//! - No heap, no allocations, no dependencies.
//!
//! ## Example
//!
//! ```rust
//! use dtmf_tones::{DtmfTable, DtmfKey};
//!
//! // Construct a zero-sized table instance
//! let table = DtmfTable::new();
//!
//! // Forward lookup from key to canonical frequencies
//! let (low, high) = DtmfTable::lookup_key(DtmfKey::K8);
//! assert_eq!((low, high), (852, 1336));
//!
//! // Reverse lookup with tolerance (e.g. from FFT bin centres)
//! let key = table.from_pair_tol_f64(770.2, 1335.6, 6.0).unwrap();
//! assert_eq!(key.to_char(), '5');
//!
//! // Nearest snapping for noisy estimates
//! let (k, snapped_low, snapped_high) = table.nearest_u32(768, 1342);
//! assert_eq!(k.to_char(), '5');
//! assert_eq!((snapped_low, snapped_high), (770, 1336));
//! ```
//!
//! This makes it easy to integrate DTMF tone detection directly into audio
//! processing pipelines (e.g., FFT bin peak picking) with robust tolerance handling
//! and compile-time validation of key mappings.


use core::cmp::Ordering;

/// Type-safe, closed set of DTMF keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DtmfKey {
    K1, K2, K3, A,
    K4, K5, K6, B,
    K7, K8, K9, C,
    Star, K0, Hash, D,
}

impl DtmfKey {
    /// Strict constructor from `char` (const).
    pub const fn from_char(c: char) -> Option<Self> {
        match c {
            '1' => Some(Self::K1),
            '2' => Some(Self::K2),
            '3' => Some(Self::K3),
            'A' => Some(Self::A),
            '4' => Some(Self::K4),
            '5' => Some(Self::K5),
            '6' => Some(Self::K6),
            'B' => Some(Self::B),
            '7' => Some(Self::K7),
            '8' => Some(Self::K8),
            '9' => Some(Self::K9),
            'C' => Some(Self::C),
            '*' => Some(Self::Star),
            '0' => Some(Self::K0),
            '#' => Some(Self::Hash),
            'D' => Some(Self::D),
            _ => None,
        }
    }

    /// Panic-on-invalid (const), useful with char literals at compile time.
    pub const fn from_char_or_panic(c: char) -> Self {
        match Self::from_char(c) {
            Some(k) => k,
            None => panic!("invalid DTMF char"),
        }
    }

    /// Back to char (const).
    pub const fn to_char(self) -> char {
        match self {
            Self::K1 => '1', Self::K2 => '2', Self::K3 => '3', Self::A => 'A',
            Self::K4 => '4', Self::K5 => '5', Self::K6 => '6', Self::B => 'B',
            Self::K7 => '7', Self::K8 => '8', Self::K9 => '9', Self::C => 'C',
            Self::Star => '*', Self::K0 => '0', Self::Hash => '#', Self::D => 'D',
        }
    }

    /// Canonical (low, high) frequencies in Hz (const).
    pub const fn freqs(self) -> (u16, u16) {
        match self {
            Self::K1 => (697, 1209),
            Self::K2 => (697, 1336),
            Self::K3 => (697, 1477),
            Self::A  => (697, 1633),

            Self::K4 => (770, 1209),
            Self::K5 => (770, 1336),
            Self::K6 => (770, 1477),
            Self::B  => (770, 1633),

            Self::K7 => (852, 1209),
            Self::K8 => (852, 1336),
            Self::K9 => (852, 1477),
            Self::C  => (852, 1633),

            Self::Star => (941, 1209),
            Self::K0   => (941, 1336),
            Self::Hash => (941, 1477),
            Self::D    => (941, 1633),
        }
    }
}

/// Tone record (ties the key to its canonical freqs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DtmfTone {
    pub key: DtmfKey,
    pub low_hz: u16,
    pub high_hz: u16,
}

/// Zero-sized table wrapper for const and runtime utilities.
pub struct DtmfTable;

impl DtmfTable {
    /// Canonical low-/high-band frequencies (Hz).
    pub const LOWS: [u16; 4]  = [697, 770, 852, 941];
    pub const HIGHS: [u16; 4] = [1209, 1336, 1477, 1633];

    /// All keys in keypad order (row-major).
    pub const ALL_KEYS: [DtmfKey; 16] = [
        DtmfKey::K1, DtmfKey::K2, DtmfKey::K3, DtmfKey::A,
        DtmfKey::K4, DtmfKey::K5, DtmfKey::K6, DtmfKey::B,
        DtmfKey::K7, DtmfKey::K8, DtmfKey::K9, DtmfKey::C,
        DtmfKey::Star, DtmfKey::K0, DtmfKey::Hash, DtmfKey::D,
    ];

    /// All tones as (key, low, high). Kept explicit to stay `const`.
    pub const ALL_TONES: [DtmfTone; 16] = [
        DtmfTone { key: DtmfKey::K1,   low_hz: 697, high_hz: 1209 },
        DtmfTone { key: DtmfKey::K2,   low_hz: 697, high_hz: 1336 },
        DtmfTone { key: DtmfKey::K3,   low_hz: 697, high_hz: 1477 },
        DtmfTone { key: DtmfKey::A,    low_hz: 697, high_hz: 1633 },

        DtmfTone { key: DtmfKey::K4,   low_hz: 770, high_hz: 1209 },
        DtmfTone { key: DtmfKey::K5,   low_hz: 770, high_hz: 1336 },
        DtmfTone { key: DtmfKey::K6,   low_hz: 770, high_hz: 1477 },
        DtmfTone { key: DtmfKey::B,    low_hz: 770, high_hz: 1633 },

        DtmfTone { key: DtmfKey::K7,   low_hz: 852, high_hz: 1209 },
        DtmfTone { key: DtmfKey::K8,   low_hz: 852, high_hz: 1336 },
        DtmfTone { key: DtmfKey::K9,   low_hz: 852, high_hz: 1477 },
        DtmfTone { key: DtmfKey::C,    low_hz: 852, high_hz: 1633 },

        DtmfTone { key: DtmfKey::Star, low_hz: 941, high_hz: 1209 },
        DtmfTone { key: DtmfKey::K0,   low_hz: 941, high_hz: 1336 },
        DtmfTone { key: DtmfKey::Hash, low_hz: 941, high_hz: 1477 },
        DtmfTone { key: DtmfKey::D,    low_hz: 941, high_hz: 1633 },
    ];

    /// Constructor (zero-sized instance).
    pub const fn new() -> Self { DtmfTable }

    /* ---------------------- Const utilities ---------------------- */

    /// Forward: key → (low, high) (const).
    pub const fn lookup_key(key: DtmfKey) -> (u16, u16) { key.freqs() }

    /// Reverse: exact (low, high) → key (const). Order-sensitive.
    pub const fn from_pair_exact(low: u16, high: u16) -> Option<DtmfKey> {
        match (low, high) {
            (697, 1209) => Some(DtmfKey::K1),
            (697, 1336) => Some(DtmfKey::K2),
            (697, 1477) => Some(DtmfKey::K3),
            (697, 1633) => Some(DtmfKey::A),
            (770, 1209) => Some(DtmfKey::K4),
            (770, 1336) => Some(DtmfKey::K5),
            (770, 1477) => Some(DtmfKey::K6),
            (770, 1633) => Some(DtmfKey::B),
            (852, 1209) => Some(DtmfKey::K7),
            (852, 1336) => Some(DtmfKey::K8),
            (852, 1477) => Some(DtmfKey::K9),
            (852, 1633) => Some(DtmfKey::C),
            (941, 1209) => Some(DtmfKey::Star),
            (941, 1336) => Some(DtmfKey::K0),
            (941, 1477) => Some(DtmfKey::Hash),
            (941, 1633) => Some(DtmfKey::D),
            _ => None,
        }
    }

    /// Reverse with normalisation (const): accepts (high, low) as well.
    pub const fn from_pair_normalised(a: u16, b: u16) -> Option<DtmfKey> {
        let (low, high) = if a <= b { (a, b) } else { (b, a) };
        Self::from_pair_exact(low, high)
    }

    /* ---------------------- Runtime helpers ---------------------- */

    /// Iterate keys in keypad order (no allocation).
    pub fn iter_keys(&self) -> core::slice::Iter<'static, DtmfKey> {
        Self::ALL_KEYS.iter()
    }

    /// Iterate tones (key + freqs) in keypad order (no allocation).
    pub fn iter_tones(&self) -> core::slice::Iter<'static, DtmfTone> {
        Self::ALL_TONES.iter()
    }

    /// Reverse lookup with tolerance in Hz (integer inputs).
    /// Matches only when *both* low and high fall within `±tol_hz` of a canonical pair.
    pub fn from_pair_tol_u32(&self, low: u32, high: u32, tol_hz: u32) -> Option<DtmfKey> {
        let (lo, hi) = normalise_u32_pair(low, high);
        for t in Self::ALL_TONES {
            if abs_diff_u32(lo, t.low_hz as u32) <= tol_hz &&
               abs_diff_u32(hi, t.high_hz as u32) <= tol_hz {
                return Some(t.key);
            }
        }
        None
    }

    /// Reverse lookup with tolerance for floating-point estimates (e.g., FFT bin centres).
    pub fn from_pair_tol_f64(&self, low: f64, high: f64, tol_hz: f64) -> Option<DtmfKey> {
        let (lo, hi) = normalise_f64_pair(low, high);
        for t in Self::ALL_TONES {
            if (lo - t.low_hz as f64).abs() <= tol_hz &&
               (hi - t.high_hz as f64).abs() <= tol_hz {
                return Some(t.key);
            }
        }
        None
    }

    /// Snap an arbitrary (low, high) estimate to the nearest canonical pair and return (key, snapped_low, snapped_high).
    /// Uses absolute distance independently on low and high bands.
    pub fn nearest_u32(&self, low: u32, high: u32) -> (DtmfKey, u16, u16) {
        let (lo, hi) = normalise_u32_pair(low, high);
        let nearest_low  = nearest_in_set_u32(lo,  &Self::LOWS);
        let nearest_high = nearest_in_set_u32(hi, &Self::HIGHS);
        let key = Self::from_pair_exact(nearest_low, nearest_high)
            .expect("canonical pair must map to a key");
        (key, nearest_low, nearest_high)
    }

    /// Floating-point variant of nearest snap.
    pub fn nearest_f64(&self, low: f64, high: f64) -> (DtmfKey, u16, u16) {
        let (lo, hi) = normalise_f64_pair(low, high);
        let nearest_low  = nearest_in_set_f64(lo,  &Self::LOWS);
        let nearest_high = nearest_in_set_f64(hi, &Self::HIGHS);
        let key = Self::from_pair_exact(nearest_low, nearest_high)
            .expect("canonical pair must map to a key");
        (key, nearest_low, nearest_high)
    }
}

/* --------------------------- Small helpers --------------------------- */

const fn abs_diff_u32(a: u32, b: u32) -> u32 {
    if a >= b { a - b } else { b - a }
}

fn nearest_in_set_u32(x: u32, set: &[u16]) -> u16 {
    let mut best = set[0];
    let mut best_d = abs_diff_u32(x, best as u32);
    let mut i = 1;
    while i < set.len() {
        let d = abs_diff_u32(x, set[i] as u32);
        if d < best_d { best = set[i]; best_d = d; }
        i += 1;
    }
    best
}

fn nearest_in_set_f64(x: f64, set: &[u16]) -> u16 {
    let mut best = set[0];
    let mut best_d = (x - best as f64).abs();
    let mut i = 1;
    while i < set.len() {
        let d = (x - set[i] as f64).abs();
        if d < best_d { best = set[i]; best_d = d; }
        i += 1;
    }
    best
}

const fn normalise_u32_pair(a: u32, b: u32) -> (u32, u32) {
    if a <= b { (a, b) } else { (b, a) }
}

fn normalise_f64_pair(a: f64, b: f64) -> (f64, f64) {
    match a.partial_cmp(&b) {
        Some(Ordering::Greater) => (b, a),
        _ => (a, b),
    }
}
