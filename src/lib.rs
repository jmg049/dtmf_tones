#![cfg_attr(not(feature = "std"), no_std)]
// Correctness and logic
#![warn(clippy::unit_cmp)] // Detects comparing unit types
#![warn(clippy::match_same_arms)] // Duplicate match arms
#![allow(clippy::result_large_err)] // Allow large error types for comprehensive error handling
#![allow(clippy::missing_const_for_fn)] // Functions may need mutations in the future
#![allow(clippy::collapsible_if)] // Sometimes clearer to have separate conditions
#![allow(clippy::missing_panics_doc)] // Panics are converted to proper errors where needed
#![allow(clippy::needless_borrows_for_generic_args)] // Sometimes clearer with explicit borrows
#![allow(clippy::if_same_then_else)] // Similar blocks may diverge in the future
#![allow(clippy::unnecessary_cast)] // Explicit casts for clarity
#![allow(clippy::identity_op)] // Explicit operations for clarity

// Performance-focused
#![warn(clippy::inefficient_to_string)] // `format!("{}", x)` vs `x.to_string()`
#![warn(clippy::map_clone)] // Cloning inside `map()` unnecessarily
#![warn(clippy::unnecessary_to_owned)] // Detects redundant `.to_owned()` or `.clone()`
#![warn(clippy::large_stack_arrays)] // Helps avoid stack overflows
#![warn(clippy::box_collection)] // Warns on boxed `Vec`, `String`, etc.
#![warn(clippy::vec_box)] // Avoids using `Vec<Box<T>>` when unnecessary
#![warn(clippy::needless_collect)] // Avoids `.collect().iter()` chains

// Style and idiomatic Rust
#![warn(clippy::redundant_clone)] // Detects unnecessary `.clone()`
#![warn(clippy::identity_op)] // e.g., `x + 0`, `x * 1`
#![warn(clippy::needless_return)] // Avoids `return` at the end of functions
#![warn(clippy::let_unit_value)] // Avoids binding `()` to variables
#![warn(clippy::manual_map)] // Use `.map()` instead of manual `match`
#![warn(clippy::unwrap_used)] // Avoids using `unwrap()`
#![warn(clippy::panic)] // Avoids using `panic!` in production code

// Maintainability
#![warn(clippy::missing_panics_doc)] // Docs for functions that might panic
#![warn(clippy::missing_safety_doc)] // Docs for `unsafe` functions
#![warn(clippy::missing_const_for_fn)] // Suggests making eligible functions `const`
#![allow(clippy::too_many_arguments)] // Allow functions with many parameters (very few and far between)
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
//! use dtmf_table::{DtmfTable, DtmfKey};
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
use core::fmt::Display;

// Python bindings (optional, feature-gated)
#[cfg(feature = "python")]
mod python;

/// Type-safe, closed set of DTMF keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DtmfKey {
    K1,
    K2,
    K3,
    A,
    K4,
    K5,
    K6,
    B,
    K7,
    K8,
    K9,
    C,
    Star,
    K0,
    Hash,
    D,
}

impl Display for DtmfKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if f.alternate() {
            // Alternate format: show enum variant name
            write!(f, "DtmfKey::{:?}", self)
        } else {
            // Normal format: just the character
            write!(f, "{}", self.to_char())
        }
    }
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
    ///
    /// # Panics
    ///
    /// Panics if the character is not a valid DTMF key character.
    #[allow(clippy::panic)]
    pub const fn from_char_or_panic(c: char) -> Self {
        match Self::from_char(c) {
            Some(k) => k,
            None => panic!("invalid DTMF char"),
        }
    }

    /// Back to char (const).
    pub const fn to_char(self) -> char {
        match self {
            Self::K1 => '1',
            Self::K2 => '2',
            Self::K3 => '3',
            Self::A => 'A',
            Self::K4 => '4',
            Self::K5 => '5',
            Self::K6 => '6',
            Self::B => 'B',
            Self::K7 => '7',
            Self::K8 => '8',
            Self::K9 => '9',
            Self::C => 'C',
            Self::Star => '*',
            Self::K0 => '0',
            Self::Hash => '#',
            Self::D => 'D',
        }
    }

    /// Canonical (low, high) frequencies in Hz (const).
    pub const fn freqs(self) -> (u16, u16) {
        match self {
            Self::K1 => (697, 1209),
            Self::K2 => (697, 1336),
            Self::K3 => (697, 1477),
            Self::A => (697, 1633),

            Self::K4 => (770, 1209),
            Self::K5 => (770, 1336),
            Self::K6 => (770, 1477),
            Self::B => (770, 1633),

            Self::K7 => (852, 1209),
            Self::K8 => (852, 1336),
            Self::K9 => (852, 1477),
            Self::C => (852, 1633),

            Self::Star => (941, 1209),
            Self::K0 => (941, 1336),
            Self::Hash => (941, 1477),
            Self::D => (941, 1633),
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

impl Display for DtmfTone {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if f.alternate() {
            // Alternate format: structured representation
            write!(
                f,
                "DtmfTone {{ key: {}, low: {} Hz, high: {} Hz }}",
                self.key, self.low_hz, self.high_hz
            )
        } else {
            // Normal format: human-readable
            write!(f, "{}: ({} Hz, {} Hz)", self.key, self.low_hz, self.high_hz)
        }
    }
}

/// Zero-sized table wrapper for const and runtime utilities.
#[derive(Debug)]
pub struct DtmfTable;

impl Default for DtmfTable {
    fn default() -> Self {
        Self::new()
    }
}

impl DtmfTable {
    /// Canonical low-/high-band frequencies (Hz).
    pub const LOWS: [u16; 4] = [697, 770, 852, 941];
    pub const HIGHS: [u16; 4] = [1209, 1336, 1477, 1633];

    /// All keys in keypad order (row-major).
    pub const ALL_KEYS: [DtmfKey; 16] = [
        DtmfKey::K1,
        DtmfKey::K2,
        DtmfKey::K3,
        DtmfKey::A,
        DtmfKey::K4,
        DtmfKey::K5,
        DtmfKey::K6,
        DtmfKey::B,
        DtmfKey::K7,
        DtmfKey::K8,
        DtmfKey::K9,
        DtmfKey::C,
        DtmfKey::Star,
        DtmfKey::K0,
        DtmfKey::Hash,
        DtmfKey::D,
    ];

    /// All tones as (key, low, high). Kept explicit to stay `const`.
    pub const ALL_TONES: [DtmfTone; 16] = [
        DtmfTone {
            key: DtmfKey::K1,
            low_hz: 697,
            high_hz: 1209,
        },
        DtmfTone {
            key: DtmfKey::K2,
            low_hz: 697,
            high_hz: 1336,
        },
        DtmfTone {
            key: DtmfKey::K3,
            low_hz: 697,
            high_hz: 1477,
        },
        DtmfTone {
            key: DtmfKey::A,
            low_hz: 697,
            high_hz: 1633,
        },
        DtmfTone {
            key: DtmfKey::K4,
            low_hz: 770,
            high_hz: 1209,
        },
        DtmfTone {
            key: DtmfKey::K5,
            low_hz: 770,
            high_hz: 1336,
        },
        DtmfTone {
            key: DtmfKey::K6,
            low_hz: 770,
            high_hz: 1477,
        },
        DtmfTone {
            key: DtmfKey::B,
            low_hz: 770,
            high_hz: 1633,
        },
        DtmfTone {
            key: DtmfKey::K7,
            low_hz: 852,
            high_hz: 1209,
        },
        DtmfTone {
            key: DtmfKey::K8,
            low_hz: 852,
            high_hz: 1336,
        },
        DtmfTone {
            key: DtmfKey::K9,
            low_hz: 852,
            high_hz: 1477,
        },
        DtmfTone {
            key: DtmfKey::C,
            low_hz: 852,
            high_hz: 1633,
        },
        DtmfTone {
            key: DtmfKey::Star,
            low_hz: 941,
            high_hz: 1209,
        },
        DtmfTone {
            key: DtmfKey::K0,
            low_hz: 941,
            high_hz: 1336,
        },
        DtmfTone {
            key: DtmfKey::Hash,
            low_hz: 941,
            high_hz: 1477,
        },
        DtmfTone {
            key: DtmfKey::D,
            low_hz: 941,
            high_hz: 1633,
        },
    ];

    /// Constructor (zero-sized instance).
    pub const fn new() -> Self {
        DtmfTable
    }

    /* ---------------------- Const utilities ---------------------- */

    /// Forward: key → (low, high) (const).
    pub const fn lookup_key(key: DtmfKey) -> (u16, u16) {
        key.freqs()
    }

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
            if abs_diff_u32(lo, t.low_hz as u32) <= tol_hz
                && abs_diff_u32(hi, t.high_hz as u32) <= tol_hz
            {
                return Some(t.key);
            }
        }
        None
    }

    /// Reverse lookup with tolerance for floating-point estimates (e.g., FFT bin centres).
    pub fn from_pair_tol_f64(&self, low: f64, high: f64, tol_hz: f64) -> Option<DtmfKey> {
        let (lo, hi) = normalise_f64_pair(low, high);
        for t in Self::ALL_TONES {
            if (lo - t.low_hz as f64).abs() <= tol_hz && (hi - t.high_hz as f64).abs() <= tol_hz {
                return Some(t.key);
            }
        }
        None
    }

    /// Snap an arbitrary (low, high) estimate to the nearest canonical pair and return (key, snapped_low, snapped_high).
    /// Uses absolute distance independently on low and high bands.
    ///
    /// # Panics
    ///
    /// This function should never panic as it always snaps to canonical frequency pairs which are guaranteed to have valid keys.
    pub fn nearest_u32(&self, low: u32, high: u32) -> (DtmfKey, u16, u16) {
        let (lo, hi) = normalise_u32_pair(low, high);
        let nearest_low = nearest_in_set_u32(lo, &Self::LOWS);
        let nearest_high = nearest_in_set_u32(hi, &Self::HIGHS);
        let key = Self::from_pair_exact(nearest_low, nearest_high)
            .expect("canonical pair must map to a key");
        (key, nearest_low, nearest_high)
    }

    /// Floating-point variant of nearest snap.
    ///
    /// # Panics
    ///
    /// This function should never panic as it always snaps to canonical frequency pairs which are guaranteed to have valid keys.
    pub fn nearest_f64(&self, low: f64, high: f64) -> (DtmfKey, u16, u16) {
        let (lo, hi) = normalise_f64_pair(low, high);
        let nearest_low = nearest_in_set_f64(lo, &Self::LOWS);
        let nearest_high = nearest_in_set_f64(hi, &Self::HIGHS);
        let key = Self::from_pair_exact(nearest_low, nearest_high)
            .expect("canonical pair must map to a key");
        (key, nearest_low, nearest_high)
    }
}

impl Display for DtmfTable {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if f.alternate() {
            // Alternate format: compact keypad grid layout
            writeln!(f, "DTMF Keypad Layout:")?;
            writeln!(f, "  1209 Hz  1336 Hz  1477 Hz  1633 Hz")?;
            writeln!(f, "697 Hz:  1       2       3       A")?;
            writeln!(f, "770 Hz:  4       5       6       B")?;
            writeln!(f, "852 Hz:  7       8       9       C")?;
            write!(f, "941 Hz:  *       0       #       D")
        } else {
            // Normal format: detailed list
            writeln!(f, "DTMF Table:")?;
            for tone in Self::ALL_TONES.iter() {
                writeln!(f, "  {}", tone)?;
            }
            Ok(())
        }
    }
}

/* --------------------------- Small helpers --------------------------- */

const fn abs_diff_u32(a: u32, b: u32) -> u32 {
    a.abs_diff(b)
}

fn nearest_in_set_u32(x: u32, set: &[u16]) -> u16 {
    let mut best = set[0];
    let mut best_d = abs_diff_u32(x, best as u32);
    let mut i = 1;
    while i < set.len() {
        let d = abs_diff_u32(x, set[i] as u32);
        if d < best_d {
            best = set[i];
            best_d = d;
        }
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
        if d < best_d {
            best = set[i];
            best_d = d;
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    // These tests require std for format! macro
    #[cfg(feature = "std")]
    mod std_tests {
        use super::*;
        extern crate std;
        use std::format;

        #[test]
        fn test_dtmf_key_normal_display() {
            assert_eq!(format!("{}", DtmfKey::K5), "5");
            assert_eq!(format!("{}", DtmfKey::Star), "*");
            assert_eq!(format!("{}", DtmfKey::Hash), "#");
            assert_eq!(format!("{}", DtmfKey::A), "A");
        }

        #[test]
        fn test_dtmf_key_alternate_display() {
            assert_eq!(format!("{:#}", DtmfKey::K5), "DtmfKey::K5");
            assert_eq!(format!("{:#}", DtmfKey::Star), "DtmfKey::Star");
            assert_eq!(format!("{:#}", DtmfKey::Hash), "DtmfKey::Hash");
            assert_eq!(format!("{:#}", DtmfKey::A), "DtmfKey::A");
        }

        #[test]
        fn test_dtmf_tone_normal_display() {
            let tone = DtmfTone {
                key: DtmfKey::K5,
                low_hz: 770,
                high_hz: 1336,
            };
            assert_eq!(format!("{}", tone), "5: (770 Hz, 1336 Hz)");
        }

        #[test]
        fn test_dtmf_tone_alternate_display() {
            let tone = DtmfTone {
                key: DtmfKey::K5,
                low_hz: 770,
                high_hz: 1336,
            };
            assert_eq!(
                format!("{:#}", tone),
                "DtmfTone { key: 5, low: 770 Hz, high: 1336 Hz }"
            );
        }

        #[test]
        fn test_dtmf_table_normal_display() {
            let table = DtmfTable::new();
            let output = format!("{}", table);
            assert!(output.contains("DTMF Table:"));
            assert!(output.contains("1: (697 Hz, 1209 Hz)"));
            assert!(output.contains("5: (770 Hz, 1336 Hz)"));
            assert!(output.contains("D: (941 Hz, 1633 Hz)"));
        }

        #[test]
        fn test_dtmf_table_alternate_display() {
            let table = DtmfTable::new();
            let output = format!("{:#}", table);
            assert!(output.contains("DTMF Keypad Layout:"));
            assert!(output.contains("1209 Hz"));
            assert!(output.contains("697 Hz:"));
            assert!(output.contains("941 Hz:"));
            // Check that all keys are present in the grid
            assert!(output.contains("1"));
            assert!(output.contains("5"));
            assert!(output.contains("*"));
            assert!(output.contains("#"));
        }

        #[test]
        fn test_all_keys_have_alternate_format() {
            // Verify all keys can be formatted with alternate format
            for key in DtmfTable::ALL_KEYS.iter() {
                let normal = format!("{}", key);
                let alternate = format!("{:#}", key);

                // Normal should be single character
                assert_eq!(normal.len(), 1);

                // Alternate should contain "DtmfKey::"
                assert!(alternate.starts_with("DtmfKey::"));

                // They should be different
                assert_ne!(normal, alternate);
            }
        }

        #[test]
        fn test_all_tones_have_alternate_format() {
            // Verify all tones can be formatted with alternate format
            for tone in DtmfTable::ALL_TONES.iter() {
                let normal = format!("{}", tone);
                let alternate = format!("{:#}", tone);

                // Normal should contain "Hz"
                assert!(normal.contains("Hz"));

                // Alternate should contain "DtmfTone"
                assert!(alternate.contains("DtmfTone"));

                // They should be different
                assert_ne!(normal, alternate);
            }
        }
    }
}
