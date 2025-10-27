# DTMF Table 

A zero-heap, `no_std` friendly, **const-first** implementation of the standard DTMF (Dual-Tone Multi-Frequency) keypad used in telephony systems.  
This crate provides compile-time safe mappings between keypad keys and their canonical low/high frequencies, along with **runtime helpers** for practical audio processing.

---

## Features

- **Const-evaluated forward and reverse mappings** between DTMF keys and frequencies  
- **Closed enum for keys** — invalid keys are unrepresentable  
- **Zero allocations**, works in `no_std` environments  
- Runtime helpers:
  - Tolerance-based reverse lookup (e.g., from FFT peaks)
  - Nearest snapping for noisy frequency estimates
  - Iteration over all tones and keys

---

## Installation

```bash
cargo add dtmf_tones
```


This crate is `no_std` by default and does not pull in any dependencies.

---

## Quick Example

```rust
use dtmf_table::{DtmfTable, DtmfKey};

fn main() {
    // Construct a zero-sized table instance
    let table = DtmfTable::new();

    // Forward lookup from key to canonical frequencies
    let (low, high) = DtmfTable::lookup_key(DtmfKey::K8);
    assert_eq!((low, high), (852, 1336));

    // Reverse lookup with tolerance (e.g., from FFT bin centres)
    let key = table.from_pair_tol_f64(770.2, 1335.6, 6.0).unwrap();
    assert_eq!(key.to_char(), '5');

    // Nearest snapping for noisy estimates
    let (k, snapped_low, snapped_high) = table.nearest_u32(768, 1342);
    assert_eq!(k.to_char(), '5');
    assert_eq!((snapped_low, snapped_high), (770, 1336));
}
```

---

## Why Const-First?

Most DTMF tone mappings are fixed, known at compile time, and tiny (4×4 keypad).
By making the mapping fully `const`, you can:

* Use it **inside `const fn`**, static initialisers, or `const` generic contexts
* Catch invalid keys **at compile time**
* Eliminate runtime table lookups entirely

---

## API Overview

| Function                          | Description                                                | `const` |
| --------------------------------- | ---------------------------------------------------------- | :-----: |
| `DtmfKey::from_char`              | Convert a char to a key (fallible)                         |    ✅    |
| `DtmfKey::from_char_or_panic`     | Convert a char to a key, panics at compile time if invalid |    ✅    |
| `DtmfKey::to_char`                | Convert key to char                                        |    ✅    |
| `DtmfTable::lookup_key`           | Forward lookup: key → (low, high)                          |    ✅    |
| `DtmfTable::from_pair_exact`      | Reverse lookup: exact pair → key                           |    ✅    |
| `DtmfTable::from_pair_normalised` | Reverse lookup: order-insensitive                          |    ✅    |
| `DtmfTable::from_pair_tol_f64`    | Reverse lookup with tolerance                              |    ❌    |
| `DtmfTable::nearest_u32`          | Snap noisy frequencies to nearest canonical pair           |    ❌    |
| `DtmfTable::iter_tones`           | Iterate over all tones                                     |    ❌    |

---

## Integration Example

This crate pairs naturally with audio analysis pipelines. For example:

* Take an audio segment
* Compute FFT magnitude
* Pick two frequency peaks
* Use `from_pair_tol_f64` or `nearest_f64` to resolve the DTMF key

```rust
// freq1 and freq2 are the peak frequencies extracted from your FFT
let key = table.from_pair_tol_f64(freq1, freq2, 5.0);
if let Some(k) = key {
    println!("Detected key: {}", k.to_char());
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
