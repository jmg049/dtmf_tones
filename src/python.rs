use crate::{DtmfKey as RustDtmfKey, DtmfTable as RustDtmfTable, DtmfTone as RustDtmfTone};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
extern crate std;
use std::collections::hash_map::DefaultHasher;
use std::format;
use std::hash::{Hash, Hasher};

/// Python wrapper for DtmfKey enum
#[pyclass(name = "DtmfKey", module = "dtmf_table")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PyDtmfKey {
    inner: RustDtmfKey,
}

#[pymethods]
impl PyDtmfKey {
    /// Create a DtmfKey from a character.
    ///
    /// Args:
    ///     c (str): Single character representing the DTMF key ('0'-'9', '*', '#', 'A'-'D')
    ///
    /// Returns:
    ///     DtmfKey: The corresponding DTMF key
    ///
    /// Raises:
    ///     ValueError: If the character is not a valid DTMF key
    #[staticmethod]
    fn from_char(c: char) -> PyResult<Self> {
        match RustDtmfKey::from_char(c) {
            Some(key) => Ok(PyDtmfKey { inner: key }),
            None => Err(PyValueError::new_err(format!(
                "Invalid DTMF character: '{}'",
                c
            ))),
        }
    }

    /// Convert the DtmfKey to its character representation.
    ///
    /// Returns:
    ///     str: Single character representing the key
    fn to_char(&self) -> char {
        self.inner.to_char()
    }

    /// Get the canonical frequencies for this DTMF key.
    ///
    /// Returns:
    ///     tuple[int, int]: (low_frequency_hz, high_frequency_hz)
    fn freqs(&self) -> (u16, u16) {
        self.inner.freqs()
    }

    fn __str__(&self) -> String {
        self.inner.to_char().to_string()
    }

    fn __repr__(&self) -> String {
        format!("DtmfKey('{}')", self.inner.to_char())
    }

    fn __eq__(&self, other: &PyDtmfKey) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }
}

/// Python wrapper for DtmfTone struct
#[pyclass(name = "DtmfTone", module = "dtmf_table")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PyDtmfTone {
    inner: RustDtmfTone,
}

#[pymethods]
impl PyDtmfTone {
    /// Create a new DtmfTone.
    ///
    /// Args:
    ///     key (DtmfKey): The DTMF key
    ///     low_hz (int): Low frequency in Hz
    ///     high_hz (int): High frequency in Hz
    #[new]
    fn new(key: PyDtmfKey, low_hz: u16, high_hz: u16) -> Self {
        PyDtmfTone {
            inner: RustDtmfTone {
                key: key.inner,
                low_hz,
                high_hz,
            },
        }
    }

    /// The DTMF key for this tone.
    #[getter]
    fn key(&self) -> PyDtmfKey {
        PyDtmfKey {
            inner: self.inner.key,
        }
    }

    /// Low frequency in Hz.
    #[getter]
    fn low_hz(&self) -> u16 {
        self.inner.low_hz
    }

    /// High frequency in Hz.
    #[getter]
    fn high_hz(&self) -> u16 {
        self.inner.high_hz
    }

    fn __str__(&self) -> String {
        format!(
            "{}: ({} Hz, {} Hz)",
            self.inner.key.to_char(),
            self.inner.low_hz,
            self.inner.high_hz
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "DtmfTone(key=DtmfKey('{}'), low_hz={}, high_hz={})",
            self.inner.key.to_char(),
            self.inner.low_hz,
            self.inner.high_hz
        )
    }

    fn __eq__(&self, other: &PyDtmfTone) -> bool {
        self.inner == other.inner
    }
}

/// Python wrapper for DtmfTable
#[pyclass(name = "DtmfTable", module = "dtmf_table")]
#[derive(Debug)]
pub struct PyDtmfTable {
    inner: RustDtmfTable,
}

#[pymethods]
impl PyDtmfTable {
    /// Create a new DTMF table instance.
    #[new]
    fn new() -> Self {
        PyDtmfTable {
            inner: RustDtmfTable::new(),
        }
    }

    /// Get all DTMF keys in keypad order.
    ///
    /// Returns:
    ///     list[DtmfKey]: All 16 DTMF keys
    #[staticmethod]
    fn all_keys() -> Vec<PyDtmfKey> {
        RustDtmfTable::ALL_KEYS
            .iter()
            .map(|&key| PyDtmfKey { inner: key })
            .collect()
    }

    /// Get all DTMF tones in keypad order.
    ///
    /// Returns:
    ///     list[DtmfTone]: All 16 DTMF tones
    #[staticmethod]
    fn all_tones() -> Vec<PyDtmfTone> {
        RustDtmfTable::ALL_TONES
            .iter()
            .map(|&tone| PyDtmfTone { inner: tone })
            .collect()
    }

    /// Look up frequencies for a given key.
    ///
    /// Args:
    ///     key (DtmfKey): The DTMF key to look up
    ///
    /// Returns:
    ///     tuple[int, int]: (low_frequency_hz, high_frequency_hz)
    #[staticmethod]
    fn lookup_key(key: PyDtmfKey) -> (u16, u16) {
        RustDtmfTable::lookup_key(key.inner)
    }

    /// Find DTMF key from exact frequency pair.
    ///
    /// Args:
    ///     low (int): Low frequency in Hz
    ///     high (int): High frequency in Hz
    ///
    /// Returns:
    ///     DtmfKey or None: The matching key, or None if no exact match
    #[staticmethod]
    fn from_pair_exact(low: u16, high: u16) -> Option<PyDtmfKey> {
        RustDtmfTable::from_pair_exact(low, high).map(|key| PyDtmfKey { inner: key })
    }

    /// Find DTMF key from frequency pair with automatic order normalization.
    ///
    /// Args:
    ///     a (int): First frequency in Hz
    ///     b (int): Second frequency in Hz
    ///
    /// Returns:
    ///     DtmfKey or None: The matching key, or None if no exact match
    #[staticmethod]
    fn from_pair_normalised(a: u16, b: u16) -> Option<PyDtmfKey> {
        RustDtmfTable::from_pair_normalised(a, b).map(|key| PyDtmfKey { inner: key })
    }

    /// Find DTMF key from frequency pair with tolerance (integer version).
    ///
    /// Args:
    ///     low (int): Low frequency in Hz
    ///     high (int): High frequency in Hz
    ///     tol_hz (int): Tolerance in Hz
    ///
    /// Returns:
    ///     DtmfKey or None: The matching key within tolerance, or None
    fn from_pair_tol_u32(&self, low: u32, high: u32, tol_hz: u32) -> Option<PyDtmfKey> {
        self.inner
            .from_pair_tol_u32(low, high, tol_hz)
            .map(|key| PyDtmfKey { inner: key })
    }

    /// Find DTMF key from frequency pair with tolerance (float version).
    ///
    /// Args:
    ///     low (float): Low frequency in Hz
    ///     high (float): High frequency in Hz
    ///     tol_hz (float): Tolerance in Hz
    ///
    /// Returns:
    ///     DtmfKey or None: The matching key within tolerance, or None
    fn from_pair_tol_f64(&self, low: f64, high: f64, tol_hz: f64) -> Option<PyDtmfKey> {
        self.inner
            .from_pair_tol_f64(low, high, tol_hz)
            .map(|key| PyDtmfKey { inner: key })
    }

    /// Find the nearest DTMF key and snap frequencies to canonical values (integer version).
    ///
    /// Args:
    ///     low (int): Low frequency estimate in Hz
    ///     high (int): High frequency estimate in Hz
    ///
    /// Returns:
    ///     tuple[DtmfKey, int, int]: (key, snapped_low_hz, snapped_high_hz)
    fn nearest_u32(&self, low: u32, high: u32) -> (PyDtmfKey, u16, u16) {
        let (key, snapped_low, snapped_high) = self.inner.nearest_u32(low, high);
        (PyDtmfKey { inner: key }, snapped_low, snapped_high)
    }

    /// Find the nearest DTMF key and snap frequencies to canonical values (float version).
    ///
    /// Args:
    ///     low (float): Low frequency estimate in Hz
    ///     high (float): High frequency estimate in Hz
    ///
    /// Returns:
    ///     tuple[DtmfKey, int, int]: (key, snapped_low_hz, snapped_high_hz)
    fn nearest_f64(&self, low: f64, high: f64) -> (PyDtmfKey, u16, u16) {
        let (key, snapped_low, snapped_high) = self.inner.nearest_f64(low, high);
        (PyDtmfKey { inner: key }, snapped_low, snapped_high)
    }

    fn __str__(&self) -> String {
        format!("{}", self.inner)
    }

    fn __repr__(&self) -> String {
        "DtmfTable()".to_string()
    }
}

/// Initialize the Python module
#[pymodule]
fn dtmf_table(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDtmfKey>()?;
    m.add_class::<PyDtmfTone>()?;
    m.add_class::<PyDtmfTable>()?;

    // Add module-level constants
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add(
        "__doc__",
        "DTMF (Dual-Tone Multi-Frequency) frequency table for telephony applications",
    )?;

    Ok(())
}
