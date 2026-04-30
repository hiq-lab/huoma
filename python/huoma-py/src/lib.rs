//! Python bindings for huoma — TTN/MPS quantum simulator.
//!
//! Scope (v0.2, VQ-109 D1 days 0–4):
//! - `Mps` class: construct, apply 1- and 2-qubit gates, query expectation
//!   values, bond dimensions, statevector.
//! - `Mps.expectation_z_string(positions)` — multi-qubit Z-product expectation.
//! - `Mps.expectation_pauli_string(spec)` — full Pauli-string expectation
//!   for chemistry Hamiltonians (single-sweep evaluation, no Python-side
//!   round-trip per Pauli).
//! - `chi_allocation_sinc` module-level function.
//!
//! Intentionally NOT exposed:
//! - `Ttn` and tree topologies — VQE production runs use 1D MPS.
//!
//! Design: bindings are thin. Chemistry-side logic (PySCF, OpenFermion,
//! VQE loop, ansatz construction) lives in `services/qmmm-fep/` Python,
//! NOT in this crate. Huoma stays standalone.

use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use huoma::allocator::chi_allocation_sinc as huoma_chi_allocation_sinc;
use huoma::mps::Mps as HuomaMps;

/// Matrix Product State simulator with adaptive per-bond truncation.
///
/// Bond dimension χ is set per-call on `apply_two_qubit`; `chi_allocation_sinc`
/// at the module level produces a per-bond χ schedule.
#[pyclass(name = "Mps", module = "huoma_py")]
struct PyMps {
    inner: HuomaMps,
}

#[pymethods]
impl PyMps {
    /// Initialise an MPS in |0...0⟩ on `n_qubits`.
    #[new]
    fn new(n_qubits: usize) -> PyResult<Self> {
        if n_qubits == 0 {
            return Err(PyValueError::new_err("n_qubits must be >= 1"));
        }
        Ok(Self {
            inner: HuomaMps::new(n_qubits),
        })
    }

    /// Number of qubits in the chain.
    #[getter]
    fn n_qubits(&self) -> usize {
        self.inner.n_qubits
    }

    /// Apply a single-qubit gate `u` (2x2 complex matrix) to qubit `q`.
    fn apply_single(&mut self, q: usize, u: PyReadonlyArray2<Complex64>) -> PyResult<()> {
        if q >= self.inner.n_qubits {
            return Err(PyValueError::new_err(format!(
                "qubit index {} out of range (n_qubits = {})",
                q, self.inner.n_qubits
            )));
        }
        let arr = u.as_array();
        if arr.shape() != [2, 2] {
            return Err(PyValueError::new_err(format!(
                "expected 2x2 matrix, got shape {:?}",
                arr.shape()
            )));
        }
        let gate: [[Complex64; 2]; 2] =
            [[arr[[0, 0]], arr[[0, 1]]], [arr[[1, 0]], arr[[1, 1]]]];
        self.inner.apply_single(q, gate);
        Ok(())
    }

    /// Apply a two-qubit gate `u` (4x4 complex matrix) on qubits `q` and `q+1`.
    /// `max_bond` caps the SVD truncation at the affected bond.
    fn apply_two_qubit(
        &mut self,
        q: usize,
        u: PyReadonlyArray2<Complex64>,
        max_bond: usize,
    ) -> PyResult<()> {
        if q + 1 >= self.inner.n_qubits {
            return Err(PyValueError::new_err(format!(
                "qubit pair ({}, {}) out of range (n_qubits = {})",
                q,
                q + 1,
                self.inner.n_qubits
            )));
        }
        let arr = u.as_array();
        if arr.shape() != [4, 4] {
            return Err(PyValueError::new_err(format!(
                "expected 4x4 matrix, got shape {:?}",
                arr.shape()
            )));
        }
        let mut gate = [[Complex64::new(0.0, 0.0); 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                gate[i][j] = arr[[i, j]];
            }
        }
        self.inner
            .apply_two_qubit(q, gate, max_bond)
            .map_err(|e| PyRuntimeError::new_err(format!("apply_two_qubit failed: {}", e)))?;
        Ok(())
    }

    /// ⟨ψ| Z_target |ψ⟩.
    fn expectation_z(&self, target: usize) -> PyResult<f64> {
        if target >= self.inner.n_qubits {
            return Err(PyValueError::new_err(format!(
                "qubit index {} out of range (n_qubits = {})",
                target, self.inner.n_qubits
            )));
        }
        Ok(self.inner.expectation_z(target))
    }

    /// ⟨ψ| ∏_q Z_q |ψ⟩ over the given qubit positions.
    ///
    /// Duplicates cancel (Z² = I). Single-sweep evaluation, normalized by ⟨ψ|ψ⟩.
    fn expectation_z_string(&self, positions: Vec<usize>) -> PyResult<f64> {
        for &p in &positions {
            if p >= self.inner.n_qubits {
                return Err(PyValueError::new_err(format!(
                    "position {} out of range (n_qubits = {})",
                    p, self.inner.n_qubits
                )));
            }
        }
        Ok(self.inner.expectation_z_string(&positions))
    }

    /// ⟨ψ| P |ψ⟩ for a Pauli string `P = ⊗_q P_q`.
    ///
    /// `spec` is a string of length `n_qubits`. Each character is one of
    /// `I`, `X`, `Y`, `Z` (case-insensitive). Returns a real number; Pauli
    /// strings are Hermitian.
    ///
    /// Single-sweep evaluation: for each X/Y site, applies a basis rotation
    /// to a clone, then measures the multi-Z product. Avoids Python-side
    /// round-trips, suitable for chemistry Hamiltonians with thousands of
    /// Pauli terms per VQE step.
    fn expectation_pauli_string(&self, spec: &str) -> PyResult<f64> {
        self.inner
            .expectation_pauli_string(spec)
            .map_err(|e| PyValueError::new_err(format!("{e}")))
    }

    /// ⟨ψ|ψ⟩.
    fn norm_squared(&self) -> f64 {
        self.inner.norm_squared()
    }

    /// Per-bond χ list (length n_qubits - 1).
    fn bond_dims(&self) -> Vec<usize> {
        self.inner.bond_dims()
    }

    /// Cumulative discarded spectral weight summed over all bonds.
    fn total_discarded_weight(&self) -> f64 {
        self.inner.total_discarded_weight()
    }

    /// Dense statevector of length 2^n_qubits.
    /// Memory cost: O(2^n complex doubles) — only safe for n ≲ 28 on M4.
    fn to_statevector<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Complex64>> {
        self.inner.to_statevector().into_pyarray(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "Mps(n_qubits={}, χ_max={}, ‖ψ‖²={:.6})",
            self.inner.n_qubits,
            self.inner.bond_dims().iter().max().copied().unwrap_or(1),
            self.inner.norm_squared()
        )
    }
}

/// sin(C/2) commensurability-driven χ allocation across bonds.
///
/// Given site-local frequencies and a total χ budget, produce a per-bond
/// allocation that concentrates χ on volatile (incommensurate) bonds and
/// spares it on stable (commensurate) ones.
#[pyfunction]
#[pyo3(signature = (frequencies, total_budget, chi_min=2, chi_max=16))]
fn chi_allocation_sinc(
    frequencies: PyReadonlyArray1<f64>,
    total_budget: usize,
    chi_min: usize,
    chi_max: usize,
) -> PyResult<Vec<usize>> {
    if chi_min == 0 || chi_max == 0 {
        return Err(PyValueError::new_err("chi_min and chi_max must be >= 1"));
    }
    if chi_min > chi_max {
        return Err(PyValueError::new_err("chi_min must be <= chi_max"));
    }
    let freqs: Vec<f64> = frequencies.as_array().iter().copied().collect();
    Ok(huoma_chi_allocation_sinc(
        &freqs,
        total_budget,
        chi_min,
        chi_max,
    ))
}

#[pymodule]
fn huoma_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMps>()?;
    m.add_function(wrap_pyfunction!(chi_allocation_sinc, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
