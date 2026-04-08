//! Topology-agnostic dense statevector reference.
//!
//! Test-only (`#[cfg(test)]`). Used by the D.2 anchors (Y-junction + degree-4
//! star) to validate the native `Ttn` contraction path against an independent
//! implementation that makes no assumption about qubit connectivity — every
//! two-qubit gate can land on any pair of qubits, not just neighbours in a
//! chain.
//!
//! The existing `kicked_ising::reference_kim_run` is KIM-specific and assumes
//! a linear chain. This module is the general-purpose dense reference that
//! non-1D Huoma tests need. Caps out at N ≈ 24 for memory reasons; N = 4 and
//! N = 5 are our anchors and trivially fit.

#![cfg(test)]

use num_complex::Complex64;

type C = Complex64;

/// Full dense statevector on `n` qubits. Qubit `q` corresponds to bit `q`
/// in the basis index (bit 0 = LSB in the mask, but by convention we treat
/// qubit 0 as the most significant bit in display / comparison so the
/// little/big-endian direction matches how tests are written). The amplitude
/// at index `k` is `<k|ψ>`.
#[derive(Debug, Clone)]
pub struct DenseState {
    pub n: usize,
    pub amps: Vec<C>,
}

impl DenseState {
    /// Product state |0…0⟩.
    pub fn zero(n: usize) -> Self {
        assert!(n <= 24, "DenseState is test-only and caps at n ≤ 24");
        let mut amps = vec![C::new(0.0, 0.0); 1 << n];
        amps[0] = C::new(1.0, 0.0);
        Self { n, amps }
    }

    /// Apply a single-qubit gate `u` to qubit `q`.
    ///
    /// Convention: qubit 0 is the *most significant* bit of the basis index.
    /// That is, for n = 4 and index k = 0b1010, qubit 0 is bit 3 (= 1),
    /// qubit 1 is bit 2 (= 0), qubit 2 is bit 1 (= 1), qubit 3 is bit 0
    /// (= 0). This matches `Mps::to_statevector` in mps.rs, so the two
    /// references can be cross-checked in the 1D case if desired.
    pub fn apply_single(&mut self, q: usize, u: [[C; 2]; 2]) {
        assert!(q < self.n);
        let shift = self.n - 1 - q;
        let mask = 1usize << shift;
        let dim = self.amps.len();
        let mut out = self.amps.clone();
        for k in 0..dim {
            let bit = (k >> shift) & 1;
            // The "paired" index has the q-bit flipped.
            if bit == 0 {
                let k1 = k | mask;
                let a0 = self.amps[k];
                let a1 = self.amps[k1];
                out[k] = u[0][0] * a0 + u[0][1] * a1;
                out[k1] = u[1][0] * a0 + u[1][1] * a1;
            }
        }
        self.amps = out;
    }

    /// Apply a two-qubit gate `u` to qubits `(qa, qb)`. The 4×4 unitary is
    /// indexed in little-endian order on the pair `(sigma_a, sigma_b)` where
    /// `row = sigma_a·2 + sigma_b` and `col = sigma'_a·2 + sigma'_b`. This
    /// matches `Mps::apply_two_qubit`.
    pub fn apply_two_qubit(&mut self, qa: usize, qb: usize, u: [[C; 4]; 4]) {
        assert!(qa < self.n && qb < self.n && qa != qb);
        let sa = self.n - 1 - qa;
        let sb = self.n - 1 - qb;
        let ma = 1usize << sa;
        let mb = 1usize << sb;
        let dim = self.amps.len();
        let mut out = vec![C::new(0.0, 0.0); dim];
        for k in 0..dim {
            // Only iterate the (0,0) corner of each 4-tuple to avoid
            // double-counting.
            let ba = (k >> sa) & 1;
            let bb = (k >> sb) & 1;
            if ba != 0 || bb != 0 {
                continue;
            }
            let k00 = k;
            let k01 = k | mb;
            let k10 = k | ma;
            let k11 = k | ma | mb;
            let a00 = self.amps[k00];
            let a01 = self.amps[k01];
            let a10 = self.amps[k10];
            let a11 = self.amps[k11];
            // out_row = Σ_col u[row][col] · a_col, with col/row ordered
            // (sa, sb) = (0,0), (0,1), (1,0), (1,1).
            out[k00] = u[0][0] * a00 + u[0][1] * a01 + u[0][2] * a10 + u[0][3] * a11;
            out[k01] = u[1][0] * a00 + u[1][1] * a01 + u[1][2] * a10 + u[1][3] * a11;
            out[k10] = u[2][0] * a00 + u[2][1] * a01 + u[2][2] * a10 + u[2][3] * a11;
            out[k11] = u[3][0] * a00 + u[3][1] * a01 + u[3][2] * a10 + u[3][3] * a11;
        }
        self.amps = out;
    }

    /// Single-qubit ⟨Z⟩ at qubit `target`, normalized by the state's norm².
    pub fn expectation_z(&self, target: usize) -> f64 {
        assert!(target < self.n);
        let shift = self.n - 1 - target;
        let mut num = 0.0_f64;
        let mut denom = 0.0_f64;
        for (k, a) in self.amps.iter().enumerate() {
            let prob = a.norm_sqr();
            denom += prob;
            if ((k >> shift) & 1) == 0 {
                num += prob;
            } else {
                num -= prob;
            }
        }
        if denom < 1e-30 {
            0.0
        } else {
            num / denom
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hadamard() -> [[C; 2]; 2] {
        let s = 1.0 / 2.0_f64.sqrt();
        [
            [C::new(s, 0.0), C::new(s, 0.0)],
            [C::new(s, 0.0), C::new(-s, 0.0)],
        ]
    }

    fn cnot() -> [[C; 4]; 4] {
        let o = C::new(1.0, 0.0);
        let z = C::new(0.0, 0.0);
        [
            [o, z, z, z],
            [z, o, z, z],
            [z, z, z, o],
            [z, z, o, z],
        ]
    }

    #[test]
    fn zero_state_has_plus_one_expectation_z() {
        let state = DenseState::zero(3);
        for q in 0..3 {
            assert!((state.expectation_z(q) - 1.0).abs() < 1e-14);
        }
    }

    #[test]
    fn hadamard_makes_zero_expectation_z() {
        let mut state = DenseState::zero(3);
        state.apply_single(1, hadamard());
        assert!((state.expectation_z(0) - 1.0).abs() < 1e-14);
        assert!(state.expectation_z(1).abs() < 1e-14);
        assert!((state.expectation_z(2) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn bell_state_has_zero_expectation_z_on_both_qubits() {
        // H on 0, then CNOT(0,1) → (|00⟩ + |11⟩)/√2.
        let mut state = DenseState::zero(2);
        state.apply_single(0, hadamard());
        state.apply_two_qubit(0, 1, cnot());
        assert!(state.expectation_z(0).abs() < 1e-14);
        assert!(state.expectation_z(1).abs() < 1e-14);
    }

    #[test]
    fn norm_preserved_under_unitary() {
        let mut state = DenseState::zero(4);
        state.apply_single(0, hadamard());
        state.apply_single(2, hadamard());
        state.apply_two_qubit(1, 3, cnot());
        let norm: f64 = state.amps.iter().map(|a| a.norm_sqr()).sum();
        assert!((norm - 1.0).abs() < 1e-14);
    }
}
