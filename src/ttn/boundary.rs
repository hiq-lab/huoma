//! Analytical boundary tensors for the projected TTN.
//!
//! At each edge where a volatile island meets the stable exterior, the
//! stable side contributes a **boundary tensor** that captures the
//! time-evolved state of the stable subtree projected onto the boundary
//! edge. The volatile-side TTN treats this tensor as a fixed, non-dynamical
//! input at the interface — it participates in observables but not in gate
//! application.
//!
//! Track F milestone **F.3**.
//!
//! # Boundary modes
//!
//! - [`BoundaryMode::ProductState`] — each stable qubit evolves
//!   independently under its local single-qubit Hamiltonian
//!   `exp(-i h_x dt X) · exp(-i h_z dt Z)` for `n_steps`. The boundary
//!   tensor is the evolved single-qubit state at the boundary qubit,
//!   `bond_dim = 1`. Exact when the stable side has zero entanglement
//!   (perfect commensurability). This is the default.
//!
//! - [`BoundaryMode::LowRankMps`] — *(stub, not yet implemented)* — for
//!   each boundary qubit, build a small MPS along the stable side of the
//!   boundary edge, evolve it, and extract the boundary tensor as the
//!   Schmidt decomposition at `bond_dim` 2–4. Captures leading corrections
//!   beyond product state.

use num_complex::Complex64;

use crate::kicked_ising::KimParams;
use crate::ttn::kim_heavy_hex::{rx_gate, rz_gate};

type C = Complex64;

/// How the boundary tensor is computed.
#[derive(Debug, Clone, Copy)]
pub enum BoundaryMode {
    /// Product-state approximation: each stable qubit evolves under its
    /// local Hamiltonian independently. Bond dimension = 1. Exact for
    /// perfectly commensurate stable regions.
    ProductState,
    /// *(Stub — not yet implemented.)* Low-rank MPS correction with the
    /// given number of stable-side hops. Bond dimension = 2–4.
    #[allow(dead_code)]
    LowRankMps {
        depth: usize,
    },
}

/// A fixed boundary tensor at a stable/volatile interface edge.
///
/// Shape: `[bond_dim, 2]` in row-major order, where the first axis is the
/// bond leg connecting into the volatile TTN and the second axis is the
/// physical leg (σ ∈ {0, 1}) of the boundary qubit.
///
/// For `ProductState` mode: `bond_dim = 1` and `data = [α, β]` where
/// `|ψ⟩ = α|0⟩ + β|1⟩` is the time-evolved single-qubit state.
#[derive(Debug, Clone)]
pub struct BoundaryTensor {
    /// Flat row-major data, length = `bond_dim × 2`.
    pub data: Vec<C>,
    /// Bond dimension (number of rows in the `[bond, phys]` matrix).
    pub bond_dim: usize,
}

impl BoundaryTensor {
    /// Compute the local ⟨Z⟩ of the boundary qubit from this tensor alone.
    ///
    /// For a product-state boundary this is just
    /// `(|α|² − |β|²) / (|α|² + |β|²)`.
    #[must_use]
    pub fn expectation_z(&self) -> f64 {
        let mut p0 = 0.0_f64;
        let mut p1 = 0.0_f64;
        for row in 0..self.bond_dim {
            p0 += self.data[row * 2].norm_sqr();
            p1 += self.data[row * 2 + 1].norm_sqr();
        }
        let denom = p0 + p1;
        if denom < 1e-30 {
            0.0
        } else {
            (p0 - p1) / denom
        }
    }
}

/// Compute the boundary tensor for one stable-side qubit at the
/// stable/volatile interface.
///
/// # Arguments
///
/// * `omega` — the boundary qubit's natural frequency (h_x for the KIM,
///   or whatever per-qubit coupling drives the single-qubit kick).
/// * `params` — KIM parameters for the Floquet evolution.
/// * `n_steps` — number of Floquet steps to time-evolve the boundary
///   qubit's state.
/// * `mode` — how to compute the boundary tensor (see [`BoundaryMode`]).
///
/// # Returns
///
/// A [`BoundaryTensor`] representing the time-evolved state of the
/// stable-side qubit at the boundary edge.
///
/// # Panics
///
/// Panics on `BoundaryMode::LowRankMps` (stub, not yet implemented).
#[must_use]
pub fn compute_boundary_tensor(
    omega: f64,
    params: KimParams,
    n_steps: usize,
    mode: BoundaryMode,
) -> BoundaryTensor {
    match mode {
        BoundaryMode::ProductState => {
            compute_product_state_boundary(omega, params, n_steps)
        }
        BoundaryMode::LowRankMps { .. } => {
            unimplemented!(
                "BoundaryMode::LowRankMps is stubbed — ship with F.3 refinement"
            )
        }
    }
}

/// Product-state boundary: evolve `|0⟩` under the single-qubit KIM
/// Hamiltonian for `n_steps` and return the resulting 2-vector as a
/// `bond_dim = 1` boundary tensor.
///
/// Each step applies:
///   1. Rz(2 · h_z · dt) — skipped if h_z = 0
///   2. Rx(2 · h_x · dt) — where h_x is overridden to `omega` if the
///      caller provides a per-qubit frequency, or falls back to `params.h_x`
///
/// We use `omega` as the per-qubit h_x value, matching the convention
/// that the frequency vector in the partitioner represents the per-qubit
/// kick strength.
fn compute_product_state_boundary(
    omega: f64,
    params: KimParams,
    n_steps: usize,
) -> BoundaryTensor {
    // Initial state: |0⟩ = [1, 0].
    let mut state = [C::new(1.0, 0.0), C::new(0.0, 0.0)];

    let rx = rx_gate(2.0 * omega * params.dt);
    let has_rz = params.h_z != 0.0;
    let rz = if has_rz {
        rz_gate(2.0 * params.h_z * params.dt)
    } else {
        [[C::new(1.0, 0.0), C::new(0.0, 0.0)],
         [C::new(0.0, 0.0), C::new(1.0, 0.0)]]
    };

    for _ in 0..n_steps {
        // Note: in the full KIM the ZZ layer comes first, but on a single
        // qubit in the stable region the ZZ acts as a global phase (the
        // stable side is product-state by assumption), so we skip it.
        if has_rz {
            let a0 = state[0];
            let a1 = state[1];
            state[0] = rz[0][0] * a0 + rz[0][1] * a1;
            state[1] = rz[1][0] * a0 + rz[1][1] * a1;
        }
        let a0 = state[0];
        let a1 = state[1];
        state[0] = rx[0][0] * a0 + rx[0][1] * a1;
        state[1] = rx[1][0] * a0 + rx[1][1] * a1;
    }

    BoundaryTensor {
        data: state.to_vec(),
        bond_dim: 1,
    }
}

/// Compute boundary tensors for all boundary edges of a set of volatile
/// islands, using the per-qubit frequency as each stable qubit's kick
/// strength.
///
/// Returns one `BoundaryTensor` per boundary edge across all islands,
/// in the same order as `islands[i].boundary_edges[j]` flattened.
#[must_use]
pub fn compute_all_boundary_tensors(
    frequencies: &[f64],
    islands: &[crate::ttn::subtree::VolatileIsland],
    params: KimParams,
    n_steps: usize,
    mode: BoundaryMode,
) -> Vec<Vec<BoundaryTensor>> {
    islands
        .iter()
        .map(|island| {
            island
                .boundary_edges
                .iter()
                .map(|be| {
                    let omega = frequencies
                        .get(be.global_stable_qubit)
                        .copied()
                        .unwrap_or(params.h_x);
                    compute_boundary_tensor(omega, params, n_steps, mode)
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn product_state_initial_is_z_plus_one() {
        let params = KimParams::self_dual();
        let bt = compute_boundary_tensor(0.0, params, 0, BoundaryMode::ProductState);
        assert_eq!(bt.bond_dim, 1);
        assert_eq!(bt.data.len(), 2);
        // No steps → still |0⟩ → ⟨Z⟩ = +1.
        assert!((bt.expectation_z() - 1.0).abs() < 1e-14);
    }

    #[test]
    fn product_state_one_step_matches_cos_theta() {
        // After one Rx(θ) on |0⟩: ⟨Z⟩ = cos(θ).
        let theta_h = 0.8;
        let params = KimParams {
            j: 0.0,
            h_x: 1.0, // overridden by omega below
            h_z: 0.0,
            dt: 1.0,
        };
        let bt = compute_boundary_tensor(
            theta_h / 2.0, // omega = h_x, so 2·omega·dt = θ_h
            params,
            1,
            BoundaryMode::ProductState,
        );
        let expected = theta_h.cos();
        assert!(
            (bt.expectation_z() - expected).abs() < 1e-14,
            "expected cos({theta_h}) = {expected}, got {}",
            bt.expectation_z()
        );
    }

    #[test]
    fn product_state_multi_step_decays() {
        // Multiple Rx(0.8) steps compound rotation. ⟨Z⟩ should oscillate
        // and not blow up.
        let params = KimParams {
            j: 0.0,
            h_x: 0.4,
            h_z: 0.0,
            dt: 1.0,
        };
        let bt = compute_boundary_tensor(0.4, params, 10, BoundaryMode::ProductState);
        assert!(bt.expectation_z().is_finite());
        assert!((-1.0..=1.0).contains(&bt.expectation_z()));
    }

    #[test]
    fn product_state_with_hz_kick() {
        let params = KimParams {
            j: 0.0,
            h_x: 0.4,
            h_z: 0.2,
            dt: 1.0,
        };
        let bt = compute_boundary_tensor(0.4, params, 5, BoundaryMode::ProductState);
        assert!(bt.expectation_z().is_finite());
        assert!((-1.0..=1.0).contains(&bt.expectation_z()));
    }

    #[test]
    fn boundary_tensor_norm_is_unity() {
        let params = KimParams::self_dual();
        for n_steps in [0, 1, 5, 20] {
            let bt = compute_boundary_tensor(0.4, params, n_steps, BoundaryMode::ProductState);
            let norm_sq: f64 = bt.data.iter().map(|c| c.norm_sqr()).sum();
            assert!(
                (norm_sq - 1.0).abs() < 1e-14,
                "boundary tensor norm² = {norm_sq} at {n_steps} steps"
            );
        }
    }

    #[test]
    fn compute_all_boundary_tensors_empty_islands() {
        let result = compute_all_boundary_tensors(
            &[],
            &[],
            KimParams::self_dual(),
            5,
            BoundaryMode::ProductState,
        );
        assert!(result.is_empty());
    }
}
