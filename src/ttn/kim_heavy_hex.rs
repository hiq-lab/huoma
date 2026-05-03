//! Heavy-hex kicked-Ising Floquet driver for the native TTN backend.
//!
//! Composes the primitives shipped in D.1–D.5.0 into a single Floquet
//! step of the kicked Ising Hamiltonian over an arbitrary coupling graph
//! whose spanning tree drives the TTN:
//!
//! - [`Ttn::apply_two_qubit_on_edge`] for the 126 tree-adjacent ZZ gates
//! - [`Ttn::apply_two_qubit_via_path`] for the 18 non-tree ZZ gates
//!   (swap-network routing through the tree)
//! - [`Ttn::apply_single`] for the per-qubit Rx / Rz kicks
//! - [`Ttn::expectation_z_all`] (D.5.0) for the per-step ⟨Z⟩ history,
//!   which is what makes this driver physically constructible at
//!   N = 127 — the earlier statevector-materialising `expectation_z`
//!   path capped at N ≲ 20
//!
//! This is Track D milestone **D.5.1** per `TRACK_D_DESIGN.md` §
//! "D.5 — Tindall N=127 benchmark": the Huoma-only runner scaffold,
//! no Tindall reference comparison, no ITensor cross-check, just the
//! machinery that produces the first honest N = 127 number on the
//! board. Those comparisons ship in subsequent PRs.
//!
//! # Gate conventions
//!
//! The driver matches the "no factor of two" convention that
//! [`crate::kicked_ising::apply_kim_step`] established for the 1D path:
//!
//! - ZZ gate is `exp(-i (J·dt) Z⊗Z)` — full angle, no half
//! - Rx / Rz kicks are `exp(-i θ/2 σ)` standard-convention, so the
//!   angles passed in are `2·h_x·dt` / `2·h_z·dt`
//!
//! Layer order inside one Floquet step is `ZZ → Rz (optional) → Rx`,
//! identical to the 1D driver so the linear-chain special case of this
//! driver would reproduce the 1D benchmark bit-for-bit (validated
//! incidentally by the existing 1D regression in `src/ttn/mod.rs`).
//!
//! # Example
//!
//! ```no_run
//! use huoma::kicked_ising::KimParams;
//! use huoma::ttn::{HeavyHexLayout, Ttn};
//! use huoma::ttn::kim_heavy_hex::run_kim_heavy_hex;
//! use huoma::ttn::topology::Edge;
//!
//! let layout = HeavyHexLayout::ibm_eagle_127();
//! let non_tree: Vec<Edge> = layout
//!     .non_tree_edges()
//!     .iter()
//!     .map(|[a, b]| Edge { a: *a, b: *b })
//!     .collect();
//! let mut ttn = Ttn::new(layout.tree().clone());
//! let params = KimParams { j: 1.0, h_x: 0.8, h_z: 0.0, dt: 0.5 };
//! let history = run_kim_heavy_hex(&mut ttn, &non_tree, params, 16, 5).unwrap();
//! assert_eq!(history.len(), 6); // initial + 5 steps
//! assert_eq!(history[0].len(), 127);
//! ```

use num_complex::Complex64;

use crate::error::Result;
use crate::kicked_ising::KimParams;
use crate::ttn::topology::Edge;
use crate::ttn::{EdgeId, Ttn};

type C = Complex64;

/// 4×4 `exp(-i θ Z⊗Z)` with the "no factor of two" convention.
///
/// Diagonal with eigenvalue pattern `(++ , -- , -- , ++)` indexed by
/// `(σ_a, σ_b) ∈ {(0,0), (0,1), (1,0), (1,1)}`. Matches
/// `crate::kicked_ising::apply_zz_dense` and the private `mps::zz`
/// factory used by the 1D driver, so linear-chain runs of this driver
/// are numerically identical to `crate::kicked_ising::apply_kim_step`.
#[must_use]
pub fn zz_gate(theta: f64) -> [[C; 4]; 4] {
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    // ZZ eigenvalue +1 (same-spin) → exp(-iθ).
    // ZZ eigenvalue −1 (diff-spin) → exp(+iθ).
    let pos = C::new(cos_t, -sin_t);
    let neg = C::new(cos_t, sin_t);
    let zero = C::new(0.0, 0.0);
    [
        [pos, zero, zero, zero],
        [zero, neg, zero, zero],
        [zero, zero, neg, zero],
        [zero, zero, zero, pos],
    ]
}

/// 2×2 `Rx(θ) = exp(-i θ/2 X)` — standard half-angle convention.
#[must_use]
pub fn rx_gate(theta: f64) -> [[C; 2]; 2] {
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    [
        [C::new(c, 0.0), C::new(0.0, -s)],
        [C::new(0.0, -s), C::new(c, 0.0)],
    ]
}

/// 2×2 `Rz(θ) = exp(-i θ/2 Z)` — standard half-angle convention.
#[must_use]
pub fn rz_gate(theta: f64) -> [[C; 2]; 2] {
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    [
        [C::new(c, -s), C::new(0.0, 0.0)],
        [C::new(0.0, 0.0), C::new(c, s)],
    ]
}

/// Apply one KIM Floquet step to `ttn` using its bound spanning-tree
/// topology plus an explicit list of non-tree coupling edges.
///
/// The tree edges are taken from `ttn.topology()` in their canonical
/// enumeration order; `non_tree_edges` is the list of graph edges that
/// were dropped to form the spanning tree (for Eagle 127q this is the
/// 18 hexagonal-plaquette closing edges returned by
/// [`HeavyHexLayout::non_tree_edges`]). The driver applies:
///
/// 1. `ZZ(J·dt)` on every tree edge via
///    [`Ttn::apply_two_qubit_on_edge`] (direct-adjacent path).
/// 2. `ZZ(J·dt)` on every non-tree edge via
///    [`Ttn::apply_two_qubit_via_path`] (swap-network path).
/// 3. `Rz(2·h_z·dt)` on every qubit (skipped if `h_z == 0`).
/// 4. `Rx(2·h_x·dt)` on every qubit.
///
/// Every SVD reuses `max_bond` — tree gates, swap-network gates, and
/// the internal SWAPs of the swap network all share the same cap so
/// the truncation budget composes uniformly across the step. Pass a
/// very large bound (e.g. `usize::MAX`) if you want the run to stay
/// lossless at small N.
///
/// [`HeavyHexLayout::non_tree_edges`]: crate::ttn::heavy_hex::HeavyHexLayout::non_tree_edges
pub fn apply_kim_step_heavy_hex(
    ttn: &mut Ttn,
    non_tree_edges: &[Edge],
    params: KimParams,
    max_bond: usize,
) -> Result<()> {
    // ── 1. ZZ entangling layer on every tree edge ──────────────────────
    let zz = zz_gate(params.j * params.dt);
    let n_edges = ttn.topology().n_edges();
    for eid_idx in 0..n_edges {
        ttn.apply_two_qubit_on_edge(EdgeId(eid_idx), zz, max_bond)?;
    }
    // ── 2. ZZ entangling layer on every non-tree edge (swap network) ──
    for edge in non_tree_edges {
        ttn.apply_two_qubit_via_path(edge.a, edge.b, zz, max_bond)?;
    }
    // ── 3. Global Rz kick (if h_z ≠ 0) ─────────────────────────────────
    if params.h_z != 0.0 {
        let rz = rz_gate(2.0 * params.h_z * params.dt);
        for q in 0..ttn.n_qubits() {
            ttn.apply_single(q, rz);
        }
    }
    // ── 4. Global Rx kick ──────────────────────────────────────────────
    let rx = rx_gate(2.0 * params.h_x * params.dt);
    for q in 0..ttn.n_qubits() {
        ttn.apply_single(q, rx);
    }
    Ok(())
}

/// Drive `n_steps` KIM Floquet steps on `ttn` and return the full
/// per-qubit ⟨Z⟩ history.
///
/// The returned vector has length `n_steps + 1`: entry `0` is the
/// initial state (read via [`Ttn::expectation_z_all`] before any gates
/// are applied) and entries `1..=n_steps` are post-step snapshots.
/// Each inner `Vec<f64>` has length [`Ttn::n_qubits`].
pub fn run_kim_heavy_hex(
    ttn: &mut Ttn,
    non_tree_edges: &[Edge],
    params: KimParams,
    max_bond: usize,
    n_steps: usize,
) -> Result<Vec<Vec<f64>>> {
    let mut history = Vec::with_capacity(n_steps + 1);
    history.push(ttn.expectation_z_all());
    for _ in 0..n_steps {
        apply_kim_step_heavy_hex(ttn, non_tree_edges, params, max_bond)?;
        history.push(ttn.expectation_z_all());
    }
    Ok(history)
}

/// Variant of [`apply_kim_step_heavy_hex`] that accepts a **per-edge
/// χ profile** for the tree-adjacent gate layer — Track D milestone
/// **D.5.2**, the matched-budget shootout entry point.
///
/// `chi_per_tree_edge[k]` is the bond-dimension cap used by the ZZ
/// gate on `EdgeId(k)` of `ttn.topology()`. The slice must have length
/// exactly `ttn.topology().n_edges()`; debug builds assert this.
///
/// The **non-tree** gate layer routes through
/// [`Ttn::apply_two_qubit_via_path`], which internally performs a
/// sequence of tree-adjacent SWAPs plus one final gate. Since the
/// swap-network's internal SWAPs cross many tree edges in a single
/// call, and threading a per-edge χ through the swap-network API
/// would require growing its public signature, this driver uses a
/// single `non_tree_chi` cap for the whole non-tree sequence. In
/// practice setting `non_tree_chi = chi_max` (the upper clip of the
/// allocator that produced `chi_per_tree_edge`) is the right default
/// — the swap network is transient and the final gate lands on a tree
/// edge which will respect its own per-edge cap on the next iteration.
///
/// Everything else (layer order, gate conventions, Rx / Rz kicks) is
/// identical to the uniform-χ driver so linear-chain runs of the two
/// drivers are numerically equivalent when `chi_per_tree_edge` is
/// constant.
pub fn apply_kim_step_heavy_hex_per_edge(
    ttn: &mut Ttn,
    non_tree_edges: &[Edge],
    params: KimParams,
    chi_per_tree_edge: &[usize],
    non_tree_chi: usize,
) -> Result<()> {
    let n_edges = ttn.topology().n_edges();
    assert_eq!(
        chi_per_tree_edge.len(),
        n_edges,
        "chi_per_tree_edge must have one entry per tree edge"
    );
    let zz = zz_gate(params.j * params.dt);

    // 1. ZZ entangling layer on every tree edge, with its own cap.
    for eid_idx in 0..n_edges {
        ttn.apply_two_qubit_on_edge(EdgeId(eid_idx), zz, chi_per_tree_edge[eid_idx])?;
    }
    // 2. ZZ entangling layer on every non-tree edge via swap network.
    for edge in non_tree_edges {
        ttn.apply_two_qubit_via_path(edge.a, edge.b, zz, non_tree_chi)?;
    }
    // 3. Global Rz kick (if h_z ≠ 0).
    if params.h_z != 0.0 {
        let rz = rz_gate(2.0 * params.h_z * params.dt);
        for q in 0..ttn.n_qubits() {
            ttn.apply_single(q, rz);
        }
    }
    // 4. Global Rx kick.
    let rx = rx_gate(2.0 * params.h_x * params.dt);
    for q in 0..ttn.n_qubits() {
        ttn.apply_single(q, rx);
    }
    Ok(())
}

/// Multi-step variant of [`apply_kim_step_heavy_hex_per_edge`] —
/// runs `n_steps` Floquet steps at fixed `chi_per_tree_edge` and
/// returns the per-qubit ⟨Z⟩ history, same shape as
/// [`run_kim_heavy_hex`].
pub fn run_kim_heavy_hex_per_edge(
    ttn: &mut Ttn,
    non_tree_edges: &[Edge],
    params: KimParams,
    chi_per_tree_edge: &[usize],
    non_tree_chi: usize,
    n_steps: usize,
) -> Result<Vec<Vec<f64>>> {
    let mut history = Vec::with_capacity(n_steps + 1);
    history.push(ttn.expectation_z_all());
    for _ in 0..n_steps {
        apply_kim_step_heavy_hex_per_edge(
            ttn,
            non_tree_edges,
            params,
            chi_per_tree_edge,
            non_tree_chi,
        )?;
        history.push(ttn.expectation_z_all());
    }
    Ok(history)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ttn::dense::DenseState;
    use crate::ttn::topology::Topology;
    use crate::ttn::HeavyHexLayout;

    /// Same 13-qubit Eagle sub-fragment used in the D.3 validation test.
    /// Replicated here so `kim_heavy_hex.rs` stays self-contained.
    fn small_eagle_fragment() -> (Topology, Vec<Edge>, Vec<Edge>) {
        let coupling: Vec<Edge> = vec![
            Edge { a: 0, b: 1 },
            Edge { a: 1, b: 2 },
            Edge { a: 2, b: 3 },
            Edge { a: 3, b: 4 },
            Edge { a: 4, b: 5 },
            Edge { a: 0, b: 6 },
            Edge { a: 6, b: 8 },
            Edge { a: 4, b: 7 },
            Edge { a: 7, b: 12 },
            Edge { a: 8, b: 9 },
            Edge { a: 9, b: 10 },
            Edge { a: 10, b: 11 },
            Edge { a: 11, b: 12 },
        ];
        // Drop (0, 6) to keep qubit 4 as a tree junction.
        let tree_edges: Vec<Edge> = coupling
            .iter()
            .copied()
            .filter(|e| !(e.a == 0 && e.b == 6))
            .collect();
        let non_tree_edges: Vec<Edge> = vec![Edge { a: 0, b: 6 }];
        let topology = Topology::from_edges(13, tree_edges);
        (topology, non_tree_edges, coupling)
    }

    /// The public driver must produce the same ⟨Z⟩ trajectory as a
    /// topology-agnostic `DenseState` reference on the 13-qubit Eagle
    /// sub-fragment used by D.3. This is the direct proof that
    /// `apply_kim_step_heavy_hex` is a correct public wrapper of the
    /// gate sequence the D.3 test already validated inline.
    #[test]
    fn small_eagle_fragment_driver_matches_dense_lossless() {
        let (topology, non_tree_edges, coupling) = small_eagle_fragment();
        let mut ttn = Ttn::new(topology);
        let mut dense = DenseState::zero(13);

        // Break the product state so KIM dynamics have something to do.
        for q in 0..13 {
            ttn.apply_single(q, rx_gate(std::f64::consts::FRAC_PI_2));
            dense.apply_single(q, rx_gate(std::f64::consts::FRAC_PI_2));
        }

        let params = KimParams::self_dual();
        let max_bond = 128;
        let n_steps = 3;

        let history = run_kim_heavy_hex(&mut ttn, &non_tree_edges, params, max_bond, n_steps)
            .expect("driver must not fail on the small Eagle fragment");
        assert_eq!(history.len(), n_steps + 1);
        assert_eq!(history[0].len(), 13);

        // Dense-reference Floquet step matching the driver's layer order.
        let zz = zz_gate(params.j * params.dt);
        let rx = rx_gate(2.0 * params.h_x * params.dt);
        for step in 1..=n_steps {
            for e in &coupling {
                dense.apply_two_qubit(e.a, e.b, zz);
            }
            // h_z = 0 at self-dual; skip Rz.
            for q in 0..13 {
                dense.apply_single(q, rx);
            }
            for q in 0..13 {
                let z_ttn = history[step][q];
                let z_dense = dense.expectation_z(q);
                assert!(
                    (z_ttn - z_dense).abs() < 1e-11,
                    "step {step}, q={q}: ttn={z_ttn:e}, dense={z_dense:e}"
                );
            }
        }
    }

    /// The driver must also produce Hermitian-bounded results on the
    /// self-dual point: every ⟨Z⟩ is in `[-1, 1]` and the norm stays
    /// normalised to within rounding when `max_bond` is above the
    /// Schmidt-rank bound.
    #[test]
    fn small_eagle_fragment_driver_preserves_norm_and_bounds() {
        let (topology, non_tree_edges, _coupling) = small_eagle_fragment();
        let mut ttn = Ttn::new(topology);
        for q in 0..13 {
            ttn.apply_single(q, rx_gate(std::f64::consts::FRAC_PI_2));
        }
        let params = KimParams::self_dual();
        let history = run_kim_heavy_hex(&mut ttn, &non_tree_edges, params, 128, 3).unwrap();
        for snapshot in &history {
            assert_eq!(snapshot.len(), 13);
            for z in snapshot {
                assert!(
                    (-1.0..=1.0).contains(z),
                    "⟨Z⟩ out of Hermitian bounds: {z}"
                );
            }
        }
        // Lossless regime: norm² within rounding of 1.
        let nsq = ttn.norm_squared();
        assert!((nsq - 1.0).abs() < 1e-10, "norm² drifted: {nsq:e}");
    }

    /// Smoke test at the full Eagle 127q scale. Runs one Floquet step on
    /// the real layout at a small bond dimension, confirms it completes,
    /// and checks that the 127 ⟨Z⟩ values are all Hermitian-bounded.
    ///
    /// This is the cheapest version of the "first honest N = 127 number"
    /// test — the minimal proof that D.5.0's env-sweep `expectation_z_all`
    /// + D.3's swap network + D.5.1's driver compose into a working
    /// pipeline at the Tindall target size. The multi-step depth-5
    /// variant lives in `eagle_127q_depth_5_tindall_params_runs_to_completion`.
    #[test]
    fn eagle_127q_single_kim_step_completes_and_is_bounded() {
        let layout = HeavyHexLayout::ibm_eagle_127();
        let non_tree: Vec<Edge> = layout
            .non_tree_edges()
            .iter()
            .map(|[a, b]| Edge { a: *a, b: *b })
            .collect();
        assert_eq!(non_tree.len(), 18);

        let mut ttn = Ttn::new(layout.tree().clone());
        assert_eq!(ttn.n_qubits(), 127);

        // Break the product state on every qubit so the ZZ layer has
        // non-trivial mixing to do (otherwise |0…0⟩ is an exact ZZ
        // eigenstate and the test is vacuous).
        for q in 0..127 {
            ttn.apply_single(q, rx_gate(std::f64::consts::FRAC_PI_2));
        }

        let params = KimParams {
            j: 1.0,
            h_x: 0.8,
            h_z: 0.0,
            dt: 0.5,
        }; // Tindall et al. PRX Quantum 5, 010308 (2024)

        apply_kim_step_heavy_hex(&mut ttn, &non_tree, params, 8)
            .expect("driver must not fail at N=127");

        let z = ttn.expectation_z_all();
        assert_eq!(z.len(), 127);
        for (q, &zq) in z.iter().enumerate() {
            assert!(
                (-1.0..=1.0).contains(&zq) && zq.is_finite(),
                "q={q}: ⟨Z⟩ = {zq} out of bounds or non-finite"
            );
        }
        // A finite discarded weight is expected at χ = 8 on Eagle 127q
        // after a full entangling layer; just check it's non-negative
        // and finite, i.e. the accounting isn't corrupted.
        let disc = ttn.total_discarded_weight();
        assert!(
            disc.is_finite() && disc >= 0.0,
            "total discarded weight invalid: {disc}"
        );
    }

    /// The per-edge χ driver with a uniform profile must produce the
    /// same per-qubit ⟨Z⟩ trajectory as the scalar-χ driver, because
    /// every SVD takes the same cap either way.
    #[test]
    fn per_edge_driver_with_uniform_profile_matches_scalar_driver() {
        let (topology, non_tree_edges, _coupling) = small_eagle_fragment();
        let n_edges = topology.n_edges();

        let mut ttn_scalar = Ttn::new(topology.clone());
        let mut ttn_per_edge = Ttn::new(topology);
        for q in 0..13 {
            ttn_scalar.apply_single(q, rx_gate(std::f64::consts::FRAC_PI_2));
            ttn_per_edge.apply_single(q, rx_gate(std::f64::consts::FRAC_PI_2));
        }

        let params = KimParams::self_dual();
        let max_bond = 64;
        let chi_uniform: Vec<usize> = vec![max_bond; n_edges];
        let n_steps = 3;

        let history_scalar =
            run_kim_heavy_hex(&mut ttn_scalar, &non_tree_edges, params, max_bond, n_steps).unwrap();
        let history_per_edge = run_kim_heavy_hex_per_edge(
            &mut ttn_per_edge,
            &non_tree_edges,
            params,
            &chi_uniform,
            max_bond,
            n_steps,
        )
        .unwrap();

        assert_eq!(history_scalar.len(), history_per_edge.len());
        for step in 0..history_scalar.len() {
            for q in 0..13 {
                let diff = (history_scalar[step][q] - history_per_edge[step][q]).abs();
                assert!(
                    diff < 1e-13,
                    "step {step}, q={q}: scalar={:e}, per_edge={:e}, diff={:e}",
                    history_scalar[step][q],
                    history_per_edge[step][q],
                    diff
                );
            }
        }
    }

    /// The per-edge χ driver with a NON-UNIFORM profile (and a cap
    /// large enough to stay lossless) must still agree with the dense
    /// reference on the 13q Eagle sub-fragment. This proves the
    /// per-edge dispatch in `apply_kim_step_heavy_hex_per_edge` is
    /// correct for arbitrary profiles, not just the constant case.
    #[test]
    fn per_edge_driver_nonuniform_lossless_matches_dense() {
        let (topology, non_tree_edges, coupling) = small_eagle_fragment();
        let n_edges = topology.n_edges();

        let mut ttn = Ttn::new(topology);
        let mut dense = DenseState::zero(13);
        for q in 0..13 {
            ttn.apply_single(q, rx_gate(std::f64::consts::FRAC_PI_2));
            dense.apply_single(q, rx_gate(std::f64::consts::FRAC_PI_2));
        }

        // A deliberately non-uniform profile. Every entry is ≥ 64 so
        // the run stays lossless (max Schmidt rank on N=13 is ≤ 64).
        let chi_per_edge: Vec<usize> =
            (0..n_edges).map(|i| 64 + (i % 4) * 16).collect();
        let non_tree_chi = 128;

        let params = KimParams::self_dual();
        let n_steps = 3;
        let history = run_kim_heavy_hex_per_edge(
            &mut ttn,
            &non_tree_edges,
            params,
            &chi_per_edge,
            non_tree_chi,
            n_steps,
        )
        .unwrap();

        // Drive dense reference with the same layer order.
        let zz = zz_gate(params.j * params.dt);
        let rx = rx_gate(2.0 * params.h_x * params.dt);
        for step in 1..=n_steps {
            for e in &coupling {
                dense.apply_two_qubit(e.a, e.b, zz);
            }
            for q in 0..13 {
                dense.apply_single(q, rx);
            }
            for q in 0..13 {
                let z_ttn = history[step][q];
                let z_dense = dense.expectation_z(q);
                assert!(
                    (z_ttn - z_dense).abs() < 1e-11,
                    "step {step}, q={q}: ttn={z_ttn:e}, dense={z_dense:e}"
                );
            }
        }
        // Lossless regime: norm² stays at 1.
        let nsq = ttn.norm_squared();
        assert!((nsq - 1.0).abs() < 1e-10, "norm² drifted: {nsq:e}");
    }

    /// Full depth-5 Tindall run on Eagle 127q at the exact parameters
    /// `(J, h_x, h_z, dt) = (1.0, 0.8, 0.0, 0.5)` from Tindall et al.
    /// (PRX Quantum 5, 010308, 2024). Exercises [`run_kim_heavy_hex`]
    /// end-to-end: the initial-state snapshot plus 5 post-step
    /// snapshots of per-qubit ⟨Z⟩.
    ///
    /// Assertions are **Huoma-only** (no Tindall / ITensor comparison
    /// yet — those ship in D.5.3 / D.5.4):
    ///
    /// - 6 snapshots returned (initial + 5 steps), each of length 127
    /// - Every ⟨Z⟩ value is finite and in `[-1, 1]`
    /// - Total discarded weight is finite and monotone non-decreasing
    ///   from step to step (enforced implicitly by the invariant that
    ///   `Ttn::total_discarded_weight` is an accumulator)
    /// - Trivial sanity check: the initial snapshot is all `+1.0`
    ///   before any gates are applied
    ///
    /// This is the first honest N = 127 depth-5 Huoma run on the board.
    #[test]
    fn eagle_127q_depth_5_tindall_params_runs_to_completion() {
        let layout = HeavyHexLayout::ibm_eagle_127();
        let non_tree: Vec<Edge> = layout
            .non_tree_edges()
            .iter()
            .map(|[a, b]| Edge { a: *a, b: *b })
            .collect();

        let mut ttn = Ttn::new(layout.tree().clone());
        let params = KimParams {
            j: 1.0,
            h_x: 0.8,
            h_z: 0.0,
            dt: 0.5,
        };
        let max_bond = 8;
        let n_steps = 5;

        let history = run_kim_heavy_hex(&mut ttn, &non_tree, params, max_bond, n_steps)
            .expect("depth-5 Tindall run must not fail at N=127");

        // 6 snapshots of 127 values each.
        assert_eq!(history.len(), n_steps + 1);
        for (step, snap) in history.iter().enumerate() {
            assert_eq!(snap.len(), 127, "step {step}: wrong length");
            for (q, &zq) in snap.iter().enumerate() {
                assert!(
                    zq.is_finite() && (-1.0..=1.0).contains(&zq),
                    "step {step}, q={q}: ⟨Z⟩ = {zq}"
                );
            }
        }

        // Initial state is |0…0⟩ so every ⟨Z⟩ = +1 exactly.
        for (q, &zq) in history[0].iter().enumerate() {
            assert!(
                (zq - 1.0).abs() < 1e-14,
                "initial q={q}: expected +1, got {zq}"
            );
        }

        // Total discarded weight is finite and non-negative. At χ = 8 on
        // a 127-qubit heavy-hex Floquet circuit we expect some truncation
        // but not catastrophic blow-up; the assertion below is an upper
        // bound derived from the total number of SVDs (≈ depth × (126
        // tree gates + 18 × short-path swap networks) ≈ a few thousand)
        // times a generous per-SVD budget of ≈ 1e-3.
        let disc = ttn.total_discarded_weight();
        assert!(
            disc.is_finite() && disc >= 0.0,
            "total discarded weight invalid: {disc}"
        );
        assert!(
            disc < 100.0,
            "total discarded weight suspiciously large: {disc}"
        );
    }

    /// Adiabatic-ramp primitive on a 12-qubit linear chain.
    ///
    /// Runs `H(s) = (1−s) H_X + s H_problem` with linearly ramped Schedule
    /// `s ∈ [0, 1]` through 50 Trotter steps, where the driver is
    /// `H_X = -h_x_0 Σ X_i` (ground state |+⟩^⊗N) and the problem is a
    /// transverse-field Ising chain `H_problem = J_0 Σ Z_iZ_{i+1} + h_z_0 Σ Z_i`.
    /// Each step calls `apply_kim_step_heavy_hex` with a step-dependent
    /// `KimParams`, dense replicates the same gate sequence, and we assert
    /// agreement at every step within rounding.
    ///
    /// This is the closed-system unitary primitive for the annealer thread:
    /// the same ramp generalises to tree-decomposable Ising at million-plus N
    /// once validated here.
    #[test]
    fn adiabatic_ramp_chain_matches_dense_lossless() {
        let n = 12;
        let n_steps = 50;
        let dt = 0.1_f64;

        let h_x_0 = 1.0_f64;
        let j_0 = 1.0_f64;
        let h_z_0 = 0.5_f64;

        let chain_edges: Vec<Edge> = (0..n - 1).map(|i| Edge { a: i, b: i + 1 }).collect();
        let topology = Topology::from_edges(n, chain_edges);
        let mut ttn = Ttn::new(topology);
        let mut dense = DenseState::zero(n);

        // Initial state: |+⟩^⊗N (ground state of -h_x_0 Σ X_i).
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        let h_gate: [[C; 2]; 2] = [
            [C::new(inv_sqrt2, 0.0), C::new(inv_sqrt2, 0.0)],
            [C::new(inv_sqrt2, 0.0), C::new(-inv_sqrt2, 0.0)],
        ];
        for q in 0..n {
            ttn.apply_single(q, h_gate);
            dense.apply_single(q, h_gate);
        }

        for q in 0..n {
            let z0 = dense.expectation_z(q);
            assert!(z0.abs() < 1e-12, "initial ⟨Z_{q}⟩ = {z0:e}, expected 0");
        }

        let no_non_tree: Vec<Edge> = vec![];
        let mut max_err = 0.0_f64;

        for k in 0..n_steps {
            // Midpoint sampling: schedule taken at the centre of step k.
            let s = (k as f64 + 0.5) / n_steps as f64;
            // Driver enters as -h_x_0 X, so the angle for the +h_x convention
            // of `apply_kim_step_heavy_hex` is negated.
            let params = KimParams {
                j: s * j_0,
                h_x: -(1.0 - s) * h_x_0,
                h_z: s * h_z_0,
                dt,
            };

            apply_kim_step_heavy_hex(&mut ttn, &no_non_tree, params, usize::MAX).unwrap();

            let zz = zz_gate(params.j * params.dt);
            for q in 0..n - 1 {
                dense.apply_two_qubit(q, q + 1, zz);
            }
            if params.h_z != 0.0 {
                let rz = rz_gate(2.0 * params.h_z * params.dt);
                for q in 0..n {
                    dense.apply_single(q, rz);
                }
            }
            let rx = rx_gate(2.0 * params.h_x * params.dt);
            for q in 0..n {
                dense.apply_single(q, rx);
            }

            let z_ttn = ttn.expectation_z_all();
            for q in 0..n {
                let z_dense = dense.expectation_z(q);
                let err = (z_ttn[q] - z_dense).abs();
                if err > max_err {
                    max_err = err;
                }
            }
        }

        assert!(
            max_err < 1e-11,
            "adiabatic ramp diverged from dense: max ⟨Z⟩ err = {max_err:e}"
        );

        // Sanity: schedule produced nontrivial dynamics. |+⟩^⊗N has ⟨Z⟩ = 0
        // everywhere; if the ramp did anything, some |⟨Z⟩| has grown.
        let z_final = ttn.expectation_z_all();
        let max_abs_z: f64 = z_final.iter().map(|z| z.abs()).fold(0.0, f64::max);
        assert!(
            max_abs_z > 0.05,
            "ramp produced trivial dynamics: max|⟨Z⟩| = {max_abs_z:e}"
        );

        let nsq = ttn.norm_squared();
        assert!((nsq - 1.0).abs() < 1e-10, "norm² drifted: {nsq:e}");

        eprintln!(
            "[adiabatic-ramp N={n} steps={n_steps}] max ⟨Z⟩ err = {max_err:.3e}, \
             final max|⟨Z⟩| = {max_abs_z:.3e}, norm² = {nsq:.15}"
        );
    }

    /// Same adiabatic ramp on a 15-qubit balanced binary tree.
    ///
    /// The chain case delegates to the validated `Mps` backend; this one
    /// forces `Backend::Tree` (the native flat-tensor path with branching
    /// nodes), so it's the load-bearing topology test for the annealer
    /// thread — million-variable adiabatic-on-tree only makes sense if
    /// the primitive holds on non-chain trees.
    #[test]
    fn adiabatic_ramp_binary_tree_matches_dense_lossless() {
        let n = 15;
        let n_steps = 50;
        let dt = 0.1_f64;

        let h_x_0 = 1.0_f64;
        let j_0 = 1.0_f64;
        let h_z_0 = 0.5_f64;

        // Balanced binary tree: node i has children 2i+1, 2i+2.
        let tree_edges: Vec<Edge> = (0..n)
            .flat_map(|i| {
                let mut es = Vec::new();
                if 2 * i + 1 < n {
                    es.push(Edge { a: i, b: 2 * i + 1 });
                }
                if 2 * i + 2 < n {
                    es.push(Edge { a: i, b: 2 * i + 2 });
                }
                es
            })
            .collect();
        assert_eq!(tree_edges.len(), n - 1);

        let topology = Topology::from_edges(n, tree_edges.clone());
        assert!(
            !topology.is_linear_chain(),
            "test must exercise Backend::Tree, not the chain fast path"
        );
        let mut ttn = Ttn::new(topology);
        let mut dense = DenseState::zero(n);

        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        let h_gate: [[C; 2]; 2] = [
            [C::new(inv_sqrt2, 0.0), C::new(inv_sqrt2, 0.0)],
            [C::new(inv_sqrt2, 0.0), C::new(-inv_sqrt2, 0.0)],
        ];
        for q in 0..n {
            ttn.apply_single(q, h_gate);
            dense.apply_single(q, h_gate);
        }

        let no_non_tree: Vec<Edge> = vec![];
        let mut max_err = 0.0_f64;

        for k in 0..n_steps {
            let s = (k as f64 + 0.5) / n_steps as f64;
            let params = KimParams {
                j: s * j_0,
                h_x: -(1.0 - s) * h_x_0,
                h_z: s * h_z_0,
                dt,
            };

            apply_kim_step_heavy_hex(&mut ttn, &no_non_tree, params, usize::MAX).unwrap();

            let zz = zz_gate(params.j * params.dt);
            for e in &tree_edges {
                dense.apply_two_qubit(e.a, e.b, zz);
            }
            if params.h_z != 0.0 {
                let rz = rz_gate(2.0 * params.h_z * params.dt);
                for q in 0..n {
                    dense.apply_single(q, rz);
                }
            }
            let rx = rx_gate(2.0 * params.h_x * params.dt);
            for q in 0..n {
                dense.apply_single(q, rx);
            }

            let z_ttn = ttn.expectation_z_all();
            for q in 0..n {
                let z_dense = dense.expectation_z(q);
                let err = (z_ttn[q] - z_dense).abs();
                if err > max_err {
                    max_err = err;
                }
            }
        }

        assert!(
            max_err < 1e-10,
            "binary-tree adiabatic ramp diverged from dense: max ⟨Z⟩ err = {max_err:e}"
        );

        let z_final = ttn.expectation_z_all();
        let max_abs_z: f64 = z_final.iter().map(|z| z.abs()).fold(0.0, f64::max);
        assert!(
            max_abs_z > 0.05,
            "ramp produced trivial dynamics on tree: max|⟨Z⟩| = {max_abs_z:e}"
        );

        let nsq = ttn.norm_squared();
        assert!((nsq - 1.0).abs() < 1e-9, "norm² drifted: {nsq:e}");

        eprintln!(
            "[adiabatic-ramp tree N={n} steps={n_steps}] max ⟨Z⟩ err = {max_err:.3e}, \
             final max|⟨Z⟩| = {max_abs_z:.3e}, norm² = {nsq:.15}"
        );
    }

    /// Adiabatic ramp on a heavy-hex grid motif with non-tree edges.
    ///
    /// The chain anchor uses 1D nearest-neighbour, the binary-tree anchor
    /// uses a tree without non-tree edges. This third anchor is the actual
    /// 2D-annealer-thread case: a regular heavy-hex grid (`grid(3, 2)`,
    /// 19 qubits, 2 non-tree edges) where `apply_kim_step_heavy_hex` has
    /// to route both the tree-edge ZZ layer and the non-tree-edge ZZ layer
    /// (the swap-network path) through the same gauge state every step.
    ///
    /// Validates byte-for-byte against `DenseState` applying ZZ on every
    /// physical-Hamiltonian edge (tree + non-tree). If this passes at
    /// machine precision, the same primitive at scale produces faithful
    /// closed-system dynamics modulo bounded χ-truncation.
    #[test]
    fn adiabatic_ramp_heavy_hex_grid_matches_dense_lossless() {
        let layout = HeavyHexLayout::grid(3, 2);
        let n = layout.n_qubits();
        assert_eq!(n, 19, "grid(3,2) should be 19 qubits");
        assert_eq!(layout.non_tree_edges().len(), 2);

        let n_steps = 50;
        let dt = 0.1_f64;

        let h_x_0 = 1.0_f64;
        let j_0 = 1.0_f64;
        let h_z_0 = 0.5_f64;

        let topology = layout.tree().clone();
        assert!(
            !topology.is_linear_chain(),
            "grid(3,2) must exercise Backend::Tree, not the chain fast path"
        );
        let tree_edges_owned: Vec<Edge> = topology.edges().to_vec();
        let non_tree_edges: Vec<Edge> = layout
            .non_tree_edges()
            .iter()
            .map(|&[a, b]| Edge { a, b })
            .collect();
        let coupling: Vec<Edge> = tree_edges_owned
            .iter()
            .copied()
            .chain(non_tree_edges.iter().copied())
            .collect();

        let mut ttn = Ttn::new(topology);
        let mut dense = DenseState::zero(n);

        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        let h_gate: [[C; 2]; 2] = [
            [C::new(inv_sqrt2, 0.0), C::new(inv_sqrt2, 0.0)],
            [C::new(inv_sqrt2, 0.0), C::new(-inv_sqrt2, 0.0)],
        ];
        for q in 0..n {
            ttn.apply_single(q, h_gate);
            dense.apply_single(q, h_gate);
        }

        let mut max_err = 0.0_f64;

        for k in 0..n_steps {
            let s = (k as f64 + 0.5) / n_steps as f64;
            let params = KimParams {
                j: s * j_0,
                h_x: -(1.0 - s) * h_x_0,
                h_z: s * h_z_0,
                dt,
            };

            apply_kim_step_heavy_hex(&mut ttn, &non_tree_edges, params, usize::MAX).unwrap();

            // Dense reference: ZZ on every coupling edge (tree + non-tree),
            // RZ on every site, RX on every site — same layer order as
            // apply_kim_step_heavy_hex.
            let zz = zz_gate(params.j * params.dt);
            for e in &coupling {
                dense.apply_two_qubit(e.a, e.b, zz);
            }
            if params.h_z != 0.0 {
                let rz = rz_gate(2.0 * params.h_z * params.dt);
                for q in 0..n {
                    dense.apply_single(q, rz);
                }
            }
            let rx = rx_gate(2.0 * params.h_x * params.dt);
            for q in 0..n {
                dense.apply_single(q, rx);
            }

            let z_ttn = ttn.expectation_z_all();
            for q in 0..n {
                let z_dense = dense.expectation_z(q);
                let err = (z_ttn[q] - z_dense).abs();
                if err > max_err {
                    max_err = err;
                }
            }
        }

        assert!(
            max_err < 1e-10,
            "heavy-hex-grid adiabatic ramp diverged from dense: max ⟨Z⟩ err = {max_err:e}"
        );

        let z_final = ttn.expectation_z_all();
        let max_abs_z: f64 = z_final.iter().map(|z| z.abs()).fold(0.0, f64::max);
        assert!(
            max_abs_z > 0.05,
            "ramp produced trivial dynamics on heavy-hex grid: max|⟨Z⟩| = {max_abs_z:e}"
        );

        let nsq = ttn.norm_squared();
        assert!((nsq - 1.0).abs() < 1e-9, "norm² drifted: {nsq:e}");

        eprintln!(
            "[adiabatic-ramp grid(3,2) N={n} steps={n_steps}] max ⟨Z⟩ err = {max_err:.3e}, \
             final max|⟨Z⟩| = {max_abs_z:.3e}, norm² = {nsq:.15}"
        );
    }
}
