//! Integration-level smoke test for the Tindall N = 127 heavy-hex KIM
//! runner (design-doc D.5.1).
//!
//! This file exercises only the public API of the `huoma` crate — no
//! `#[cfg(test)]` helpers, no private types — and runs the full
//! Tindall-parameters depth-5 Floquet circuit on the real Eagle 127q
//! spanning tree, confirming that the public entry points compose into
//! a working pipeline at the target size.
//!
//! The stronger Huoma-only correctness checks (bounds on every
//! snapshot, initial |0…0⟩ sanity, discarded-weight accounting) live in
//! the unit tests inside `src/ttn/kim_heavy_hex.rs`; this integration
//! test just proves the public surface is sufficient for a downstream
//! user to reproduce the run without reaching into crate internals.
//!
//! The Tindall vs. published-numbers assertion and the ITensor
//! cross-reference comparison ship in follow-up PRs (D.5.3 / D.5.4).

use huoma::kicked_ising::KimParams;
use huoma::ttn::kim_heavy_hex::{run_kim_heavy_hex, rx_gate};
use huoma::ttn::topology::Edge;
use huoma::ttn::{HeavyHexLayout, Ttn};

/// Tindall et al., PRX Quantum 5, 010308 (2024) parameters:
/// `J = 1`, `h_x = 0.8`, `h_z = 0`, `dt = 0.5`.
fn tindall_params() -> KimParams {
    KimParams {
        j: 1.0,
        h_x: 0.8,
        h_z: 0.0,
        dt: 0.5,
    }
}

#[test]
fn tindall_eagle_127q_depth_5_runs_to_completion_via_public_api() {
    // 1. Build the Eagle 127q heavy-hex layout from the public constructor.
    let layout = HeavyHexLayout::ibm_eagle_127();
    assert_eq!(layout.n_qubits(), 127);
    assert_eq!(layout.tree().n_edges(), 126);
    assert_eq!(layout.non_tree_edges().len(), 18);

    // 2. Convert the layout's serde-friendly `[a, b]` non-tree list to
    //    the `Edge` type the TTN gate API consumes.
    let non_tree: Vec<Edge> = layout
        .non_tree_edges()
        .iter()
        .map(|[a, b]| Edge { a: *a, b: *b })
        .collect();

    // 3. Initialise the TTN from the spanning tree. `Ttn::new` on the
    //    127q heavy-hex produces a product |0…0⟩ state in canonical form.
    let mut ttn = Ttn::new(layout.tree().clone());
    assert_eq!(ttn.n_qubits(), 127);

    // 4. Break the product state so the ZZ layer has non-trivial work —
    //    otherwise |0…0⟩ is an exact ZZ eigenstate and the test is
    //    vacuous. `Rx(π/2)` on every qubit rotates to |+⟩^⊗127.
    for q in 0..127 {
        ttn.apply_single(q, rx_gate(std::f64::consts::FRAC_PI_2));
    }

    // 5. Run 5 Floquet steps at Tindall's parameters, χ = 8 cap.
    let params = tindall_params();
    let max_bond = 8;
    let n_steps = 5;
    let history = run_kim_heavy_hex(&mut ttn, &non_tree, params, max_bond, n_steps)
        .expect("run_kim_heavy_hex must succeed on Eagle 127q");

    // 6. Verify the shape of the returned history: `n_steps + 1` snapshots
    //    (one before any gates, then one per step) of 127 per-qubit
    //    ⟨Z⟩ values each.
    assert_eq!(history.len(), n_steps + 1);
    for snap in &history {
        assert_eq!(snap.len(), 127);
        for &z in snap {
            assert!(z.is_finite(), "⟨Z⟩ must be finite");
            assert!((-1.0..=1.0).contains(&z), "⟨Z⟩ must be Hermitian-bounded");
        }
    }

    // 7. Discarded weight is finite and non-negative (truncation
    //    accounting still sane after 5 full Floquet layers at χ = 8).
    let disc = ttn.total_discarded_weight();
    assert!(disc.is_finite() && disc >= 0.0);
}
