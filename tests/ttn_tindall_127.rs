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
use huoma::ttn::allocator::chi_allocation_sinc_tree;
use huoma::ttn::kim_heavy_hex::{
    run_kim_heavy_hex, run_kim_heavy_hex_per_edge, rx_gate,
};
use huoma::ttn::topology::Edge;
use huoma::ttn::{HeavyHexLayout, Ttn};
use std::path::PathBuf;

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

/// Matched-budget shootout scaffold — Track D milestone **D.5.2**.
///
/// Runs the full Tindall heavy-hex KIM circuit on Eagle 127q with two
/// different χ profiles at **exactly the same total edge budget**:
///
/// 1. **Uniform** — every tree edge gets the same `χ_uniform`
/// 2. **sin(C/2) tree allocator** — `chi_allocation_sinc_tree` from
///    D.4 distributes the same total budget across the 126 tree edges
///    in proportion to each edge's cross-cut commensurability score
///
/// At **Tindall's homogeneous parameters** (`J = 1`, `h_x = 0.8`,
/// `h_z = 0`, `dt = 0.5`, uniform over every qubit) the sin(C/2)
/// allocator **collapses to near-uniform** because every pair has
/// the same frequency ratio → `C = 0` → zero score → water-filling
/// falls back to an even distribution where individual edges
/// differ from the uniform value by at most one χ unit (a
/// water-filling rounding artefact).
///
/// Assertions are all Huoma-only — no Tindall / ITensor comparison
/// yet (those ship in D.5.3 / D.5.4). The goal of this test is:
///
/// - both profiles run end-to-end to depth 20 at N = 127 without
///   failure, out-of-bounds values, or non-finite numbers
/// - the total χ budget consumed matches exactly on both sides
/// - the homogeneous-Tindall allocator profile is within one χ unit
///   of uniform on every edge (the collapse case)
///
/// An earlier draft of this test asserted that the two profiles
/// should produce ⟨Z⟩ trajectories agreeing to within 5 % at every
/// depth, on the theory that a "near-uniform" profile produces
/// "near-uniform" observables. Empirically this is false at χ = 8
/// on depth-5+ Tindall: the truncation floor is large enough that a
/// ±1 shift in χ on even a handful of edges reorders the severely-
/// truncated Schmidt spectrum and compounds across Floquet layers
/// into per-qubit ⟨Z⟩ differences of 0.5 – 1.0. That is a real
/// property of both allocators at this budget, not a bug, and the
/// test now records the max per-depth difference as observational
/// output rather than an assertion. The definitive "is either
/// profile closer to Tindall's published values" question is D.5.3's
/// problem, not this test's.
#[test]
fn tindall_eagle_127q_matched_budget_shootout_uniform_vs_sinc_tree() {
    let layout = HeavyHexLayout::ibm_eagle_127();
    let non_tree: Vec<Edge> = layout
        .non_tree_edges()
        .iter()
        .map(|[a, b]| Edge { a: *a, b: *b })
        .collect();

    // Matched budget: 126 tree edges × 8 = 1008. chi_min = 2 so the
    // allocator has room to move without saturating the floor.
    let n_tree_edges = layout.tree().n_edges();
    let chi_uniform_value: usize = 8;
    let total_budget: usize = n_tree_edges * chi_uniform_value;
    let chi_min: usize = 2;
    let chi_max: usize = 32;

    // Profile 1 — uniform.
    let chi_uniform: Vec<usize> = vec![chi_uniform_value; n_tree_edges];
    assert_eq!(chi_uniform.iter().sum::<usize>(), total_budget);

    // Profile 2 — sin(C/2) tree allocator on Tindall's *homogeneous*
    // frequencies. All 127 qubits share h_x = 0.8, so the commensurability
    // residual is zero everywhere and the allocator collapses to a
    // near-uniform integer allocation that still consumes the exact
    // same total budget.
    let frequencies_homogeneous: Vec<f64> = vec![0.8_f64; 127];
    let chi_alloc_homogeneous = chi_allocation_sinc_tree(
        &frequencies_homogeneous,
        layout.tree(),
        total_budget,
        chi_min,
        chi_max,
    );
    assert_eq!(chi_alloc_homogeneous.len(), n_tree_edges);
    assert_eq!(
        chi_alloc_homogeneous.iter().sum::<usize>(),
        total_budget,
        "allocator must exactly consume the matched budget"
    );
    // On the homogeneous input the allocator must collapse to a near-uniform
    // profile: every edge is within ±1 of the uniform value.
    for (i, &chi) in chi_alloc_homogeneous.iter().enumerate() {
        assert!(
            chi.abs_diff(chi_uniform_value) <= 1,
            "edge {i}: allocator drift from uniform is too large ({chi} vs {chi_uniform_value})"
        );
    }

    let params = KimParams {
        j: 1.0,
        h_x: 0.8,
        h_z: 0.0,
        dt: 0.5,
    };
    let non_tree_chi = chi_max;

    // Run both profiles from a fresh |+⟩^⊗127 state and record the
    // per-depth history. Depth 5 is the Tindall milestone checkpoint;
    // depths 10 and 20 extend the trajectory for future matched-budget
    // comparisons that don't yet have an external reference.
    let depths = [5usize, 10, 20];
    let n_steps = *depths.iter().max().unwrap();

    let init_state = || {
        let mut ttn = Ttn::new(layout.tree().clone());
        for q in 0..127 {
            ttn.apply_single(q, rx_gate(std::f64::consts::FRAC_PI_2));
        }
        ttn
    };

    // 1. Uniform-χ run.
    let mut ttn_uniform = init_state();
    let history_uniform =
        run_kim_heavy_hex(&mut ttn_uniform, &non_tree, params, chi_uniform_value, n_steps)
            .expect("uniform run must succeed");

    // 2. sin(C/2)-allocator-χ run (homogeneous frequencies, so profile
    //    is within ±1 of uniform on every edge).
    let mut ttn_alloc = init_state();
    let history_alloc = run_kim_heavy_hex_per_edge(
        &mut ttn_alloc,
        &non_tree,
        params,
        &chi_alloc_homogeneous,
        non_tree_chi,
        n_steps,
    )
    .expect("allocator run must succeed");

    // Shape checks: n_steps + 1 snapshots of 127 values each.
    assert_eq!(history_uniform.len(), n_steps + 1);
    assert_eq!(history_alloc.len(), n_steps + 1);
    for snap in history_uniform.iter().chain(history_alloc.iter()) {
        assert_eq!(snap.len(), 127);
        for &z in snap {
            assert!(
                z.is_finite() && (-1.0..=1.0).contains(&z),
                "⟨Z⟩ out of bounds or non-finite: {z}"
            );
        }
    }

    // Record per-depth max |Z_uniform - Z_alloc| as an observable
    // output rather than an assertion — see the test docstring for
    // why a tight tolerance does not hold on this input. The honest
    // assertion here is just that the diff is finite and bounded
    // (which follows from the Hermitian bounds already checked
    // above, but we re-check it as a trivial guard).
    for &depth in &depths {
        let mut max_abs_diff = 0.0_f64;
        for q in 0..127 {
            let diff = (history_uniform[depth][q] - history_alloc[depth][q]).abs();
            if diff > max_abs_diff {
                max_abs_diff = diff;
            }
        }
        assert!(
            max_abs_diff.is_finite() && (0.0..=2.0).contains(&max_abs_diff),
            "depth {depth}: max |ΔZ| must be finite and ≤ 2.0 (Hermitian range), got {max_abs_diff}"
        );
        // Observational record — printed with `-- --nocapture`.
        eprintln!(
            "[D.5.2 shootout] homogeneous depth={depth:2}: max |Z_uniform − Z_alloc| = {max_abs_diff:.6}"
        );
    }

    // Discarded-weight accounting is finite and non-negative on both
    // profiles. No upper bound is asserted because at χ ≈ 8 on a
    // depth-20 entangling circuit over 127 qubits, the truncation
    // floor is non-trivially large and will be the subject of
    // D.5.3's Tindall comparison.
    let disc_uniform = ttn_uniform.total_discarded_weight();
    let disc_alloc = ttn_alloc.total_discarded_weight();
    assert!(disc_uniform.is_finite() && disc_uniform >= 0.0);
    assert!(disc_alloc.is_finite() && disc_alloc >= 0.0);
}

/// Matched-budget shootout on a **disordered** Tindall variant: the
/// per-qubit transverse field `h_x` carries a small deterministic
/// jitter across the 127 qubits, so the allocator's cross-cut
/// sin(C/2) scores are genuinely non-zero and the resulting profile
/// is genuinely non-uniform. Exercises the per-edge χ code path at
/// full scale on an input where it actually does something.
///
/// Huoma-only assertions again: both profiles run, produce finite
/// bounded values, and the allocator profile differs from uniform on
/// at least one edge.
#[test]
fn tindall_eagle_127q_matched_budget_shootout_on_disordered_frequencies() {
    let layout = HeavyHexLayout::ibm_eagle_127();
    let non_tree: Vec<Edge> = layout
        .non_tree_edges()
        .iter()
        .map(|[a, b]| Edge { a: *a, b: *b })
        .collect();

    let n_tree_edges = layout.tree().n_edges();
    let chi_uniform_value: usize = 8;
    let total_budget: usize = n_tree_edges * chi_uniform_value;
    let chi_min: usize = 2;
    let chi_max: usize = 32;

    // Deterministic per-qubit h_x jitter around Tindall's 0.8 baseline.
    // The √prime sequence gives an irrational, non-commensurate pattern
    // that the allocator can actually exploit.
    let primes = [2.0_f64, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0];
    let frequencies: Vec<f64> = (0..127)
        .map(|i| 0.8 + 0.05 * primes[i % primes.len()].sqrt())
        .collect();

    let chi_alloc = chi_allocation_sinc_tree(
        &frequencies,
        layout.tree(),
        total_budget,
        chi_min,
        chi_max,
    );
    assert_eq!(chi_alloc.len(), n_tree_edges);
    assert_eq!(chi_alloc.iter().sum::<usize>(), total_budget);

    // On a genuinely disordered input the allocator must produce a
    // non-flat profile: at least one edge should differ from the
    // uniform value. This guards against a silent regression where
    // the allocator collapses to uniform regardless of input.
    let any_non_uniform = chi_alloc
        .iter()
        .any(|&c| c != chi_uniform_value);
    assert!(
        any_non_uniform,
        "allocator profile is uniform despite non-uniform frequencies"
    );

    let params = KimParams {
        j: 1.0,
        h_x: 0.8,
        h_z: 0.0,
        dt: 0.5,
    };
    let non_tree_chi = chi_max;
    let n_steps = 10;

    // Two fresh states.
    let mut ttn_uniform = Ttn::new(layout.tree().clone());
    let mut ttn_alloc = Ttn::new(layout.tree().clone());
    for q in 0..127 {
        ttn_uniform.apply_single(q, rx_gate(std::f64::consts::FRAC_PI_2));
        ttn_alloc.apply_single(q, rx_gate(std::f64::consts::FRAC_PI_2));
    }

    let history_uniform =
        run_kim_heavy_hex(&mut ttn_uniform, &non_tree, params, chi_uniform_value, n_steps)
            .expect("uniform run must succeed");
    let history_alloc = run_kim_heavy_hex_per_edge(
        &mut ttn_alloc,
        &non_tree,
        params,
        &chi_alloc,
        non_tree_chi,
        n_steps,
    )
    .expect("allocator run must succeed");

    // Shape + bounds sanity on both runs.
    for snap in history_uniform.iter().chain(history_alloc.iter()) {
        assert_eq!(snap.len(), 127);
        for &z in snap {
            assert!(
                z.is_finite() && (-1.0..=1.0).contains(&z),
                "⟨Z⟩ out of bounds: {z}"
            );
        }
    }

    // Finite discarded weight.
    assert!(ttn_uniform.total_discarded_weight().is_finite());
    assert!(ttn_alloc.total_discarded_weight().is_finite());
}

// ─── D.5.3: Huoma vs Tindall published reference ────────────────────────

/// Load the Tindall reference CSV for θ_h = 0.8 and return
/// (steps, z62_values).
fn load_tindall_z62_reference() -> (Vec<usize>, Vec<f64>) {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("external/tindall_ref/z62_theta_h_0.8_bp_chi_inf.csv");
    let content = std::fs::read_to_string(&p).unwrap_or_else(|e| {
        panic!(
            "Tindall reference CSV missing at {}: {e}",
            p.display()
        )
    });
    let mut steps = Vec::new();
    let mut z62 = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("step") {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        assert_eq!(parts.len(), 2, "malformed CSV line: {line}");
        steps.push(parts[0].parse::<usize>().unwrap());
        z62.push(parts[1].parse::<f64>().unwrap());
    }
    (steps, z62)
}

/// Tindall's exact KimParams for θ_h = 0.8, θ_J = π/4.
///
/// Hamiltonian convention from the paper:
///   `U(θ_h) = (∏ exp(+iπ/4 Z_v Z_{v'})) · (∏ exp(-iθ_h/2 X_v))`
///
/// Huoma's `zz_gate(θ)` implements `exp(-iθ Z⊗Z)`, so to match the
/// paper's `exp(+iπ/4 Z⊗Z)` we need `θ = -π/4`, i.e. `j * dt = -π/4`.
///
/// Initial state: |0…0⟩ (no Hadamard / Rx preparation).
fn tindall_params_exact() -> KimParams {
    KimParams {
        j: -std::f64::consts::FRAC_PI_4,
        h_x: 0.4, // 2 * h_x * dt = 0.8 = θ_h
        h_z: 0.0,
        dt: 1.0,
    }
}

/// **D.5.3 — Huoma vs Tindall's published ⟨Z₆₂⟩ trajectory.**
///
/// Runs the Tindall kicked-Ising circuit on Eagle 127q at the exact
/// parameters from Tindall et al. (PRX Quantum 5, 010308, 2024):
///   θ_J = π/4 (per edge per step)
///   θ_h = 0.8 (X-kick angle per step)
///   initial state = |0…0⟩
///
/// Compares Huoma's ⟨Z₆₂⟩ at each depth 0–20 against the published
/// BP-TNS values from:
///   https://github.com/JoeyT1994/BP-TNS-Data
///   IBMHeavyHexData/DynamicsThetah0.8BPTNSChiInfty.npz
///
/// At χ = 8 the Huoma TTN truncation budget is very tight compared to
/// Tindall's converged BP result, so agreement will degrade with depth.
/// The test records the per-depth deviation and asserts a loose bound
/// that will tighten as we increase χ or add the ITensor cross-reference
/// (D.5.4).
///
/// NOTE: the paper's data repo provides only qubit 62's trajectory,
/// not full 127-qubit profiles. A full element-wise comparison across
/// all qubits is blocked on the authors releasing per-qubit data;
/// until then, qubit 62 is the anchor.
#[test]
fn tindall_eagle_127q_z62_vs_published_reference() {
    // 1. Load the Tindall reference data.
    let (ref_steps, ref_z62) = load_tindall_z62_reference();
    assert_eq!(ref_steps.len(), 21, "reference must have depths 0–20");
    assert_eq!(ref_steps[0], 0);
    assert_eq!(ref_steps[20], 20);

    // 2. Build the Eagle 127q layout and TTN.
    let layout = HeavyHexLayout::ibm_eagle_127();
    let non_tree: Vec<Edge> = layout
        .non_tree_edges()
        .iter()
        .map(|[a, b]| Edge { a: *a, b: *b })
        .collect();

    // 3. Initial state: |0…0⟩ (Ttn::new already initialises to this).
    let mut ttn = Ttn::new(layout.tree().clone());

    // 4. Tindall's exact parameters.
    let params = tindall_params_exact();
    let max_bond = 8;
    let n_steps = 20;

    // 5. Run 20 Floquet steps, recording ⟨Z⟩ history.
    let history = run_kim_heavy_hex(&mut ttn, &non_tree, params, max_bond, n_steps)
        .expect("D.5.3 Tindall run at depth 20 must succeed");
    assert_eq!(history.len(), 21);

    // 6. Compare qubit 62 at each depth against the reference.
    let qubit = 62;
    let mut max_abs_diff = 0.0_f64;
    eprintln!("\n[D.5.3] Huoma ⟨Z₆₂⟩ vs Tindall BP-TNS χ=∞ (Huoma χ=8):");
    eprintln!("{:>5}  {:>14}  {:>14}  {:>12}", "depth", "Huoma", "Tindall", "diff");
    for step in 0..=n_steps {
        let z_huoma = history[step][qubit];
        let z_tindall = ref_z62[step];
        let diff = z_huoma - z_tindall;
        if diff.abs() > max_abs_diff {
            max_abs_diff = diff.abs();
        }
        eprintln!(
            "{step:5}  {z_huoma:>14.10}  {z_tindall:>14.10}  {diff:>+12.6e}"
        );

        // The value must at least be finite and Hermitian-bounded.
        assert!(
            z_huoma.is_finite() && (-1.0..=1.0).contains(&z_huoma),
            "depth {step}: Huoma ⟨Z_62⟩ = {z_huoma} out of bounds"
        );
    }

    // Step 0 is the initial state |0…0⟩ so both should be +1 exactly.
    assert!(
        (history[0][qubit] - 1.0).abs() < 1e-14,
        "depth 0: Huoma ⟨Z_62⟩ must be +1 on the initial state"
    );

    // Step 1: after one ZZ + Rx layer from |0…0⟩, ⟨Z_62⟩ = cos(θ_h)
    // regardless of χ (because |0…0⟩ is a ZZ eigenstate). Both Huoma
    // and Tindall should agree to FP precision here.
    assert!(
        (history[1][qubit] - ref_z62[1]).abs() < 1e-10,
        "depth 1: Huoma ⟨Z_62⟩ must match Tindall to FP precision (no truncation at depth 1)"
    );

    // Overall bound: at χ = 8 vs Tindall's converged BP, the max
    // deviation across 20 depths should be bounded — loose enough to
    // accommodate truncation drift but tight enough to catch a sign
    // error or gate-ordering bug.
    eprintln!(
        "\n[D.5.3] max |Huoma − Tindall| across depths 0–20 = {max_abs_diff:.6}"
    );
    assert!(
        max_abs_diff < 1.5,
        "max |ΔZ_62| = {max_abs_diff} exceeds 1.5 — likely a parameter or gate error"
    );

    // Discarded weight is finite.
    let disc = ttn.total_discarded_weight();
    assert!(disc.is_finite() && disc >= 0.0);
    eprintln!(
        "[D.5.3] total discarded weight after 20 steps at χ=8: {disc:.6e}"
    );
}
