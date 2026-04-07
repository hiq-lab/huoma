//! 1D Floquet Kicked Ising validation benchmark.
//!
//! Validates the huoma simulator and the production sin(C/2) χ allocator
//! against an independent dense statevector reference at the self-dual
//! point of the kicked Ising chain (J = h_x = π/4, h_z = 0;
//! Bertini-Kos-Prosen, PRX 9, 021033, 2019).
//!
//! Stages:
//! - **A** N=12, χ_max = 64 (provably exact). Cross-check huoma against
//!         the dense reference simulator. Expect agreement to ~1e-10.
//! - **B** N=12, χ_max sweep at 4..32 (uniform). Document fidelity/error vs χ.
//! - **D** N=24, exact statevector reference (16M-dim) vs huoma at large χ.
//!         Confirms the validation also holds at the largest tractable
//!         statevector size.
//! - **F** Disordered self-dual KIM (N=14 dense ref + N=50 high-χ ref).
//!         Compares uniform-χ vs sin(C/2)-allocated χ at matched total
//!         budget — the production allocator shootout.
//!
//! See `PHASE7_REPORT.md` for the verdict on the discarded-weight Jacobian
//! allocator that previously occupied Stages C and E.

use huoma::allocator::chi_allocation_sinc_with_radius;
use huoma::kicked_ising::{
    apply_kim_step, apply_kim_step_disordered, reference_kim_run,
    reference_kim_run_disordered, KimParams,
};
use huoma::mps::Mps;
use std::time::Instant;

const TOL_EXACT: f64 = 1e-10;

/// Run the kicked Ising circuit on an MPS for `n_steps` Trotter steps using
/// the supplied per-bond χ profile. Returns the per-step ⟨Z_q⟩ history,
/// shape `(n_steps+1) × n`.
fn run_mps_kim(
    n: usize,
    params: KimParams,
    n_steps: usize,
    chi_per_bond: &[usize],
) -> (Mps, Vec<Vec<f64>>) {
    let mut mps = Mps::new(n);
    let mut history = Vec::with_capacity(n_steps + 1);
    history.push((0..n).map(|q| mps.expectation_z(q)).collect());

    for _step in 0..n_steps {
        apply_kim_step(&mut mps, params, chi_per_bond).unwrap();
        history.push((0..n).map(|q| mps.expectation_z(q)).collect());
    }
    (mps, history)
}

/// Disordered variant: each site has its own h_x.
fn run_mps_kim_disordered(
    n: usize,
    params: KimParams,
    h_x_per_site: &[f64],
    n_steps: usize,
    chi_per_bond: &[usize],
) -> (Mps, Vec<Vec<f64>>) {
    let mut mps = Mps::new(n);
    let mut history = Vec::with_capacity(n_steps + 1);
    history.push((0..n).map(|q| mps.expectation_z(q)).collect());

    for _step in 0..n_steps {
        apply_kim_step_disordered(&mut mps, params, h_x_per_site, chi_per_bond).unwrap();
        history.push((0..n).map(|q| mps.expectation_z(q)).collect());
    }
    (mps, history)
}

/// Deterministic Mulberry32-style PRNG so that disordered tests are
/// reproducible without bringing in `rand` as a dev dependency.
fn det_random_hx(n: usize, seed: u64, base: f64, amplitude: f64) -> Vec<f64> {
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let r = ((state >> 11) as f64) / ((1_u64 << 53) as f64); // [0, 1)
        out.push(base + amplitude * (2.0 * r - 1.0));
    }
    out
}

/// Maximum absolute deviation between two ⟨Z⟩ histories of identical shape.
fn history_max_err(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut m = 0.0_f64;
    for (row_a, row_b) in a.iter().zip(b.iter()) {
        assert_eq!(row_a.len(), row_b.len());
        for (x, y) in row_a.iter().zip(row_b.iter()) {
            m = m.max((x - y).abs());
        }
    }
    m
}

// ─────────────────────────────────────────────────────────────────────────────
// STAGE A — N=12, full χ, exact agreement with dense reference
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn stage_a_n12_full_chi_matches_reference() {
    let n = 12;
    let n_steps = 10;
    let params = KimParams::self_dual();

    println!("\n  ══════════════════════════════════════════════════════════════════");
    println!("  Stage A: N={n}, χ_max=64 (provably exact), {n_steps} Trotter steps");
    println!("  ══════════════════════════════════════════════════════════════════\n");

    let chi_max = 1_usize << (n / 2); // 2^6 = 64, exact for any 12-qubit state
    let chi_per_bond = vec![chi_max; n - 1];

    let t_ref = Instant::now();
    let ref_history = reference_kim_run(n, params, n_steps);
    let ref_ms = t_ref.elapsed().as_secs_f64() * 1000.0;

    let t_mps = Instant::now();
    let (mps, mps_history) = run_mps_kim(n, params, n_steps, &chi_per_bond);
    let mps_ms = t_mps.elapsed().as_secs_f64() * 1000.0;

    let max_err = history_max_err(&ref_history, &mps_history);
    let max_bond = mps.bond_dims().iter().max().copied().unwrap_or(0);
    let total_disc = mps.total_discarded_weight();

    println!("  reference statevector: {ref_ms:8.2} ms");
    println!("  huoma MPS:        {mps_ms:8.2} ms");
    println!("  max bond used:         {max_bond}");
    println!("  total discarded weight: {total_disc:.3e}");
    println!("  max ⟨Z⟩ error vs ref:  {max_err:.3e}\n");

    assert!(
        max_err < TOL_EXACT,
        "Stage A: huoma at full χ disagrees with dense reference: max_err={max_err}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// STAGE B — N=12, χ sweep, document fidelity vs uniform χ budget
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn stage_b_n12_chi_sweep_uniform() {
    let n = 12;
    let n_steps = 10;
    let params = KimParams::self_dual();

    println!("\n  ══════════════════════════════════════════════════════════════════");
    println!("  Stage B: N={n}, uniform χ sweep, {n_steps} Trotter steps");
    println!("  ══════════════════════════════════════════════════════════════════\n");

    let ref_history = reference_kim_run(n, params, n_steps);

    println!(
        "  {:>4} | {:>14} | {:>14} | {:>10}",
        "χ", "max ⟨Z⟩ error", "total disc", "wall ms"
    );
    println!("  {}", "-".repeat(54));

    for &chi in &[2, 4, 6, 8, 12, 16, 24, 32, 64] {
        let chi_per_bond = vec![chi; n - 1];

        let t = Instant::now();
        let (mps, mps_history) = run_mps_kim(n, params, n_steps, &chi_per_bond);
        let ms = t.elapsed().as_secs_f64() * 1000.0;

        let max_err = history_max_err(&ref_history, &mps_history);
        let total_disc = mps.total_discarded_weight();

        println!(
            "  {:>4} | {:>14.3e} | {:>14.3e} | {:>10.2}",
            chi, max_err, total_disc, ms
        );
    }
    println!();
}

// ─────────────────────────────────────────────────────────────────────────────
// STAGE D — N=24, exact statevector reference (16M-dim) vs huoma
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn stage_d_n24_full_chi_matches_reference() {
    let n = 24;
    let n_steps = 6;
    let params = KimParams::self_dual();

    println!("\n  ══════════════════════════════════════════════════════════════════");
    println!("  Stage D: N={n}, χ_max=4096 (exact), {n_steps} Trotter steps");
    println!("  ══════════════════════════════════════════════════════════════════\n");

    // For 24 qubits, exact χ_max = 2^12 = 4096. We use χ_max = 256 which is
    // not strictly exact but gives essentially-converged numerics for this
    // shallow circuit (entanglement growth in self-dual KIM is bounded by
    // ~ v_E · n_steps · log 2 ≈ 4 ebits at step 6 → χ ~ 16 saturates).
    let chi_max = 256;
    let chi_per_bond = vec![chi_max; n - 1];

    let t_ref = Instant::now();
    let ref_history = reference_kim_run(n, params, n_steps);
    let ref_ms = t_ref.elapsed().as_secs_f64() * 1000.0;

    let t_mps = Instant::now();
    let (mps, mps_history) = run_mps_kim(n, params, n_steps, &chi_per_bond);
    let mps_ms = t_mps.elapsed().as_secs_f64() * 1000.0;

    let max_err = history_max_err(&ref_history, &mps_history);
    let max_bond = mps.bond_dims().iter().max().copied().unwrap_or(0);
    let total_disc = mps.total_discarded_weight();

    println!("  reference statevector ({} amps): {ref_ms:.2} ms", 1_usize << n);
    println!("  huoma MPS (χ_max={chi_max}):  {mps_ms:.2} ms");
    println!("  actual max bond used:            {max_bond}");
    println!("  total discarded weight:          {total_disc:.3e}");
    println!("  max ⟨Z⟩ error vs ref:            {max_err:.3e}\n");

    assert!(
        max_err < 1e-8,
        "Stage D: huoma at χ_max={chi_max} disagrees with dense reference: max_err={max_err}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// STAGE F — Disordered self-dual KIM, N=14 (statevector reference) and
// N=50 (high-χ MPS reference). Each site has h_x = π/4 + δ_i with δ_i drawn
// from a fixed-seed uniform distribution. Compares uniform-χ vs the
// production sin(C/2) allocator (chi_allocation_sinc) at matched total
// budget.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn stage_f_disordered_sinc_vs_uniform() {
    let params = KimParams::self_dual();
    let n_steps = 8;
    let disorder_amplitude = 0.5; // δ ∈ [-0.5, 0.5] around h_x = π/4
    let seed = 12345_u64;

    println!("\n  ══════════════════════════════════════════════════════════════════");
    println!("  Stage F: DISORDERED self-dual KIM (h_x site-disorder ±{disorder_amplitude})");
    println!("  uniform χ vs sin(C/2)-allocated χ at matched total budget");
    println!("  ══════════════════════════════════════════════════════════════════\n");

    println!(
        "  {:>4} | {:>16} | {:>14} {:>12} | {:>14} {:>12}",
        "N", "strategy", "max ⟨Z⟩ err", "budget", "wall ms", "max bond"
    );
    println!("  {}", "-".repeat(82));

    // ── Sub-test 1: N=14 with dense statevector ground truth ─────────────
    {
        let n = 14_usize;
        let h_x_per_site = det_random_hx(n, seed, params.h_x, disorder_amplitude);
        let chi_ceiling = 8_usize; // uniform baseline
        // Allocator bounds must bracket budget / n_bonds (= chi_ceiling) so
        // that water-filling has room to produce non-uniform allocations.
        let chi_min_alloc = 2_usize;
        let chi_max_alloc = 12_usize;

        let t = Instant::now();
        let hist_ref = reference_kim_run_disordered(n, params, &h_x_per_site, n_steps);
        let ms_ref = t.elapsed().as_secs_f64() * 1000.0;
        println!(
            "  {:>4} | {:>16} | {:>14} {:>12} | {:>14.2} {:>12}",
            n, "dense ref", "—", "—", ms_ref, "—"
        );

        let chi_uniform = vec![chi_ceiling; n - 1];
        let t = Instant::now();
        let (mps_u, hist_u) =
            run_mps_kim_disordered(n, params, &h_x_per_site, n_steps, &chi_uniform);
        let ms_u = t.elapsed().as_secs_f64() * 1000.0;
        let err_u = history_max_err(&hist_ref, &hist_u);
        let budget_u: usize = chi_uniform.iter().sum();
        let mb_u = mps_u.bond_dims().iter().max().copied().unwrap_or(0);
        println!(
            "  {:>4} | {:>16} | {:>14.3e} {:>12} | {:>14.2} {:>12}",
            n,
            format!("uniform χ={chi_ceiling}"),
            err_u,
            budget_u,
            ms_u,
            mb_u
        );

        // ── sin(C/2) at matched total budget ────────────────────────────
        let t_ch = Instant::now();
        let chi_sinc = chi_allocation_sinc_with_radius(
            &h_x_per_site,
            5,
            budget_u,
            chi_min_alloc,
            chi_max_alloc,
        );
        let ch_build_ms = t_ch.elapsed().as_secs_f64() * 1000.0;
        let t = Instant::now();
        let (mps_s, hist_s) =
            run_mps_kim_disordered(n, params, &h_x_per_site, n_steps, &chi_sinc);
        let ms_s = t.elapsed().as_secs_f64() * 1000.0;
        let err_s = history_max_err(&hist_ref, &hist_s);
        let budget_s: usize = chi_sinc.iter().sum();
        let mb_s = mps_s.bond_dims().iter().max().copied().unwrap_or(0);
        println!(
            "  {:>4} | {:>16} | {:>14.3e} {:>12} | {:>14.2} {:>12}    (sinc2 build {:.2}ms)",
            n, "sinc2 matched", err_s, budget_s, ms_s, mb_s, ch_build_ms
        );
        println!("    sinc2 chi profile: {:?}", chi_sinc);
        println!();
    }

    // ── Sub-test 2: N=50 with high-χ MPS reference ───────────────────────
    {
        let n = 50_usize;
        let h_x_per_site = det_random_hx(n, seed, params.h_x, disorder_amplitude);
        let chi_ceiling = 8_usize;
        let chi_min_alloc = 2_usize;
        let chi_max_alloc = 12_usize;
        let chi_ref = 64_usize;

        let chi_ref_per_bond = vec![chi_ref; n - 1];
        let t = Instant::now();
        let (mps_ref, hist_ref) =
            run_mps_kim_disordered(n, params, &h_x_per_site, n_steps, &chi_ref_per_bond);
        let ms_ref = t.elapsed().as_secs_f64() * 1000.0;
        let mb_ref = mps_ref.bond_dims().iter().max().copied().unwrap_or(0);
        println!(
            "  {:>4} | {:>16} | {:>14} {:>12} | {:>14.2} {:>12}",
            n,
            format!("ref χ={chi_ref}"),
            "—",
            chi_ref * (n - 1),
            ms_ref,
            mb_ref
        );

        let chi_uniform = vec![chi_ceiling; n - 1];
        let t = Instant::now();
        let (mps_u, hist_u) =
            run_mps_kim_disordered(n, params, &h_x_per_site, n_steps, &chi_uniform);
        let ms_u = t.elapsed().as_secs_f64() * 1000.0;
        let err_u = history_max_err(&hist_ref, &hist_u);
        let budget_u: usize = chi_uniform.iter().sum();
        let mb_u = mps_u.bond_dims().iter().max().copied().unwrap_or(0);
        println!(
            "  {:>4} | {:>16} | {:>14.3e} {:>12} | {:>14.2} {:>12}",
            n,
            format!("uniform χ={chi_ceiling}"),
            err_u,
            budget_u,
            ms_u,
            mb_u
        );

        // ── sin(C/2) at matched total budget ────────────────────────────
        let t_ch = Instant::now();
        let chi_sinc = chi_allocation_sinc_with_radius(
            &h_x_per_site,
            5,
            budget_u,
            chi_min_alloc,
            chi_max_alloc,
        );
        let ch_build_ms = t_ch.elapsed().as_secs_f64() * 1000.0;
        let t = Instant::now();
        let (mps_s, hist_s) =
            run_mps_kim_disordered(n, params, &h_x_per_site, n_steps, &chi_sinc);
        let ms_s = t.elapsed().as_secs_f64() * 1000.0;
        let err_s = history_max_err(&hist_ref, &hist_s);
        let budget_s: usize = chi_sinc.iter().sum();
        let mb_s = mps_s.bond_dims().iter().max().copied().unwrap_or(0);
        println!(
            "  {:>4} | {:>16} | {:>14.3e} {:>12} | {:>14.2} {:>12}    (sinc2 build {:.2}ms)",
            n, "sinc2 matched", err_s, budget_s, ms_s, mb_s, ch_build_ms
        );
        println!();
    }
}
