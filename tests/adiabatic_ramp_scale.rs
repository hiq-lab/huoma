//! Scale smoke test for the closed-system adiabatic ramp primitive on a
//! 1D chain via the validated `Mps` fast path
//! (`kicked_ising::apply_kim_step` → `apply_two_qubit_layer_parallel`).
//!
//! Annealer-thread headline: a million-variable adiabatic sweep
//! `H(s) = (1−s)H_X + s H_problem` simulated end-to-end in closed-system
//! unitarity at bounded χ — three orders of magnitude beyond what current
//! and near-term annealer hardware can embed.
//!
//! The ramp primitive is validated against `DenseState` at small N in
//! `src/ttn/kim_heavy_hex.rs::adiabatic_ramp_*_matches_dense_lossless`.
//! `apply_kim_step` is independently validated against dense in
//! `tests/kim_validation.rs` — both paths apply identical gate sequences,
//! so the small-N agreement transfers to this driver.
//!
//! Note on norm tracking: at heavy truncation (χ ≪ Schmidt rank,
//! many gates) the post-truncation MPS is not in a canonical form, and
//! the standalone `norm_squared` calculation accumulates FP drift in
//! the env-contraction. ⟨Z⟩ values remain well-conditioned because
//! `expectation_z` divides numerator-env by denominator-env, but
//! `norm_squared` itself is informational here, not load-bearing. A
//! periodic recanonicalization sweep would fix the drift; out of scope
//! for this annealer-thread headline.

use huoma::kicked_ising::{apply_kim_step, KimParams};
use huoma::mps::{self, Mps};

fn run_adiabatic_chain(n: usize, n_steps: usize, max_bond: usize, dt: f64, tag: &str) {
    let h_x_0 = 1.0_f64;
    let j_0 = 1.0_f64;
    let h_z_0 = 0.1_f64;

    let start = std::time::Instant::now();

    let mut state = Mps::new(n);
    let t_init = start.elapsed();
    eprintln!("[{tag}] Mps::new ({n}): {t_init:.2?}");

    state.apply_single_all(mps::h());
    let t_prep = start.elapsed();
    eprintln!("[{tag}] |+⟩^⊗N preparation: {:?}", t_prep - t_init);

    let chi_per_bond: Vec<usize> = vec![max_bond; n.saturating_sub(1)];

    let ramp_start = std::time::Instant::now();
    for k in 0..n_steps {
        let s = (k as f64 + 0.5) / n_steps as f64;
        let params = KimParams {
            j: s * j_0,
            h_x: -(1.0 - s) * h_x_0,
            h_z: s * h_z_0,
            dt,
        };
        apply_kim_step(&mut state, params, &chi_per_bond).unwrap();
    }
    let t_ramp = ramp_start.elapsed();
    eprintln!(
        "[{tag}] {n_steps} ramp steps at χ={max_bond}: {t_ramp:.2?} ({:.2?}/step)",
        t_ramp / n_steps as u32
    );

    let measure_start = std::time::Instant::now();
    let z = state.expectation_z_all();
    let t_measure = measure_start.elapsed();
    assert_eq!(z.len(), n);

    let max_abs_z: f64 = z.iter().map(|x| x.abs()).fold(0.0, f64::max);
    let mean_abs_z: f64 = z.iter().map(|x| x.abs()).sum::<f64>() / n as f64;
    let mean_z: f64 = z.iter().sum::<f64>() / n as f64;

    for (q, &zq) in z.iter().enumerate() {
        assert!(
            zq.is_finite() && (-1.0..=1.0).contains(&zq),
            "q={q}: ⟨Z⟩ = {zq}"
        );
    }
    assert!(
        max_abs_z > 1e-3,
        "ramp produced trivial dynamics: max|⟨Z⟩| = {max_abs_z:e}"
    );

    let nsq_start = std::time::Instant::now();
    let nsq = state.norm_squared();
    let t_norm = nsq_start.elapsed();
    let disc = state.total_discarded_weight();
    let total = start.elapsed();

    eprintln!(
        "[{tag}] expectation_z all ({n}): {t_measure:.2?}, \
         max|⟨Z⟩|={max_abs_z:.3}, mean|⟨Z⟩|={mean_abs_z:.3}, mean⟨Z⟩={mean_z:+.3}"
    );
    eprintln!(
        "\n[{tag}] DONE: {n} qubits, χ={max_bond}, {n_steps} steps, \
         total = {total:.2?}, norm² = {nsq:.6} (in {t_norm:.2?}, informational), \
         discarded = {disc:.4e}\n"
    );
}

/// χ-scaling calibration spike — confirms `apply_kim_step` per-bond cost
/// scales as χ³ on this hardware, so we can extrapolate to scale.
#[test]
#[ignore = "calibration spike, ignored to keep CI lean"]
fn adiabatic_ramp_chi_scaling_spike() {
    for &chi in &[8, 16, 32] {
        let n = 1000;
        let n_steps = 10;
        let mut state = Mps::new(n);
        state.apply_single_all(mps::h());
        let chi_per_bond: Vec<usize> = vec![chi; n - 1];
        let t0 = std::time::Instant::now();
        for k in 0..n_steps {
            let s = (k as f64 + 0.5) / n_steps as f64;
            apply_kim_step(
                &mut state,
                KimParams {
                    j: s,
                    h_x: -(1.0 - s),
                    h_z: 0.1 * s,
                    dt: 0.1,
                },
                &chi_per_bond,
            )
            .unwrap();
        }
        let t = t0.elapsed();
        eprintln!(
            "[spike] N={n} χ={chi} steps={n_steps}: {t:.2?} ({:.2?}/step)",
            t / n_steps as u32
        );
    }
}

/// Calibration run at N = 10K — exercises the ramp at nontrivial scale.
#[test]
#[ignore = "10K calibration, ~50 seconds"]
fn adiabatic_ramp_10k_qubits_chain_completes() {
    run_adiabatic_chain(10_000, 50, 16, 0.1, "adiabatic 10K");
}

/// 100K intermediate run — useful midpoint to verify the linear-in-N
/// extrapolation before the 1M headline.
#[test]
#[ignore = "100K adiabatic ramp, multi-minute"]
fn adiabatic_ramp_100k_qubits_chain_completes() {
    run_adiabatic_chain(100_000, 50, 8, 0.1, "adiabatic 100K");
}

/// One-million-qubit closed-system adiabatic ramp — annealer-thread
/// headline. Three orders of magnitude beyond near-term annealer
/// hardware's logical capacity.
#[test]
#[ignore = "1M-qubit adiabatic ramp, multi-minute"]
fn adiabatic_ramp_1m_qubits_chain_completes() {
    run_adiabatic_chain(1_000_000, 50, 8, 0.1, "adiabatic 1M");
}
