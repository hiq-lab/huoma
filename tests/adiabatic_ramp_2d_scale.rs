//! Scale smoke test for the closed-system adiabatic ramp on a 2D
//! heavy-hex grid via the validated `apply_kim_step_heavy_hex` driver.
//!
//! Annealer-thread chapter 2: the unprojected-2D headline. The chain
//! version (`adiabatic_ramp_scale.rs`) lives on the 1D fast path
//! (`Mps + apply_kim_step`); this one drives the native tree backend
//! (`Ttn + apply_kim_step_heavy_hex`) with an explicit non-tree-edge
//! list, exercising both the tree-edge ZZ layer and the swap-network
//! path that routes the non-tree-edge ZZ layer through the spanning
//! tree every step.
//!
//! The heavy-hex grid generator `HeavyHexLayout::grid(rows, b)` is
//! validated structurally in `src/ttn/heavy_hex.rs`. The time-varying
//! schedule on a heavy-hex motif (with 2 non-tree edges) is validated
//! against `DenseState` to machine precision in
//! `src/ttn/kim_heavy_hex.rs::adiabatic_ramp_heavy_hex_grid_matches_dense_lossless`.
//! `Ttn::canonicalize_and_normalize` keeps the gauge from drifting at
//! scale, mirroring what saved the 1M-qubit chain run from FP overflow.
//!
//! 2D entanglement obeys perimeter-law not chain-area-law, so the same
//! χ that worked on a 1M chain (χ=8) saturates at much smaller N here.
//! Sized tests escalate from 1K calibration to as-large-as-tractable
//! headline; the χ-scaling spike anchors the per-bond cost on this
//! hardware so the headline size can be picked honestly.

use huoma::error::Result;
use huoma::kicked_ising::KimParams;
use huoma::ttn::heavy_hex::HeavyHexLayout;
use huoma::ttn::kim_heavy_hex::apply_kim_step_heavy_hex;
use huoma::ttn::topology::Edge;
use huoma::ttn::Ttn;
use num_complex::Complex64;

type C = Complex64;

fn h_gate() -> [[C; 2]; 2] {
    let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
    [
        [C::new(inv_sqrt2, 0.0), C::new(inv_sqrt2, 0.0)],
        [C::new(inv_sqrt2, 0.0), C::new(-inv_sqrt2, 0.0)],
    ]
}

fn run_adiabatic_grid(
    rows: usize,
    bridges_per_row: usize,
    n_steps: usize,
    max_bond: usize,
    dt: f64,
    canon_every: usize,
    tag: &str,
) -> Result<()> {
    let h_x_0 = 1.0_f64;
    let j_0 = 1.0_f64;
    let h_z_0 = 0.1_f64;

    let start = std::time::Instant::now();

    let layout = HeavyHexLayout::grid(rows, bridges_per_row);
    let n = layout.n_qubits();
    let topology = layout.tree().clone();
    let non_tree_edges: Vec<Edge> = layout
        .non_tree_edges()
        .iter()
        .map(|&[a, b]| Edge { a, b })
        .collect();
    let n_tree_edges = topology.n_edges();
    let n_non_tree = non_tree_edges.len();
    let t_layout = start.elapsed();
    eprintln!(
        "[{tag}] HeavyHexLayout::grid({rows}, {bridges_per_row}): \
         {n} qubits, {n_tree_edges} tree edges, {n_non_tree} non-tree edges \
         ({t_layout:.2?})"
    );

    let mut ttn = Ttn::new(topology);
    let t_init = start.elapsed();
    eprintln!("[{tag}] Ttn::new: {:?}", t_init - t_layout);

    let h = h_gate();
    for q in 0..n {
        ttn.apply_single(q, h);
    }
    let t_prep = start.elapsed();
    eprintln!("[{tag}] |+⟩^⊗N preparation: {:?}", t_prep - t_init);

    let mut t_canon_total = std::time::Duration::ZERO;
    let ramp_start = std::time::Instant::now();
    for k in 0..n_steps {
        let s = (k as f64 + 0.5) / n_steps as f64;
        let params = KimParams {
            j: s * j_0,
            h_x: -(1.0 - s) * h_x_0,
            h_z: s * h_z_0,
            dt,
        };
        apply_kim_step_heavy_hex(&mut ttn, &non_tree_edges, params, max_bond)?;
        if (k + 1) % canon_every == 0 || k + 1 == n_steps {
            let canon_start = std::time::Instant::now();
            ttn.canonicalize_and_normalize()?;
            t_canon_total += canon_start.elapsed();
        }
    }
    let t_ramp = ramp_start.elapsed();

    eprintln!(
        "[{tag}] canonicalize: every {canon_every} steps + after last, total = {t_canon_total:.2?}"
    );
    eprintln!(
        "[{tag}] {n_steps} ramp steps at χ={max_bond}: {t_ramp:.2?} ({:.2?}/step)",
        t_ramp / n_steps as u32
    );

    let measure_start = std::time::Instant::now();
    let z = ttn.expectation_z_all();
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
    let nsq = ttn.norm_squared();
    let t_norm = nsq_start.elapsed();
    let disc = ttn.total_discarded_weight();
    let total = start.elapsed();

    eprintln!(
        "[{tag}] expectation_z all ({n}): {t_measure:.2?}, \
         max|⟨Z⟩|={max_abs_z:.3}, mean|⟨Z⟩|={mean_abs_z:.3}, mean⟨Z⟩={mean_z:+.3}"
    );
    eprintln!(
        "\n[{tag}] DONE: {n} qubits, χ={max_bond}, {n_steps} steps, \
         total = {total:.2?}, norm² = {nsq:.6} (in {t_norm:.2?}), \
         discarded = {disc:.4e}\n"
    );

    Ok(())
}

/// χ-scaling calibration spike on the 2D heavy-hex driver. Confirms the
/// per-bond cost as a function of χ on this hardware so the headline
/// size can be picked honestly.
#[test]
#[ignore = "calibration spike, ignored to keep CI lean"]
fn adiabatic_ramp_2d_chi_scaling_spike() {
    let rows = 4_usize;
    let bridges_per_row = 4_usize;
    let layout = HeavyHexLayout::grid(rows, bridges_per_row);
    let n = layout.n_qubits();
    let topology = layout.tree().clone();
    let non_tree_edges: Vec<Edge> = layout
        .non_tree_edges()
        .iter()
        .map(|&[a, b]| Edge { a, b })
        .collect();

    let h = h_gate();
    let n_steps = 5;
    let dt = 0.1_f64;

    for &chi in &[8_usize, 16, 32] {
        let mut ttn = Ttn::new(topology.clone());
        for q in 0..n {
            ttn.apply_single(q, h);
        }
        let t0 = std::time::Instant::now();
        for k in 0..n_steps {
            let s = (k as f64 + 0.5) / n_steps as f64;
            apply_kim_step_heavy_hex(
                &mut ttn,
                &non_tree_edges,
                KimParams {
                    j: s,
                    h_x: -(1.0 - s),
                    h_z: 0.1 * s,
                    dt,
                },
                chi,
            )
            .unwrap();
        }
        let t = t0.elapsed();
        eprintln!(
            "[2D spike] grid({rows},{bridges_per_row}) N={n} χ={chi} steps={n_steps}: \
             {t:.2?} ({:.2?}/step)",
            t / n_steps as u32
        );
    }
}

/// 1K calibration on a near-square 2D heavy-hex grid. grid(20, 20) =
/// 1200 qubits, ~1140 tree edges, ~360 non-tree edges; χ=16 keeps the
/// per-step cost tractable on a desktop while having enough bond
/// dimension headroom for the modest entanglement growth in a 50-step
/// ramp.
#[test]
#[ignore = "1K 2D adiabatic ramp, multi-minute"]
fn adiabatic_ramp_2d_grid_1k_completes() {
    run_adiabatic_grid(20, 20, 50, 16, 0.1, 5, "2D-1K").unwrap();
}

/// 4K intermediate run — useful midpoint to verify the per-bond
/// extrapolation before any larger headline. grid(45, 30) = 4065 qubits.
#[test]
#[ignore = "4K 2D adiabatic ramp, multi-minute"]
fn adiabatic_ramp_2d_grid_5k_completes() {
    run_adiabatic_grid(45, 30, 50, 8, 0.1, 5, "2D-4K").unwrap();
}

/// 9.5K headline. grid(63, 50) = 9463 qubits at χ=8 to keep per-step
/// cost in the ~25-second-per-step regime. Honest scope: 2D entanglement
/// saturates χ=8 faster than chain χ=8 did, so discarded weight is the
/// load-bearing quality measure here.
#[test]
#[ignore = "9.5K 2D adiabatic ramp, ~23 min"]
fn adiabatic_ramp_2d_grid_10k_completes() {
    run_adiabatic_grid(63, 50, 50, 8, 0.1, 5, "2D-9.5K").unwrap();
}
