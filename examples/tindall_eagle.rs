//! Tindall N=127 Eagle kicked-Ising benchmark — runnable demo.
//!
//! Reproduces the Tindall et al. (PRX Quantum 5, 010308, 2024) kicked-Ising
//! Floquet circuit on the IBM Eagle 127q heavy-hex topology using Huoma's
//! native tree-tensor-network backend.
//!
//! # Usage
//!
//! ```sh
//! cargo run --release --example tindall_eagle
//! cargo run --release --example tindall_eagle -- --chi 16 --depth 10
//! ```
//!
//! # Parameters
//!
//! The circuit matches Tindall's published convention:
//!
//! - ZZ coupling: `exp(+iπ/4 Z⊗Z)` per edge per Trotter step
//! - X kick: `exp(-iθ_h/2 X)` with `θ_h = 0.8` per qubit per step
//! - Initial state: `|0…0⟩`
//!
//! The output is the per-qubit ⟨Z⟩ magnetization profile at the requested
//! depth, plus a summary comparing qubit 62 against the published reference.

use huoma::kicked_ising::KimParams;
use huoma::ttn::kim_heavy_hex::run_kim_heavy_hex;
use huoma::ttn::topology::Edge;
use huoma::ttn::{HeavyHexLayout, Ttn};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let chi: usize = parse_arg(&args, "--chi").unwrap_or(8);
    let depth: usize = parse_arg(&args, "--depth").unwrap_or(5);

    eprintln!("╔══════════════════════════════════════════════════════════╗");
    eprintln!("║  Huoma — Tindall N=127 Eagle Kicked-Ising Benchmark    ║");
    eprintln!("╚══════════════════════════════════════════════════════════╝");
    eprintln!();
    eprintln!("  χ_max  = {chi}");
    eprintln!("  depth  = {depth}");
    eprintln!("  θ_J    = π/4  (ZZ coupling per edge)");
    eprintln!("  θ_h    = 0.8  (X kick per qubit)");
    eprintln!("  N      = 127  (Eagle heavy-hex)");
    eprintln!();

    // 1. Build the Eagle 127q layout.
    let layout = HeavyHexLayout::ibm_eagle_127();
    let non_tree: Vec<Edge> = layout
        .non_tree_edges()
        .iter()
        .map(|[a, b]| Edge { a: *a, b: *b })
        .collect();

    // 2. Initialise the TTN in |0…0⟩.
    let mut ttn = Ttn::new(layout.tree().clone());

    // 3. Tindall's exact parameters.
    //    Huoma's zz_gate(θ) = exp(-iθ Z⊗Z), so j·dt = -π/4 matches
    //    Tindall's exp(+iπ/4 Z⊗Z).
    let params = KimParams {
        j: -std::f64::consts::FRAC_PI_4,
        h_x: 0.4, // 2·h_x·dt = 0.8 = θ_h
        h_z: 0.0,
        dt: 1.0,
    };

    // 4. Run.
    let start = std::time::Instant::now();
    let history = run_kim_heavy_hex(&mut ttn, &non_tree, params, chi, depth)
        .expect("run_kim_heavy_hex failed");
    let elapsed = start.elapsed();

    eprintln!("  completed in {elapsed:.2?}");
    eprintln!("  total discarded weight = {:.6e}", ttn.total_discarded_weight());
    eprintln!();

    // 5. Print the final magnetization profile.
    let z_final = &history[depth];
    println!("# qubit,z_depth_{depth}");
    for (q, &z) in z_final.iter().enumerate() {
        println!("{q},{z:.15e}");
    }

    // 6. Qubit 62 summary vs Tindall reference (depth ≤ 20).
    // Reference values from external/tindall_ref/z62_theta_h_0.8_bp_chi_inf.csv
    let tindall_z62 = [
        1.0, 0.69670671, 0.48540024, 0.51220978, 0.46572520,
        0.49816486, 0.46486951, 0.44385249, 0.44197507, 0.45166789,
        0.38500141, 0.30495499, 0.35656340, 0.32484345, 0.30828285,
        0.31078596, 0.28361069, 0.27211137, 0.25336712, 0.24548963,
        0.22957524,
    ];

    eprintln!("── Qubit 62 trajectory ──");
    eprintln!("{:>5}  {:>14}  {:>14}  {:>12}", "depth", "Huoma", "Tindall", "diff");
    for step in 0..=depth.min(20) {
        let z_huoma = history[step][62];
        let z_ref = if step < tindall_z62.len() {
            tindall_z62[step]
        } else {
            f64::NAN
        };
        let diff = z_huoma - z_ref;
        eprintln!(
            "{step:5}  {z_huoma:>14.10}  {z_ref:>14.10}  {diff:>+12.6e}"
        );
    }
}

fn parse_arg<T: std::str::FromStr>(args: &[String], flag: &str) -> Option<T> {
    args.windows(2).find_map(|w| {
        if w[0] == flag {
            w[1].parse().ok()
        } else {
            None
        }
    })
}
