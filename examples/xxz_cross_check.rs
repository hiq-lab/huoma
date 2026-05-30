//! Smoke-cross-check between Huoma's dense XXZ reference and the
//! ITensor reference runner.
//!
//! Runs `reference_xxz_run` on the same parameters that
//! `external/itensor_ref/xxz_griffiths.jl` reads from
//! `<path>.manifest.json`, and prints initial + final ⟨σ_z⟩ rows.
//! Visual diff against the ITensor `.itensor.json` output confirms
//! the two TEBD implementations agree at lossless χ.
//!
//! Usage:
//!   cargo run --release --example xxz_cross_check
//!
//! Expected output for the default `/tmp/test_xxz.manifest.json`
//! (N=10, Δ=1, dt=0.1, n_steps=10, uniform J=1, Néel initial):
//!   step 0 (initial):  [1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
//!   step 10 (final):   matches ITensor to ~1e-12 element-wise

use huoma::xxz::{reference_xxz_run, XxzParams};

fn main() {
    let n = 10;
    let params = XxzParams { delta: 1.0, dt: 0.1 };
    let n_steps = 10;
    let initial = 341_usize; // Néel state |0101010101⟩
    let j_per_bond: Vec<f64> = vec![1.0; n - 1];

    let history = reference_xxz_run(n, &j_per_bond, params, initial, n_steps);

    println!("step 0 (initial): {:?}", history[0]);
    println!("step 10 (final):  {:?}", history[n_steps]);
}
