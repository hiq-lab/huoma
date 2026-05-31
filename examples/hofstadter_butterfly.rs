//! Generate Hofstadter-butterfly CSVs for square lattice (canonical
//! validation) and {7, 3} hyperbolic tiling (Track F headline result).
//!
//! Run:
//!   cargo run --release --example hofstadter_butterfly
//!
//! Output:
//!   results/hofstadter/square_8x8.csv
//!   results/hofstadter/hyperbolic_7_3_r2.csv
//!
//! Plot via `python external/hofstadter_ref/plot_butterfly.py`.

use huoma::hyperbolic::HyperbolicLayout;
use huoma::magnetic::{
    EmbeddedGraph, ButterflyRow, butterfly_to_csv, hofstadter_butterfly,
};

fn main() -> std::io::Result<()> {
    let n_steps = 500;

    // -- Square 8×8: canonical Hofstadter butterfly --
    let sq = EmbeddedGraph::square_lattice(8, 8);
    let t0 = std::time::Instant::now();
    let sq_rows = hofstadter_butterfly(&sq, 1.0, 0.0, 1.0, n_steps);
    let sq_wall = t0.elapsed();
    let sq_path = "results/hofstadter/square_8x8.csv";
    std::fs::write(sq_path, butterfly_to_csv(&sq_rows))?;
    summarise("square 8×8", &sq_rows, sq_wall);

    // -- {7, 3} hyperbolic at radius 2 (112 vertices) --
    let layout = HyperbolicLayout::pq_tiling(7, 3, 2);
    let hyp: EmbeddedGraph = (&layout).into();
    let t1 = std::time::Instant::now();
    let hyp_rows = hofstadter_butterfly(&hyp, 1.0, 0.0, 1.0, n_steps);
    let hyp_wall = t1.elapsed();
    let hyp_path = "results/hofstadter/hyperbolic_7_3_r2.csv";
    std::fs::write(hyp_path, butterfly_to_csv(&hyp_rows))?;
    summarise(&format!("{{7,3}} hyperbolic r=2 (N={})", layout.n_qubits()), &hyp_rows, hyp_wall);

    println!("\nWrote:");
    println!("  {sq_path}");
    println!("  {hyp_path}");
    Ok(())
}

fn summarise(name: &str, rows: &[ButterflyRow], wall: std::time::Duration) {
    let n_per_flux = rows[0].eigenvalues.len();
    let n_flux = rows.len();
    let mut e_min = f64::INFINITY;
    let mut e_max = f64::NEG_INFINITY;
    for row in rows {
        for &e in &row.eigenvalues {
            e_min = e_min.min(e);
            e_max = e_max.max(e);
        }
    }
    println!(
        "{name:30}  N={n_per_flux:4}  flux={n_flux:4}  pts={:6}  E∈[{e_min:.4},{e_max:.4}]  {:.2?}",
        n_per_flux * n_flux,
        wall
    );
}
