//! Phase 4 — Pocket+Ligand Adiabatic Sweep auf Huoma's TTN.
//!
//! Toy-Test der first-principles-Hypothese:
//!   "Beim Bindungs-Adiabatic-Sweep zeigt das per-step discarded_weight-Profil
//!    einen Spike bei kritischem s* genau dann wenn die Bindung im kollektiven
//!    Regime ist. Bei perturbativer Bindung bleibt das Profil monoton."
//!
//! Setup:
//! - Pocket = linear chain 0..11 (Backbone) — 12 Sites, 11 Tree-Edges
//! - Ligand = Site 12, Tree-Edge zu Site 6
//! - Apo-Edges: alle 12 Tree-Edges, gemeinsame Kopplung j_apo(s)
//! - Lig-Coupling-Edges: separate Liste, j_lig(s)
//!     • Perturbativ:   [(12, 6)] — 1 Edge, alle Coupling lokal
//!     • Kollektiv:     [(12, 6), (12, 0), (12, 3), (12, 9), (12, 11)] — 5 Edges,
//!                       Coupling über die ganze Pocket verteilt
//!     Beide haben das gleiche integrierte Lig-Coupling (Σ j_lig = J_LIG_TOTAL).
//!
//! Schedule (50 Steps, dt=0.1):
//!   Phase 1, s ∈ [0, 0.5]:  Driver → Apo
//!     h_x(s) = (1 − 2s) · H_X_0
//!     j_apo(s) = 2s · J_APO_0
//!     j_lig(s) = 0
//!   Phase 2, s ∈ [0.5, 1]:  Apo → Holo
//!     h_x = 0
//!     j_apo = J_APO_0
//!     j_lig(s) = (2s − 1) · J_LIG_TOTAL / N_lig_edges
//!
//! Pro Step: kumulatives discarded_weight(eid) abfragen, Differenz zum
//! vorherigen Step bilden = Per-Step-Beitrag jeder Edge.
//! Output: TSV-Datei mit (step, s, h_x, j_apo, j_lig, edge_id, edge_a, edge_b,
//!         is_lig_attachment, disc_step, disc_cum).

#![allow(clippy::too_many_arguments)]

use huoma::error::Result;
use huoma::ttn::heavy_hex::HeavyHexLayout;
use huoma::ttn::kim_heavy_hex::{rx_gate, rz_gate, zz_gate};
use huoma::ttn::topology::{Edge, Topology};
use huoma::ttn::{EdgeId, Ttn};
use num_complex::Complex64;
use std::fs::File;
use std::io::Write;

type C = Complex64;

const N_POCKET: usize = 12;
const LIG_SITE: usize = N_POCKET; // = 12
const N_TOTAL: usize = N_POCKET + 1;
const LIG_ATTACH: usize = 6; // Tree-Anchor des Liganden
const N_STEPS: usize = 50;
const DT: f64 = 0.1;
const H_X_0: f64 = 1.0;
const J_APO_0: f64 = 1.0;
const J_LIG_TOTAL: f64 = 3.0; // gleiche Integral-Coupling für beide Configs
const H_Z_0: f64 = 0.05; // klein, sodass Lig-Coupling-Variationen dominant werden
const MAX_BOND: usize = 6; // klein genug für meaningful truncation auf 13 Qubits
const CANON_EVERY: usize = 10;

fn build_topology() -> (Topology, Vec<Edge>, Vec<usize>) {
    // Tree: linear chain 0..11 + ligand attached to site 6
    let mut edges: Vec<Edge> = Vec::with_capacity(N_TOTAL - 1);
    for i in 0..N_POCKET - 1 {
        edges.push(Edge { a: i, b: i + 1 });
    }
    edges.push(Edge {
        a: LIG_ATTACH,
        b: LIG_SITE,
    });
    debug_assert_eq!(edges.len(), N_TOTAL - 1);
    let edge_descriptions: Vec<usize> = (0..edges.len()).collect();
    let topology = Topology::from_edges(N_TOTAL, edges.clone());
    (topology, edges, edge_descriptions)
}

fn h_gate() -> [[C; 2]; 2] {
    let s = std::f64::consts::FRAC_1_SQRT_2;
    [
        [C::new(s, 0.0), C::new(s, 0.0)],
        [C::new(s, 0.0), C::new(-s, 0.0)],
    ]
}

#[derive(Clone, Copy)]
struct ScheduleVal {
    h_x: f64,
    j_apo: f64,
    j_lig_per_edge: f64,
    h_z: f64,
}

fn schedule(s: f64, n_lig_edges: usize) -> ScheduleVal {
    if s < 0.5 {
        let s2 = 2.0 * s;
        ScheduleVal {
            h_x: (1.0 - s2) * H_X_0,
            j_apo: s2 * J_APO_0,
            j_lig_per_edge: 0.0,
            h_z: s2 * H_Z_0, // longitudinal field rampt mit Apo-Coupling
        }
    } else {
        let s2 = 2.0 * (s - 0.5);
        ScheduleVal {
            h_x: 0.0,
            j_apo: J_APO_0,
            j_lig_per_edge: s2 * J_LIG_TOTAL / n_lig_edges as f64,
            h_z: H_Z_0,
        }
    }
}

fn apply_step(
    ttn: &mut Ttn,
    tree_edges: &[Edge],
    lig_edges: &[(usize, usize)],
    sched: ScheduleVal,
    max_bond: usize,
) -> Result<()> {
    // 1. Apo-ZZ on all tree edges with j_apo
    if sched.j_apo.abs() > 1e-15 {
        let zz = zz_gate(sched.j_apo * DT);
        for eid in 0..tree_edges.len() {
            ttn.apply_two_qubit_on_edge(EdgeId(eid), zz, max_bond)?;
        }
    }
    // 2. Lig-ZZ — these may overlap with tree edges (e.g. edge (6, 12) is tree
    //    *and* the perturbativ-list). We apply the lig coupling regardless via
    //    the path API which works for both tree and non-tree.
    if sched.j_lig_per_edge.abs() > 1e-15 {
        let zz = zz_gate(sched.j_lig_per_edge * DT);
        for &(a, b) in lig_edges {
            ttn.apply_two_qubit_via_path(a, b, zz, max_bond)?;
        }
    }
    // 3. Rz if needed
    if sched.h_z.abs() > 1e-15 {
        let rz = rz_gate(2.0 * sched.h_z * DT);
        for q in 0..ttn.n_qubits() {
            ttn.apply_single(q, rz);
        }
    }
    // 4. Rx
    if sched.h_x.abs() > 1e-15 {
        let rx = rx_gate(2.0 * sched.h_x * DT);
        for q in 0..ttn.n_qubits() {
            ttn.apply_single(q, rx);
        }
    }
    Ok(())
}

fn run_config(
    label: &str,
    lig_edges: &[(usize, usize)],
    out_path: &str,
) -> Result<()> {
    eprintln!("\n=== Config: {label} ({} lig edges) ===", lig_edges.len());

    let t_start = std::time::Instant::now();
    let (topology, tree_edges, _) = build_topology();
    let n_tree_edges = tree_edges.len();

    let mut ttn = Ttn::new(topology);
    let h = h_gate();
    for q in 0..N_TOTAL {
        ttn.apply_single(q, h);
    }

    let mut prev_disc: Vec<f64> = vec![0.0; n_tree_edges];
    let mut prev_total: f64 = 0.0;

    let mut log_lines: Vec<String> = Vec::new();
    log_lines.push("step\ts\th_x\tj_apo\tj_lig_per_edge\tedge_id\tedge_a\tedge_b\tis_lig_attach\tdisc_step\tdisc_cum\ttotal_disc_step".to_string());

    for k in 0..N_STEPS {
        let s = (k as f64 + 0.5) / N_STEPS as f64;
        let sched = schedule(s, lig_edges.len());

        apply_step(&mut ttn, &tree_edges, lig_edges, sched, MAX_BOND)?;

        if (k + 1) % CANON_EVERY == 0 || k + 1 == N_STEPS {
            ttn.canonicalize_and_normalize()?;
        }

        let total = ttn.total_discarded_weight();
        let total_step = total - prev_total;
        prev_total = total;

        for eid in 0..n_tree_edges {
            let cum = ttn.discarded_weight(EdgeId(eid));
            let step_disc = cum - prev_disc[eid];
            prev_disc[eid] = cum;
            let edge = tree_edges[eid];
            let is_lig = (edge.a == LIG_SITE || edge.b == LIG_SITE) as u8;
            log_lines.push(format!(
                "{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{}\t{}\t{}\t{}\t{:.6e}\t{:.6e}\t{:.6e}",
                k, s, sched.h_x, sched.j_apo, sched.j_lig_per_edge,
                eid, edge.a, edge.b, is_lig,
                step_disc, cum, total_step
            ));
        }
    }

    let z = ttn.expectation_z_all();
    let max_abs_z: f64 = z.iter().map(|x| x.abs()).fold(0.0, f64::max);
    let z_lig = z[LIG_SITE];
    let z_pocket_mean: f64 = z[..N_POCKET].iter().sum::<f64>() / N_POCKET as f64;
    let total_disc = ttn.total_discarded_weight();
    let z_pocket: Vec<f64> = z[..N_POCKET].to_vec();
    eprintln!("[{label}] per-site ⟨Z⟩:");
    for (i, &z_i) in z_pocket.iter().enumerate() {
        eprintln!("  site {:>2}: {:+.4}", i, z_i);
    }
    eprintln!("  site 12 (lig): {:+.4}", z_lig);

    eprintln!(
        "[{label}] {} steps, χ={MAX_BOND}, {:.2?}",
        N_STEPS,
        t_start.elapsed()
    );
    eprintln!(
        "[{label}] final: max|⟨Z⟩|={max_abs_z:.4}  ⟨Z_lig⟩={z_lig:.4}  ⟨Z_pocket⟩={z_pocket_mean:.4}  total_discarded={total_disc:.4}"
    );

    let mut f = File::create(out_path).expect("cannot write log");
    for line in &log_lines {
        writeln!(f, "{line}").unwrap();
    }
    eprintln!("[{label}] log → {out_path}");

    Ok(())
}

#[test]
#[ignore = "scale test, run explicitly"]
fn pocket_ligand_perturbativ_vs_kollektiv() -> Result<()> {
    let _ = HeavyHexLayout::ibm_eagle_127(); // smoke

    // Perturbativ: 1 Lig-Edge (Tree)
    let lig_perturbativ: Vec<(usize, usize)> = vec![(LIG_ATTACH, LIG_SITE)];
    run_config(
        "perturbativ",
        &lig_perturbativ,
        "results/VQ-110/pocket_ligand_perturbativ.tsv",
    )?;

    // Kollektiv: 5 Lig-Edges (1 tree + 4 non-tree)
    let lig_kollektiv: Vec<(usize, usize)> = vec![
        (LIG_ATTACH, LIG_SITE),
        (0, LIG_SITE),
        (3, LIG_SITE),
        (9, LIG_SITE),
        (11, LIG_SITE),
    ];
    run_config(
        "kollektiv",
        &lig_kollektiv,
        "results/VQ-110/pocket_ligand_kollektiv.tsv",
    )?;

    Ok(())
}
