//! Phase 5 — Realistic Pocket+Ligand Adiabatic Sweep.
//!
//! Im Gegensatz zum Toy-System (1D-Linear-Chain mit synthetischer Lig-Coupling-
//! Verteilung) holt dieser Test eine echte Pocket aus 1L63 (T4-Lysozym L99A
//! Apo) als Topology + Coupling-Edges. Drei Liganden (BNZ schwach, IND stark,
//! N4B stark) als separate Konfigurationen.
//!
//! Erwartung aus der Literatur: T4-L99A ist eine *rigide hydrophobe Pocket*,
//! NICHT allosterisch. Daher sollten alle drei Liganden ein **perturbatives**
//! Profil zeigen (flaches per-step discarded weight, kein Spike). Wenn der
//! Test diese Erwartung bestätigt, ist die Methode konsistent mit unabhängiger
//! biophysikalischer Klassifikation. Wenn IND oder N4B plötzlich ein Spike
//! zeigen, müssen wir prüfen ob das ein Methoden-Artefakt ist oder ein echter
//! Befund (z.B. lokale Pocket-Reorganisation für größere Liganden).
//!
//! Plus: Vergleich gegen Phase-4-Toy-System (1 Edge perturbativ, 5 Edges
//! kollektiv) — die realistische BNZ-Pocket hat 11 Lig-Edges, also strukturell
//! kollektiv-aussehend. Wenn das Profil trotzdem flach ist, lernen wir:
//! Anzahl der Edges ist NICHT der einzige Faktor, die *Stärke* pro Edge und
//! die geometrische Anordnung zählen auch.

#![allow(clippy::too_many_arguments)]

use huoma::error::Result;
use huoma::ttn::kim_heavy_hex::{rx_gate, rz_gate, zz_gate};
use huoma::ttn::topology::{Edge, Topology};
use huoma::ttn::{EdgeId, Ttn};
use num_complex::Complex64;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Write;

type C = Complex64;

const N_STEPS: usize = 50;
const DT: f64 = 0.1;
const H_X_0: f64 = 1.0;
const J_APO_0: f64 = 1.0;
const J_LIG_TOTAL: f64 = 3.0;
const H_Z_0: f64 = 0.05;
const MAX_BOND: usize = 6;
const CANON_EVERY: usize = 10;

#[derive(Deserialize)]
struct PocketSite {
    site_id: usize,
    residue_seq: i32,
    #[allow(dead_code)]
    residue_name: String,
    #[allow(dead_code)]
    coords: [f64; 3],
    min_lig_distance: f64,
}

#[derive(Deserialize)]
struct LigandRecord {
    site_id: usize,
    #[allow(dead_code)]
    resname: String,
    #[allow(dead_code)]
    centroid: [f64; 3],
    n_heavy_atoms: usize,
}

#[derive(Deserialize)]
struct EdgeRec {
    a: usize,
    b: usize,
    distance_a: f64,
}

#[derive(Deserialize)]
struct PocketTopology {
    name: String,
    n_total_sites: usize,
    pocket_residues: Vec<PocketSite>,
    ligand: LigandRecord,
    apo_native_contact_edges: Vec<EdgeRec>,
    lig_pocket_edges: Vec<EdgeRec>,
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
            h_z: s2 * H_Z_0,
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

/// Build a spanning tree from the apo native contacts via simple BFS,
/// returning (tree_edges, non_tree_edges). Tree-Edges sind eine
/// kontiguierliche Teilmenge der Apo-Native-Contacts. Lig-Pocket-Edges
/// werden separat behandelt (alle via swap-network applied).
fn split_tree(
    n_sites: usize,
    apo_edges: &[EdgeRec],
    lig_attach_site: usize,
    lig_site_id: usize,
) -> (Vec<Edge>, Vec<(usize, usize)>) {
    // Adjacency from apo edges
    let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
    for e in apo_edges {
        adj.entry(e.a).or_default().push(e.b);
        adj.entry(e.b).or_default().push(e.a);
    }

    // BFS from site 0 to build spanning tree
    let mut tree_edges: Vec<Edge> = Vec::new();
    let mut visited: HashSet<usize> = HashSet::new();
    let mut queue: Vec<usize> = vec![0];
    visited.insert(0);
    let mut tree_set: HashSet<(usize, usize)> = HashSet::new();
    while let Some(u) = queue.pop() {
        if let Some(neigh) = adj.get(&u) {
            for &v in neigh {
                if !visited.contains(&v) {
                    visited.insert(v);
                    let edge = (u.min(v), u.max(v));
                    tree_set.insert(edge);
                    tree_edges.push(Edge {
                        a: edge.0,
                        b: edge.1,
                    });
                    queue.push(v);
                }
            }
        }
    }

    // Verify all pocket sites reachable; if not, add a few missing edges
    if visited.len() < n_sites - 1 {
        // Brute-force: for any unvisited pocket site, find nearest visited
        // pocket via apo_edges and add it. Should not happen for our T4 data.
        eprintln!("WARN: BFS missed sites; pocket-graph may be disconnected");
    }

    // Add ligand site to the tree via lig_attach_site (closest pocket residue)
    tree_edges.push(Edge {
        a: lig_attach_site.min(lig_site_id),
        b: lig_attach_site.max(lig_site_id),
    });
    tree_set.insert((lig_attach_site.min(lig_site_id), lig_attach_site.max(lig_site_id)));

    // Non-tree apo edges (these will be ZZ-applied via swap network)
    let mut non_tree_apo: Vec<(usize, usize)> = Vec::new();
    for e in apo_edges {
        let key = (e.a.min(e.b), e.a.max(e.b));
        if !tree_set.contains(&key) {
            non_tree_apo.push(key);
        }
    }

    (tree_edges, non_tree_apo)
}

/// Variant: how to use the lig-pocket edges
#[derive(Clone, Copy)]
enum LigMode {
    /// All N lig edges with shared coupling (kollektiv-realistic)
    All,
    /// Only the closest single edge with full coupling (perturbativ-realistic)
    OnlyClosest,
    /// All N lig edges enumerated, but coupling = 0 (decoy: tests whether the
    /// step-25 spike is genuine ligand-onset signal or schedule artifact from
    /// h_x dropping discontinuously to 0 at s=0.5).
    Decoy,
}

fn run_pocket_variant(json_path: &str, out_path: &str, mode: LigMode) -> Result<(String, f64, f64, usize)> {
    let raw = std::fs::read_to_string(json_path).expect("read json");
    let topo: PocketTopology = serde_json::from_str(&raw).expect("parse json");

    eprintln!("\n=== {} ===", topo.name);
    eprintln!(
        "  pocket={}, lig_atoms={}, apo_edges={}, lig_edges={}",
        topo.pocket_residues.len(),
        topo.ligand.n_heavy_atoms,
        topo.apo_native_contact_edges.len(),
        topo.lig_pocket_edges.len()
    );

    let n_sites = topo.n_total_sites;
    let lig_id = topo.ligand.site_id;

    // Ligand attach site: pocket residue mit kleinster lig-distance
    let lig_attach = topo
        .pocket_residues
        .iter()
        .min_by(|a, b| {
            a.min_lig_distance
                .partial_cmp(&b.min_lig_distance)
                .unwrap()
        })
        .map(|s| s.site_id)
        .expect("non-empty pocket");
    eprintln!(
        "  lig_attach_site = {} (residue {}, distance {:.2}Å)",
        lig_attach,
        topo.pocket_residues
            .iter()
            .find(|s| s.site_id == lig_attach)
            .map(|s| s.residue_seq)
            .unwrap_or(-1),
        topo.pocket_residues
            .iter()
            .find(|s| s.site_id == lig_attach)
            .map(|s| s.min_lig_distance)
            .unwrap_or(0.0)
    );

    let (tree_edges, non_tree_apo_edges) =
        split_tree(n_sites, &topo.apo_native_contact_edges, lig_attach, lig_id);

    eprintln!(
        "  tree_edges={}, non_tree_apo={}, lig_edges={}",
        tree_edges.len(),
        non_tree_apo_edges.len(),
        topo.lig_pocket_edges.len()
    );

    let topology = Topology::from_edges(n_sites, tree_edges.clone());
    let mut ttn = Ttn::new(topology);

    let h = h_gate();
    for q in 0..n_sites {
        ttn.apply_single(q, h);
    }

    let (lig_edges_pairs, j_lig_scale): (Vec<(usize, usize)>, f64) = match mode {
        LigMode::All => (topo.lig_pocket_edges.iter().map(|e| (e.a, e.b)).collect(), 1.0),
        LigMode::OnlyClosest => {
            let closest = topo
                .lig_pocket_edges
                .iter()
                .min_by(|a, b| a.distance_a.partial_cmp(&b.distance_a).unwrap())
                .expect("non-empty lig edges");
            (vec![(closest.a, closest.b)], 1.0)
        }
        LigMode::Decoy => {
            // same edge enumeration as All, but coupling magnitude = 0
            (topo.lig_pocket_edges.iter().map(|e| (e.a, e.b)).collect(), 0.0)
        }
    };
    let mode_label = match mode {
        LigMode::All => "All",
        LigMode::OnlyClosest => "OnlyClosest",
        LigMode::Decoy => "Decoy(j_lig=0)",
    };
    eprintln!(
        "  lig_mode = {}, n_active_lig_edges = {}, j_lig_scale = {}",
        mode_label, lig_edges_pairs.len(), j_lig_scale
    );

    let n_tree_edges = tree_edges.len();
    let mut prev_disc: Vec<f64> = vec![0.0; n_tree_edges];
    let mut prev_total: f64 = 0.0;

    let mut log_lines: Vec<String> = Vec::new();
    log_lines.push("step\ts\th_x\tj_apo\tj_lig_per_edge\tedge_id\tedge_a\tedge_b\tis_lig_attach\tdisc_step\tdisc_cum\ttotal_disc_step".to_string());

    let t_start = std::time::Instant::now();
    for k in 0..N_STEPS {
        let s = (k as f64 + 0.5) / N_STEPS as f64;
        let mut sched = schedule(s, lig_edges_pairs.len());
        sched.j_lig_per_edge *= j_lig_scale;

        // 1. Apo-ZZ on tree edges
        if sched.j_apo.abs() > 1e-15 {
            let zz = zz_gate(sched.j_apo * DT);
            for eid in 0..tree_edges.len() {
                ttn.apply_two_qubit_on_edge(EdgeId(eid), zz, MAX_BOND)?;
            }
        }
        // 2. Apo-ZZ on non-tree apo edges (swap-network)
        if sched.j_apo.abs() > 1e-15 {
            let zz = zz_gate(sched.j_apo * DT);
            for &(a, b) in &non_tree_apo_edges {
                ttn.apply_two_qubit_via_path(a, b, zz, MAX_BOND)?;
            }
        }
        // 3. Lig-ZZ
        if sched.j_lig_per_edge.abs() > 1e-15 {
            let zz = zz_gate(sched.j_lig_per_edge * DT);
            for &(a, b) in &lig_edges_pairs {
                ttn.apply_two_qubit_via_path(a, b, zz, MAX_BOND)?;
            }
        }
        // 4. Rz
        if sched.h_z.abs() > 1e-15 {
            let rz = rz_gate(2.0 * sched.h_z * DT);
            for q in 0..n_sites {
                ttn.apply_single(q, rz);
            }
        }
        // 5. Rx
        if sched.h_x.abs() > 1e-15 {
            let rx = rx_gate(2.0 * sched.h_x * DT);
            for q in 0..n_sites {
                ttn.apply_single(q, rx);
            }
        }

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
            let is_lig = (edge.a == lig_id || edge.b == lig_id) as u8;
            log_lines.push(format!(
                "{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{}\t{}\t{}\t{}\t{:.6e}\t{:.6e}\t{:.6e}",
                k, s, sched.h_x, sched.j_apo, sched.j_lig_per_edge,
                eid, edge.a, edge.b, is_lig,
                step_disc, cum, total_step
            ));
        }
    }
    let elapsed = t_start.elapsed();

    let z = ttn.expectation_z_all();
    let max_abs_z: f64 = z.iter().map(|x| x.abs()).fold(0.0, f64::max);
    let total_disc = ttn.total_discarded_weight();

    eprintln!(
        "  {N_STEPS} steps, χ={MAX_BOND}, {elapsed:.2?}; max|⟨Z⟩|={max_abs_z:.4}, total_disc={total_disc:.4}"
    );

    let mut f = File::create(out_path).expect("write log");
    for line in &log_lines {
        writeln!(f, "{line}").unwrap();
    }
    eprintln!("  log → {out_path}");

    Ok((topo.name, total_disc, max_abs_z, lig_edges_pairs.len()))
}

#[test]
#[ignore = "scale test, run explicitly"]
fn realistic_pocket_three_ligands() -> Result<()> {
    std::fs::create_dir_all("results/VQ-110").ok();

    let configs = [
        ("results/VQ-110/inputs/T4_L99A_BNZ.pocket.json", "BNZ"),
        ("results/VQ-110/inputs/T4_L99A_IND.pocket.json", "IND"),
        ("results/VQ-110/inputs/T4_L99A_N4B.pocket.json", "N4B"),
    ];

    let mut results: Vec<(String, f64, f64, usize)> = Vec::new();
    for (json, label) in configs {
        // Kollektiv-Variante: alle Lig-Edges aktiv
        let out_all = format!("results/VQ-110/realistic_{label}_all.tsv");
        let r_all = run_pocket_variant(json, &out_all, LigMode::All)?;
        results.push((format!("{}_all", r_all.0), r_all.1, r_all.2, r_all.3));

        // Perturbativ-Variante: nur dichteste Edge
        let out_one = format!("results/VQ-110/realistic_{label}_one.tsv");
        let r_one = run_pocket_variant(json, &out_one, LigMode::OnlyClosest)?;
        results.push((format!("{}_one", r_one.0), r_one.1, r_one.2, r_one.3));

        // Decoy-Variante: j_lig=0 (keine Bindung) als Negativ-Kontrolle
        let out_dec = format!("results/VQ-110/realistic_{label}_decoy.tsv");
        let r_dec = run_pocket_variant(json, &out_dec, LigMode::Decoy)?;
        results.push((format!("{}_decoy", r_dec.0), r_dec.1, r_dec.2, r_dec.3));
    }

    println!("\n=== Phase 5 Zusammenfassung ===");
    println!(
        "{:<20} {:>10} {:>12} {:>10}",
        "Ligand", "n_lig_edges", "total_disc", "max|Z|"
    );
    for (name, td, mz, ne) in &results {
        println!("{:<20} {:>10} {:>12.6} {:>10.4}", name, ne, td, mz);
    }

    Ok(())
}
