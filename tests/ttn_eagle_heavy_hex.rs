//! Integration-level invariants + golden-file regression test for the
//! `ttn::heavy_hex::HeavyHexLayout::ibm_eagle_127()` constructor.
//!
//! The golden file lives at `tests/golden/ibm_eagle_127.json` and is the
//! canonical serialisation of the layout. To regenerate after an intentional
//! change to the constructor, run:
//!
//! ```sh
//! UPDATE_GOLDEN=1 cargo test --test ttn_eagle_heavy_hex
//! ```
//!
//! Without that env var the test compares byte-for-byte and fails on any
//! drift, so any change to `ibm_eagle_127()` that would silently rewrite
//! the spanning tree is caught by CI.

use huoma::ttn::heavy_hex::{
    eagle_127_coupling_edges, EAGLE_BRIDGES, EAGLE_ROWS, EAGLE_THROUGH_BRIDGES,
};
use huoma::ttn::{HeavyHexLayout, Ttn};
use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

fn golden_path() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("golden");
    p.push("ibm_eagle_127.json");
    p
}

#[test]
fn eagle_structural_facts() {
    let layout = HeavyHexLayout::ibm_eagle_127();
    assert_eq!(layout.n_qubits(), 127);
    assert_eq!(layout.tree().n_qubits(), 127);
    assert_eq!(layout.tree().n_edges(), 126);
    assert_eq!(layout.non_tree_edges().len(), 18);

    // Tree edges are canonical with a ≤ b and lexicographically sorted.
    for w in layout.tree_edges.windows(2) {
        assert!(w[0] < w[1], "tree edges must be strictly sorted");
    }
    for e in &layout.tree_edges {
        assert!(e[0] < e[1], "tree edge {e:?} must satisfy a < b");
    }
    for e in layout.non_tree_edges() {
        assert!(e[0] < e[1], "non-tree edge {e:?} must satisfy a < b");
    }
}

#[test]
fn eagle_row_structure_matches_ibm_numbering() {
    // Guard against any accidental reshuffle of the row / bridge ranges.
    assert_eq!(
        EAGLE_ROWS,
        [
            (0, 14),
            (18, 33),
            (37, 52),
            (56, 71),
            (75, 90),
            (94, 109),
            (113, 127),
        ]
    );

    let row_qubits: usize = EAGLE_ROWS.iter().map(|(s, e)| e - s).sum();
    assert_eq!(row_qubits, 103);
    assert_eq!(EAGLE_BRIDGES.len(), 24);
    assert_eq!(row_qubits + EAGLE_BRIDGES.len(), 127);
    assert_eq!(EAGLE_THROUGH_BRIDGES.len(), 6);
}

#[test]
fn eagle_coupling_graph_matches_fake_sherbrooke_count() {
    // Sanity: `eagle_127_coupling_edges()` must produce exactly 144 unique
    // undirected edges, matching the FakeSherbrooke coupling map.
    let edges = eagle_127_coupling_edges();
    let set: BTreeSet<[usize; 2]> = edges
        .iter()
        .map(|e| if e.a < e.b { [e.a, e.b] } else { [e.b, e.a] })
        .collect();
    assert_eq!(edges.len(), 144);
    assert_eq!(set.len(), 144);
}

#[test]
fn eagle_tree_plus_non_tree_equals_full_coupling_graph() {
    let layout = HeavyHexLayout::ibm_eagle_127();
    let mut union: BTreeSet<[usize; 2]> = layout.tree_edges.iter().copied().collect();
    for &e in layout.non_tree_edges() {
        assert!(union.insert(e), "non-tree edge {e:?} duplicates a tree edge");
    }
    let full: BTreeSet<[usize; 2]> = eagle_127_coupling_edges()
        .iter()
        .map(|e| if e.a < e.b { [e.a, e.b] } else { [e.b, e.a] })
        .collect();
    assert_eq!(union, full);
}

#[test]
fn eagle_heavy_paths_partition_every_qubit() {
    let layout = HeavyHexLayout::ibm_eagle_127();
    let mut coverage = vec![0u32; layout.n_qubits()];
    for path in layout.heavy_paths() {
        for &q in path {
            coverage[q] += 1;
        }
    }
    for (q, c) in coverage.iter().enumerate() {
        assert_eq!(*c, 1, "qubit {q} covered {c} times by heavy paths");
    }
}

#[test]
fn eagle_heavy_paths_rows_are_row_ranges_then_bridge_singletons() {
    let layout = HeavyHexLayout::ibm_eagle_127();
    let paths = layout.heavy_paths();
    assert_eq!(paths.len(), 7 + 24);

    // First 7 entries are the 7 horizontal rows in row-major order.
    let expected_lengths = [14, 15, 15, 15, 15, 15, 14];
    for (i, exp) in expected_lengths.iter().enumerate() {
        assert_eq!(paths[i].len(), *exp, "heavy path {i} has wrong length");
        let (start, end) = EAGLE_ROWS[i];
        let row: Vec<usize> = (start..end).collect();
        assert_eq!(paths[i], row, "heavy path {i} is not row [{start}, {end})");
    }

    // Next 24 entries are bridge singletons, one per bridge qubit in the
    // declared order.
    for (i, (bridge, _, _)) in EAGLE_BRIDGES.iter().enumerate() {
        let p = &paths[7 + i];
        assert_eq!(p.len(), 1);
        assert_eq!(p[0], *bridge);
    }
}

#[test]
fn eagle_through_bridges_are_interior_tree_nodes() {
    let layout = HeavyHexLayout::ibm_eagle_127();
    for b in EAGLE_THROUGH_BRIDGES {
        assert_eq!(
            layout.tree().degree(b),
            2,
            "through-bridge {b} must be degree-2 in the spanning tree"
        );
    }
}

#[test]
fn eagle_leaf_bridges_are_tree_leaves() {
    let layout = HeavyHexLayout::ibm_eagle_127();
    let throughs: BTreeSet<usize> = EAGLE_THROUGH_BRIDGES.into_iter().collect();
    for (bridge, _, _) in EAGLE_BRIDGES {
        if throughs.contains(&bridge) {
            continue;
        }
        assert_eq!(
            layout.tree().degree(bridge),
            1,
            "leaf bridge {bridge} must be a tree leaf"
        );
    }
}

#[test]
fn eagle_cycle_count_matches_independent_cycle_count() {
    // |E| - |V| + 1 = 144 - 127 + 1 = 18 for a connected graph.
    let e = eagle_127_coupling_edges().len();
    let v = 127;
    assert_eq!(e - v + 1, 18);
    let layout = HeavyHexLayout::ibm_eagle_127();
    assert_eq!(layout.non_tree_edges().len(), 18);
}

#[test]
fn eagle_tree_feeds_ttn_constructor() {
    // Sanity: the `Topology` produced by `layout.tree()` is accepted by
    // `Ttn::new` without panicking. This is the contract the TTN engine
    // consumes in D.3 § "Heavy-hex mapping" of TRACK_D_DESIGN.md.
    let layout = HeavyHexLayout::ibm_eagle_127();
    let ttn = Ttn::new(layout.tree().clone());
    assert_eq!(ttn.n_qubits(), 127);
}

#[test]
fn eagle_json_matches_golden_file() {
    let layout = HeavyHexLayout::ibm_eagle_127();
    let actual = layout.to_json();

    let path = golden_path();
    if std::env::var("UPDATE_GOLDEN").is_ok() {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, &actual).expect("failed to write golden file");
        println!("wrote golden file: {}", path.display());
        return;
    }

    let expected = fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "golden file {} missing: {e}\n\
             To generate it, run: UPDATE_GOLDEN=1 cargo test --test ttn_eagle_heavy_hex",
            path.display()
        )
    });

    assert_eq!(
        actual,
        expected,
        "Eagle 127q layout JSON drifted from the golden file.\n\
         If this change is intentional, regenerate with:\n\
             UPDATE_GOLDEN=1 cargo test --test ttn_eagle_heavy_hex"
    );
}

#[test]
fn eagle_json_round_trips_back_into_a_valid_layout() {
    let layout = HeavyHexLayout::ibm_eagle_127();
    let j = layout.to_json();
    let back = HeavyHexLayout::from_json(&j).expect("deserialisation failed");
    assert_eq!(back.name, "ibm_eagle_127");
    assert_eq!(back.n_qubits, 127);
    assert_eq!(back.tree_edges, layout.tree_edges);
    assert_eq!(back.non_tree_edges, layout.non_tree_edges);
    assert_eq!(back.heavy_paths, layout.heavy_paths);
    // And the reconstructed tree is consumable by `Ttn`.
    let _ttn = Ttn::new(back.tree().clone());
}
