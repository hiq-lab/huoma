//! Integration-level invariants + golden-file regression test for the
//! IBM Eagle 127q heavy-hex topology.
//!
//! The golden file at `tests/golden/ibm_eagle_127.json` is the canonical
//! serialisation of `huoma::topology::ibm_eagle_127()`. To regenerate it
//! after an intentional change, run:
//!
//! ```sh
//! UPDATE_GOLDEN=1 cargo test --test topology_eagle
//! ```
//!
//! The test will write the current serialisation to disk and pass. Without
//! that env var the test compares byte-for-byte and fails on any mismatch.

use huoma::topology::heavy_hex::{
    eagle_127_coupling_edges, EAGLE_BRIDGES, EAGLE_ROWS, EAGLE_THROUGH_BRIDGES,
};
use huoma::topology::{ibm_eagle_127, Topology, Tree};
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
    let t = ibm_eagle_127();

    // 127 qubits, 126 tree edges, 18 non-tree edges — the forced counts.
    assert_eq!(t.n_qubits(), 127);
    assert_eq!(t.tree_edges().len(), 126);
    assert_eq!(t.non_tree_edges().len(), 18);

    // Tree edges are lexicographically sorted with a < b.
    let edges = t.tree_edges();
    for w in edges.windows(2) {
        assert!(w[0] < w[1], "tree_edges must be strictly sorted");
    }
    for &(a, b) in &edges {
        assert!(a < b, "tree edge {a}-{b} must be canonical with a<b");
    }
}

#[test]
fn eagle_row_structure_matches_ibm_numbering() {
    // The 7 row ranges encode IBM's row-major qubit numbering (top=0..13,
    // bottom=113..126). Guard against any accidental reshuffling.
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
    let bridge_qubits = EAGLE_BRIDGES.len();
    assert_eq!(row_qubits + bridge_qubits, 127);
    assert_eq!(bridge_qubits, 24);
    assert_eq!(EAGLE_THROUGH_BRIDGES.len(), 6);
}

#[test]
fn eagle_coupling_graph_matches_fake_sherbrooke() {
    // Sanity: the full undirected edge set must have exactly 144 unique
    // entries, matching the conf_sherbrooke.json coupling map.
    let edges = eagle_127_coupling_edges();
    let set: BTreeSet<_> = edges.iter().collect();
    assert_eq!(edges.len(), 144);
    assert_eq!(set.len(), 144);
}

#[test]
fn eagle_tree_plus_non_tree_equals_full_coupling_graph() {
    let t = ibm_eagle_127();
    let mut union: BTreeSet<_> = t.tree_edges().into_iter().collect();
    for e in t.non_tree_edges() {
        assert!(union.insert(*e), "non-tree edge {e:?} duplicates a tree edge");
    }
    let all: BTreeSet<_> = eagle_127_coupling_edges().into_iter().collect();
    assert_eq!(union, all);
}

#[test]
fn eagle_heavy_paths_partition_every_qubit() {
    let t = ibm_eagle_127();
    let mut coverage = vec![0u32; t.n_qubits()];
    for path in t.heavy_paths() {
        for &q in path {
            coverage[q] += 1;
        }
    }
    for (q, c) in coverage.iter().enumerate() {
        assert_eq!(*c, 1, "qubit {q} covered {c} times by heavy paths");
    }
}

#[test]
fn eagle_heavy_paths_are_row_plus_bridge_singletons() {
    let t = ibm_eagle_127();
    let paths = t.heavy_paths();
    assert_eq!(paths.len(), 7 + 24);

    // First 7 are the 7 horizontal rows in row-major order.
    let expected_lengths = [14, 15, 15, 15, 15, 15, 14];
    for (i, exp) in expected_lengths.iter().enumerate() {
        assert_eq!(paths[i].len(), *exp, "heavy path {i} has wrong length");
        let (start, end) = EAGLE_ROWS[i];
        let row: Vec<usize> = (start..end).collect();
        assert_eq!(paths[i], row, "heavy path {i} is not row {start}..{end}");
    }

    // Remaining 24 are bridge singletons, one per bridge qubit.
    for (i, (bridge, _, _)) in EAGLE_BRIDGES.iter().enumerate() {
        let p = &paths[7 + i];
        assert_eq!(p.len(), 1);
        assert_eq!(p[0], *bridge);
    }
}

#[test]
fn eagle_through_bridges_are_interior_tree_nodes() {
    let t = ibm_eagle_127();
    // A through-bridge has both its vertical edges in the tree, so it
    // has degree 2 in the tree (one parent, one child, or two children).
    for b in EAGLE_THROUGH_BRIDGES {
        let total_degree = t.children()[b].len() + if t.parent()[b].is_some() { 1 } else { 0 };
        assert_eq!(total_degree, 2, "through-bridge {b} should have tree degree 2");
    }
}

#[test]
fn eagle_leaf_bridges_are_tree_leaves() {
    let t = ibm_eagle_127();
    let throughs: BTreeSet<usize> = EAGLE_THROUGH_BRIDGES.into_iter().collect();
    for (bridge, _, _) in EAGLE_BRIDGES {
        if throughs.contains(&bridge) {
            continue;
        }
        // A leaf bridge is connected to the row above and nothing else in
        // the tree. Since row 0 is the root side, the "up" direction is
        // towards the root; therefore a leaf bridge must be a tree leaf.
        assert!(
            t.children()[bridge].is_empty(),
            "leaf bridge {bridge} must be a tree leaf"
        );
    }
}

#[test]
fn eagle_cycle_count_matches_independent_cycle_count() {
    // |E| - |V| + 1 == number of independent cycles for a connected graph.
    let e = eagle_127_coupling_edges().len();
    let v = 127;
    assert_eq!(e - v + 1, 18);
    // And exactly that many edges must be dropped to form the spanning tree.
    let t = ibm_eagle_127();
    assert_eq!(t.non_tree_edges().len(), 18);
}

#[test]
fn eagle_json_matches_golden_file() {
    let t = ibm_eagle_127();
    let actual = t.to_json().expect("serialisation failed");

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
             To generate it, run: UPDATE_GOLDEN=1 cargo test --test topology_eagle",
            path.display()
        )
    });

    assert_eq!(
        actual,
        expected,
        "Eagle 127q topology JSON drifted from the golden file.\n\
         If this change is intentional, regenerate with:\n\
             UPDATE_GOLDEN=1 cargo test --test topology_eagle"
    );
}

#[test]
fn eagle_json_round_trips_back_into_a_valid_tree() {
    let t = ibm_eagle_127();
    let j = t.to_json().unwrap();
    let back = Tree::from_json(&j).unwrap();
    assert_eq!(back.name, "ibm_eagle_127");
    assert_eq!(back.n_qubits, 127);
    assert_eq!(back.tree_edges(), t.tree_edges());
    assert_eq!(back.non_tree_edges(), t.non_tree_edges());
    assert_eq!(back.heavy_paths, t.heavy_paths);
    back.validate().expect("deserialised tree must revalidate");
}
