//! IBM Eagle 127-qubit heavy-hex topology (D.2 / D.3).
//!
//! The coupling graph is taken byte-for-byte from the [`FakeSherbrooke`]
//! `conf_sherbrooke.json` shipped with `qiskit-ibm-runtime`. Eagle r3 is a
//! heavy-hexagonal lattice with:
//!
//! - **127 qubits** arranged in 7 horizontal rows of lengths
//!   `[14, 15, 15, 15, 15, 15, 14]`,
//! - **144 undirected coupling edges** (96 horizontal + 48 vertical bridge),
//! - **6 bridge rows** of 4 bridge qubits each; every bridge qubit has one
//!   neighbour in the row above and one in the row below,
//! - **18 independent cycles** (one per internal hexagon of the lattice).
//!
//! # Spanning-tree policy
//!
//! The spanning tree used by `ibm_eagle_127` is the **row-major heavy-path
//! decomposition**:
//!
//! 1. The 7 horizontal rows become the 7 heavy paths (96 tree edges).
//! 2. In each bridge row, the **leftmost** bridge qubit (lowest qubit index)
//!    is the **through-bridge**: both of its vertical edges are kept, so it
//!    becomes an interior node of the spanning tree linking the row above to
//!    the row below. The concrete through-bridges are
//!    `[14, 33, 52, 71, 90, 109]`.
//! 3. The remaining 3 bridges per bridge row are **leaf bridges**: we keep
//!    only their "up" edge (to the row above) and drop their "down" edge.
//!    This gives `6 × 3 = 18` dropped edges, exactly the number of
//!    independent cycles of the heavy-hex graph.
//!
//! The resulting spanning tree has `96 + 6*5 = 126` edges and the root is
//! qubit 0 (the top-left corner).
//!
//! This policy is deterministic and purely local, so the golden JSON file
//! regenerates byte-identically from this constructor on every build.
//!
//! [`FakeSherbrooke`]: https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/fake-provider-fake-sherbrooke

use super::{canon, Edge, Tree};

/// Row ranges `[start, end)` of Eagle 127q, top to bottom.
///
/// Row 0 and row 6 have 14 qubits; the five interior rows have 15 qubits each.
pub const EAGLE_ROWS: [(usize, usize); 7] = [
    (0, 14),    // row 0: 14 qubits (top)
    (18, 33),   // row 2: 15 qubits
    (37, 52),   // row 4: 15 qubits
    (56, 71),   // row 6: 15 qubits
    (75, 90),   // row 8: 15 qubits
    (94, 109),  // row 10: 15 qubits
    (113, 127), // row 12: 14 qubits (bottom)
];

/// Bridge qubit `(bridge_qubit, up_neighbour, down_neighbour)` triples.
///
/// Every bridge qubit has degree 2 in the heavy-hex graph: one edge to the
/// row above (`up`) and one to the row below (`down`).
pub const EAGLE_BRIDGES: [(usize, usize, usize); 24] = [
    // bridge row 0→2 (qubits 14..18)
    (14, 0, 18),
    (15, 4, 22),
    (16, 8, 26),
    (17, 12, 30),
    // bridge row 2→4 (qubits 33..37)
    (33, 20, 39),
    (34, 24, 43),
    (35, 28, 47),
    (36, 32, 51),
    // bridge row 4→6 (qubits 52..56)
    (52, 37, 56),
    (53, 41, 60),
    (54, 45, 64),
    (55, 49, 68),
    // bridge row 6→8 (qubits 71..75)
    (71, 58, 77),
    (72, 62, 81),
    (73, 66, 85),
    (74, 70, 89),
    // bridge row 8→10 (qubits 90..94)
    (90, 75, 94),
    (91, 79, 98),
    (92, 83, 102),
    (93, 87, 106),
    // bridge row 10→12 (qubits 109..113)
    (109, 96, 114),
    (110, 100, 118),
    (111, 104, 122),
    (112, 108, 126),
];

/// Leftmost bridge qubit in each bridge row — these are kept as
/// **through-bridges**: both the up and down edges go into the spanning tree.
pub const EAGLE_THROUGH_BRIDGES: [usize; 6] = [14, 33, 52, 71, 90, 109];

/// Full undirected coupling graph of Eagle 127q (144 edges).
///
/// Edges are canonicalised with `a < b` but are returned in the order
/// `horizontal (row 0 left-to-right, row 2 left-to-right, ...) then
/// vertical (bridge-row 0→2 top-to-bottom, ...)` to make debug prints
/// legible.
pub fn eagle_127_coupling_edges() -> Vec<Edge> {
    let mut edges = Vec::with_capacity(144);
    // horizontal row edges
    for (start, end) in EAGLE_ROWS {
        for q in start..end - 1 {
            edges.push(canon(q, q + 1));
        }
    }
    // vertical bridge edges
    for (bridge, up, down) in EAGLE_BRIDGES {
        edges.push(canon(bridge, up));
        edges.push(canon(bridge, down));
    }
    edges
}

/// Build the IBM Eagle 127q `Tree` with the row-major heavy-path spanning
/// tree (see module docs for the policy).
pub fn ibm_eagle_127() -> Tree {
    let n_qubits = 127;

    // 1. Tree edges: every horizontal edge plus, per bridge row,
    //    - for the through-bridge: both the up and down edges,
    //    - for each leaf bridge: only the up edge.
    let mut tree_edges: Vec<Edge> = Vec::with_capacity(126);

    for (start, end) in EAGLE_ROWS {
        for q in start..end - 1 {
            tree_edges.push(canon(q, q + 1));
        }
    }
    for (bridge, up, down) in EAGLE_BRIDGES {
        if EAGLE_THROUGH_BRIDGES.contains(&bridge) {
            tree_edges.push(canon(bridge, up));
            tree_edges.push(canon(bridge, down));
        } else {
            tree_edges.push(canon(bridge, up));
        }
    }
    tree_edges.sort_unstable();
    tree_edges.dedup();
    debug_assert_eq!(tree_edges.len(), 126, "spanning tree must have N-1 edges");

    // 2. Non-tree edges: everything in the coupling graph minus the tree edges.
    let all_edges = eagle_127_coupling_edges();
    let tree_set: std::collections::BTreeSet<Edge> = tree_edges.iter().copied().collect();
    let mut non_tree_edges: Vec<Edge> = all_edges
        .into_iter()
        .filter(|e| !tree_set.contains(e))
        .collect();
    non_tree_edges.sort_unstable();
    non_tree_edges.dedup();
    debug_assert_eq!(
        non_tree_edges.len(),
        18,
        "Eagle has 18 independent cycles, so 18 non-tree edges"
    );

    // 3. Heavy paths: the 7 horizontal rows, top to bottom, left to right.
    let heavy_paths: Vec<Vec<usize>> = EAGLE_ROWS
        .iter()
        .map(|&(start, end)| (start..end).collect())
        .collect();

    // 4. Bridge qubits are not yet covered by the heavy paths — they need to
    //    be appended as singleton paths (or as extensions of the row they
    //    hang off). We represent them as **one-qubit heavy paths** so that
    //    they appear in the partition but do not introduce any path-internal
    //    edges. This keeps `heavy_paths` semantically clean: each inner Vec
    //    is a maximal linear run inside the tree whose *interior* nodes have
    //    degree ≤ 2 in the tree, and a single qubit trivially satisfies that.
    let mut heavy_paths = heavy_paths;
    for (bridge, _, _) in EAGLE_BRIDGES {
        heavy_paths.push(vec![bridge]);
    }
    debug_assert_eq!(
        heavy_paths.iter().map(|p| p.len()).sum::<usize>(),
        n_qubits
    );

    // Root at qubit 0 — the top-left corner of row 0.
    Tree::from_edges(
        "ibm_eagle_127",
        n_qubits,
        0,
        &tree_edges,
        &non_tree_edges,
        heavy_paths,
    )
    .expect("Eagle 127q spanning tree is known-good by construction")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::Topology;

    #[test]
    fn eagle_has_127_qubits() {
        let t = ibm_eagle_127();
        assert_eq!(t.n_qubits(), 127);
    }

    #[test]
    fn eagle_coupling_graph_has_144_edges() {
        let edges = eagle_127_coupling_edges();
        let unique: std::collections::BTreeSet<_> = edges.iter().collect();
        assert_eq!(edges.len(), 144, "raw edge list has 144 entries");
        assert_eq!(unique.len(), 144, "all entries are unique");
    }

    #[test]
    fn eagle_spanning_tree_has_126_edges() {
        let t = ibm_eagle_127();
        assert_eq!(t.tree_edges().len(), 126);
    }

    #[test]
    fn eagle_has_exactly_18_non_tree_edges() {
        let t = ibm_eagle_127();
        assert_eq!(t.non_tree_edges().len(), 18);
    }

    #[test]
    fn eagle_has_7_row_heavy_paths_plus_24_bridge_singletons() {
        let t = ibm_eagle_127();
        assert_eq!(t.heavy_paths().len(), 7 + 24);
        let row_lengths: Vec<usize> =
            t.heavy_paths().iter().take(7).map(|p| p.len()).collect();
        assert_eq!(row_lengths, vec![14, 15, 15, 15, 15, 15, 14]);
        for path in t.heavy_paths().iter().skip(7) {
            assert_eq!(path.len(), 1);
        }
    }

    #[test]
    fn every_coupling_edge_is_either_tree_or_non_tree() {
        let t = ibm_eagle_127();
        let all: std::collections::BTreeSet<Edge> =
            eagle_127_coupling_edges().into_iter().collect();
        let mut accounted: std::collections::BTreeSet<Edge> = t.tree_edges().into_iter().collect();
        for e in t.non_tree_edges() {
            accounted.insert(*e);
        }
        assert_eq!(accounted, all);
    }

    #[test]
    fn through_bridges_have_both_vertical_edges_in_tree() {
        let t = ibm_eagle_127();
        let tree_set: std::collections::BTreeSet<Edge> = t.tree_edges().into_iter().collect();
        for (bridge, up, down) in EAGLE_BRIDGES {
            let is_through = EAGLE_THROUGH_BRIDGES.contains(&bridge);
            let up_in = tree_set.contains(&canon(bridge, up));
            let down_in = tree_set.contains(&canon(bridge, down));
            if is_through {
                assert!(up_in && down_in, "through-bridge {bridge} must keep both edges");
            } else {
                assert!(up_in && !down_in, "leaf bridge {bridge} must keep up only");
            }
        }
    }

    #[test]
    fn root_is_qubit_0_and_deg1_qubits_are_leaves() {
        let t = ibm_eagle_127();
        assert_eq!(t.root(), 0);
        // Qubits 13 and 113 have degree 1 in the heavy-hex graph (end of the
        // short rows) and must also be leaves of the spanning tree.
        assert!(t.children()[13].is_empty());
        assert!(t.children()[113].is_empty());
    }

    #[test]
    fn eagle_tree_round_trips_through_json() {
        let t = ibm_eagle_127();
        let j = t.to_json().unwrap();
        let back = Tree::from_json(&j).unwrap();
        assert_eq!(back.name, t.name);
        assert_eq!(back.n_qubits, t.n_qubits);
        assert_eq!(back.parent, t.parent);
        assert_eq!(back.children, t.children);
        assert_eq!(back.non_tree_edges, t.non_tree_edges);
        assert_eq!(back.heavy_paths, t.heavy_paths);
    }
}
