//! IBM Eagle 127q heavy-hex layout for the TTN backend.
//!
//! Provides the hard-coded coupling graph of the 127-qubit IBM Eagle r3
//! processor (as shipped by the `FakeSherbrooke` `conf_sherbrooke.json`
//! fixture in `qiskit-ibm-runtime`) together with a deterministic spanning
//! tree that the [`crate::ttn::Ttn`] simulator can consume.
//!
//! # Graph facts
//!
//! - **127 qubits** arranged in 7 horizontal rows of lengths
//!   `[14, 15, 15, 15, 15, 15, 14]`, with 6 bridge rows of 4 bridge qubits
//!   each between them.
//! - **144 undirected coupling edges** (96 horizontal inside the rows plus
//!   48 vertical across the bridge rows).
//! - **18 independent cycles** (|E| − |V| + 1 = 18), one per inner hexagonal
//!   plaquette.
//!
//! # Spanning-tree policy
//!
//! The layout uses a **row-major heavy-path spanning tree**:
//!
//! 1. Every horizontal row edge is a tree edge — 96 edges, one heavy path
//!    per row.
//! 2. In each of the 6 bridge rows the **leftmost** bridge qubit
//!    `[14, 33, 52, 71, 90, 109]` is the **through-bridge**: both of its
//!    vertical edges (up + down) are tree edges, so it becomes an interior
//!    tree node that links the row above to the row below.
//! 3. The remaining 3 bridges per bridge row keep only their "up" edge (to
//!    the row above) and drop the "down" edge. That is `6 × 3 = 18` dropped
//!    edges, exactly matching the independent-cycle count.
//!
//! Result: a valid spanning tree with `96 + 6·5 = 126` edges, a fixed set
//! of 18 `non_tree_edges`, and root qubit 0 in the top-left corner. The
//! 18 non-tree edges are handled at gate time by the swap-network path
//! through the tree (`TRACK_D_DESIGN.md` § "Heavy-hex mapping").
//!
//! This policy is deterministic and local, so the byte-stable golden file
//! at `tests/golden/ibm_eagle_127.json` regenerates identically on every
//! build and regression-tests any accidental drift in the constructor.
//!
//! # Example
//!
//! ```no_run
//! use huoma::ttn::{HeavyHexLayout, Topology, Ttn};
//!
//! let layout = HeavyHexLayout::ibm_eagle_127();
//! assert_eq!(layout.tree().n_qubits(), 127);
//! assert_eq!(layout.tree().n_edges(), 126);
//! assert_eq!(layout.non_tree_edges().len(), 18);
//! assert_eq!(layout.heavy_paths().len(), 7 + 24);
//!
//! // The spanning tree is a normal `Topology` the TTN can consume.
//! let ttn = Ttn::new(layout.tree().clone());
//! # let _ = ttn;
//! ```

use serde::{Deserialize, Serialize};

use super::topology::{Edge, Topology};

/// Row ranges `[start, end)` of Eagle 127q, top to bottom.
///
/// Row 0 and row 6 have 14 qubits; the five interior rows have 15 each.
pub const EAGLE_ROWS: [(usize, usize); 7] = [
    (0, 14),    // row 0: 14 qubits (top)
    (18, 33),   // row 2: 15 qubits
    (37, 52),   // row 4: 15 qubits
    (56, 71),   // row 6: 15 qubits
    (75, 90),   // row 8: 15 qubits
    (94, 109),  // row 10: 15 qubits
    (113, 127), // row 12: 14 qubits (bottom)
];

/// Bridge qubits as `(bridge_qubit, up_neighbour, down_neighbour)` triples.
///
/// Every bridge qubit has degree 2 in the heavy-hex graph: one edge to the
/// row above and one to the row below.
pub const EAGLE_BRIDGES: [(usize, usize, usize); 24] = [
    // bridge row 0 → 2 (qubits 14..18)
    (14, 0, 18),
    (15, 4, 22),
    (16, 8, 26),
    (17, 12, 30),
    // bridge row 2 → 4 (qubits 33..37)
    (33, 20, 39),
    (34, 24, 43),
    (35, 28, 47),
    (36, 32, 51),
    // bridge row 4 → 6 (qubits 52..56)
    (52, 37, 56),
    (53, 41, 60),
    (54, 45, 64),
    (55, 49, 68),
    // bridge row 6 → 8 (qubits 71..75)
    (71, 58, 77),
    (72, 62, 81),
    (73, 66, 85),
    (74, 70, 89),
    // bridge row 8 → 10 (qubits 90..94)
    (90, 75, 94),
    (91, 79, 98),
    (92, 83, 102),
    (93, 87, 106),
    // bridge row 10 → 12 (qubits 109..113)
    (109, 96, 114),
    (110, 100, 118),
    (111, 104, 122),
    (112, 108, 126),
];

/// Leftmost bridge qubit in each bridge row — kept as a **through-bridge**:
/// both the up and down edges enter the spanning tree, so the bridge is an
/// interior tree node.
pub const EAGLE_THROUGH_BRIDGES: [usize; 6] = [14, 33, 52, 71, 90, 109];

/// Canonical `[usize; 2]` encoding of an undirected edge (sorted so that
/// `a ≤ b`). Used inside [`HeavyHexLayout`] to make the serde shape small
/// and byte-stable; converted to the ttn [`Edge`] type at the public API.
pub type SerdeEdge = [usize; 2];

fn canon(a: usize, b: usize) -> SerdeEdge {
    if a <= b {
        [a, b]
    } else {
        [b, a]
    }
}

fn edge_to_pair(e: Edge) -> SerdeEdge {
    canon(e.a, e.b)
}

/// Full undirected coupling graph of Eagle 127q (144 edges).
///
/// Edges are returned canonicalised with `a < b`, in a human-ordered
/// sequence: horizontal edges row-by-row left-to-right first, then vertical
/// bridge edges top-to-bottom. The order is not load-bearing — the golden
/// file stores the sorted form — but it makes debug prints legible.
#[must_use]
pub fn eagle_127_coupling_edges() -> Vec<Edge> {
    let mut edges = Vec::with_capacity(144);
    // horizontal row edges
    for (start, end) in EAGLE_ROWS {
        for q in start..end - 1 {
            let [a, b] = canon(q, q + 1);
            edges.push(Edge { a, b });
        }
    }
    // vertical bridge edges (both up and down per bridge)
    for (bridge, up, down) in EAGLE_BRIDGES {
        let [a, b] = canon(bridge, up);
        edges.push(Edge { a, b });
        let [a, b] = canon(bridge, down);
        edges.push(Edge { a, b });
    }
    edges
}

/// Hardware layout for a heavy-hex backend.
///
/// Owns the spanning-tree [`Topology`] that the TTN simulator consumes, plus
/// the two pieces of structural metadata the `Topology` itself deliberately
/// does not carry:
///
/// - `non_tree_edges` — the original heavy-hex edges that had to be dropped
///   to turn the graph into a tree. Gate application on these pairs goes
///   through the swap-network path in [`crate::ttn::Ttn`].
/// - `heavy_paths` — the heavy-path decomposition of the spanning tree. For
///   Eagle 127q this is 7 horizontal rows plus 24 bridge-qubit singletons.
///   Used by the tree-generalised `chi_allocation_sinc` that lands in D.4.
///
/// The layout is `serde`-serialisable to a canonical, byte-stable JSON and
/// is pinned against `tests/golden/ibm_eagle_127.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeavyHexLayout {
    /// Human-readable layout name (e.g. `"ibm_eagle_127"`).
    pub name: String,
    /// Number of physical qubits.
    pub n_qubits: usize,
    /// Spanning-tree edges in canonical `[min, max]` form, sorted
    /// lexicographically. Length = `n_qubits - 1`.
    pub tree_edges: Vec<SerdeEdge>,
    /// Heavy-hex coupling edges dropped to form the spanning tree, in
    /// canonical form, sorted lexicographically.
    pub non_tree_edges: Vec<SerdeEdge>,
    /// Heavy-path decomposition: ordered qubit sequences along each path.
    /// Consecutive entries in an inner vector are tree-adjacent. Paths
    /// partition every qubit exactly once.
    pub heavy_paths: Vec<Vec<usize>>,

    /// Lazily rebuilt from `tree_edges` on first access; `Topology` itself
    /// is not `serde` (programmer-error-on-invalid panics are not serde).
    #[serde(skip)]
    tree: std::cell::OnceCell<Topology>,
}

impl HeavyHexLayout {
    /// Build the IBM Eagle 127q `HeavyHexLayout` via the row-major
    /// heavy-path spanning-tree policy (see module docs).
    #[must_use]
    pub fn ibm_eagle_127() -> Self {
        let n_qubits = 127;

        // Step 1: enumerate spanning-tree edges.
        let mut tree_edges: Vec<SerdeEdge> = Vec::with_capacity(126);
        for (start, end) in EAGLE_ROWS {
            for q in start..end - 1 {
                tree_edges.push(canon(q, q + 1));
            }
        }
        for (bridge, up, down) in EAGLE_BRIDGES {
            tree_edges.push(canon(bridge, up));
            if EAGLE_THROUGH_BRIDGES.contains(&bridge) {
                tree_edges.push(canon(bridge, down));
            }
        }
        tree_edges.sort_unstable();
        tree_edges.dedup();
        debug_assert_eq!(
            tree_edges.len(),
            126,
            "spanning tree on 127 qubits must have 126 edges"
        );

        // Step 2: non-tree edges = full coupling graph minus tree edges.
        let tree_set: std::collections::BTreeSet<SerdeEdge> = tree_edges.iter().copied().collect();
        let mut non_tree_edges: Vec<SerdeEdge> = eagle_127_coupling_edges()
            .into_iter()
            .map(edge_to_pair)
            .filter(|e| !tree_set.contains(e))
            .collect();
        non_tree_edges.sort_unstable();
        non_tree_edges.dedup();
        debug_assert_eq!(
            non_tree_edges.len(),
            18,
            "Eagle has |E| - |V| + 1 = 18 independent cycles"
        );

        // Step 3: heavy-path decomposition = 7 rows then bridge singletons.
        let mut heavy_paths: Vec<Vec<usize>> =
            EAGLE_ROWS.iter().map(|&(s, e)| (s..e).collect()).collect();
        for (bridge, _, _) in EAGLE_BRIDGES {
            heavy_paths.push(vec![bridge]);
        }
        debug_assert_eq!(
            heavy_paths.iter().map(Vec::len).sum::<usize>(),
            n_qubits,
            "heavy paths must cover every qubit exactly once"
        );

        Self {
            name: "ibm_eagle_127".to_string(),
            n_qubits,
            tree_edges,
            non_tree_edges,
            heavy_paths,
            tree: std::cell::OnceCell::new(),
        }
    }

    /// Borrow the spanning-tree [`Topology`] the TTN can consume.
    ///
    /// Built lazily on first call from `tree_edges`; cached for the lifetime
    /// of the layout. Panics (via `Topology::from_edges`) if the stored
    /// `tree_edges` are not a valid tree — which can only happen to a
    /// hand-edited or corrupted deserialised layout.
    #[must_use]
    pub fn tree(&self) -> &Topology {
        self.tree.get_or_init(|| {
            let edges: Vec<Edge> = self
                .tree_edges
                .iter()
                .map(|&[a, b]| Edge { a, b })
                .collect();
            Topology::from_edges(self.n_qubits, edges)
        })
    }

    /// Heavy-hex coupling edges that were dropped to form the spanning tree.
    /// Length = 18 for Eagle 127q.
    #[must_use]
    pub fn non_tree_edges(&self) -> &[SerdeEdge] {
        &self.non_tree_edges
    }

    /// Heavy-path decomposition of the spanning tree. 7 row-paths plus
    /// 24 bridge singletons = 31 entries for Eagle 127q.
    #[must_use]
    pub fn heavy_paths(&self) -> &[Vec<usize>] {
        &self.heavy_paths
    }

    /// Number of qubits in the layout.
    #[must_use]
    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// Canonical JSON representation — stable across runs because
    /// `tree_edges` and `non_tree_edges` are lexicographically sorted and
    /// `heavy_paths` is in construction order.
    ///
    /// # Panics
    ///
    /// Panics if the layout fails to serialise. `serde_json::to_string_pretty`
    /// can only fail on custom `Serialize` impls that return errors; the
    /// derived impls on this type cannot.
    #[must_use]
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self)
            .expect("HeavyHexLayout serialisation with derived serde is infallible")
    }

    /// Parse from canonical JSON and force the spanning-tree cache, which
    /// validates every tree invariant in the process.
    ///
    /// # Errors
    ///
    /// Returns the underlying `serde_json` error if the input is not valid
    /// JSON of the expected shape. A structurally invalid layout (bad tree
    /// edges, out-of-range qubits, etc.) panics inside `tree()`, matching
    /// the panic-on-programmer-error discipline of the rest of `ttn`.
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        let layout: Self = serde_json::from_str(s)?;
        // Force tree construction to run all invariant checks.
        let _ = layout.tree();
        Ok(layout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eagle_coupling_graph_has_144_edges() {
        let edges = eagle_127_coupling_edges();
        assert_eq!(edges.len(), 144);
        let set: std::collections::BTreeSet<SerdeEdge> =
            edges.iter().copied().map(edge_to_pair).collect();
        assert_eq!(set.len(), 144, "all 144 edges must be unique");
    }

    #[test]
    fn eagle_has_127_qubits_and_126_tree_edges() {
        let layout = HeavyHexLayout::ibm_eagle_127();
        assert_eq!(layout.n_qubits(), 127);
        assert_eq!(layout.tree_edges.len(), 126);
        assert_eq!(layout.tree().n_qubits(), 127);
        assert_eq!(layout.tree().n_edges(), 126);
    }

    #[test]
    fn eagle_has_exactly_18_non_tree_edges() {
        let layout = HeavyHexLayout::ibm_eagle_127();
        assert_eq!(layout.non_tree_edges().len(), 18);
    }

    #[test]
    fn eagle_tree_plus_non_tree_equals_full_coupling_graph() {
        let layout = HeavyHexLayout::ibm_eagle_127();
        let mut union: std::collections::BTreeSet<SerdeEdge> =
            layout.tree_edges.iter().copied().collect();
        for e in layout.non_tree_edges() {
            assert!(union.insert(*e), "non-tree edge {e:?} duplicates a tree edge");
        }
        let full: std::collections::BTreeSet<SerdeEdge> = eagle_127_coupling_edges()
            .into_iter()
            .map(edge_to_pair)
            .collect();
        assert_eq!(union, full);
    }

    #[test]
    fn eagle_has_7_row_heavy_paths_plus_24_bridge_singletons() {
        let layout = HeavyHexLayout::ibm_eagle_127();
        assert_eq!(layout.heavy_paths().len(), 7 + 24);
        let row_lengths: Vec<usize> =
            layout.heavy_paths().iter().take(7).map(Vec::len).collect();
        assert_eq!(row_lengths, vec![14, 15, 15, 15, 15, 15, 14]);
        for path in layout.heavy_paths().iter().skip(7) {
            assert_eq!(path.len(), 1);
        }
    }

    #[test]
    fn eagle_heavy_paths_partition_every_qubit_once() {
        let layout = HeavyHexLayout::ibm_eagle_127();
        let mut count = vec![0u32; layout.n_qubits()];
        for path in layout.heavy_paths() {
            for &q in path {
                count[q] += 1;
            }
        }
        assert!(count.iter().all(|&c| c == 1));
    }

    #[test]
    fn through_bridges_have_degree_2_in_spanning_tree() {
        let layout = HeavyHexLayout::ibm_eagle_127();
        for b in EAGLE_THROUGH_BRIDGES {
            assert_eq!(
                layout.tree().degree(b),
                2,
                "through-bridge {b} should be an interior tree node"
            );
        }
    }

    #[test]
    fn leaf_bridges_have_degree_1_in_spanning_tree() {
        let layout = HeavyHexLayout::ibm_eagle_127();
        let throughs: std::collections::BTreeSet<usize> =
            EAGLE_THROUGH_BRIDGES.iter().copied().collect();
        for (bridge, _, _) in EAGLE_BRIDGES {
            if throughs.contains(&bridge) {
                continue;
            }
            assert_eq!(
                layout.tree().degree(bridge),
                1,
                "leaf bridge {bridge} should be a tree leaf"
            );
        }
    }

    #[test]
    fn degree_1_qubits_13_and_113_are_tree_leaves() {
        // Qubits 13 and 113 are the degree-1 ends of rows 0 and 12 in the
        // heavy-hex graph; they must also be leaves of the spanning tree.
        let layout = HeavyHexLayout::ibm_eagle_127();
        assert_eq!(layout.tree().degree(13), 1);
        assert_eq!(layout.tree().degree(113), 1);
    }

    #[test]
    fn json_round_trip_preserves_structure() {
        let layout = HeavyHexLayout::ibm_eagle_127();
        let j = layout.to_json();
        let back = HeavyHexLayout::from_json(&j).unwrap();
        assert_eq!(back.name, layout.name);
        assert_eq!(back.n_qubits, layout.n_qubits);
        assert_eq!(back.tree_edges, layout.tree_edges);
        assert_eq!(back.non_tree_edges, layout.non_tree_edges);
        assert_eq!(back.heavy_paths, layout.heavy_paths);
        assert_eq!(back.tree().n_edges(), 126);
    }

    #[test]
    fn cycle_count_matches_euler_formula() {
        // |E| - |V| + 1 = 144 - 127 + 1 = 18 independent cycles,
        // so the spanning tree must drop exactly 18 edges.
        let layout = HeavyHexLayout::ibm_eagle_127();
        assert_eq!(
            eagle_127_coupling_edges().len() - layout.n_qubits() + 1,
            18
        );
        assert_eq!(layout.non_tree_edges().len(), 18);
    }
}
