//! Physical-topology abstractions for the tensor-network backend.
//!
//! This module is the foundation of Track D (see `ROADMAP.md`). It generalises
//! the simulator beyond the 1D chain that `mps.rs` assumes, so that Huoma can
//! represent the heavy-hex graphs shipped by IBM Eagle/Heron and the 2D grids
//! shipped by IQM.
//!
//! # Structure
//!
//! A [`Topology`] exposes:
//!
//! - a qubit count,
//! - an **oriented spanning tree** (parent pointers, children lists, root),
//! - the **heavy-path decomposition** of that tree — the sequence of maximal
//!   linear paths whose internal nodes have degree ≤ 2 in the tree. For
//!   heavy-hex these are exactly the horizontal rows of the physical lattice,
//!   which keeps as many two-qubit gates as possible "along a path" rather
//!   than "around a corner".
//! - the set of original coupling edges that were **dropped** to turn the
//!   graph into a tree. On Eagle 127q this is exactly 18 edges (one per
//!   independent cycle of the heavy-hex lattice). Track D.4 will handle these
//!   via SVD-mediated non-adjacent gates; D.3 only records them.
//!
//! # Determinism
//!
//! The concrete [`Tree`] is `Serialize + Deserialize` with a canonical,
//! byte-stable JSON representation. `ibm_eagle_127()` is pinned against a
//! golden file in `tests/golden/ibm_eagle_127.json` so any change to the
//! spanning-tree construction is caught by CI.

pub mod heavy_hex;

use serde::{Deserialize, Serialize};

pub use heavy_hex::ibm_eagle_127;

/// An undirected edge of the physical coupling graph, stored with `a < b`
/// so that equal edges compare equal.
pub type Edge = (usize, usize);

fn canon(a: usize, b: usize) -> Edge {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

/// A tensor-network topology: an oriented spanning tree over `n_qubits`
/// qubits, plus the heavy-hex / grid edges that the tree had to drop.
///
/// Invariants (enforced by [`Tree::validate`], checked on every
/// constructor and on deserialisation):
///
/// - `parent.len() == children.len() == n_qubits`
/// - exactly one node has `parent == None` (the root)
/// - the parent relation is acyclic and reaches every node from the root
/// - `tree_edges().count() == n_qubits - 1`
/// - each non-tree edge connects two qubits both in `0..n_qubits`
/// - heavy paths partition every qubit exactly once
/// - heavy-path internal edges are all tree edges
pub trait Topology {
    /// Number of physical qubits.
    fn n_qubits(&self) -> usize;

    /// Root of the spanning tree (the node with `parent == None`).
    fn root(&self) -> usize;

    /// Parent pointer array. `parent[root] == None`; every other entry is
    /// `Some(p)` where `p` is strictly closer to the root.
    fn parent(&self) -> &[Option<usize>];

    /// Children lists. `children[p]` contains every `c` with `parent[c] == Some(p)`,
    /// in a deterministic order (sorted ascending by qubit index).
    fn children(&self) -> &[Vec<usize>];

    /// Undirected tree edges, sorted lexicographically with `a < b`.
    /// Length = `n_qubits - 1`.
    fn tree_edges(&self) -> Vec<Edge> {
        let mut es: Vec<Edge> = (0..self.n_qubits())
            .filter_map(|c| self.parent()[c].map(|p| canon(p, c)))
            .collect();
        es.sort_unstable();
        es
    }

    /// Coupling-graph edges that were dropped to form the spanning tree,
    /// sorted lexicographically. On Eagle 127q this has exactly 18 entries.
    fn non_tree_edges(&self) -> &[Edge];

    /// Heavy-path decomposition of the spanning tree. Each inner `Vec` is an
    /// ordered list of qubit indices along one path, consecutive entries are
    /// tree edges. Paths partition all `n_qubits` qubits.
    fn heavy_paths(&self) -> &[Vec<usize>];
}

/// Concrete, serialisable implementation of [`Topology`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tree {
    /// Human-readable topology name (e.g. `"ibm_eagle_127"`).
    pub name: String,
    pub n_qubits: usize,
    pub root: usize,
    pub parent: Vec<Option<usize>>,
    pub children: Vec<Vec<usize>>,
    pub non_tree_edges: Vec<Edge>,
    pub heavy_paths: Vec<Vec<usize>>,
}

/// Errors raised by topology construction and validation.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum TopologyError {
    #[error("shape mismatch: {0}")]
    Shape(String),
    #[error("multiple roots found: {0:?}")]
    MultipleRoots(Vec<usize>),
    #[error("no root (every node has a parent)")]
    NoRoot,
    #[error("cycle detected at node {0}")]
    Cycle(usize),
    #[error("node {0} unreachable from root")]
    Unreachable(usize),
    #[error("qubit index {0} out of range (n_qubits = {1})")]
    OutOfRange(usize, usize),
    #[error("heavy paths do not partition all qubits: missing={missing:?} duplicates={duplicates:?}")]
    HeavyPathPartition {
        missing: Vec<usize>,
        duplicates: Vec<usize>,
    },
    #[error("heavy-path edge ({0},{1}) is not a tree edge")]
    HeavyPathEdgeNotInTree(usize, usize),
    #[error("non-tree edge ({0},{1}) duplicates a tree edge")]
    NonTreeEdgeIsTreeEdge(usize, usize),
    #[error("json: {0}")]
    Json(String),
}

impl Tree {
    /// Build a `Tree` from an undirected edge list + a precomputed heavy-path
    /// decomposition, rooted at `root`. Performs full invariant validation.
    pub fn from_edges(
        name: impl Into<String>,
        n_qubits: usize,
        root: usize,
        tree_edges: &[Edge],
        non_tree_edges: &[Edge],
        heavy_paths: Vec<Vec<usize>>,
    ) -> Result<Self, TopologyError> {
        if root >= n_qubits {
            return Err(TopologyError::OutOfRange(root, n_qubits));
        }
        if tree_edges.len() + 1 != n_qubits {
            return Err(TopologyError::Shape(format!(
                "expected {} tree edges for {} qubits, got {}",
                n_qubits - 1,
                n_qubits,
                tree_edges.len()
            )));
        }

        // Build undirected adjacency from the tree edges.
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n_qubits];
        let mut tree_edge_set: std::collections::BTreeSet<Edge> = std::collections::BTreeSet::new();
        for &(a, b) in tree_edges {
            if a >= n_qubits {
                return Err(TopologyError::OutOfRange(a, n_qubits));
            }
            if b >= n_qubits {
                return Err(TopologyError::OutOfRange(b, n_qubits));
            }
            adj[a].push(b);
            adj[b].push(a);
            tree_edge_set.insert(canon(a, b));
        }
        for list in &mut adj {
            list.sort_unstable();
            list.dedup();
        }

        // BFS from root to assign parent/children and catch cycles + unreachable nodes.
        let mut parent: Vec<Option<usize>> = vec![None; n_qubits];
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); n_qubits];
        let mut seen = vec![false; n_qubits];
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(root);
        seen[root] = true;
        while let Some(u) = queue.pop_front() {
            for &v in &adj[u] {
                if !seen[v] {
                    seen[v] = true;
                    parent[v] = Some(u);
                    children[u].push(v);
                    queue.push_back(v);
                } else if parent[u] != Some(v) {
                    // Back-edge in an undirected BFS from a tree means the graph
                    // actually has a cycle (not a tree). We catch the case of
                    // the caller passing more edges than a tree should have.
                    if parent[u] == Some(v) {
                        continue;
                    }
                    // `seen[v]` and `v` is not our parent -> v lies on a different
                    // branch, so the edge closes a cycle.
                    if v != u {
                        return Err(TopologyError::Cycle(u));
                    }
                }
            }
        }
        for (q, s) in seen.iter().enumerate() {
            if !s {
                return Err(TopologyError::Unreachable(q));
            }
        }
        for list in &mut children {
            list.sort_unstable();
        }

        // Sort non-tree edges canonically and reject any that duplicate a tree edge.
        let mut nte: Vec<Edge> = non_tree_edges.iter().map(|&(a, b)| canon(a, b)).collect();
        nte.sort_unstable();
        nte.dedup();
        for &(a, b) in &nte {
            if a >= n_qubits {
                return Err(TopologyError::OutOfRange(a, n_qubits));
            }
            if b >= n_qubits {
                return Err(TopologyError::OutOfRange(b, n_qubits));
            }
            if tree_edge_set.contains(&(a, b)) {
                return Err(TopologyError::NonTreeEdgeIsTreeEdge(a, b));
            }
        }

        // Verify heavy paths partition all qubits and use only tree edges.
        let mut coverage = vec![0u32; n_qubits];
        for path in &heavy_paths {
            for &q in path {
                if q >= n_qubits {
                    return Err(TopologyError::OutOfRange(q, n_qubits));
                }
                coverage[q] += 1;
            }
            for w in path.windows(2) {
                let (a, b) = (w[0], w[1]);
                if !tree_edge_set.contains(&canon(a, b)) {
                    return Err(TopologyError::HeavyPathEdgeNotInTree(a, b));
                }
            }
        }
        let missing: Vec<usize> = coverage
            .iter()
            .enumerate()
            .filter_map(|(q, &c)| if c == 0 { Some(q) } else { None })
            .collect();
        let duplicates: Vec<usize> = coverage
            .iter()
            .enumerate()
            .filter_map(|(q, &c)| if c > 1 { Some(q) } else { None })
            .collect();
        if !missing.is_empty() || !duplicates.is_empty() {
            return Err(TopologyError::HeavyPathPartition {
                missing,
                duplicates,
            });
        }

        Ok(Tree {
            name: name.into(),
            n_qubits,
            root,
            parent,
            children,
            non_tree_edges: nte,
            heavy_paths,
        })
    }

    /// Re-run the full invariant check. Useful after hand-constructing a
    /// `Tree` or after `from_json`.
    pub fn validate(&self) -> Result<(), TopologyError> {
        let tree_edges = self.tree_edges();
        // Delegate by rebuilding, which performs the same checks.
        let rebuilt = Tree::from_edges(
            self.name.clone(),
            self.n_qubits,
            self.root,
            &tree_edges,
            &self.non_tree_edges,
            self.heavy_paths.clone(),
        )?;
        if rebuilt.parent != self.parent || rebuilt.children != self.children {
            return Err(TopologyError::Shape(
                "parent/children inconsistent with tree_edges".into(),
            ));
        }
        Ok(())
    }

    /// Canonical JSON representation. Stable across runs: keys are in the
    /// order declared above, `heavy_paths` are in construction order,
    /// `non_tree_edges` and tree edges are lexicographically sorted.
    pub fn to_json(&self) -> Result<String, TopologyError> {
        serde_json::to_string_pretty(self).map_err(|e| TopologyError::Json(e.to_string()))
    }

    /// Parse from the canonical JSON and revalidate every invariant.
    pub fn from_json(s: &str) -> Result<Self, TopologyError> {
        let tree: Tree = serde_json::from_str(s).map_err(|e| TopologyError::Json(e.to_string()))?;
        tree.validate()?;
        Ok(tree)
    }
}

impl Topology for Tree {
    fn n_qubits(&self) -> usize {
        self.n_qubits
    }
    fn root(&self) -> usize {
        self.root
    }
    fn parent(&self) -> &[Option<usize>] {
        &self.parent
    }
    fn children(&self) -> &[Vec<usize>] {
        &self.children
    }
    fn non_tree_edges(&self) -> &[Edge] {
        &self.non_tree_edges
    }
    fn heavy_paths(&self) -> &[Vec<usize>] {
        &self.heavy_paths
    }
}

// -----------------------------------------------------------------------------
// Unit tests — tiny hand-built topologies for the trait, heavy-hex specifics
// live in tests/topology_eagle.rs and src/topology/heavy_hex.rs.
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A 5-node linear chain 0-1-2-3-4. Tree edges are all four bonds.
    fn chain5() -> Tree {
        Tree::from_edges(
            "chain5",
            5,
            0,
            &[(0, 1), (1, 2), (2, 3), (3, 4)],
            &[],
            vec![vec![0, 1, 2, 3, 4]],
        )
        .unwrap()
    }

    #[test]
    fn chain_has_expected_parents_and_children() {
        let t = chain5();
        assert_eq!(t.n_qubits(), 5);
        assert_eq!(t.root(), 0);
        assert_eq!(t.parent()[0], None);
        assert_eq!(t.parent()[1], Some(0));
        assert_eq!(t.parent()[4], Some(3));
        assert_eq!(t.children()[0], vec![1]);
        assert_eq!(t.children()[4], Vec::<usize>::new());
    }

    #[test]
    fn chain_tree_edges_sorted() {
        let t = chain5();
        assert_eq!(t.tree_edges(), vec![(0, 1), (1, 2), (2, 3), (3, 4)]);
    }

    #[test]
    fn chain_heavy_path_covers_everything() {
        let t = chain5();
        assert_eq!(t.heavy_paths(), &[vec![0, 1, 2, 3, 4]]);
    }

    #[test]
    fn wrong_edge_count_is_rejected() {
        let err = Tree::from_edges("bad", 5, 0, &[(0, 1), (1, 2)], &[], vec![vec![0, 1, 2]])
            .unwrap_err();
        assert!(matches!(err, TopologyError::Shape(_)));
    }

    #[test]
    fn disconnected_tree_is_rejected() {
        // 5 nodes, 4 edges — but 0-1, 1-2 is one component and 3-4 is another,
        // with a duplicate (0,1) padding the count. `from_edges` dedups, so we
        // instead construct 0-1, 0-1 (dup), 2-3, 3-4 which still yields a
        // disconnected graph from the root's perspective after dedup.
        let err = Tree::from_edges(
            "bad",
            5,
            0,
            &[(0, 1), (0, 1), (2, 3), (3, 4)],
            &[],
            vec![vec![0, 1], vec![2, 3, 4]],
        );
        // Either a shape error (duplicate -> fewer than N-1 unique tree edges
        // after adjacency dedup) or an unreachable-node error is acceptable.
        assert!(err.is_err());
    }

    #[test]
    fn heavy_path_must_use_tree_edges() {
        let err = Tree::from_edges(
            "bad",
            4,
            0,
            &[(0, 1), (1, 2), (2, 3)],
            &[],
            vec![vec![0, 2, 3]], // (0,2) is not a tree edge
        )
        .unwrap_err();
        assert!(matches!(err, TopologyError::HeavyPathEdgeNotInTree(0, 2)));
    }

    #[test]
    fn heavy_path_must_partition_all_qubits() {
        let err = Tree::from_edges(
            "bad",
            4,
            0,
            &[(0, 1), (1, 2), (2, 3)],
            &[],
            vec![vec![0, 1, 2]], // missing qubit 3
        )
        .unwrap_err();
        match err {
            TopologyError::HeavyPathPartition { missing, .. } => {
                assert_eq!(missing, vec![3]);
            }
            _ => panic!("expected HeavyPathPartition, got {err:?}"),
        }
    }

    #[test]
    fn non_tree_edges_cannot_duplicate_tree_edges() {
        let err = Tree::from_edges(
            "bad",
            4,
            0,
            &[(0, 1), (1, 2), (2, 3)],
            &[(0, 1)],
            vec![vec![0, 1, 2, 3]],
        )
        .unwrap_err();
        assert!(matches!(err, TopologyError::NonTreeEdgeIsTreeEdge(0, 1)));
    }

    #[test]
    fn chain_round_trips_through_json() {
        let t = chain5();
        let j = t.to_json().unwrap();
        let back = Tree::from_json(&j).unwrap();
        assert_eq!(back.n_qubits, t.n_qubits);
        assert_eq!(back.parent, t.parent);
        assert_eq!(back.heavy_paths, t.heavy_paths);
    }
}
