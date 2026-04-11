//! Tree topology for TTN — edge set, connectivity, cut partitions, tree paths.
//!
//! This module owns the graph-theoretic layer of the TTN: who is adjacent to
//! whom, what path connects two qubits, and what partition is induced by
//! removing a given edge. The tree invariants are validated once at
//! construction, caches are computed once at construction, and everything
//! downstream (gauge sweeps, two-site contraction, sin(C/2) tree allocator)
//! consumes the cached views.
//!
//! Track D milestone D.2.

/// Stable handle to an edge in a [`Topology`]. The integer is an index into
/// `Topology::edges` and is preserved for the lifetime of the topology, the
/// same way bond indices are preserved across the lifetime of an `Mps`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EdgeId(pub usize);

/// Undirected edge between two qubits in a [`Topology`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Edge {
    pub a: usize,
    pub b: usize,
}

impl Edge {
    /// Return the vertex at the other end of the edge.
    #[must_use]
    pub fn other(&self, from: usize) -> usize {
        if self.a == from {
            self.b
        } else {
            debug_assert_eq!(self.b, from, "from must be an endpoint");
            self.a
        }
    }
}

/// Tree topology over `n_qubits` qubits.
///
/// Invariants enforced at construction:
/// - `edges.len() == n_qubits - 1` (tree, not graph)
/// - Every edge's endpoints are in `0..n_qubits`
/// - No self-loops
/// - No duplicate edges
/// - The graph is connected (→ together with the edge-count invariant this
///   is equivalent to "tree", no cycle check needed separately)
///
/// Caches computed at construction and stable for the lifetime of the
/// topology:
/// - `neighbours[v]`: edges incident on qubit `v`
/// - `cut_partitions[e]`: the (A, B) partition of vertices induced by
///   removing edge `e`, each sorted ascending for determinism
#[derive(Debug, Clone)]
pub struct Topology {
    n_qubits: usize,
    edges: Vec<Edge>,
    neighbours: Vec<Vec<EdgeId>>,
    cut_partitions: Vec<(Vec<usize>, Vec<usize>)>,
}

impl Topology {
    /// Linear-chain topology with edges `(0,1), (1,2), …, (n-2, n-1)`.
    /// This is the degenerate-tree special case used by the D.1 1D regression.
    #[must_use]
    pub fn linear_chain(n_qubits: usize) -> Self {
        assert!(n_qubits >= 1, "Topology::linear_chain requires n_qubits ≥ 1");
        let edges: Vec<Edge> = (0..n_qubits.saturating_sub(1))
            .map(|i| Edge { a: i, b: i + 1 })
            .collect();
        Self::build(n_qubits, edges, true).expect("linear chain is always a valid tree")
    }

    /// General-tree constructor — milestone D.2.
    ///
    /// Validates all tree invariants. Panics with a descriptive message if
    /// any are violated; the panic path is the contract because a malformed
    /// topology is a programmer error, not a runtime condition. If you need
    /// to recover from invalid input, validate upstream.
    ///
    /// Pre-computes [`Self::cut_partition`] for every edge, which costs
    /// O(N²) memory and O(N · |E|) time. For small topologies (N ≤ ~10K)
    /// this is fine; for million-qubit graphs use
    /// [`Self::from_edges_lightweight`] instead.
    #[must_use]
    pub fn from_edges(n_qubits: usize, edges: Vec<Edge>) -> Self {
        Self::build(n_qubits, edges, true).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Lightweight tree constructor — Track F milestone F.0.
    ///
    /// Validates the same tree invariants as [`Self::from_edges`] but
    /// **skips** the O(N²) cut-partition precomputation. Calling
    /// [`Self::cut_partition`] on a lightweight topology will panic;
    /// use [`Self::has_cut_partitions`] to check first.
    ///
    /// All other methods (`neighbours`, `degree`, `path`, `n_qubits`,
    /// `n_edges`, `edges`, `is_linear_chain`) work identically.
    ///
    /// Use this for million-qubit spanning trees where the per-edge
    /// scoring is done via radius-bounded local BFS (see
    /// [`crate::ttn::allocator::edge_sinc_score_local`]) rather than
    /// the full cross-cut sum.
    #[must_use]
    pub fn from_edges_lightweight(n_qubits: usize, edges: Vec<Edge>) -> Self {
        Self::build(n_qubits, edges, false).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Whether this topology has pre-computed cut partitions.
    #[must_use]
    pub fn has_cut_partitions(&self) -> bool {
        !self.cut_partitions.is_empty()
    }

    /// Shared construction path. Separated from `from_edges` so the
    /// linear-chain constructor can assume success without the panic shim.
    /// When `compute_cuts` is false the O(N²) cut-partition computation
    /// is skipped (Track F lightweight mode).
    fn build(n_qubits: usize, edges: Vec<Edge>, compute_cuts: bool) -> Result<Self, String> {
        if n_qubits == 0 {
            return Err("Topology requires n_qubits ≥ 1".to_string());
        }
        // Edge count: a tree on n vertices has exactly n-1 edges.
        if edges.len() + 1 != n_qubits {
            return Err(format!(
                "tree on {} qubits must have exactly {} edges, got {}",
                n_qubits,
                n_qubits - 1,
                edges.len()
            ));
        }
        // Endpoint + self-loop check.
        for (i, e) in edges.iter().enumerate() {
            if e.a >= n_qubits || e.b >= n_qubits {
                return Err(format!(
                    "edge {i} ({}, {}) has endpoint out of range [0, {n_qubits})",
                    e.a, e.b
                ));
            }
            if e.a == e.b {
                return Err(format!("edge {i} is a self-loop on vertex {}", e.a));
            }
        }
        // Duplicate check (unordered). Uses a hash set for O(N) instead
        // of the previous O(N²) nested loop, which was the bottleneck at
        // 1M qubits (233s → should be <1s after this fix).
        {
            let mut seen = std::collections::HashSet::with_capacity(edges.len());
            for (i, e) in edges.iter().enumerate() {
                let canonical = if e.a <= e.b { (e.a, e.b) } else { (e.b, e.a) };
                if !seen.insert(canonical) {
                    return Err(format!(
                        "duplicate edge at index {i}: ({}, {})",
                        e.a, e.b
                    ));
                }
            }
        }

        // Neighbours table.
        let mut neighbours: Vec<Vec<EdgeId>> = vec![Vec::new(); n_qubits];
        for (i, e) in edges.iter().enumerate() {
            neighbours[e.a].push(EdgeId(i));
            neighbours[e.b].push(EdgeId(i));
        }

        // Connectivity via BFS from vertex 0. Combined with |E| = n-1 this
        // rules out cycles: a connected graph with exactly n-1 edges is a
        // tree.
        let mut seen = vec![false; n_qubits];
        let mut queue = std::collections::VecDeque::new();
        seen[0] = true;
        queue.push_back(0usize);
        while let Some(v) = queue.pop_front() {
            for &eid in &neighbours[v] {
                let w = edges[eid.0].other(v);
                if !seen[w] {
                    seen[w] = true;
                    queue.push_back(w);
                }
            }
        }
        if !seen.iter().all(|&s| s) {
            let missing: Vec<_> = seen
                .iter()
                .enumerate()
                .filter_map(|(i, &s)| (!s).then_some(i))
                .collect();
            return Err(format!("disconnected vertices: {missing:?}"));
        }

        // Cut partitions. Removing edge e disconnects the tree into exactly
        // two subtrees (A_e, B_e). Compute each via BFS on the graph with
        // edge e removed, starting from e.a. Everything BFS reaches is A;
        // everything else is B.
        //
        // Skipped in lightweight mode (Track F) — O(N²) memory at large N.
        let cut_partitions = if compute_cuts {
            let mut parts = Vec::with_capacity(edges.len());
            for (e_idx, e) in edges.iter().enumerate() {
                let mut in_a = vec![false; n_qubits];
                in_a[e.a] = true;
                let mut queue = std::collections::VecDeque::new();
                queue.push_back(e.a);
                while let Some(v) = queue.pop_front() {
                    for &eid in &neighbours[v] {
                        if eid.0 == e_idx {
                            continue; // pretend this edge is removed
                        }
                        let w = edges[eid.0].other(v);
                        if !in_a[w] {
                            in_a[w] = true;
                            queue.push_back(w);
                        }
                    }
                }
                let mut a_side: Vec<usize> = (0..n_qubits).filter(|&v| in_a[v]).collect();
                let mut b_side: Vec<usize> = (0..n_qubits).filter(|&v| !in_a[v]).collect();
                a_side.sort_unstable();
                b_side.sort_unstable();
                parts.push((a_side, b_side));
            }
            parts
        } else {
            Vec::new()
        };

        Ok(Self {
            n_qubits,
            edges,
            neighbours,
            cut_partitions,
        })
    }

    #[must_use]
    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    #[must_use]
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    #[must_use]
    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    #[must_use]
    pub fn edge(&self, id: EdgeId) -> Edge {
        self.edges[id.0]
    }

    /// Edges incident on vertex `v`.
    #[must_use]
    pub fn neighbours(&self, v: usize) -> &[EdgeId] {
        &self.neighbours[v]
    }

    /// Degree of vertex `v` (number of incident edges).
    #[must_use]
    pub fn degree(&self, v: usize) -> usize {
        self.neighbours[v].len()
    }

    /// The two sorted vertex sets induced by removing edge `id`. The side
    /// containing `edge(id).a` is returned first.
    ///
    /// # Panics
    ///
    /// Panics if the topology was built with [`Self::from_edges_lightweight`]
    /// (which skips the O(N²) cut-partition precomputation). Check
    /// [`Self::has_cut_partitions`] first if unsure.
    #[must_use]
    pub fn cut_partition(&self, id: EdgeId) -> (&[usize], &[usize]) {
        assert!(
            !self.cut_partitions.is_empty(),
            "cut_partition called on a lightweight topology (built with \
             from_edges_lightweight). Use from_edges for small topologies \
             or edge_sinc_score_local for radius-bounded scoring."
        );
        let (a, b) = &self.cut_partitions[id.0];
        (a, b)
    }

    /// Unique tree path between two vertices, returned as an ordered list of
    /// edges from `from` to `to`. Empty iff `from == to`.
    ///
    /// Computed on demand via BFS with parent tracking. O(n) per call; no
    /// caching because the Tindall-scale call count is bounded by the total
    /// number of gates, not by a nested-loop structure.
    #[must_use]
    pub fn path(&self, from: usize, to: usize) -> Vec<EdgeId> {
        assert!(from < self.n_qubits && to < self.n_qubits, "path: vertex out of range");
        if from == to {
            return Vec::new();
        }
        let mut parent: Vec<Option<(usize, EdgeId)>> = vec![None; self.n_qubits];
        let mut seen = vec![false; self.n_qubits];
        seen[from] = true;
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(from);
        while let Some(v) = queue.pop_front() {
            if v == to {
                break;
            }
            for &eid in &self.neighbours[v] {
                let w = self.edges[eid.0].other(v);
                if !seen[w] {
                    seen[w] = true;
                    parent[w] = Some((v, eid));
                    queue.push_back(w);
                }
            }
        }
        debug_assert!(seen[to], "tree must be connected, so every pair has a path");
        let mut path = Vec::new();
        let mut cur = to;
        while cur != from {
            let (prev, eid) = parent[cur].expect("parent must be set on a reached node");
            path.push(eid);
            cur = prev;
        }
        path.reverse();
        path
    }

    /// True if the topology is a linear chain `0—1—2—…—(n-1)`.
    #[must_use]
    pub fn is_linear_chain(&self) -> bool {
        if self.edges.len() + 1 != self.n_qubits {
            return false;
        }
        self.edges
            .iter()
            .enumerate()
            .all(|(i, e)| (e.a == i && e.b == i + 1) || (e.a == i + 1 && e.b == i))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_chain_basic() {
        let t = Topology::linear_chain(5);
        assert_eq!(t.n_qubits(), 5);
        assert_eq!(t.n_edges(), 4);
        assert!(t.is_linear_chain());
        for (i, e) in t.edges().iter().enumerate() {
            assert_eq!((e.a, e.b), (i, i + 1));
        }
    }

    #[test]
    fn y_junction_construction() {
        // 3 leaves (1, 2, 3) connected to centre 0.
        let edges = vec![
            Edge { a: 0, b: 1 },
            Edge { a: 0, b: 2 },
            Edge { a: 0, b: 3 },
        ];
        let t = Topology::from_edges(4, edges);
        assert_eq!(t.n_qubits(), 4);
        assert_eq!(t.n_edges(), 3);
        assert!(!t.is_linear_chain());
        assert_eq!(t.degree(0), 3);
        assert_eq!(t.degree(1), 1);
        assert_eq!(t.degree(2), 1);
        assert_eq!(t.degree(3), 1);
    }

    #[test]
    fn y_junction_cut_partitions() {
        let edges = vec![
            Edge { a: 0, b: 1 },
            Edge { a: 0, b: 2 },
            Edge { a: 0, b: 3 },
        ];
        let t = Topology::from_edges(4, edges);
        // Removing (0,1) splits into {0,2,3} and {1}.
        let (a, b) = t.cut_partition(EdgeId(0));
        assert_eq!(a, &[0, 2, 3]);
        assert_eq!(b, &[1]);
        // Removing (0,2) splits into {0,1,3} and {2}.
        let (a, b) = t.cut_partition(EdgeId(1));
        assert_eq!(a, &[0, 1, 3]);
        assert_eq!(b, &[2]);
    }

    #[test]
    fn linear_chain_cut_partitions_are_prefixes() {
        let t = Topology::linear_chain(5);
        for b in 0..4 {
            let (left, right) = t.cut_partition(EdgeId(b));
            let expected_left: Vec<usize> = (0..=b).collect();
            let expected_right: Vec<usize> = (b + 1..5).collect();
            assert_eq!(left, expected_left.as_slice(), "bond {b} left");
            assert_eq!(right, expected_right.as_slice(), "bond {b} right");
        }
    }

    #[test]
    fn path_in_linear_chain() {
        let t = Topology::linear_chain(5);
        let p = t.path(0, 3);
        assert_eq!(p, vec![EdgeId(0), EdgeId(1), EdgeId(2)]);
        // Reverse direction.
        let p = t.path(4, 1);
        assert_eq!(p, vec![EdgeId(3), EdgeId(2), EdgeId(1)]);
        // Empty path.
        assert!(t.path(2, 2).is_empty());
    }

    #[test]
    fn path_in_y_junction() {
        // 1—0—2, with a third leaf 3 hanging off 0.
        let edges = vec![
            Edge { a: 0, b: 1 },
            Edge { a: 0, b: 2 },
            Edge { a: 0, b: 3 },
        ];
        let t = Topology::from_edges(4, edges);
        let p = t.path(1, 2);
        assert_eq!(p, vec![EdgeId(0), EdgeId(1)]);
        let p = t.path(1, 3);
        assert_eq!(p, vec![EdgeId(0), EdgeId(2)]);
        let p = t.path(2, 3);
        assert_eq!(p, vec![EdgeId(1), EdgeId(2)]);
    }

    #[test]
    #[should_panic(expected = "disconnected vertices")]
    fn disconnected_rejected() {
        // n=4, 3 edges, but (2,3) isolated from (0,1).
        // This is 3 edges for 4 vertices → edge-count check passes, but the
        // connectivity check must catch it: e.g. 0-1, 0-1, 2-3 triggers the
        // duplicate check first, so we need a case that slips past that.
        // 0-1, 0-1 is a duplicate; instead use 0-1, 2-3, 0-2 (4 edges, wrong).
        // To get a disconnected case with exactly 3 edges on 4 vertices we
        // need a cycle on one side: 0-1, 1-2, 0-2 (triangle) leaves 3 isolated.
        let edges = vec![
            Edge { a: 0, b: 1 },
            Edge { a: 1, b: 2 },
            Edge { a: 0, b: 2 },
        ];
        let _ = Topology::from_edges(4, edges);
    }

    #[test]
    #[should_panic(expected = "must have exactly")]
    fn wrong_edge_count_rejected() {
        let edges = vec![Edge { a: 0, b: 1 }];
        let _ = Topology::from_edges(4, edges);
    }

    #[test]
    #[should_panic(expected = "self-loop")]
    fn self_loop_rejected() {
        let edges = vec![
            Edge { a: 0, b: 0 },
            Edge { a: 0, b: 1 },
            Edge { a: 1, b: 2 },
        ];
        let _ = Topology::from_edges(4, edges);
    }

    #[test]
    #[should_panic(expected = "duplicate edge")]
    fn duplicate_edge_rejected() {
        let edges = vec![
            Edge { a: 0, b: 1 },
            Edge { a: 1, b: 0 },
            Edge { a: 2, b: 3 },
        ];
        let _ = Topology::from_edges(4, edges);
    }

    #[test]
    #[should_panic(expected = "endpoint out of range")]
    fn out_of_range_vertex_rejected() {
        let edges = vec![
            Edge { a: 0, b: 1 },
            Edge { a: 1, b: 2 },
            Edge { a: 2, b: 7 },
        ];
        let _ = Topology::from_edges(4, edges);
    }
}
