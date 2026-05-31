//! Closed orientable surfaces tiled by regular `{p,q}` maps, with
//! periodic boundary conditions — the prerequisite for **bulk**
//! (boundary-free) magnetic spectra. On a hyperbolic lattice the
//! boundary is an O(1) fraction of all sites, so an open finite patch
//! (the OBC `EmbeddedGraph` path in `magnetic.rs`) is edge-mode
//! dominated and not a valid bulk spectrum. A closed surface removes
//! the boundary entirely.
//!
//! # Status (M1)
//!
//! This module starts with the **Euclidean torus** (`{4,4}`, genus 1),
//! the analytically-anchored validation case. Hofstadter's original
//! 1976 calculation *is* the `{4,4}` torus with rational flux `p/q`
//! and the two Bloch phases playing the role of the torus's two
//! Aharonov–Bohm twist holonomies — so this is the gold-standard PBC
//! validation, not a warm-up. Hyperbolic closed surfaces (genus ≥ 2:
//! Klein quartic `{7,3}`, Bolza `{8,8}`) come from verified quotient
//! data (M1b) and are deliberately *not* hand-faked here.
//!
//! # Homology cuts and twist flux
//!
//! A genus-`g` surface has `2g` independent non-contractible cycles.
//! We represent them as **cuts**: each directed edge `u → v` carries
//! an integer `crossing[k] ∈ {−1, 0, +1}` recording how it crosses
//! the `k`-th cut. Threading Aharonov–Bohm flux `θ_k` through cycle
//! `k` (the twist averaging of M2/M3) adds `crossing[k] · θ_k` to that
//! edge's Peierls phase.
//!
//! The load-bearing consistency condition: around **every face** the
//! net crossing of each cut is zero (a contractible loop crosses any
//! cut a net-zero number of times). This is what makes the
//! per-plaquette flux well-defined independent of the twists, and it
//! is checked by `verify` / the acceptance tests.

use crate::ttn::topology::{Edge, Topology};

/// An undirected edge on a closed surface, stored with a canonical
/// orientation `u → v` and annotated with its homology-cut crossings
/// for that orientation.
#[derive(Clone, Debug)]
pub struct SurfaceEdge {
    pub u: usize,
    pub v: usize,
    /// `crossing[k]` ∈ {−1, 0, +1}: signed crossing of the k-th
    /// homology cut when traversing `u → v`. Length `2g`.
    pub crossing: Vec<i32>,
}

/// A regular `{p, q}` tiling of a closed orientable genus-`g` surface.
#[derive(Clone, Debug)]
pub struct ClosedSurface {
    pub p: usize,
    pub q: usize,
    pub genus: usize,
    /// Vertex positions in the fundamental domain (for plotting and
    /// the embedding-chart gauge). Wraparound is encoded in the edge
    /// crossings, not the positions.
    pub positions: Vec<(f64, f64)>,
    pub edges: Vec<SurfaceEdge>,
    /// Faces as ordered vertex cycles (each of length `p`).
    pub faces: Vec<Vec<usize>>,
}

impl ClosedSurface {
    #[must_use]
    pub fn n_vertices(&self) -> usize {
        self.positions.len()
    }

    #[must_use]
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    #[must_use]
    pub fn n_faces(&self) -> usize {
        self.faces.len()
    }

    /// Number of homology cuts (`2g`).
    #[must_use]
    pub fn n_cuts(&self) -> usize {
        2 * self.genus
    }

    /// Euler characteristic `V − E + F`.
    #[must_use]
    pub fn euler_characteristic(&self) -> i64 {
        self.n_vertices() as i64 - self.n_edges() as i64 + self.n_faces() as i64
    }

    /// Per-vertex degree, computed from the edge list.
    #[must_use]
    pub fn degrees(&self) -> Vec<usize> {
        let mut d = vec![0_usize; self.n_vertices()];
        for e in &self.edges {
            d[e.u] += 1;
            d[e.v] += 1;
        }
        d
    }

    /// `L × L` regular `{4,4}` square tiling of the torus (genus 1).
    ///
    /// Vertices on integer coordinates `(i, j)`, `0 ≤ i, j < L`,
    /// indexed `idx = j·L + i`. Each vertex has a "right" edge to
    /// `((i+1) mod L, j)` and an "up" edge to `(i, (j+1) mod L)`;
    /// these wrap at the boundary, and the wrap edges carry the
    /// homology crossings (cut 0 = the x-loop, cut 1 = the y-loop).
    /// Faces are the `L²` unit squares.
    ///
    /// # Panics
    /// If `l < 2`.
    #[must_use]
    pub fn torus_44(l: usize) -> Self {
        assert!(l >= 2, "torus_44 requires L ≥ 2");
        let idx = |i: usize, j: usize| -> usize { j * l + i };
        let positions: Vec<(f64, f64)> = (0..l)
            .flat_map(|j| (0..l).map(move |i| (i as f64, j as f64)))
            .collect();

        let mut edges = Vec::with_capacity(2 * l * l);
        for j in 0..l {
            for i in 0..l {
                let u = idx(i, j);
                // Right edge u → ((i+1) mod L, j): crosses x-cut (cut 0)
                // when it wraps (i+1 == L).
                let i_r = (i + 1) % l;
                let wrap_x = i32::from(i + 1 == l);
                edges.push(SurfaceEdge {
                    u,
                    v: idx(i_r, j),
                    crossing: vec![wrap_x, 0],
                });
                // Up edge u → (i, (j+1) mod L): crosses y-cut (cut 1)
                // when it wraps (j+1 == L).
                let j_u = (j + 1) % l;
                let wrap_y = i32::from(j + 1 == l);
                edges.push(SurfaceEdge {
                    u,
                    v: idx(i, j_u),
                    crossing: vec![0, wrap_y],
                });
            }
        }

        // Faces: unit squares (i,j) → (i+1,j) → (i+1,j+1) → (i,j+1),
        // all indices mod L. Ordered CCW.
        let mut faces = Vec::with_capacity(l * l);
        for j in 0..l {
            for i in 0..l {
                let i_r = (i + 1) % l;
                let j_u = (j + 1) % l;
                faces.push(vec![
                    idx(i, j),
                    idx(i_r, j),
                    idx(i_r, j_u),
                    idx(i, j_u),
                ]);
            }
        }

        Self {
            p: 4,
            q: 4,
            genus: 1,
            positions,
            edges,
            faces,
        }
    }

    /// Look up the directed crossing for traversing `a → b`. Returns
    /// the crossing vector with the sign appropriate to the traversal
    /// direction (negated if the stored edge is `b → a`). Returns
    /// `None` if no edge connects `a` and `b`.
    #[must_use]
    fn directed_crossing(&self, a: usize, b: usize) -> Option<Vec<i32>> {
        for e in &self.edges {
            if e.u == a && e.v == b {
                return Some(e.crossing.clone());
            }
            if e.u == b && e.v == a {
                return Some(e.crossing.iter().map(|&c| -c).collect());
            }
        }
        None
    }

    /// Verify the closed-surface invariants. Returns `Ok(())` or a
    /// descriptive error. Checks:
    /// 1. Euler characteristic `V − E + F = 2 − 2g`.
    /// 2. Every vertex has degree `q`.
    /// 3. Every face has exactly `p` vertices.
    /// 4. Around every face, the net crossing of each cut is zero
    ///    (cuts are closed cochains → per-plaquette flux is
    ///    well-defined under twists).
    ///
    /// # Errors
    /// Returns a message describing the first invariant violated.
    pub fn verify(&self) -> Result<(), String> {
        let chi = self.euler_characteristic();
        let expected_chi = 2 - 2 * self.genus as i64;
        if chi != expected_chi {
            return Err(format!(
                "Euler characteristic {chi} != 2 - 2g = {expected_chi} (g={})",
                self.genus
            ));
        }
        for (v, &d) in self.degrees().iter().enumerate() {
            if d != self.q {
                return Err(format!("vertex {v} has degree {d}, expected q={}", self.q));
            }
        }
        for (f, face) in self.faces.iter().enumerate() {
            if face.len() != self.p {
                return Err(format!(
                    "face {f} has {} vertices, expected p={}",
                    face.len(),
                    self.p
                ));
            }
            // Net crossing around the face must be zero for every cut.
            let n_cuts = self.n_cuts();
            let mut net = vec![0_i32; n_cuts];
            let m = face.len();
            for k in 0..m {
                let a = face[k];
                let b = face[(k + 1) % m];
                let cr = self.directed_crossing(a, b).ok_or_else(|| {
                    format!("face {f}: no edge between consecutive vertices {a} and {b}")
                })?;
                for (acc, c) in net.iter_mut().zip(cr.iter()) {
                    *acc += c;
                }
            }
            if net.iter().any(|&c| c != 0) {
                return Err(format!(
                    "face {f}: net homology crossing {net:?} is nonzero — \
                     per-plaquette flux is not well-defined under twists"
                ));
            }
        }
        Ok(())
    }

    /// Build a spanning-tree `Topology` for compatibility with the TTN
    /// backend. (Not used by the spectral magnetic path, which works
    /// on the full edge list; provided for parity with `HeavyHexLayout`
    /// / `HyperbolicLayout`.) Returns the tree plus the non-tree edges.
    #[must_use]
    pub fn spanning_tree(&self) -> (Topology, Vec<Edge>) {
        let n = self.n_vertices();
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for e in &self.edges {
            adj[e.u].push(e.v);
            adj[e.v].push(e.u);
        }
        for nbrs in &mut adj {
            nbrs.sort_unstable();
        }
        let mut seen = vec![false; n];
        let mut tree_edges = Vec::new();
        let mut in_tree = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        seen[0] = true;
        queue.push_back(0_usize);
        while let Some(x) = queue.pop_front() {
            for &y in &adj[x] {
                if !seen[y] {
                    seen[y] = true;
                    let (a, b) = (x.min(y), x.max(y));
                    tree_edges.push(Edge { a, b });
                    in_tree.insert((a, b));
                    queue.push_back(y);
                }
            }
        }
        let mut non_tree = Vec::new();
        for e in &self.edges {
            let (a, b) = (e.u.min(e.v), e.u.max(e.v));
            if !in_tree.contains(&(a, b)) {
                non_tree.push(Edge { a, b });
            }
        }
        non_tree.sort_by_key(|e| (e.a, e.b));
        non_tree.dedup();
        let tree = Topology::from_edges_lightweight(n, tree_edges);
        (tree, non_tree)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `{4,4}` torus at several sizes: full §2.1-style acceptance.
    /// V = L², E = 2L², F = L², χ = 0 = 2 − 2·1 (genus 1), degree 4,
    /// 4-gon faces, 2 homology cuts, per-face crossing cancellation.
    #[test]
    fn torus_44_topology_is_valid_genus_1() {
        for l in [2_usize, 3, 4, 6, 10] {
            let s = ClosedSurface::torus_44(l);
            assert_eq!(s.n_vertices(), l * l, "L={l} V");
            assert_eq!(s.n_edges(), 2 * l * l, "L={l} E");
            assert_eq!(s.n_faces(), l * l, "L={l} F");
            assert_eq!(s.euler_characteristic(), 0, "L={l} χ");
            assert_eq!(s.genus, 1);
            assert_eq!(s.n_cuts(), 2);
            // verify() bundles Euler + degree + face-size + per-face
            // crossing cancellation.
            s.verify().unwrap_or_else(|e| panic!("L={l}: {e}"));
        }
    }

    /// Degree is exactly 4 at every vertex (no boundary — the whole
    /// point of PBC).
    #[test]
    fn torus_44_is_boundary_free_degree_4() {
        let s = ClosedSurface::torus_44(5);
        let d = s.degrees();
        assert!(d.iter().all(|&x| x == 4), "degrees: {d:?}");
        // No vertex has degree < 4 (which is what a boundary would
        // produce) — contrast with the OBC square lattice where
        // corners have degree 2 and edges degree 3.
        assert_eq!(*d.iter().min().unwrap(), 4);
        assert_eq!(*d.iter().max().unwrap(), 4);
    }

    /// Each homology cut is non-trivial: exactly L edges cross it
    /// (one wrap edge per row for the x-cut, per column for the
    /// y-cut), and the two cuts are crossed by disjoint edge sets.
    #[test]
    fn torus_44_homology_cuts_are_nontrivial_and_correct_size() {
        let l = 7;
        let s = ClosedSurface::torus_44(l);
        let mut cross0 = 0;
        let mut cross1 = 0;
        let mut both = 0;
        for e in &s.edges {
            let c0 = e.crossing[0] != 0;
            let c1 = e.crossing[1] != 0;
            if c0 {
                cross0 += 1;
            }
            if c1 {
                cross1 += 1;
            }
            if c0 && c1 {
                both += 1;
            }
        }
        assert_eq!(cross0, l, "x-cut should be crossed by L wrap edges");
        assert_eq!(cross1, l, "y-cut should be crossed by L wrap edges");
        assert_eq!(both, 0, "no edge crosses both cuts on the {{4,4}} torus");
    }

    /// The spanning-tree helper produces a valid tree with the right
    /// non-tree edge count: for genus-1 {4,4}, non-tree edges =
    /// E − (V − 1) = 2L² − (L² − 1) = L² + 1.
    #[test]
    fn torus_44_spanning_tree_edge_counts() {
        let l = 5;
        let s = ClosedSurface::torus_44(l);
        let (tree, non_tree) = s.spanning_tree();
        assert_eq!(tree.n_qubits(), l * l);
        assert_eq!(tree.n_edges(), l * l - 1);
        assert_eq!(non_tree.len(), l * l + 1);
        // Tree + non-tree = total edges.
        assert_eq!(tree.n_edges() + non_tree.len(), s.n_edges());
    }

    /// `verify()` catches a corrupted crossing annotation (per-face
    /// cancellation broken) — proves the check is load-bearing, not
    /// decorative.
    #[test]
    fn verify_rejects_broken_crossing() {
        let mut s = ClosedSurface::torus_44(4);
        // Corrupt one edge's crossing so a face no longer cancels.
        s.edges[0].crossing[0] = 5;
        let err = s.verify().unwrap_err();
        assert!(
            err.contains("net homology crossing"),
            "expected crossing-cancellation failure, got: {err}"
        );
    }
}
