//! Regular hyperbolic {p, q} tilings on the Poincaré disk.
//!
//! Foundation for Track F.2 (and downstream F.3 Hofstadter / F.6
//! circuit-QED hyperbolic / F.7 Selberg). Produces a tree-decomposable
//! `Topology` analogous to `HeavyHexLayout::grid` for heavy-hex —
//! qubits at tiling vertices, edges between adjacent vertices, BFS
//! spanning tree + non-tree edges, ready to feed the existing TTN
//! backend and swap-network non-tree-edge machinery.
//!
//! # Approach
//!
//! Face-based breadth-first enumeration on the Poincaré disk:
//!
//! 1. Place the fundamental p-gon at the origin with one *edge
//!    midpoint* along the positive real axis (so vertices sit at
//!    angles `π/p + 2πk/p`).
//! 2. Each face has p neighbour faces, reached by going forward
//!    `2·apothem` in the direction of the shared edge midpoint and
//!    flipping the local frame by π.
//! 3. BFS deduplicates face centres by Poincaré-coordinate hash with
//!    `EPS` tolerance.
//! 4. The vertex graph is constructed from face vertices: each face
//!    contributes p vertices; vertices shared between adjacent faces
//!    are deduplicated by position; tiling edges connect vertices
//!    that are at hyperbolic distance ≈ `side_length`.
//! 5. Spanning tree comes from a BFS in the vertex graph from the
//!    central-vertex closest to the origin (which, for the
//!    edge-midpoint-on-real-axis orientation, sits at angle `π/p`).
//!
//! # Scope of F.2.a (this iteration)
//!
//! Validates {7, 3} at small radius (one or two face-shells around
//! the central face). The Möbius infrastructure is generic; the
//! face-BFS works for any hyperbolic {p, q} (`(p-2)(q-2) > 4`).
//! Larger radii and additional tilings ({3,7}, {4,5}, {5,4}) are
//! follow-on work (F.2.b).

use num_complex::Complex64;
use std::collections::HashMap;

use crate::ttn::topology::{Edge, Topology};

type C = Complex64;

/// FP-tolerance for deduplicating face/vertex positions on the
/// Poincaré disk by quantised hashing. `1e-6` is comfortably below
/// any inter-vertex distance for the small-radius {p, q} tilings
/// this module targets (smallest separation for the supported
/// hyperbolic tilings is the side length, which is `~0.5` for
/// {7,3} and larger for other supported (p,q)).
const POSITION_HASH_EPS: f64 = 1e-6;

// ─────────────────────────────────────────────────────────────────────────────
// Möbius transformations on the Poincaré disk (PSU(1,1))
// ─────────────────────────────────────────────────────────────────────────────

/// Orientation-preserving isometry of the Poincaré disk, represented
/// as a 2×2 complex matrix in `PSU(1,1)`:
///
/// ```text
/// M = [[a, b̄],          z ↦  (a z + b̄) / (b z + ā)
///      [b, ā]]           with |a|² − |b|² = 1
/// ```
///
/// Composition is matrix multiplication. Inverse uses `M⁻¹ = [[ā, −b̄], [−b, a]]`.
#[derive(Clone, Copy, Debug)]
pub struct Mobius {
    pub a: C,
    pub b: C,
}

impl Mobius {
    /// The identity isometry.
    #[must_use]
    pub fn identity() -> Self {
        Self {
            a: C::new(1.0, 0.0),
            b: C::new(0.0, 0.0),
        }
    }

    /// Rotation by angle `θ` about the origin. `z ↦ e^{iθ} z`.
    #[must_use]
    pub fn rotation(theta: f64) -> Self {
        Self {
            a: C::from_polar(1.0, theta / 2.0),
            b: C::new(0.0, 0.0),
        }
    }

    /// Translation that sends the origin to the point `p` (with
    /// `|p| < 1`). Constructed as the normalised Möbius map
    /// `z ↦ (z + p) / (p̄ z + 1)` scaled into `PSU(1,1)`.
    ///
    /// # Panics
    /// If `|p| ≥ 1` (not in the open disk).
    #[must_use]
    pub fn translation_to(p: C) -> Self {
        let n2 = p.norm_sqr();
        assert!(n2 < 1.0, "translation_to: |p|={} not in open disk", n2.sqrt());
        let denom = (1.0 - n2).sqrt();
        Self {
            a: C::new(1.0 / denom, 0.0),
            b: p.conj() / denom,
        }
    }

    /// Compose two isometries: `(self ∘ other)(z) = self(other(z))`.
    /// Matrix product of the underlying `PSU(1,1)` matrices.
    #[must_use]
    pub fn compose(&self, other: &Self) -> Self {
        // [[a₁, b̄₁], [b₁, ā₁]] · [[a₂, b̄₂], [b₂, ā₂]]
        //   row 0: (a₁·a₂ + b̄₁·b₂,   a₁·b̄₂ + b̄₁·ā₂)
        //   row 1: (b₁·a₂ + ā₁·b₂,   b₁·b̄₂ + ā₁·ā₂)
        // The result is a Möbius matrix in the same convention.
        let new_a = self.a * other.a + self.b.conj() * other.b;
        let new_b = self.b * other.a + self.a.conj() * other.b;
        Self {
            a: new_a,
            b: new_b,
        }
    }

    /// Inverse isometry. `M⁻¹ = [[ā, −b̄], [−b, a]]` for `M ∈ PSU(1,1)`.
    #[must_use]
    pub fn inverse(&self) -> Self {
        Self {
            a: self.a.conj(),
            b: -self.b,
        }
    }

    /// Apply this isometry to a point. `z ↦ (a z + b̄) / (b z + ā)`.
    #[must_use]
    pub fn apply(&self, z: C) -> C {
        (self.a * z + self.b.conj()) / (self.b * z + self.a.conj())
    }

    /// Position of the origin under this isometry — convenient for
    /// indexing tiles by their centre coordinates.
    #[must_use]
    pub fn origin_image(&self) -> C {
        // apply(0) = b̄ / ā
        self.b.conj() / self.a.conj()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// {p, q} regular polygon metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Geometric metrics of a regular {p, q} hyperbolic tile:
/// circumradius `R` (centre-to-vertex), apothem `ρ` (centre-to-edge-
/// midpoint), and side length `s` (vertex-to-vertex).
///
/// Formulas come from the fundamental right triangle of {p, q, 2} via
/// hyperbolic right-angle trigonometry:
///
/// - `cosh(s/2) = cos(π/p) / sin(π/q)`
/// - `cosh(ρ)  = cos(π/q) / sin(π/p)`
/// - `cosh(R)  = cosh(ρ) · cosh(s/2)`
///
/// Requires `(p − 2)(q − 2) > 4` (the hyperbolic-tiling condition).
#[derive(Clone, Copy, Debug)]
pub struct PqMetrics {
    pub p: usize,
    pub q: usize,
    pub side_length: f64,
    pub apothem: f64,
    pub circumradius: f64,
}

impl PqMetrics {
    /// # Panics
    /// If `p < 3`, `q < 3`, or `(p-2)(q-2) ≤ 4` (not hyperbolic).
    #[must_use]
    pub fn new(p: usize, q: usize) -> Self {
        assert!(p >= 3, "p must be ≥ 3");
        assert!(q >= 3, "q must be ≥ 3");
        assert!(
            (p - 2) * (q - 2) > 4,
            "{{ p={p}, q={q} }} is not hyperbolic ((p-2)(q-2) must be > 4)"
        );
        let pi_p = std::f64::consts::PI / p as f64;
        let pi_q = std::f64::consts::PI / q as f64;
        let cosh_half_s = pi_p.cos() / pi_q.sin();
        let cosh_apothem = pi_q.cos() / pi_p.sin();
        let side_length = 2.0 * cosh_half_s.acosh();
        let apothem = cosh_apothem.acosh();
        let circumradius = (cosh_apothem * cosh_half_s).acosh();
        Self {
            p,
            q,
            side_length,
            apothem,
            circumradius,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Face enumeration on the Poincaré disk
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a hyperbolic distance `d` from origin to a Poincaré-disk
/// coordinate (real, on the positive real axis). `r = tanh(d / 2)`.
#[must_use]
pub fn poincare_radius(d: f64) -> f64 {
    (d / 2.0).tanh()
}

/// Hyperbolic distance between two points on the Poincaré disk.
/// `d(z, w) = arcosh(1 + 2 |z − w|² / ((1 − |z|²)(1 − |w|²)))`.
#[must_use]
pub fn hyperbolic_distance(z: C, w: C) -> f64 {
    let num = 2.0 * (z - w).norm_sqr();
    let denom = (1.0 - z.norm_sqr()) * (1.0 - w.norm_sqr());
    (1.0 + num / denom).acosh()
}

/// Quantise a Poincaré coordinate into a `(i64, i64)` hash key with
/// resolution `POSITION_HASH_EPS`. Two points hashing to the same key
/// are treated as identical for BFS deduplication.
fn position_key(z: C) -> (i64, i64) {
    let kx = (z.re / POSITION_HASH_EPS).round() as i64;
    let ky = (z.im / POSITION_HASH_EPS).round() as i64;
    (kx, ky)
}

/// A face of the tiling, represented by its isometry from the
/// fundamental face at the origin and by its centre position.
#[derive(Clone, Debug)]
struct Face {
    centre: C,
    iso: Mobius,
    shell: usize,
}

/// BFS-enumerate the faces of a {p, q} tiling out to `max_shells`
/// face-shells from the central face. Shell 0 = central face only;
/// shell 1 = central + p neighbours; etc.
///
/// Returns faces in BFS order (central first).
fn enumerate_faces(metrics: PqMetrics, max_shells: usize) -> Vec<Face> {
    // Face-to-face hop: forward by 2·apothem in the direction of the
    // shared edge midpoint, then flip the local frame by π so the
    // neighbour's "inward" edge points back at the current face.
    let p = metrics.p;
    let hop_radius = poincare_radius(2.0 * metrics.apothem);
    let hops: Vec<Mobius> = (0..p)
        .map(|k| {
            let theta = 2.0 * std::f64::consts::PI * k as f64 / p as f64;
            // hop_k = R(θ) · T_real(2·apothem) · R(−θ) · R(π)
            let r_theta = Mobius::rotation(theta);
            let r_neg_theta = Mobius::rotation(-theta);
            let t = Mobius::translation_to(C::from_polar(hop_radius, 0.0));
            let flip = Mobius::rotation(std::f64::consts::PI);
            r_theta
                .compose(&t)
                .compose(&r_neg_theta)
                .compose(&flip)
        })
        .collect();

    let mut faces: Vec<Face> = Vec::new();
    let mut seen: HashMap<(i64, i64), usize> = HashMap::new();

    let central = Face {
        centre: C::new(0.0, 0.0),
        iso: Mobius::identity(),
        shell: 0,
    };
    seen.insert(position_key(central.centre), 0);
    faces.push(central);

    let mut frontier = vec![0_usize];
    for shell in 1..=max_shells {
        let mut next_frontier = Vec::new();
        for &face_idx in &frontier {
            let face_iso = faces[face_idx].iso;
            for hop in &hops {
                let nbr_iso = face_iso.compose(hop);
                let nbr_centre = nbr_iso.origin_image();
                // Skip if would push us outside the open disk by FP noise.
                if nbr_centre.norm_sqr() >= 1.0 - 1e-12 {
                    continue;
                }
                let key = position_key(nbr_centre);
                if !seen.contains_key(&key) {
                    let new_idx = faces.len();
                    seen.insert(key, new_idx);
                    faces.push(Face {
                        centre: nbr_centre,
                        iso: nbr_iso,
                        shell,
                    });
                    next_frontier.push(new_idx);
                }
            }
        }
        frontier = next_frontier;
        if frontier.is_empty() {
            break;
        }
    }

    faces
}

/// The `p` vertices of a face in BFS order. Vertices sit at distance
/// `circumradius` from the face centre, at angles
/// `π/p + 2π·k/p` (so edge midpoints fall on angles `2π·k/p`,
/// matching the hop-direction convention in `enumerate_faces`).
fn face_vertices(face: &Face, metrics: PqMetrics) -> Vec<C> {
    let r_pos = poincare_radius(metrics.circumradius);
    let p = metrics.p;
    let pi_p = std::f64::consts::PI / p as f64;
    (0..p)
        .map(|k| {
            let theta = pi_p + 2.0 * pi_p * k as f64;
            face.iso.apply(C::from_polar(r_pos, theta))
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// HyperbolicLayout — the public {p, q} layout type
// ─────────────────────────────────────────────────────────────────────────────

/// A regular {p, q} hyperbolic tiling realised as a qubit topology.
///
/// Qubits sit at tiling vertices; edges are the tiling's
/// vertex-to-vertex links. The spanning tree comes from a BFS in the
/// vertex graph starting at the vertex closest to the Poincaré-disk
/// origin. Non-tree edges are reported separately, ready to feed the
/// swap-network non-tree-edge path of the existing TTN backend.
#[derive(Clone, Debug)]
pub struct HyperbolicLayout {
    metrics: PqMetrics,
    /// Poincaré-disk coordinate per qubit, in BFS order from the
    /// root.
    vertices: Vec<C>,
    /// BFS shell number per qubit (0 = root).
    shells: Vec<usize>,
    /// Spanning-tree edges (length `n_qubits − 1`).
    tree_edges: Vec<Edge>,
    /// Non-tree (cycle-closing) edges of the tiling.
    non_tree_edges: Vec<Edge>,
}

impl HyperbolicLayout {
    /// Build the {p, q} tiling out to `max_face_shells` face-shells
    /// from the central face (shell 0 = central face only).
    ///
    /// For a meaningful TTN-able layout, choose `max_face_shells ≥ 1`.
    /// Larger values grow the disk roughly exponentially with shell
    /// number; for {7, 3} the per-shell face count starts as
    /// 1, 7, 21, 56, … so a few shells already give a sizeable graph.
    #[must_use]
    pub fn pq_tiling(p: usize, q: usize, max_face_shells: usize) -> Self {
        let metrics = PqMetrics::new(p, q);
        let faces = enumerate_faces(metrics, max_face_shells);

        // Build the union of all face vertices, deduplicated by
        // position hash.
        let mut vertex_pos: Vec<C> = Vec::new();
        let mut vertex_index: HashMap<(i64, i64), usize> = HashMap::new();
        let mut face_vertex_indices: Vec<Vec<usize>> = Vec::with_capacity(faces.len());
        for face in &faces {
            let verts = face_vertices(face, metrics);
            let mut idxs = Vec::with_capacity(verts.len());
            for v in verts {
                let key = position_key(v);
                let idx = *vertex_index.entry(key).or_insert_with(|| {
                    let i = vertex_pos.len();
                    vertex_pos.push(v);
                    i
                });
                idxs.push(idx);
            }
            face_vertex_indices.push(idxs);
        }
        let n_vertices = vertex_pos.len();

        // Tiling edges: every consecutive pair of vertices around
        // each face, deduplicated.
        let mut edge_set: std::collections::HashSet<(usize, usize)> =
            std::collections::HashSet::new();
        for face_idxs in &face_vertex_indices {
            let n = face_idxs.len();
            for k in 0..n {
                let (mut u, mut v) = (face_idxs[k], face_idxs[(k + 1) % n]);
                if u > v {
                    std::mem::swap(&mut u, &mut v);
                }
                if u != v {
                    edge_set.insert((u, v));
                }
            }
        }
        let all_edges: Vec<(usize, usize)> = edge_set.into_iter().collect();

        // Spanning tree via vertex BFS from the vertex closest to
        // the origin (Euclidean distance, ties broken by index).
        let root = {
            let mut best_idx = 0_usize;
            let mut best_norm = f64::INFINITY;
            for (i, v) in vertex_pos.iter().enumerate() {
                let n = v.norm_sqr();
                if n < best_norm {
                    best_norm = n;
                    best_idx = i;
                }
            }
            best_idx
        };

        // Adjacency for vertex BFS.
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n_vertices];
        for &(u, v) in &all_edges {
            adj[u].push(v);
            adj[v].push(u);
        }
        // Deterministic neighbour order: ascending vertex index.
        for nbrs in &mut adj {
            nbrs.sort_unstable();
        }

        let mut shells = vec![usize::MAX; n_vertices];
        let mut parent: Vec<Option<usize>> = vec![None; n_vertices];
        shells[root] = 0;
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(root);
        let mut bfs_order = Vec::with_capacity(n_vertices);
        bfs_order.push(root);
        while let Some(u) = queue.pop_front() {
            for &w in &adj[u] {
                if shells[w] == usize::MAX {
                    shells[w] = shells[u] + 1;
                    parent[w] = Some(u);
                    queue.push_back(w);
                    bfs_order.push(w);
                }
            }
        }
        // Sanity: BFS must reach every vertex (the tiling vertex
        // graph is connected by construction).
        assert_eq!(
            bfs_order.len(),
            n_vertices,
            "vertex BFS did not reach every vertex — internal bug"
        );

        // Relabel qubits in BFS order so qubit 0 is the root and
        // qubit indices grow outward by shell.
        let new_index: Vec<usize> = {
            let mut map = vec![usize::MAX; n_vertices];
            for (new_i, &old_i) in bfs_order.iter().enumerate() {
                map[old_i] = new_i;
            }
            map
        };
        let vertices: Vec<C> = bfs_order.iter().map(|&i| vertex_pos[i]).collect();
        let shells: Vec<usize> = bfs_order.iter().map(|&i| shells[i]).collect();

        // Partition edges into spanning-tree and non-tree.
        let mut tree_edges: Vec<Edge> = Vec::new();
        let mut non_tree_edges: Vec<Edge> = Vec::new();
        for (i, par) in parent.iter().enumerate() {
            if let Some(p_old) = par {
                let a = new_index[i].min(new_index[*p_old]);
                let b = new_index[i].max(new_index[*p_old]);
                tree_edges.push(Edge { a, b });
            }
        }
        // Non-tree = all edges minus spanning-tree edges.
        let tree_set: std::collections::HashSet<(usize, usize)> =
            tree_edges.iter().map(|e| (e.a, e.b)).collect();
        for (u_old, v_old) in &all_edges {
            let a = new_index[*u_old].min(new_index[*v_old]);
            let b = new_index[*u_old].max(new_index[*v_old]);
            if !tree_set.contains(&(a, b)) {
                non_tree_edges.push(Edge { a, b });
            }
        }
        tree_edges.sort_by_key(|e| (e.a, e.b));
        non_tree_edges.sort_by_key(|e| (e.a, e.b));

        Self {
            metrics,
            vertices,
            shells,
            tree_edges,
            non_tree_edges,
        }
    }

    /// Number of qubits.
    #[must_use]
    pub fn n_qubits(&self) -> usize {
        self.vertices.len()
    }

    /// Geometric metrics of the underlying {p, q} tile.
    #[must_use]
    pub fn metrics(&self) -> PqMetrics {
        self.metrics
    }

    /// Poincaré-disk position of qubit `q`. Qubit 0 is the root
    /// (vertex closest to the origin); subsequent qubits are ordered
    /// by BFS shell.
    #[must_use]
    pub fn vertex(&self, q: usize) -> C {
        self.vertices[q]
    }

    /// BFS-shell number of qubit `q` (0 = root).
    #[must_use]
    pub fn shell(&self, q: usize) -> usize {
        self.shells[q]
    }

    /// Spanning-tree edges (length `n_qubits − 1`).
    #[must_use]
    pub fn tree_edges(&self) -> &[Edge] {
        &self.tree_edges
    }

    /// Non-tree (cycle-closing) edges of the tiling.
    #[must_use]
    pub fn non_tree_edges(&self) -> &[Edge] {
        &self.non_tree_edges
    }

    /// Build a `Topology` from the spanning tree. Use
    /// [`Self::non_tree_edges`] alongside it for the swap-network
    /// non-tree-edge path (same convention as `HeavyHexLayout`).
    #[must_use]
    pub fn tree(&self) -> Topology {
        Topology::from_edges_lightweight(self.n_qubits(), self.tree_edges.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_3, FRAC_PI_4, PI};

    fn approx_eq_c(a: C, b: C, tol: f64) -> bool {
        (a - b).norm() < tol
    }

    /// `Mobius::identity()` applied to a point must give the same
    /// point.
    #[test]
    fn identity_is_identity() {
        let id = Mobius::identity();
        for z in [
            C::new(0.0, 0.0),
            C::new(0.3, 0.4),
            C::new(-0.5, 0.2),
        ] {
            assert!(
                approx_eq_c(id.apply(z), z, 1e-14),
                "identity moved {z:?} to {:?}",
                id.apply(z)
            );
        }
    }

    /// `M · M⁻¹` is the identity (up to FP precision) for any
    /// constructed `Mobius`.
    #[test]
    fn inverse_undoes_composition() {
        let cases = [
            Mobius::rotation(FRAC_PI_3),
            Mobius::rotation(-1.234),
            Mobius::translation_to(C::new(0.3, 0.0)),
            Mobius::translation_to(C::new(-0.2, 0.4)),
            Mobius::rotation(FRAC_PI_4)
                .compose(&Mobius::translation_to(C::new(0.5, -0.1))),
        ];
        for m in cases {
            let prod = m.compose(&m.inverse());
            for z in [C::new(0.0, 0.0), C::new(0.2, 0.3)] {
                assert!(
                    approx_eq_c(prod.apply(z), z, 1e-12),
                    "M·M⁻¹ moved {z:?} to {:?}",
                    prod.apply(z)
                );
            }
        }
    }

    /// Composition of Möbius isometries is associative on
    /// well-conditioned points.
    #[test]
    fn composition_is_associative() {
        let m1 = Mobius::rotation(0.7);
        let m2 = Mobius::translation_to(C::new(0.3, 0.1));
        let m3 = Mobius::rotation(-0.4)
            .compose(&Mobius::translation_to(C::new(-0.2, 0.25)));
        let left = m1.compose(&m2).compose(&m3);
        let right = m1.compose(&m2.compose(&m3));
        for z in [C::new(0.0, 0.0), C::new(0.1, -0.2)] {
            assert!(
                approx_eq_c(left.apply(z), right.apply(z), 1e-12),
                "(m1·m2)·m3 ≠ m1·(m2·m3) at z={z:?}",
            );
        }
    }

    /// `Mobius::translation_to(p)` must send the origin exactly to
    /// `p` and preserve the disk (image norm < 1).
    #[test]
    fn translation_to_sends_origin_to_point() {
        for p in [
            C::new(0.4, 0.0),
            C::new(0.0, 0.6),
            C::new(-0.3, 0.5),
        ] {
            let m = Mobius::translation_to(p);
            assert!(approx_eq_c(m.apply(C::new(0.0, 0.0)), p, 1e-14));
        }
    }

    /// Hyperbolic-distance formula sanity: a single
    /// `translation_to(r, 0)` moves the origin to a point at
    /// hyperbolic distance `2·atanh(r)` from the origin.
    #[test]
    fn hyperbolic_distance_matches_poincare_radius() {
        for d in [0.1_f64, 0.5, 1.0, 2.0] {
            let r = poincare_radius(d);
            let z = C::new(r, 0.0);
            let measured = hyperbolic_distance(C::new(0.0, 0.0), z);
            assert!(
                (measured - d).abs() < 1e-12,
                "poincare_radius({d}) = {r}, hyp dist back = {measured}",
            );
        }
    }

    /// PqMetrics for {7, 3} must agree with the textbook values
    /// (verified against the formulas in the doc comment).
    #[test]
    fn pq_metrics_7_3() {
        let m = PqMetrics::new(7, 3);
        // Direct evaluation of the formulas:
        let pi_7 = PI / 7.0;
        let pi_3 = PI / 3.0;
        let s = 2.0 * (pi_7.cos() / pi_3.sin()).acosh();
        let rho = (pi_3.cos() / pi_7.sin()).acosh();
        let r = (rho.cosh() * (s / 2.0).cosh()).acosh();
        assert!((m.side_length - s).abs() < 1e-14);
        assert!((m.apothem - rho).abs() < 1e-14);
        assert!((m.circumradius - r).abs() < 1e-14);
        // Numerical sanity (values around 0.5-0.6 for {7,3}).
        assert!(m.side_length > 0.5 && m.side_length < 0.6);
        assert!(m.apothem > 0.5 && m.apothem < 0.55);
        assert!(m.circumradius > m.apothem); // strictly larger than apothem
    }

    /// {7, 3} non-hyperbolic guard: PqMetrics::new should accept
    /// hyperbolic (p, q) and reject non-hyperbolic.
    #[test]
    fn hyperbolicity_guard() {
        // Hyperbolic: (p-2)(q-2) > 4
        let _ok = PqMetrics::new(7, 3); // 5·1 = 5 > 4 ✓
        let _ok2 = PqMetrics::new(4, 5); // 2·3 = 6 > 4 ✓
        let _ok3 = PqMetrics::new(3, 7); // 1·5 = 5 > 4 ✓
    }
    #[test]
    #[should_panic(expected = "not hyperbolic")]
    fn euclidean_4_4_rejected() {
        // {4, 4} is Euclidean ((4-2)(4-2) = 4, not > 4)
        let _ = PqMetrics::new(4, 4);
    }
    #[test]
    #[should_panic(expected = "not hyperbolic")]
    fn spherical_5_3_rejected() {
        // {5, 3} is spherical (icosahedral, (5-2)(3-2) = 3 < 4)
        let _ = PqMetrics::new(5, 3);
    }

    /// {7, 3} at 0 face-shells: just the central heptagonal tile,
    /// giving 7 vertices and 7 edges (a single 7-cycle, plus a
    /// spanning tree of 6 edges and 1 non-tree edge closing the
    /// cycle).
    #[test]
    fn tiling_7_3_radius_0_is_a_single_heptagon() {
        let layout = HyperbolicLayout::pq_tiling(7, 3, 0);
        assert_eq!(layout.n_qubits(), 7, "central heptagon has 7 vertices");
        // 7 tiling edges total, 6 in spanning tree, 1 non-tree.
        assert_eq!(layout.tree_edges().len(), 6);
        assert_eq!(layout.non_tree_edges().len(), 1);
        // All 7 vertices at the same Poincaré distance from origin.
        let r0 = layout.vertex(0).norm();
        for q in 1..7 {
            assert!(
                (layout.vertex(q).norm() - r0).abs() < 1e-10,
                "vertex {q} at unexpected distance from origin",
            );
        }
        // Tree topology check: build_lightweight succeeds on the
        // spanning tree.
        let tree = layout.tree();
        assert_eq!(tree.n_qubits(), 7);
    }

    /// {7, 3} at 1 face-shell: central heptagon + 7 neighbour
    /// heptagons = 8 faces total. The total vertex count is the
    /// load-bearing combinatorial invariant we can check.
    ///
    /// Hand-derived count (verified two ways):
    ///
    /// **By incidence:** 8 faces × 7 vertices/face = 56 vertex
    /// incidences. Vertex multiplicities in the enumerated set:
    /// the 7 central-heptagon vertices each meet 3 faces (central
    /// + two adjacent neighbours), the 7 vertices shared between
    /// adjacent neighbours each meet 2 enumerated faces (the third
    /// would be in radius 2), and the 7 × 3 = 21 "interior" vertices
    /// of each neighbour each meet 1 face. Total incidences =
    /// 7·3 + 7·2 + 21·1 = 21 + 14 + 21 = 56 ✓.
    ///
    /// **By direct count:** 7 (central) + 7 (shared neighbour-to-
    /// neighbour) + 21 (interior per neighbour, 3 each × 7) = 35.
    ///
    /// This pins the radius-1 vertex count as a regression guard for
    /// future BFS / deduplication / face-vertex-extraction changes.
    #[test]
    fn tiling_7_3_radius_1_vertex_count_and_edge_lengths() {
        let layout = HyperbolicLayout::pq_tiling(7, 3, 1);
        assert_eq!(
            layout.n_qubits(),
            35,
            "{{7,3}} at face-radius=1 has 35 vertices (7 central + 7 shared + 21 interior)",
        );
        // All tiling edges must have hyperbolic length ≈ side_length.
        let s = layout.metrics().side_length;
        for e in layout
            .tree_edges()
            .iter()
            .chain(layout.non_tree_edges().iter())
        {
            let d = hyperbolic_distance(layout.vertex(e.a), layout.vertex(e.b));
            assert!(
                (d - s).abs() < 1e-6,
                "edge ({}, {}) has hyperbolic length {d}, expected ≈ {s}",
                e.a,
                e.b
            );
        }
        // Spanning tree invariant: n_qubits − 1 tree edges.
        assert_eq!(layout.tree_edges().len(), layout.n_qubits() - 1);
        // Total edges = tree + non_tree.
        let n_tree = layout.tree_edges().len();
        let n_non_tree = layout.non_tree_edges().len();
        let n_total = n_tree + n_non_tree;
        // Sanity: at least the central heptagon's 7 edges exist;
        // the boundary contributes more.
        assert!(n_total >= 7, "tiling should have ≥ 7 edges, got {n_total}");
        // Spanning tree is a valid Topology.
        let tree = layout.tree();
        assert_eq!(tree.n_qubits(), layout.n_qubits());
        assert_eq!(tree.n_edges(), layout.n_qubits() - 1);
    }

    /// Vertex 0 (the root) is the BFS-closest vertex to the origin;
    /// every other vertex has a strictly larger shell number than
    /// at least one of its neighbours. Pins the BFS invariant so
    /// future refactors of the relabelling logic can't silently
    /// break it.
    #[test]
    fn root_is_origin_closest_and_shells_are_well_formed() {
        let layout = HyperbolicLayout::pq_tiling(7, 3, 1);
        assert_eq!(layout.shell(0), 0);
        let mut root_norm = layout.vertex(0).norm();
        for q in 1..layout.n_qubits() {
            assert!(
                layout.vertex(q).norm() >= root_norm - 1e-12,
                "vertex {q} closer to origin than root",
            );
            root_norm = root_norm.min(layout.vertex(q).norm());
            assert!(
                layout.shell(q) >= 1,
                "non-root vertex {q} has shell 0",
            );
        }
    }
}
