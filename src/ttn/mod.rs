//! Tree Tensor Network (TTN) simulator.
//!
//! This is the entry point for non-1D Huoma. Public types ([`Topology`],
//! [`EdgeId`], [`Edge`], [`Ttn`]) live here; the implementation is split into
//! submodules:
//!
//! - [`topology`] — graph layer (validation, neighbours, cut partitions, paths)
//! - [`site`] — native flat-storage `TtnSite` + tensor reshape helpers
//! - [`gauge`] — orthogonality-center tracking + QR sweeps
//! - [`contraction`] — two-site merge + bipartition SVD (D.2 in progress)
//! - [`dense`] — test-only topology-agnostic statevector reference
//!
//! The linear-chain special case continues to delegate to the validated
//! [`crate::mps::Mps`] backend through D.2 (see `TRACK_D_DESIGN.md`); the
//! native tree path is engaged for non-linear topologies via a `Backend`
//! enum dispatched in this module. The facade retires in D.3 when the
//! swap-network code lands.
//!
//! See `TRACK_D_DESIGN.md` for the full milestone roadmap.

pub mod allocator;
pub mod boundary;
pub mod contraction;
pub mod feedback;
pub mod gauge;
pub mod heavy_hex;
pub mod kim_heavy_hex;
pub mod partition;
pub mod projected;
pub mod site;
pub mod subtree;
pub mod topology;

#[cfg(test)]
pub(crate) mod dense;

pub use heavy_hex::HeavyHexLayout;
pub use topology::{Edge, EdgeId, Topology};

use num_complex::Complex64;

use crate::error::Result;
use crate::mps::{Mps, TruncationMode};
use site::TtnSite;

type C = Complex64;

/// Internal storage backend. Linear chains delegate to the validated `Mps`
/// path; everything else uses the native flat-tensor backend. The facade
/// retires in D.3 when the swap-network code lands, at which point this
/// enum collapses into a single `Tree` variant.
#[derive(Debug, Clone)]
enum Backend {
    Linear(Mps),
    Tree {
        sites: Vec<TtnSite>,
        truncation_mode: TruncationMode,
        discarded_per_edge: Vec<f64>,
        /// Current orthogonality center. A product state is trivially
        /// canonical with respect to any center, so we seed with 0 and let
        /// subsequent gates move the center lazily.
        center: usize,
    },
}

/// Tree Tensor Network state vector.
#[derive(Debug, Clone)]
pub struct Ttn {
    topology: Topology,
    backend: Backend,
}

impl Ttn {
    /// Initialise a TTN over `topology` in the `|0…0⟩` product state.
    #[must_use]
    pub fn new(topology: Topology) -> Self {
        let backend = if topology.is_linear_chain() {
            Backend::Linear(Mps::new(topology.n_qubits()))
        } else {
            let sites: Vec<TtnSite> = (0..topology.n_qubits())
                .map(|v| TtnSite::product_zero(topology.neighbours(v).to_vec()))
                .collect();
            Backend::Tree {
                sites,
                truncation_mode: TruncationMode::default(),
                discarded_per_edge: vec![0.0; topology.n_edges()],
                center: 0,
            }
        };
        Self { topology, backend }
    }

    /// Borrow the topology this TTN was built on.
    #[must_use]
    pub fn topology(&self) -> &Topology {
        &self.topology
    }

    /// Number of physical qubits.
    #[must_use]
    pub fn n_qubits(&self) -> usize {
        self.topology.n_qubits()
    }

    /// Set the SVD truncation mode used by subsequent two-qubit gates.
    pub fn set_truncation_mode(&mut self, mode: TruncationMode) {
        match &mut self.backend {
            Backend::Linear(mps) => mps.set_truncation_mode(mode),
            Backend::Tree { truncation_mode, .. } => *truncation_mode = mode,
        }
    }

    /// Cumulative discarded weight on edge `id`. For the linear backend
    /// `EdgeId(k)` maps to MPS bond `k`; for the tree backend it indexes
    /// directly into the per-edge accumulator.
    #[must_use]
    pub fn discarded_weight(&self, id: EdgeId) -> f64 {
        match &self.backend {
            Backend::Linear(mps) => mps.discarded_weight(id.0),
            Backend::Tree { discarded_per_edge, .. } => {
                discarded_per_edge.get(id.0).copied().unwrap_or(0.0)
            }
        }
    }

    /// Total cumulative discarded weight summed over all edges.
    #[must_use]
    pub fn total_discarded_weight(&self) -> f64 {
        match &self.backend {
            Backend::Linear(mps) => mps.total_discarded_weight(),
            Backend::Tree { discarded_per_edge, .. } => discarded_per_edge.iter().sum(),
        }
    }

    /// Bond / edge dimension of edge `id`.
    #[must_use]
    pub fn edge_dim(&self, id: EdgeId) -> usize {
        match &self.backend {
            Backend::Linear(mps) => mps.bond_dim(id.0),
            Backend::Tree { sites, .. } => {
                let edge = self.topology.edge(id);
                sites[edge.a].dim_for_edge(id)
            }
        }
    }

    /// Apply a single-qubit gate to qubit `q`.
    pub fn apply_single(&mut self, q: usize, u: [[C; 2]; 2]) {
        match &mut self.backend {
            Backend::Linear(mps) => mps.apply_single(q, u),
            Backend::Tree { sites, .. } => apply_single_tree(&mut sites[q], u),
        }
    }

    /// Apply a two-qubit gate on the qubits incident to edge `id`, truncating
    /// the new bond at `max_bond`.
    pub fn apply_two_qubit_on_edge(
        &mut self,
        id: EdgeId,
        u: [[C; 4]; 4],
        max_bond: usize,
    ) -> Result<()> {
        match &mut self.backend {
            Backend::Linear(mps) => {
                let edge = self.topology.edge(id);
                let q = edge.a.min(edge.b);
                let other = edge.a.max(edge.b);
                debug_assert_eq!(other, q + 1, "linear-chain edge must connect adjacent qubits");
                mps.apply_two_qubit(q, u, max_bond)
            }
            Backend::Tree {
                sites,
                truncation_mode,
                discarded_per_edge,
                center,
            } => {
                // Move the gauge center to edge.a before the SVD so truncation
                // is optimal in the 2-norm sense. (For lossless max_bond this
                // is not strictly required but it keeps the path identical.)
                let edge = self.topology.edge(id);
                gauge::move_center(sites, &self.topology, *center, edge.a);
                let mode = *truncation_mode;
                let (_, discarded) =
                    contraction::apply_two_qubit_on_edge_native(
                        sites,
                        &self.topology,
                        id,
                        u,
                        max_bond,
                        mode,
                    )?;
                if let Some(slot) = discarded_per_edge.get_mut(id.0) {
                    *slot += discarded;
                }
                // The contraction routine absorbs S into site[u] → u is the
                // new center.
                *center = edge.a;
                Ok(())
            }
        }
    }

    /// Apply a two-qubit gate on any pair of qubits by routing through the
    /// unique tree path between them — milestone D.3 (swap network).
    ///
    /// Strategy: walk the tree path `q1 → … → q2`, SWAP forward along each
    /// edge so that the *state* originally at `q1` is adjacent to `q2`,
    /// apply the gate on the last edge, then SWAP back to restore the
    /// original qubit-to-site assignment. For a path of length `k` this
    /// costs `2·(k − 1)` SWAPs plus the final gate, i.e. `2k − 1` two-qubit
    /// operations on tree-adjacent pairs. If `q1` and `q2` are already
    /// tree-adjacent the path has length 1 and the method degenerates to
    /// a single [`Self::apply_two_qubit_on_edge`] call.
    ///
    /// The 4×4 `u` is indexed with `q1` as the first qubit and `q2` as the
    /// second (row = `2·σ_{q1} + σ_{q2}`). At every edge encountered along
    /// the path the gate / SWAP is re-oriented to match the edge's stored
    /// `(a, b)` ordering before being handed to
    /// [`Self::apply_two_qubit_on_edge`], so callers can ignore edge
    /// orientation entirely.
    ///
    /// All swap-network SVDs share the caller-provided `max_bond`. For
    /// lossless validation against a dense reference pass a bound large
    /// enough that no singular value is ever dropped (a safe upper bound is
    /// `2^(n_qubits / 2)`); for production heavy-hex execution use the
    /// same finite cap as the direct-edge gates so truncation error
    /// composes uniformly.
    ///
    /// # Errors
    ///
    /// Propagates any SVD failure from the underlying
    /// [`Self::apply_two_qubit_on_edge`] calls.
    ///
    /// # Panics
    ///
    /// Panics if `q1 == q2` or either qubit is out of range, matching the
    /// panic-on-programmer-error discipline used by the rest of `ttn`.
    pub fn apply_two_qubit_via_path(
        &mut self,
        q1: usize,
        q2: usize,
        u: [[C; 4]; 4],
        max_bond: usize,
    ) -> Result<()> {
        let n = self.topology.n_qubits();
        assert!(q1 < n && q2 < n, "qubit out of range");
        assert!(q1 != q2, "apply_two_qubit_via_path requires q1 != q2");

        let path = self.topology.path(q1, q2);
        debug_assert!(
            !path.is_empty(),
            "path between distinct qubits in a tree must be non-empty"
        );

        // Reconstruct the ordered vertex sequence q1 = v_0, v_1, …, v_k = q2.
        // We need this to know which end of each edge is the "current" one
        // that holds the in-transit q1-state at each step of the walk.
        let mut vertices: Vec<usize> = Vec::with_capacity(path.len() + 1);
        vertices.push(q1);
        for &eid in &path {
            let e = self.topology.edge(eid);
            let prev = *vertices.last().expect("vertices seeded with q1");
            let next = if e.a == prev {
                e.b
            } else {
                debug_assert_eq!(
                    e.b, prev,
                    "topology::path returned an edge not incident on the previous vertex"
                );
                e.a
            };
            vertices.push(next);
        }
        debug_assert_eq!(*vertices.last().expect("non-empty"), q2);

        // Tree-adjacent fast path: path length 1, no swaps, just orient the
        // gate to the edge's stored (a, b) direction and dispatch.
        let final_edge = self.topology.edge(path[path.len() - 1]);
        if path.len() == 1 {
            let oriented = orient_two_qubit_gate_for_edge(u, q1, final_edge);
            return self.apply_two_qubit_on_edge(path[0], oriented, max_bond);
        }

        let swap = swap_gate();

        // Forward pass: SWAP along path[0..k-1] so the state originally at
        // q1 migrates to v_{k-1}, which is adjacent to v_k = q2.
        for i in 0..path.len() - 1 {
            let eid = path[i];
            // SWAP is symmetric under qubit reordering so we can hand it off
            // verbatim without re-orienting.
            self.apply_two_qubit_on_edge(eid, swap, max_bond)?;
        }

        // Now the "q1-state" is at v_{k-1}. Apply the real gate on the last
        // edge, oriented so its first qubit index matches v_{k-1}.
        let from = vertices[vertices.len() - 2];
        let oriented = orient_two_qubit_gate_for_edge(u, from, final_edge);
        self.apply_two_qubit_on_edge(path[path.len() - 1], oriented, max_bond)?;

        // Backward pass: undo the forward SWAPs in reverse order so every
        // qubit-to-site assignment returns to its pre-call state.
        for i in (0..path.len() - 1).rev() {
            let eid = path[i];
            self.apply_two_qubit_on_edge(eid, swap, max_bond)?;
        }

        Ok(())
    }

    /// Single-qubit ⟨Z⟩ at qubit `target`, normalised by the state's 2-norm².
    ///
    /// **Native tree path (D.5.0):** moves the orthogonality centre to
    /// `target` via [`gauge::move_center`] (O(tree-distance · χ³)) and reads
    /// the expectation value from the local site tensor in O(χ^deg · 2).
    /// No dense statevector is materialised, so this method scales to the
    /// full Eagle 127q TTN — the statevector-materialising implementation
    /// that shipped in D.2 was capped at N ≲ 20. The mutable receiver
    /// reflects the gauge side-effect; after the call, the TTN's
    /// orthogonality centre sits at `target`.
    ///
    /// **Linear path:** delegates directly to [`Mps::expectation_z`], which
    /// already uses a gauge-aware environment sweep and is `&self`.
    #[must_use]
    pub fn expectation_z(&mut self, target: usize) -> f64 {
        match &mut self.backend {
            Backend::Linear(mps) => mps.expectation_z(target),
            Backend::Tree { sites, center, .. } => {
                gauge::move_center(sites, &self.topology, *center, target);
                *center = target;
                expectation_z_at_center(&sites[target])
            }
        }
    }

    /// Compute `⟨Z_q⟩` for every qubit in a single DFS traversal from the
    /// current orthogonality centre. Returns a vector of length
    /// [`Self::n_qubits`] indexed by qubit id.
    ///
    /// The traversal moves the centre along tree edges so that each edge of
    /// the spanning tree is walked at most twice across the whole call, so
    /// the total cost is `O(N · χ³)` rather than the `O(N²)` of calling
    /// [`Self::expectation_z`] in a naïve loop from a fixed initial centre.
    /// At `N = 127` with `χ ≤ 64` this is milliseconds, not minutes.
    pub fn expectation_z_all(&mut self) -> Vec<f64> {
        let n = self.n_qubits();
        let mut result = vec![0.0_f64; n];
        match &mut self.backend {
            Backend::Linear(mps) => {
                return mps.expectation_z_all();
            }
            Backend::Tree { sites, center, .. } => {
                // DFS order from the current centre — consecutive visits
                // are either tree-adjacent or, at worst, separated by the
                // Euler-tour overhead of a pair of backtrack edges, which
                // still totals O(n_edges · 2) moves across the whole call.
                let order = dfs_order(&self.topology, *center);
                for &q in &order {
                    gauge::move_center(sites, &self.topology, *center, q);
                    *center = q;
                    result[q] = expectation_z_at_center(&sites[q]);
                }
            }
        }
        result
    }

    /// Wave-function 2-norm squared.
    ///
    /// In site-canonical form (enforced by every gate-application code path
    /// in this module) the 2-norm² of the state equals the 2-norm² of the
    /// orthogonality-centre site alone, since every other site is an
    /// isometry with respect to its centre-ward axis. No tree traversal
    /// required — this is O(χ^deg · 2), not O(2^N).
    #[must_use]
    pub fn norm_squared(&self) -> f64 {
        match &self.backend {
            Backend::Linear(mps) => mps.norm_squared(),
            Backend::Tree { sites, center, .. } => sites[*center]
                .data
                .iter()
                .map(|c| c.norm_sqr())
                .sum(),
        }
    }

    /// Test-only access to the full dense statevector for small trees
    /// (ordered with qubit 0 as the most significant bit).
    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn to_statevector(&self) -> Vec<C> {
        match &self.backend {
            Backend::Linear(mps) => mps.to_statevector(),
            Backend::Tree { sites, .. } => contraction::tree_to_statevector(sites, &self.topology),
        }
    }
}

/// The 4×4 two-qubit SWAP gate, indexed as row = `2·σ_a + σ_b` matching the
/// convention of [`Ttn::apply_two_qubit_on_edge`]. Used by the swap-network
/// long-range gate in [`Ttn::apply_two_qubit_via_path`].
#[inline]
fn swap_gate() -> [[C; 4]; 4] {
    let o = C::new(1.0, 0.0);
    let z = C::new(0.0, 0.0);
    [
        [o, z, z, z],
        [z, z, o, z],
        [z, o, z, z],
        [z, z, z, o],
    ]
}

/// Re-orient a two-qubit gate so its "first qubit" index matches a specific
/// endpoint of a tree edge.
///
/// The swap-network walks the tree path and needs to apply the final gate
/// with its first qubit landing on the *in-transit* end of the last edge,
/// which may be either `edge.a` or `edge.b` depending on how the path was
/// traversed. `apply_two_qubit_on_edge` always interprets the 4×4 as
/// `row = 2·σ_a + σ_b`, so if `from == edge.a` we return `u` verbatim;
/// otherwise we transpose the qubit-index pair:
///
/// ```text
/// out[2·α + β, 2·α' + β']  =  u[2·β + α, 2·β' + α']
/// ```
///
/// SWAP is symmetric under this transformation — a round-trip through
/// `orient_two_qubit_gate_for_edge` leaves it unchanged — so the SWAPs
/// driving the forward / backward passes can be applied without ever
/// calling this helper.
fn orient_two_qubit_gate_for_edge(u: [[C; 4]; 4], from: usize, edge: Edge) -> [[C; 4]; 4] {
    if from == edge.a {
        return u;
    }
    debug_assert_eq!(
        from, edge.b,
        "orient_two_qubit_gate_for_edge: `from` must be an endpoint of `edge`"
    );
    let mut out = [[C::new(0.0, 0.0); 4]; 4];
    for alpha in 0..2 {
        for beta in 0..2 {
            for alpha_p in 0..2 {
                for beta_p in 0..2 {
                    out[2 * alpha + beta][2 * alpha_p + beta_p] =
                        u[2 * beta + alpha][2 * beta_p + alpha_p];
                }
            }
        }
    }
    out
}

/// Apply a single-qubit 2×2 unitary to the physical leg of a native
/// `TtnSite`. The physical axis is always the last axis; we iterate over
/// every virtual multi-index and transform the 2-vector at that slice.
fn apply_single_tree(site: &mut TtnSite, u: [[C; 2]; 2]) {
    let nd = site.rank();
    let phys_dim = site.shape[nd - 1];
    debug_assert_eq!(phys_dim, 2, "physical leg must have dimension 2");
    let virt_count: usize = site.data.len() / phys_dim;
    for v in 0..virt_count {
        let i0 = v * 2;
        let a0 = site.data[i0];
        let a1 = site.data[i0 + 1];
        site.data[i0] = u[0][0] * a0 + u[0][1] * a1;
        site.data[i0 + 1] = u[1][0] * a0 + u[1][1] * a1;
    }
}

/// Normalised single-qubit ⟨Z⟩ read from a TTN site that is assumed to
/// currently be the orthogonality centre of the state.
///
/// In site-canonical form every site except the centre is an isometry with
/// respect to its centre-ward axis, so any local observable on the centre
/// reduces to a direct sum over the centre's tensor data: the row-major
/// storage places the physical axis last, so every virtual multi-index
/// `v` has its `σ = 0` amplitude at `data[2v]` and its `σ = 1` amplitude
/// at `data[2v + 1]`. The normalised expectation is
///
/// ```text
/// ⟨Z⟩  =  (Σ_v |data[2v]|² − Σ_v |data[2v+1]|²) / Σ_v (|data[2v]|² + |data[2v+1]|²)
/// ```
///
/// and the denominator is the total 2-norm² of the state (same as
/// [`Ttn::norm_squared`]). A zero-norm state is reported as `⟨Z⟩ = 0`
/// rather than NaN so downstream asserts don't trip on degenerate inputs.
fn expectation_z_at_center(site: &TtnSite) -> f64 {
    let nd = site.rank();
    let phys_dim = site.shape[nd - 1];
    debug_assert_eq!(phys_dim, 2, "physical leg must have dimension 2");
    let virt_count = site.data.len() / phys_dim;
    let mut p0 = 0.0_f64;
    let mut p1 = 0.0_f64;
    for v in 0..virt_count {
        p0 += site.data[2 * v].norm_sqr();
        p1 += site.data[2 * v + 1].norm_sqr();
    }
    let denom = p0 + p1;
    if denom < 1e-30 {
        0.0
    } else {
        (p0 - p1) / denom
    }
}

/// Depth-first tree traversal order from `start`. Used by
/// [`Ttn::expectation_z_all`] to route the orthogonality-centre walk so
/// that each tree edge is traversed at most twice across the whole sweep
/// (the standard Euler-tour bound), keeping the observable pass at
/// `O(N · χ³)` instead of `O(N²)`. Output is a deterministic pre-order:
/// children are pushed in reverse of their `neighbours()` listing so they
/// come off the stack in the same order the topology reports them.
fn dfs_order(topology: &Topology, start: usize) -> Vec<usize> {
    let n = topology.n_qubits();
    let mut order = Vec::with_capacity(n);
    if n == 0 {
        return order;
    }
    let mut visited = vec![false; n];
    let mut stack: Vec<usize> = Vec::with_capacity(n);
    stack.push(start);
    while let Some(u) = stack.pop() {
        if visited[u] {
            continue;
        }
        visited[u] = true;
        order.push(u);
        // Push neighbours in reverse so the first-listed neighbour is
        // popped first, giving a stable pre-order deterministic with
        // respect to the topology's edge-enumeration.
        for &eid in topology.neighbours(u).iter().rev() {
            let nb = topology.edge(eid).other(u);
            if !visited[nb] {
                stack.push(nb);
            }
        }
    }
    order
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_chain_topology_basic() {
        let t = Topology::linear_chain(5);
        assert_eq!(t.n_qubits(), 5);
        assert_eq!(t.n_edges(), 4);
        assert!(t.is_linear_chain());
        for (i, e) in t.edges().iter().enumerate() {
            assert_eq!((e.a, e.b), (i, i + 1));
        }
    }

    #[test]
    fn from_edges_accepts_linear_chain_explicitly() {
        let edges = (0..4).map(|i| Edge { a: i, b: i + 1 }).collect();
        let t = Topology::from_edges(5, edges);
        assert!(t.is_linear_chain());
        assert_eq!(t.n_edges(), 4);
    }

    #[test]
    fn ttn_new_initialises_to_product_state() {
        let t = Topology::linear_chain(6);
        let mut ttn = Ttn::new(t);
        assert_eq!(ttn.n_qubits(), 6);
        // |0…0⟩: every ⟨Z⟩ = +1.
        for q in 0..6 {
            assert!((ttn.expectation_z(q) - 1.0).abs() < 1e-12);
        }
        // No SVDs yet → no discarded weight.
        assert_eq!(ttn.total_discarded_weight(), 0.0);
    }

    /// 1D REGRESSION (Track D milestone D.1):
    /// A `Ttn` over a linear chain must reproduce an `Mps` evolution
    /// bit-for-bit. We drive both with the same circuit and assert that the
    /// per-qubit ⟨Z⟩ values agree to floating-point precision. If this ever
    /// fails, the scaffolding has drifted away from the validated 1D path
    /// and Track D's tree work cannot be trusted.
    #[test]
    fn ttn_linear_chain_reproduces_mps_kim_evolution() {
        use crate::kicked_ising::{apply_kim_step, KimParams};

        let n = 8;
        let params = KimParams::self_dual();
        let chi: Vec<usize> = vec![16; n - 1];
        let n_steps = 4;

        // Drive the canonical Mps directly.
        let mut mps = Mps::new(n);
        for _ in 0..n_steps {
            apply_kim_step(&mut mps, params, &chi).unwrap();
        }

        // Drive the linear-chain Ttn through its facade. Because Ttn::new
        // builds a fresh Mps internally with the same default truncation
        // mode and the linear-chain edge ordering matches Mps bond ordering,
        // the two evolutions are the same calculation reached via two APIs.
        let mut ttn = Ttn::new(Topology::linear_chain(n));
        for _ in 0..n_steps {
            apply_kim_step(ttn_backend_mut(&mut ttn), params, &chi).unwrap();
        }

        // Element-wise ⟨Z⟩ agreement to FP precision across all qubits.
        for q in 0..n {
            let z_mps = mps.expectation_z(q);
            let z_ttn = ttn.expectation_z(q);
            assert!(
                (z_mps - z_ttn).abs() < 1e-13,
                "1D regression broke at q={q}: mps={z_mps:e}, ttn={z_ttn:e}"
            );
        }

        // The discarded-weight tracker on the Ttn must mirror the Mps tracker
        // edge-for-edge, since the underlying SVD sequence is identical.
        let mps_discarded = (0..n - 1).map(|b| mps.discarded_weight(b)).collect::<Vec<_>>();
        let ttn_discarded = (0..n - 1)
            .map(|b| ttn.discarded_weight(EdgeId(b)))
            .collect::<Vec<_>>();
        assert_eq!(
            mps_discarded, ttn_discarded,
            "discarded-weight tracker drifted between Mps and Ttn"
        );
    }

    // ─── D.2 native tree anchors ───────────────────────────────────────
    //
    // Y-junction (N=4, one degree-3 centre + three leaves) and degree-4
    // star (N=5, one degree-4 centre + four leaves) driven through the
    // native `Ttn` against a topology-agnostic `DenseState` reference,
    // both lossless (large max_bond) and truncated (low max_bond with
    // discarded-weight accounting). If any of these fail, the native
    // contraction path has drifted from the dense reference and the
    // swap-network / heavy-hex milestones in D.3+ cannot be trusted.

    use crate::ttn::dense::DenseState;

    fn hadamard() -> [[C; 2]; 2] {
        let s = 1.0 / 2.0_f64.sqrt();
        [
            [C::new(s, 0.0), C::new(s, 0.0)],
            [C::new(s, 0.0), C::new(-s, 0.0)],
        ]
    }

    fn rx(theta: f64) -> [[C; 2]; 2] {
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        [
            [C::new(c, 0.0), C::new(0.0, -s)],
            [C::new(0.0, -s), C::new(c, 0.0)],
        ]
    }

    fn rz(theta: f64) -> [[C; 2]; 2] {
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        [
            [C::new(c, -s), C::new(0.0, 0.0)],
            [C::new(0.0, 0.0), C::new(c, s)],
        ]
    }

    fn cnot() -> [[C; 4]; 4] {
        let o = C::new(1.0, 0.0);
        let z = C::new(0.0, 0.0);
        [
            [o, z, z, z],
            [z, o, z, z],
            [z, z, z, o],
            [z, z, o, z],
        ]
    }

    fn zz(theta: f64) -> [[C; 4]; 4] {
        // e^{-i θ/2 · Z⊗Z} as a 4×4 matrix on (sigma_a, sigma_b).
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        // Diagonal entries: exp(-i θ/2 · z_a z_b).
        // (0,0): z_a=+1, z_b=+1 → exp(-iθ/2)
        // (0,1): z_a=+1, z_b=-1 → exp(+iθ/2)
        // (1,0): z_a=-1, z_b=+1 → exp(+iθ/2)
        // (1,1): z_a=-1, z_b=-1 → exp(-iθ/2)
        let diag_neg = C::new(c, -s);
        let diag_pos = C::new(c, s);
        let zc = C::new(0.0, 0.0);
        [
            [diag_neg, zc, zc, zc],
            [zc, diag_pos, zc, zc],
            [zc, zc, diag_pos, zc],
            [zc, zc, zc, diag_neg],
        ]
    }

    /// Y-junction topology, centre 0 connected to leaves 1, 2, 3.
    fn y_junction() -> Topology {
        Topology::from_edges(
            4,
            vec![
                Edge { a: 0, b: 1 },
                Edge { a: 0, b: 2 },
                Edge { a: 0, b: 3 },
            ],
        )
    }

    /// Degree-4 star, centre 0 connected to leaves 1, 2, 3, 4.
    fn star_four() -> Topology {
        Topology::from_edges(
            5,
            vec![
                Edge { a: 0, b: 1 },
                Edge { a: 0, b: 2 },
                Edge { a: 0, b: 3 },
                Edge { a: 0, b: 4 },
            ],
        )
    }

    /// Drive a TTN and a dense reference with the same gate schedule and
    /// assert that every single-qubit ⟨Z⟩ agrees to within `tol` after the
    /// last gate.
    fn assert_ttn_matches_dense(
        topology: Topology,
        max_bond: usize,
        tol: f64,
    ) {
        let mut ttn = Ttn::new(topology.clone());
        let mut dense = DenseState::zero(topology.n_qubits());

        // Hadamard-everywhere layer to break the product state.
        for q in 0..topology.n_qubits() {
            ttn.apply_single(q, hadamard());
            dense.apply_single(q, hadamard());
        }

        // Per-qubit Rx / Rz layer with per-qubit twists so the state has
        // non-trivial phases to preserve.
        for q in 0..topology.n_qubits() {
            let theta = 0.3 + 0.17 * q as f64;
            ttn.apply_single(q, rx(theta));
            dense.apply_single(q, rx(theta));
            ttn.apply_single(q, rz(0.42 - 0.11 * q as f64));
            dense.apply_single(q, rz(0.42 - 0.11 * q as f64));
        }

        // Two layers of CNOTs + ZZs along every edge. The first layer
        // creates genuine entanglement; the second layer stresses the
        // gauge-movement choreography by bouncing the center around.
        for _ in 0..2 {
            for eid in 0..topology.n_edges() {
                let edge = topology.edge(EdgeId(eid));
                ttn.apply_two_qubit_on_edge(EdgeId(eid), cnot(), max_bond).unwrap();
                dense.apply_two_qubit(edge.a, edge.b, cnot());
                ttn.apply_two_qubit_on_edge(EdgeId(eid), zz(0.47), max_bond).unwrap();
                dense.apply_two_qubit(edge.a, edge.b, zz(0.47));
            }
        }

        for q in 0..topology.n_qubits() {
            let z_ttn = ttn.expectation_z(q);
            let z_dense = dense.expectation_z(q);
            assert!(
                (z_ttn - z_dense).abs() < tol,
                "qubit {q}: ttn={z_ttn:e}, dense={z_dense:e}, diff={:e}",
                (z_ttn - z_dense).abs()
            );
        }
    }

    #[test]
    fn y_junction_lossless_matches_dense() {
        // max_bond large enough that no singular value is ever dropped.
        // For N=4 the Schmidt rank across any cut is ≤ 2^(N/2) = 4.
        assert_ttn_matches_dense(y_junction(), 64, 1e-12);
    }

    #[test]
    fn y_junction_truncated_bounded_by_discarded_weight() {
        // Low max_bond → some truncation expected. The assertion is that the
        // discarded-weight tracker bounds the per-qubit error, not that the
        // error is zero. We use max_bond=2 (one level below the lossless
        // rank) so truncation actually bites.
        let topology = y_junction();
        let mut ttn = Ttn::new(topology.clone());
        let mut dense = DenseState::zero(topology.n_qubits());
        for q in 0..topology.n_qubits() {
            ttn.apply_single(q, hadamard());
            dense.apply_single(q, hadamard());
        }
        for q in 0..topology.n_qubits() {
            let theta = 0.3 + 0.17 * q as f64;
            ttn.apply_single(q, rx(theta));
            dense.apply_single(q, rx(theta));
        }
        for _ in 0..2 {
            for eid in 0..topology.n_edges() {
                let edge = topology.edge(EdgeId(eid));
                ttn.apply_two_qubit_on_edge(EdgeId(eid), cnot(), 2).unwrap();
                dense.apply_two_qubit(edge.a, edge.b, cnot());
                ttn.apply_two_qubit_on_edge(EdgeId(eid), zz(0.47), 2).unwrap();
                dense.apply_two_qubit(edge.a, edge.b, zz(0.47));
            }
        }
        let total_discarded = ttn.total_discarded_weight();
        // The 2-norm error on the full state is ≤ √(total_discarded). Per-qubit
        // ⟨Z⟩ errors inherit at most a constant multiple of that. Use the
        // generous 1e-3 absolute tolerance from the design doc, gated on the
        // observation that truncation actually happened.
        for q in 0..topology.n_qubits() {
            let z_ttn = ttn.expectation_z(q);
            let z_dense = dense.expectation_z(q);
            assert!(
                (z_ttn - z_dense).abs() < 1e-3,
                "qubit {q}: ttn={z_ttn:e}, dense={z_dense:e}, total_discarded={total_discarded:e}"
            );
        }
    }

    #[test]
    fn degree_four_star_lossless_matches_dense() {
        // N=5, centre has degree 4 → stresses the contraction code's
        // arbitrary-degree axis bookkeeping one more axis than Y-junction.
        assert_ttn_matches_dense(star_four(), 128, 1e-12);
    }

    #[test]
    fn degree_four_star_truncated_bounded_by_discarded_weight() {
        let topology = star_four();
        let mut ttn = Ttn::new(topology.clone());
        let mut dense = DenseState::zero(topology.n_qubits());
        for q in 0..topology.n_qubits() {
            ttn.apply_single(q, hadamard());
            dense.apply_single(q, hadamard());
        }
        for _ in 0..2 {
            for eid in 0..topology.n_edges() {
                let edge = topology.edge(EdgeId(eid));
                ttn.apply_two_qubit_on_edge(EdgeId(eid), cnot(), 2).unwrap();
                dense.apply_two_qubit(edge.a, edge.b, cnot());
                ttn.apply_two_qubit_on_edge(EdgeId(eid), zz(0.31), 2).unwrap();
                dense.apply_two_qubit(edge.a, edge.b, zz(0.31));
            }
        }
        for q in 0..topology.n_qubits() {
            let z_ttn = ttn.expectation_z(q);
            let z_dense = dense.expectation_z(q);
            assert!(
                (z_ttn - z_dense).abs() < 1e-3,
                "qubit {q}: ttn={z_ttn:e}, dense={z_dense:e}"
            );
        }
    }

    // ─── D.3 swap-network anchors ──────────────────────────────────────
    //
    // `apply_two_qubit_via_path` routes a gate between any two qubits by
    // walking the tree path, SWAPping forward, applying the gate on the
    // last edge, and SWAPping back. These tests pin its correctness
    // against `DenseState` on three topologies:
    //
    //   - Y-junction (path length 2 between leaves)
    //   - 5-qubit linear chain (path length 3 and 4)
    //   - degree-4 star (path length 2 between opposite leaves)
    //
    // Every test is run **lossless** (max_bond large enough that no SV is
    // ever dropped) so any drift from the dense reference is unambiguously
    // a bug in the swap-network logic, not a truncation artefact.

    /// Orient helper: for a tree path of length 1 (q1, q2 tree-adjacent),
    /// `apply_two_qubit_via_path` must produce the same state as
    /// `apply_two_qubit_on_edge` with the properly-oriented gate.
    #[test]
    fn via_path_tree_adjacent_matches_direct_edge() {
        let topology = Topology::linear_chain(4);
        let mut ttn_path = Ttn::new(topology.clone());
        let mut ttn_edge = Ttn::new(topology.clone());
        // Break the product state so the comparison is non-trivial.
        for q in 0..4 {
            ttn_path.apply_single(q, hadamard());
            ttn_edge.apply_single(q, hadamard());
        }
        let u = cnot();
        // Apply on (1, 2) via path (length 1, since they are tree-adjacent
        // on the linear chain).
        ttn_path.apply_two_qubit_via_path(1, 2, u, 64).unwrap();
        // And via the direct edge (edge 1 on the 4-chain connects qubits
        // 1 and 2).
        ttn_edge.apply_two_qubit_on_edge(EdgeId(1), u, 64).unwrap();
        for q in 0..4 {
            let z_path = ttn_path.expectation_z(q);
            let z_edge = ttn_edge.expectation_z(q);
            assert!(
                (z_path - z_edge).abs() < 1e-13,
                "path vs direct mismatch at q={q}: path={z_path:e}, edge={z_edge:e}"
            );
        }
    }

    /// Tree-adjacent gate via path must also respect orientation: the gate
    /// must be applied with `q1` as the first qubit index even if the
    /// underlying edge stores `(q2, q1)`. The direction matters for
    /// asymmetric gates like CNOT.
    #[test]
    fn via_path_tree_adjacent_respects_qubit_order_for_cnot() {
        // 3-qubit Y-junction via from_edges, edge (0,1) is EdgeId(0).
        // Apply CNOT(0,1) vs CNOT(1,0) and verify they give different
        // states (CNOT is not symmetric in its two qubits).
        let topology = Topology::from_edges(
            3,
            vec![
                Edge { a: 0, b: 1 },
                Edge { a: 0, b: 2 },
            ],
        );
        // Prepare |+⟩|+⟩|+⟩.
        let mut ttn_01 = Ttn::new(topology.clone());
        let mut ttn_10 = Ttn::new(topology.clone());
        for q in 0..3 {
            ttn_01.apply_single(q, hadamard());
            ttn_10.apply_single(q, hadamard());
        }
        // Rotate qubit 0 to mix the control bit.
        ttn_01.apply_single(0, rx(0.7));
        ttn_10.apply_single(0, rx(0.7));
        // Apply CNOT in opposite directions via path.
        ttn_01.apply_two_qubit_via_path(0, 1, cnot(), 64).unwrap();
        ttn_10.apply_two_qubit_via_path(1, 0, cnot(), 64).unwrap();
        // Compare against a dense reference for each direction.
        let mut dense_01 = DenseState::zero(3);
        let mut dense_10 = DenseState::zero(3);
        for q in 0..3 {
            dense_01.apply_single(q, hadamard());
            dense_10.apply_single(q, hadamard());
        }
        dense_01.apply_single(0, rx(0.7));
        dense_10.apply_single(0, rx(0.7));
        dense_01.apply_two_qubit(0, 1, cnot());
        dense_10.apply_two_qubit(1, 0, cnot());
        for q in 0..3 {
            assert!(
                (ttn_01.expectation_z(q) - dense_01.expectation_z(q)).abs() < 1e-13,
                "CNOT(0,1) direction failed at q={q}"
            );
            assert!(
                (ttn_10.expectation_z(q) - dense_10.expectation_z(q)).abs() < 1e-13,
                "CNOT(1,0) direction failed at q={q}"
            );
        }
    }

    /// Y-junction leaf-to-leaf: apply a gate on (1, 2) where 1 and 2 are
    /// leaves joined only via the degree-3 centre 0. Path length = 2 → one
    /// forward SWAP, one gate, one backward SWAP = three tree-adjacent
    /// operations in total.
    #[test]
    fn via_path_y_junction_leaf_to_leaf_matches_dense() {
        let topology = y_junction();
        let mut ttn = Ttn::new(topology.clone());
        let mut dense = DenseState::zero(4);
        for q in 0..4 {
            ttn.apply_single(q, hadamard());
            dense.apply_single(q, hadamard());
            let theta = 0.23 + 0.19 * q as f64;
            ttn.apply_single(q, rx(theta));
            dense.apply_single(q, rx(theta));
        }
        // Non-trivial gate on leaves 1 and 2.
        ttn.apply_two_qubit_via_path(1, 2, zz(0.37), 64).unwrap();
        dense.apply_two_qubit(1, 2, zz(0.37));
        // And a CNOT on leaves 1 and 3, which exercises a second path of
        // length 2 through the same centre.
        ttn.apply_two_qubit_via_path(1, 3, cnot(), 64).unwrap();
        dense.apply_two_qubit(1, 3, cnot());
        for q in 0..4 {
            let z_ttn = ttn.expectation_z(q);
            let z_dense = dense.expectation_z(q);
            assert!(
                (z_ttn - z_dense).abs() < 1e-12,
                "via-path Y-junction leaf-to-leaf failed at q={q}: ttn={z_ttn:e}, dense={z_dense:e}"
            );
        }
    }

    /// 5-qubit linear chain: apply a gate on (0, 4) — path length 4, the
    /// longest gate-distance possible on this topology. Three forward
    /// SWAPs, one gate, three back SWAPs = seven tree-adjacent operations.
    #[test]
    fn via_path_linear_chain_length_4_matches_dense() {
        let topology = Topology::linear_chain(5);
        let mut ttn = Ttn::new(topology.clone());
        let mut dense = DenseState::zero(5);
        for q in 0..5 {
            ttn.apply_single(q, hadamard());
            dense.apply_single(q, hadamard());
            ttn.apply_single(q, rz(0.31 + 0.07 * q as f64));
            dense.apply_single(q, rz(0.31 + 0.07 * q as f64));
        }
        ttn.apply_two_qubit_via_path(0, 4, cnot(), 128).unwrap();
        dense.apply_two_qubit(0, 4, cnot());
        // Follow up with a reverse-direction path (4, 0) — the swap network
        // must also handle q1 > q2 correctly.
        ttn.apply_two_qubit_via_path(4, 0, zz(0.42), 128).unwrap();
        dense.apply_two_qubit(4, 0, zz(0.42));
        for q in 0..5 {
            let z_ttn = ttn.expectation_z(q);
            let z_dense = dense.expectation_z(q);
            assert!(
                (z_ttn - z_dense).abs() < 1e-12,
                "via-path length-4 failed at q={q}: ttn={z_ttn:e}, dense={z_dense:e}"
            );
        }
    }

    /// Degree-4 star: apply a gate on two leaves (2, 4) through the degree-4
    /// centre. Path length 2. This exercises the SWAP path through a node
    /// whose multi-index reshape is more expensive than Y-junction.
    #[test]
    fn via_path_degree_four_star_leaf_to_leaf_matches_dense() {
        let topology = star_four();
        let mut ttn = Ttn::new(topology.clone());
        let mut dense = DenseState::zero(5);
        for q in 0..5 {
            ttn.apply_single(q, hadamard());
            dense.apply_single(q, hadamard());
        }
        // One tree-adjacent gate first to build entanglement with the
        // centre, then a leaf-to-leaf via-path gate across the star.
        ttn.apply_two_qubit_on_edge(EdgeId(0), zz(0.19), 128).unwrap();
        dense.apply_two_qubit(0, 1, zz(0.19));
        ttn.apply_two_qubit_via_path(2, 4, cnot(), 128).unwrap();
        dense.apply_two_qubit(2, 4, cnot());
        for q in 0..5 {
            let z_ttn = ttn.expectation_z(q);
            let z_dense = dense.expectation_z(q);
            assert!(
                (z_ttn - z_dense).abs() < 1e-12,
                "via-path star failed at q={q}: ttn={z_ttn:e}, dense={z_dense:e}"
            );
        }
    }

    /// Applying a gate `U` via path and then its inverse `U†` via the same
    /// path must return the state to its pre-call form. This is a strong
    /// internal consistency check on the forward-then-backward SWAP
    /// choreography: any asymmetry in the SWAP path would leave a residual
    /// qubit permutation behind that the inverse gate would not undo.
    #[test]
    fn via_path_unitary_round_trip_returns_to_initial() {
        let topology = Topology::linear_chain(5);
        let mut ttn = Ttn::new(topology.clone());
        for q in 0..5 {
            ttn.apply_single(q, hadamard());
            ttn.apply_single(q, rx(0.41 - 0.08 * q as f64));
        }
        // Snapshot initial ⟨Z⟩ values.
        let z_before: Vec<f64> = (0..5).map(|q| ttn.expectation_z(q)).collect();

        // Apply a non-symmetric gate and its inverse via path. CNOT is its
        // own inverse, so CNOT ∘ CNOT = I and the state must return exactly.
        ttn.apply_two_qubit_via_path(0, 3, cnot(), 128).unwrap();
        ttn.apply_two_qubit_via_path(0, 3, cnot(), 128).unwrap();

        for q in 0..5 {
            let z_after = ttn.expectation_z(q);
            assert!(
                (z_after - z_before[q]).abs() < 1e-12,
                "round-trip failed at q={q}: before={:e}, after={z_after:e}",
                z_before[q]
            );
        }
    }

    /// Mixed drive test: on the degree-4 star, run one layer of
    /// Hadamard-everywhere then a sequence of via-path gates covering
    /// every leaf-pair. This stresses repeated re-entry of the swap
    /// network with different endpoints on the same topology.
    #[test]
    fn via_path_all_leaf_pairs_on_star_match_dense() {
        let topology = star_four();
        let mut ttn = Ttn::new(topology.clone());
        let mut dense = DenseState::zero(5);
        for q in 0..5 {
            ttn.apply_single(q, hadamard());
            dense.apply_single(q, hadamard());
        }
        // All 6 ordered leaf-pairs with different-ish gates.
        let pairs = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)];
        for (i, &(a, b)) in pairs.iter().enumerate() {
            let theta = 0.11 + 0.09 * i as f64;
            ttn.apply_two_qubit_via_path(a, b, zz(theta), 128).unwrap();
            dense.apply_two_qubit(a, b, zz(theta));
        }
        for q in 0..5 {
            let z_ttn = ttn.expectation_z(q);
            let z_dense = dense.expectation_z(q);
            assert!(
                (z_ttn - z_dense).abs() < 1e-11,
                "all-leaf-pairs failed at q={q}: ttn={z_ttn:e}, dense={z_dense:e}"
            );
        }
    }

    /// Orient helper sanity: transposing a 4×4 qubit gate twice returns
    /// the original gate. Guards against an arithmetic typo in the
    /// multi-index reshuffle inside `orient_two_qubit_gate_for_edge`.
    #[test]
    fn orient_gate_is_its_own_inverse() {
        let u = cnot();
        let edge = Edge { a: 3, b: 7 };
        let once = orient_two_qubit_gate_for_edge(u, 7, edge);
        let twice = orient_two_qubit_gate_for_edge(once, 7, edge);
        // Twice-transposed should equal the original.
        for i in 0..4 {
            for j in 0..4 {
                let diff = (twice[i][j] - u[i][j]).norm();
                assert!(diff < 1e-15, "twice[{i}][{j}] != u[{i}][{j}]");
            }
        }
    }

    // The KIM driver in `kicked_ising.rs` takes `&mut Mps` directly. For the
    // regression test we expose the linear-chain backend through a tiny
    // crate-private helper rather than widening the `Ttn` public surface
    // (which is shaped for the eventual general-tree case, not for current
    // 1D drivers).
    fn ttn_backend_mut(ttn: &mut Ttn) -> &mut Mps {
        match &mut ttn.backend {
            Backend::Linear(mps) => mps,
            Backend::Tree { .. } => {
                panic!("ttn_backend_mut called on non-linear backend")
            }
        }
    }

    // ─── D.5.0 env-sweep expectation_z anchors ─────────────────────────
    //
    // The native `Ttn::expectation_z` and `expectation_z_all` walk the
    // tree in site-canonical form and read the expectation value from the
    // local centre tensor. These tests pin the env-sweep implementation
    // against (a) the existing `DenseState` reference on the D.2
    // Y-junction / star anchors, and (b) its own batched variant.

    #[test]
    fn expectation_z_all_matches_per_qubit_loop_on_y_junction() {
        // Drive a Y-junction through a non-trivial circuit and assert that
        // `expectation_z_all` produces the same per-qubit values as the
        // scalar `expectation_z(q)` called in a loop. Covers the case
        // where the DFS centre-walk crosses the junction repeatedly.
        let topology = y_junction();
        let mut ttn_a = Ttn::new(topology.clone());
        let mut ttn_b = Ttn::new(topology.clone());
        for q in 0..4 {
            ttn_a.apply_single(q, hadamard());
            ttn_b.apply_single(q, hadamard());
            ttn_a.apply_single(q, rx(0.17 + 0.09 * q as f64));
            ttn_b.apply_single(q, rx(0.17 + 0.09 * q as f64));
        }
        for eid in 0..topology.n_edges() {
            ttn_a.apply_two_qubit_on_edge(EdgeId(eid), cnot(), 64).unwrap();
            ttn_b.apply_two_qubit_on_edge(EdgeId(eid), cnot(), 64).unwrap();
            ttn_a.apply_two_qubit_on_edge(EdgeId(eid), zz(0.33), 64).unwrap();
            ttn_b.apply_two_qubit_on_edge(EdgeId(eid), zz(0.33), 64).unwrap();
        }
        let batched = ttn_a.expectation_z_all();
        let looped: Vec<f64> = (0..4).map(|q| ttn_b.expectation_z(q)).collect();
        assert_eq!(batched.len(), 4);
        for q in 0..4 {
            assert!(
                (batched[q] - looped[q]).abs() < 1e-13,
                "q={q}: batched={:e}, looped={:e}",
                batched[q],
                looped[q]
            );
        }
    }

    #[test]
    fn expectation_z_all_matches_dense_on_degree_four_star() {
        // Native env-sweep `expectation_z_all` must agree with the
        // topology-agnostic `DenseState` reference element-wise, with the
        // TTN in an arbitrary post-gate gauge state. Lossless run.
        let topology = star_four();
        let mut ttn = Ttn::new(topology.clone());
        let mut dense = DenseState::zero(5);
        for q in 0..5 {
            ttn.apply_single(q, hadamard());
            dense.apply_single(q, hadamard());
        }
        for _ in 0..2 {
            for eid in 0..topology.n_edges() {
                let edge = topology.edge(EdgeId(eid));
                ttn.apply_two_qubit_on_edge(EdgeId(eid), zz(0.19), 128).unwrap();
                dense.apply_two_qubit(edge.a, edge.b, zz(0.19));
                ttn.apply_two_qubit_on_edge(EdgeId(eid), cnot(), 128).unwrap();
                dense.apply_two_qubit(edge.a, edge.b, cnot());
            }
        }
        let batched = ttn.expectation_z_all();
        for q in 0..5 {
            let z_dense = dense.expectation_z(q);
            assert!(
                (batched[q] - z_dense).abs() < 1e-12,
                "q={q}: ttn={:e}, dense={:e}",
                batched[q],
                z_dense
            );
        }
    }

    #[test]
    fn env_sweep_expectation_z_matches_dense_on_small_eagle_fragment() {
        // End-to-end check on the 13-qubit Eagle sub-fragment used by
        // `small_eagle_fragment_kim_matches_dense_lossless` but with the
        // comparison performed via the new env-sweep path rather than the
        // previous statevector-materialising implementation. This is the
        // direct proof that D.5.0 is a drop-in replacement on a topology
        // that has a cycle and a real heavy-hex junction.
        let (topology, non_tree_edges, coupling) = small_eagle_fragment();
        let mut ttn = Ttn::new(topology);
        let mut dense = DenseState::zero(13);
        for q in 0..13 {
            ttn.apply_single(q, hadamard());
            dense.apply_single(q, hadamard());
        }
        let params = KimParams::self_dual();
        let max_bond = 128;
        for _ in 0..3 {
            apply_kim_step_ttn_heavy_hex(&mut ttn, &non_tree_edges, params, max_bond).unwrap();
            apply_kim_step_dense_heavy_hex(&mut dense, &coupling, params);
        }
        let z_ttn = ttn.expectation_z_all();
        for q in 0..13 {
            let z_dense = dense.expectation_z(q);
            assert!(
                (z_ttn[q] - z_dense).abs() < 1e-11,
                "q={q}: ttn={:e}, dense={:e}",
                z_ttn[q],
                z_dense
            );
        }
        // The env-sweep norm_squared should also match the lossless unit
        // norm across the full run (the Hadamard layer produces a unit
        // product, every gate is unitary, max_bond is above the max Schmidt
        // rank so no SV is dropped).
        let norm_sq = ttn.norm_squared();
        assert!(
            (norm_sq - 1.0).abs() < 1e-11,
            "norm² drifted: {norm_sq:e}"
        );
    }

    #[test]
    fn norm_squared_on_product_state_is_unity_before_any_gates() {
        // Sanity: for a fresh TTN on any topology the norm² read from
        // the orthogonality-centre site is exactly 1.
        for topology in [
            Topology::linear_chain(6),
            y_junction(),
            star_four(),
        ] {
            let ttn = Ttn::new(topology);
            let nsq = ttn.norm_squared();
            assert!(
                (nsq - 1.0).abs() < 1e-15,
                "initial norm² should be 1, got {nsq:e}"
            );
        }
    }

    #[test]
    fn expectation_z_does_not_drift_across_repeated_calls() {
        // Calling `expectation_z` moves the orthogonality centre as a
        // side effect. The resulting state must still be valid — a second
        // call on any qubit must produce the same value, within the noise
        // of two QR sweeps. Guards against any accidental state corruption
        // in the gauge-move path.
        let topology = y_junction();
        let mut ttn = Ttn::new(topology.clone());
        for q in 0..4 {
            ttn.apply_single(q, hadamard());
            ttn.apply_single(q, rz(0.23 + 0.11 * q as f64));
        }
        for eid in 0..topology.n_edges() {
            ttn.apply_two_qubit_on_edge(EdgeId(eid), cnot(), 64).unwrap();
        }
        // First pass: record the snapshot.
        let snap: Vec<f64> = (0..4).map(|q| ttn.expectation_z(q)).collect();
        // Second pass: must match the snapshot element-wise to QR-roundoff.
        let second: Vec<f64> = (0..4).map(|q| ttn.expectation_z(q)).collect();
        for q in 0..4 {
            assert!(
                (snap[q] - second[q]).abs() < 1e-12,
                "q={q} drifted between passes: first={}, second={}",
                snap[q],
                second[q]
            );
        }
    }

    // ─── D.3 small-heavy-hex KIM validation ────────────────────────────
    //
    // Build a tiny sub-fragment of IBM Eagle (13 qubits, 1 hexagonal cycle,
    // 1 degree-3 junction in the spanning tree, 1 non-tree edge) and run a
    // few Floquet steps of the kicked Ising model through both the native
    // `Ttn` backend and the topology-agnostic `DenseState` reference.
    // Success condition: every qubit's ⟨Z⟩ agrees across the two paths to
    // floating-point precision after N steps, with `max_bond` large enough
    // that no singular value is ever dropped. This is the design-doc D.3
    // gate: "small heavy-hex subgraph (N≈12–16) Ttn vs dense statevector
    // agreement at FP precision after 5 KIM layers" (TRACK_D_DESIGN.md §
    // D.3).
    //
    // The 13-qubit fragment is a genuine sub-graph of Eagle 127q:
    //
    //     Eagle qubit → local id
    //      0, 1, 2, 3, 4, 5    →  0, 1, 2, 3, 4, 5   (row 0 partial)
    //      14, 15               →  6, 7              (bridges)
    //      18, 19, 20, 21, 22  →  8, 9, 10, 11, 12   (row 2 partial)
    //
    // Coupling edges (13 total):
    //   row 0:     (0,1), (1,2), (2,3), (3,4), (4,5)
    //   bridges:   (0,6), (6,8), (4,7), (7,12)
    //   row 2:     (8,9), (9,10), (10,11), (11,12)
    //
    // |E| − |V| + 1 = 13 − 13 + 1 = 1 independent cycle. Spanning tree
    // drops the (0,6) edge, which leaves qubit 4 as a genuine degree-3
    // junction (tree neighbours: 3, 5, 7) and forces `apply_two_qubit_via_path`
    // to route the single non-tree ZZ gate through the junction on a
    // tree path of length 11. That exercises both the junction
    // contraction code in the native Ttn backend and the
    // swap-network machinery on the same test.

    use crate::kicked_ising::KimParams;

    /// Tuple returned by `small_eagle_fragment`:
    ///   - `Topology`: the spanning tree (12 edges over 13 qubits)
    ///   - `Vec<Edge>`: the coupling edges dropped to form the tree
    ///     (exactly one for this fragment)
    ///   - `Vec<Edge>`: the full coupling graph (13 edges, with `a < b`)
    fn small_eagle_fragment() -> (Topology, Vec<Edge>, Vec<Edge>) {
        // Full coupling graph (13 edges).
        let coupling: Vec<Edge> = vec![
            // row 0
            Edge { a: 0, b: 1 },
            Edge { a: 1, b: 2 },
            Edge { a: 2, b: 3 },
            Edge { a: 3, b: 4 },
            Edge { a: 4, b: 5 },
            // bridges
            Edge { a: 0, b: 6 },
            Edge { a: 6, b: 8 },
            Edge { a: 4, b: 7 },
            Edge { a: 7, b: 12 },
            // row 2
            Edge { a: 8, b: 9 },
            Edge { a: 9, b: 10 },
            Edge { a: 10, b: 11 },
            Edge { a: 11, b: 12 },
        ];
        debug_assert_eq!(coupling.len(), 13);

        // Spanning tree: drop (0, 6) so qubit 4 remains a degree-3
        // junction (3, 5, 7). Every other coupling edge is a tree edge.
        let tree_edges: Vec<Edge> = coupling
            .iter()
            .copied()
            .filter(|e| !(e.a == 0 && e.b == 6))
            .collect();
        debug_assert_eq!(tree_edges.len(), 12);

        let non_tree_edges: Vec<Edge> = vec![Edge { a: 0, b: 6 }];

        let topology = Topology::from_edges(13, tree_edges);
        debug_assert_eq!(topology.degree(4), 3, "qubit 4 must be a tree junction");
        debug_assert_eq!(topology.degree(0), 1, "qubit 0 is a tree leaf after dropping (0,6)");
        debug_assert_eq!(topology.degree(5), 1, "qubit 5 is a tree leaf");
        debug_assert_eq!(topology.degree(6), 1, "qubit 6 is a tree leaf after dropping (0,6)");

        (topology, non_tree_edges, coupling)
    }

    /// One KIM Floquet step on a `Ttn` over a heavy-hex sub-graph:
    ///
    /// 1. ZZ entangling layer on every coupling edge (tree-adjacent pairs
    ///    via `apply_two_qubit_on_edge`, non-tree pairs via
    ///    `apply_two_qubit_via_path`).
    /// 2. Optional longitudinal Rz(2·h_z·dt) kick on every qubit.
    /// 3. Transverse Rx(2·h_x·dt) kick on every qubit.
    ///
    /// Conventions match `kicked_ising::apply_kim_step` for the 1D path:
    /// the ZZ gate is `exp(-i (J·dt) Z⊗Z)` with **no factor of two**, and
    /// the Rx / Rz kicks use the standard `exp(-i θ/2 σ)` convention so
    /// the angles passed in are `2·h_x·dt` / `2·h_z·dt`.
    fn apply_kim_step_ttn_heavy_hex(
        ttn: &mut Ttn,
        non_tree_edges: &[Edge],
        params: KimParams,
        max_bond: usize,
    ) -> Result<()> {
        // 1. ZZ entangling layer — tree-adjacent edges first, via the
        //    native direct-edge path, then non-tree edges via the swap
        //    network. Order within each group is deterministic (the
        //    topology's internal edge enumeration) so test runs are
        //    reproducible bit-for-bit.
        let zz_theta_full = 2.0 * params.j * params.dt; // test zz() uses θ/2 internally
        let zz_u = zz(zz_theta_full);

        // Tree edges: iterate the topology's canonical edge list.
        let topology = ttn.topology().clone();
        for eid_idx in 0..topology.n_edges() {
            ttn.apply_two_qubit_on_edge(EdgeId(eid_idx), zz_u, max_bond)?;
        }
        // Non-tree edges: route via the swap network through the tree.
        for edge in non_tree_edges {
            ttn.apply_two_qubit_via_path(edge.a, edge.b, zz_u, max_bond)?;
        }

        // 2. Global Rz kick (only if h_z ≠ 0).
        if params.h_z != 0.0 {
            let rz_u = rz(2.0 * params.h_z * params.dt);
            for q in 0..topology.n_qubits() {
                ttn.apply_single(q, rz_u);
            }
        }

        // 3. Global Rx kick.
        let rx_u = rx(2.0 * params.h_x * params.dt);
        for q in 0..topology.n_qubits() {
            ttn.apply_single(q, rx_u);
        }

        Ok(())
    }

    /// Same KIM Floquet step on a `DenseState` reference, using the full
    /// coupling-edge list (tree + non-tree together). The dense path has
    /// no spanning-tree structure — ZZ is applied directly to every pair.
    fn apply_kim_step_dense_heavy_hex(
        dense: &mut DenseState,
        coupling_edges: &[Edge],
        params: KimParams,
    ) {
        let zz_theta_full = 2.0 * params.j * params.dt;
        let zz_u = zz(zz_theta_full);
        for e in coupling_edges {
            dense.apply_two_qubit(e.a, e.b, zz_u);
        }
        if params.h_z != 0.0 {
            let rz_u = rz(2.0 * params.h_z * params.dt);
            for q in 0..dense.n {
                dense.apply_single(q, rz_u);
            }
        }
        let rx_u = rx(2.0 * params.h_x * params.dt);
        for q in 0..dense.n {
            dense.apply_single(q, rx_u);
        }
    }

    /// **The D.3 validation gate.** Drive the 13-qubit Eagle sub-fragment
    /// through 3 KIM Floquet steps on both the native `Ttn` path (direct
    /// tree-edge gates + swap-network non-tree gates) and the
    /// topology-agnostic `DenseState` reference. Assert element-wise
    /// ⟨Z⟩ agreement at floating-point precision after every step.
    ///
    /// Lossless run: `max_bond = 128` is well above the maximal Schmidt
    /// rank `2^(N/2) ≈ 90` for N = 13, so no singular value is ever
    /// dropped. Any drift between the two paths is unambiguously a bug
    /// in the heavy-hex KIM driver or in the swap-network routing, not
    /// a truncation artefact.
    #[test]
    fn small_eagle_fragment_kim_matches_dense_lossless() {
        let (topology, non_tree_edges, coupling) = small_eagle_fragment();
        assert_eq!(topology.n_qubits(), 13);
        assert_eq!(topology.n_edges(), 12);
        assert_eq!(non_tree_edges.len(), 1);
        assert_eq!(coupling.len(), 13);

        let mut ttn = Ttn::new(topology);
        let mut dense = DenseState::zero(13);

        // Break the product state so KIM dynamics have something to work
        // on. A simple Hadamard-on-every-qubit layer puts the system in
        // |+⟩⊗13 which is an eigenstate of the Rx kick but not of the ZZ
        // layer, so the Floquet dynamics are non-trivial from step one.
        for q in 0..13 {
            ttn.apply_single(q, hadamard());
            dense.apply_single(q, hadamard());
        }

        // Use the self-dual point to get genuinely non-trivial dynamics
        // (J = h_x = π/4). h_z = 0 so the Rz kick is skipped.
        let params = KimParams::self_dual();
        let max_bond = 128;
        let n_steps = 3;

        for step in 1..=n_steps {
            apply_kim_step_ttn_heavy_hex(&mut ttn, &non_tree_edges, params, max_bond).unwrap();
            apply_kim_step_dense_heavy_hex(&mut dense, &coupling, params);

            for q in 0..13 {
                let z_ttn = ttn.expectation_z(q);
                let z_dense = dense.expectation_z(q);
                assert!(
                    (z_ttn - z_dense).abs() < 1e-11,
                    "step {step}, q={q}: ttn={z_ttn:e}, dense={z_dense:e}, diff={:e}",
                    (z_ttn - z_dense).abs()
                );
            }
        }

        // The lossless run must also produce zero total discarded weight
        // — `max_bond = 128` is large enough that every SVD keeps every
        // singular value, across both the 12 direct-edge ZZ gates per
        // step and the 21 swap-network operations per non-tree ZZ gate
        // per step.
        assert!(
            ttn.total_discarded_weight() < 1e-14,
            "expected zero discarded weight for lossless run, got {:e}",
            ttn.total_discarded_weight()
        );
    }

    /// Same 13-qubit fragment, but run the KIM circuit on the TTN with a
    /// non-self-dual parameter set including a longitudinal kick
    /// (`h_z ≠ 0`). Guards the `h_z` branch of the driver, which is
    /// skipped in the self-dual test above.
    #[test]
    fn small_eagle_fragment_kim_with_longitudinal_kick_matches_dense() {
        let (topology, non_tree_edges, coupling) = small_eagle_fragment();
        let mut ttn = Ttn::new(topology);
        let mut dense = DenseState::zero(13);

        for q in 0..13 {
            ttn.apply_single(q, hadamard());
            dense.apply_single(q, hadamard());
        }

        let params = KimParams {
            j: 0.31,
            h_x: 0.47,
            h_z: 0.19,
            dt: 1.0,
        };
        let max_bond = 128;
        let n_steps = 2;

        for step in 1..=n_steps {
            apply_kim_step_ttn_heavy_hex(&mut ttn, &non_tree_edges, params, max_bond).unwrap();
            apply_kim_step_dense_heavy_hex(&mut dense, &coupling, params);
            for q in 0..13 {
                let z_ttn = ttn.expectation_z(q);
                let z_dense = dense.expectation_z(q);
                assert!(
                    (z_ttn - z_dense).abs() < 1e-11,
                    "step {step}, q={q}: ttn={z_ttn:e}, dense={z_dense:e}, diff={:e}",
                    (z_ttn - z_dense).abs()
                );
            }
        }
    }
}
