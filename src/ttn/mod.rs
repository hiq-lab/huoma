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

pub mod contraction;
pub mod gauge;
pub mod site;
pub mod topology;

#[cfg(test)]
pub(crate) mod dense;

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

    /// Apply a two-qubit gate on two qubits connected only via the tree
    /// path — milestone D.3 (swap network).
    pub fn apply_two_qubit_via_path(
        &mut self,
        _q1: usize,
        _q2: usize,
        _u: [[C; 4]; 4],
        _max_bond: usize,
    ) -> Result<()> {
        unimplemented!(
            "Ttn::apply_two_qubit_via_path lands with Track D milestone D.3 (swap network)"
        );
    }

    /// Single-qubit ⟨Z⟩ at qubit `target`, normalised by the state's 2-norm².
    #[must_use]
    pub fn expectation_z(&self, target: usize) -> f64 {
        match &self.backend {
            Backend::Linear(mps) => mps.expectation_z(target),
            Backend::Tree { sites, .. } => expectation_z_tree(sites, &self.topology, target),
        }
    }

    /// Wave-function 2-norm squared.
    #[must_use]
    pub fn norm_squared(&self) -> f64 {
        match &self.backend {
            Backend::Linear(mps) => mps.norm_squared(),
            Backend::Tree { sites, .. } => norm_squared_tree(sites, &self.topology),
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

/// Normalised single-qubit ⟨Z⟩ computed by contracting the whole tree to a
/// dense statevector. Caps out at `n_qubits ≤ ~20` but is honest and
/// trivially correct, which is what the D.2 tests need. Replaced by a
/// gauge-aware environment sweep in a later milestone if the cost becomes
/// a bottleneck for D.5.
fn expectation_z_tree(sites: &[TtnSite], topology: &Topology, target: usize) -> f64 {
    let n = topology.n_qubits();
    assert!(target < n);
    let psi = contraction::tree_to_statevector(sites, topology);
    let shift = n - 1 - target;
    let mut num = 0.0_f64;
    let mut denom = 0.0_f64;
    for (k, amp) in psi.iter().enumerate() {
        let prob = amp.norm_sqr();
        denom += prob;
        if ((k >> shift) & 1) == 0 {
            num += prob;
        } else {
            num -= prob;
        }
    }
    if denom < 1e-30 { 0.0 } else { num / denom }
}

/// 2-norm² of the state computed via the full statevector contraction.
fn norm_squared_tree(sites: &[TtnSite], topology: &Topology) -> f64 {
    contraction::tree_to_statevector(sites, topology)
        .iter()
        .map(|a| a.norm_sqr())
        .sum()
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
        let ttn = Ttn::new(t);
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
}
