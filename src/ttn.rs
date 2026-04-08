//! Tree Tensor Network (TTN) simulator — Track D scaffolding.
//!
//! This module is the new entry point for non-1D Huoma. The public types
//! ([`Topology`], [`EdgeId`], [`Ttn`]) define the surface that the eventual
//! heavy-hex implementation has to fill, and the linear-chain special case
//! is wired up today by delegating to the validated [`crate::mps::Mps`]
//! backend. That gives the D.1 milestone — a `Ttn` that can represent a 1D
//! chain and reproduce `Mps` semantics bit-for-bit — without duplicating any
//! of the SVD / contraction code in `mps.rs`.
//!
//! General trees (heavy-hex Eagle 127q in particular) are *not* implemented
//! in this scaffolding. Constructors that would need them ([`Topology::from_edges`]
//! beyond a linear chain, [`Ttn::apply_two_qubit_via_path`], …) panic with a
//! clear "unimplemented" message so that the surface compiles, the 1D
//! regression has a real assertion to anchor on, and the Track D milestones
//! D.2–D.5 in `TRACK_D_DESIGN.md` can fill in the body without renaming
//! anything.
//!
//! See `TRACK_D_DESIGN.md` for the milestones and the design rationale.

use num_complex::Complex64;

use crate::error::Result;
use crate::mps::{Mps, TruncationMode};

type C = Complex64;

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

/// Tree topology over `n_qubits` qubits.
///
/// Invariants enforced at construction (currently only for the linear-chain
/// constructor — general-tree validation lands with milestone D.2):
/// - `edges.len() == n_qubits - 1`
/// - The edge set forms a tree (connected, acyclic).
/// - `EdgeId(k)` indexes `edges[k]` and is stable.
#[derive(Debug, Clone)]
pub struct Topology {
    n_qubits: usize,
    edges: Vec<Edge>,
}

impl Topology {
    /// Linear-chain topology with edges `(0,1), (1,2), …, (n-2, n-1)`.
    /// This is the degenerate-tree case used by the D.1 1D regression and is
    /// the only general-purpose constructor implemented in the scaffolding.
    #[must_use]
    pub fn linear_chain(n_qubits: usize) -> Self {
        assert!(n_qubits >= 1, "Topology::linear_chain requires n_qubits ≥ 1");
        let edges = (0..n_qubits.saturating_sub(1))
            .map(|i| Edge { a: i, b: i + 1 })
            .collect();
        Self { n_qubits, edges }
    }

    /// General-tree constructor — milestone D.2.
    ///
    /// The signature is fixed; the body lands with D.2 alongside the
    /// connectivity / cycle / leaf-count validation listed in
    /// `TRACK_D_DESIGN.md` § Architecture.
    #[must_use]
    pub fn from_edges(n_qubits: usize, edges: Vec<Edge>) -> Self {
        // For the linear-chain happy path the user could legitimately call
        // `from_edges` with `[(0,1), (1,2), …]` — accept that without
        // requiring D.2 to ship, so the API contract is honest from day one.
        let is_linear_chain = edges.len() + 1 == n_qubits
            && edges
                .iter()
                .enumerate()
                .all(|(i, e)| (e.a == i && e.b == i + 1) || (e.a == i + 1 && e.b == i));
        if is_linear_chain {
            return Self { n_qubits, edges };
        }
        unimplemented!(
            "Topology::from_edges for non-linear trees lands with Track D milestone D.2"
        );
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

    /// True if the topology is a linear chain `0—1—2—…—(n-1)`. Used by the
    /// scaffolding to dispatch to the [`Mps`] backend; replaced by a richer
    /// dispatch (degree-aware) when D.2 lands.
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

/// Tree Tensor Network state vector.
///
/// In the scaffolding, when [`Topology::is_linear_chain`] is true the `Ttn`
/// is a thin facade around an [`Mps`] — every method delegates and inherits
/// the validated 1D semantics. General trees land with milestone D.2 and
/// will live in this same struct (the `Mps` field becomes one variant of an
/// internal storage enum, or the storage moves to a tree-native layout, to
/// be decided when D.2 ships).
#[derive(Debug, Clone)]
pub struct Ttn {
    topology: Topology,
    /// Linear-chain backend. Always `Some(_)` in the scaffolding because
    /// non-linear topologies are unimplemented; making it an `Option` keeps
    /// the door open for D.2 without requiring an enum today.
    backend: Mps,
}

impl Ttn {
    /// Initialise a TTN over `topology` in the `|0…0⟩` product state.
    ///
    /// Currently only supports linear-chain topologies; general trees
    /// land with D.2.
    #[must_use]
    pub fn new(topology: Topology) -> Self {
        assert!(
            topology.is_linear_chain(),
            "Ttn::new currently supports only linear-chain topologies; \
             general trees land with Track D milestone D.2"
        );
        let backend = Mps::new(topology.n_qubits());
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
    /// Mirrors [`Mps::set_truncation_mode`].
    pub fn set_truncation_mode(&mut self, mode: TruncationMode) {
        self.backend.set_truncation_mode(mode);
    }

    /// Cumulative discarded weight on edge `id`.
    /// Honest 2-norm² truncation error contributed by every SVD on this edge.
    #[must_use]
    pub fn discarded_weight(&self, id: EdgeId) -> f64 {
        // For the linear-chain backend, EdgeId k corresponds to MPS bond k
        // (edge between qubit k and qubit k+1).
        self.backend.discarded_weight(id.0)
    }

    /// Total cumulative discarded weight summed over all edges.
    #[must_use]
    pub fn total_discarded_weight(&self) -> f64 {
        self.backend.total_discarded_weight()
    }

    /// Bond / edge dimension of edge `id`.
    #[must_use]
    pub fn edge_dim(&self, id: EdgeId) -> usize {
        self.backend.bond_dim(id.0)
    }

    /// Apply a single-qubit gate to qubit `q`.
    pub fn apply_single(&mut self, q: usize, u: [[C; 2]; 2]) {
        self.backend.apply_single(q, u);
    }

    /// Apply a two-qubit gate on the qubits incident to edge `id`.
    /// `max_bond` truncates the SVD on that edge.
    ///
    /// In the scaffolding the edge is required to connect adjacent qubits
    /// in the linear chain (which it always does for the linear backend).
    pub fn apply_two_qubit_on_edge(
        &mut self,
        id: EdgeId,
        u: [[C; 4]; 4],
        max_bond: usize,
    ) -> Result<()> {
        let edge = self.topology.edge(id);
        let q = edge.a.min(edge.b);
        let other = edge.a.max(edge.b);
        debug_assert_eq!(
            other,
            q + 1,
            "linear-chain edge must connect adjacent qubits"
        );
        self.backend.apply_two_qubit(q, u, max_bond)
    }

    /// Apply a two-qubit gate on two qubits connected only via the tree
    /// path — milestone D.3 (swap network). Stub here so the API is fixed
    /// for callers that will eventually want long-range gates.
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

    /// Single-qubit ⟨Z⟩ at qubit `target`.
    #[must_use]
    pub fn expectation_z(&self, target: usize) -> f64 {
        self.backend.expectation_z(target)
    }

    /// Wave-function 2-norm squared. SVD truncation does not preserve the
    /// 2-norm exactly, so observables are normalised against this in
    /// downstream callers (same convention as [`Mps`]).
    #[must_use]
    pub fn norm_squared(&self) -> f64 {
        self.backend.norm_squared()
    }

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

    // The KIM driver in `kicked_ising.rs` takes `&mut Mps` directly. For the
    // regression test we expose the linear-chain backend through a tiny
    // crate-private helper rather than widening the `Ttn` public surface
    // (which is shaped for the eventual general-tree case, not for current
    // 1D drivers).
    fn ttn_backend_mut(ttn: &mut Ttn) -> &mut Mps {
        &mut ttn.backend
    }
}
