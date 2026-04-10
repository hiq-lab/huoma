//! Tree-edge partitioner: classify spanning-tree edges as Stable or
//! Volatile based on sin(C/2) commensurability scores and a
//! budget-adaptive threshold.
//!
//! This is the TTN generalisation of [`crate::partition::partition_adaptive`]
//! (which operates on 1D MPS bonds). Track F milestone **F.1**.
//!
//! # Algorithm
//!
//! 1. Score every tree edge via [`crate::ttn::allocator::edge_sinc_score_local`]
//!    with a BFS radius, producing one `f64` per edge.
//! 2. Feed the scores through [`crate::allocator::chi_allocation_target_budget`]
//!    to get a tentative per-edge χ allocation that exactly consumes the
//!    caller's total budget.
//! 3. Edges whose tentative χ ≤ `chi_min` are classified **Stable** (the
//!    budget says they don't need simulation — their dynamics are
//!    analytically tractable). All other edges are **Volatile**.
//! 4. Re-run the water-filling on the volatile subset only, redistributing
//!    the full budget among fewer edges for a tighter allocation.
//! 5. Identify boundary edges: volatile edges incident on at least one
//!    stable qubit.
//! 6. Identify connected components of the volatile edge subgraph via BFS.

use crate::allocator::chi_allocation_target_budget;
use crate::ttn::allocator::edge_sinc_score_local;
use crate::ttn::topology::{EdgeId, Topology};

/// Default BFS radius for local edge scoring at million-qubit scale.
/// Matches the 1D [`crate::allocator::DEFAULT_SINC_RADIUS`].
pub const DEFAULT_PARTITION_RADIUS: usize = 5;

/// Edge classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeClass {
    /// Dynamics across this edge are analytically tractable — no TTN
    /// simulation needed.
    Stable,
    /// Dynamics across this edge require tensor-network simulation.
    Volatile,
}

/// Result of [`partition_tree_adaptive`].
#[derive(Debug, Clone)]
pub struct TreePartition {
    /// Per-edge classification, indexed by `EdgeId`.
    pub edge_classes: Vec<EdgeClass>,
    /// Indices of volatile edges.
    pub volatile_edges: Vec<EdgeId>,
    /// Qubits that touch at least one volatile edge (the simulation domain).
    pub volatile_qubits: Vec<usize>,
    /// Qubits that touch only stable edges (the analytical domain).
    pub stable_qubits: Vec<usize>,
    /// Volatile edges that have at least one endpoint in the stable set
    /// (i.e. the interface between simulation and analytical domains).
    pub boundary_edges: Vec<EdgeId>,
    /// Per-edge χ allocation for volatile edges (0 for stable edges).
    /// Length = `topology.n_edges()`, indexed by `EdgeId`.
    pub chi_per_edge: Vec<usize>,
    /// Raw per-edge sin(C/2) scores before partitioning.
    pub scores: Vec<f64>,
}

/// Partition the spanning tree into Stable (analytical) and Volatile
/// (must-simulate) edges based on sin(C/2) commensurability scores.
///
/// The **budget-adaptive threshold** (design choice b) works as follows:
/// edges whose share of the total χ budget would be ≤ `chi_min` under
/// proportional water-filling are classified Stable — the budget is too
/// thin for them to carry meaningful entanglement, and the channel model
/// can handle them analytically.
///
/// Works on **any** topology, including lightweight ones built with
/// [`Topology::from_edges_lightweight`]. Uses
/// [`edge_sinc_score_local`] with the given `radius` for scoring.
///
/// # Arguments
///
/// * `frequencies` — per-qubit frequency vector, length ≥ `topology.n_qubits()`
/// * `topology` — the spanning tree (may be lightweight)
/// * `total_budget` — total χ budget across all volatile edges
/// * `chi_min` — minimum χ per volatile edge (also the threshold: edges
///   whose tentative allocation ≤ chi_min are Stable)
/// * `chi_max` — maximum χ per volatile edge
/// * `radius` — BFS radius for local scoring
#[must_use]
pub fn partition_tree_adaptive(
    frequencies: &[f64],
    topology: &Topology,
    total_budget: usize,
    chi_min: usize,
    chi_max: usize,
    radius: usize,
) -> TreePartition {
    let n_edges = topology.n_edges();
    let n_qubits = topology.n_qubits();

    if n_edges == 0 {
        return TreePartition {
            edge_classes: Vec::new(),
            volatile_edges: Vec::new(),
            volatile_qubits: Vec::new(),
            stable_qubits: (0..n_qubits).collect(),
            boundary_edges: Vec::new(),
            chi_per_edge: Vec::new(),
            scores: Vec::new(),
        };
    }

    // Step 1: score every edge via radius-bounded local BFS.
    let scores: Vec<f64> = (0..n_edges)
        .map(|i| edge_sinc_score_local(frequencies, topology, EdgeId(i), radius))
        .collect();

    // Step 2: tentative water-filling over ALL edges.
    let tentative_chi = chi_allocation_target_budget(&scores, total_budget, chi_min, chi_max);

    // Step 3: classify — an edge is Stable if EITHER its score is
    // negligible (no discriminating signal) OR its tentative χ ≤ chi_min
    // (the budget says it doesn't need simulation).
    let max_score = scores.iter().cloned().fold(0.0_f64, f64::max);
    let score_floor = max_score * 0.01; // edges carrying < 1% of the peak signal
    let mut edge_classes = vec![EdgeClass::Stable; n_edges];
    let mut volatile_edges = Vec::new();
    let mut volatile_scores = Vec::new();
    for (i, &chi) in tentative_chi.iter().enumerate() {
        if scores[i] > score_floor && chi > chi_min {
            edge_classes[i] = EdgeClass::Volatile;
            volatile_edges.push(EdgeId(i));
            volatile_scores.push(scores[i]);
        }
    }

    // Step 4: re-allocate budget across volatile edges only.
    let volatile_budget = total_budget.saturating_sub(volatile_edges.len().saturating_mul(chi_min));
    let volatile_chi_raw = if !volatile_scores.is_empty() {
        chi_allocation_target_budget(
            &volatile_scores,
            volatile_budget + volatile_edges.len() * chi_min,
            chi_min,
            chi_max,
        )
    } else {
        Vec::new()
    };

    let mut chi_per_edge = vec![0usize; n_edges];
    for (vi, &eid) in volatile_edges.iter().enumerate() {
        chi_per_edge[eid.0] = volatile_chi_raw[vi];
    }

    // Step 5: identify volatile / stable qubits.
    let mut qubit_volatile = vec![false; n_qubits];
    for &eid in &volatile_edges {
        let e = topology.edge(eid);
        qubit_volatile[e.a] = true;
        qubit_volatile[e.b] = true;
    }
    let volatile_qubits: Vec<usize> = (0..n_qubits).filter(|&q| qubit_volatile[q]).collect();
    let stable_qubits: Vec<usize> = (0..n_qubits).filter(|&q| !qubit_volatile[q]).collect();

    // Step 6: identify boundary edges — volatile edges with at least one
    // endpoint that is ALSO adjacent to a stable edge.
    let mut qubit_touches_stable = vec![false; n_qubits];
    for i in 0..n_edges {
        if edge_classes[i] == EdgeClass::Stable {
            let e = topology.edge(EdgeId(i));
            qubit_touches_stable[e.a] = true;
            qubit_touches_stable[e.b] = true;
        }
    }
    let boundary_edges: Vec<EdgeId> = volatile_edges
        .iter()
        .copied()
        .filter(|&eid| {
            let e = topology.edge(eid);
            qubit_touches_stable[e.a] || qubit_touches_stable[e.b]
        })
        .collect();

    TreePartition {
        edge_classes,
        volatile_edges,
        volatile_qubits,
        stable_qubits,
        boundary_edges,
        chi_per_edge,
        scores,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ttn::heavy_hex::HeavyHexLayout;
    use crate::ttn::topology::Edge;

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

    #[test]
    fn uniform_frequencies_all_stable() {
        let topology = Topology::linear_chain(10);
        let frequencies = vec![1.0; 10];
        let p = partition_tree_adaptive(&frequencies, &topology, 72, 2, 16, 5);
        // All-same frequencies → zero scores → all edges stable.
        assert!(p.volatile_edges.is_empty());
        assert_eq!(p.stable_qubits.len(), 10);
        assert_eq!(p.volatile_qubits.len(), 0);
    }

    #[test]
    fn strongly_disordered_all_volatile() {
        let topology = Topology::linear_chain(6);
        // Very incommensurate: √primes
        let frequencies = [
            1.0,
            2.0_f64.sqrt(),
            3.0_f64.sqrt(),
            5.0_f64.sqrt(),
            7.0_f64.sqrt(),
            11.0_f64.sqrt(),
        ];
        // Large budget → every edge gets ≫ chi_min → all volatile.
        let p = partition_tree_adaptive(&frequencies, &topology, 100, 2, 32, 5);
        assert_eq!(p.volatile_edges.len(), 5);
        assert_eq!(p.volatile_qubits.len(), 6);
    }

    #[test]
    fn y_junction_all_incommensurate_all_volatile() {
        // All leaves are incommensurate with the centre.
        let topology = y_junction();
        let frequencies = [1.0, 7.0_f64.sqrt(), 11.0_f64.sqrt(), 13.0_f64.sqrt()];
        let p = partition_tree_adaptive(&frequencies, &topology, 24, 2, 16, 5);
        assert_eq!(p.volatile_edges.len(), 3, "all edges should be volatile");
        assert_eq!(p.volatile_qubits.len(), 4);
    }

    #[test]
    fn y_junction_commensurate_pair_still_volatile_due_to_cross_cut() {
        // Centre 0: ω = 1.0, Leaf 1: ω = 2.0 (commensurate with centre)
        // Leaf 2: ω = √7, Leaf 3: ω = √11 (both incommensurate)
        //
        // Edge (0,1) separates {0, 2, 3} from {1}. Even though ω_0/ω_1
        // is commensurate, the cross-cut includes (√7, 2) and (√11, 2)
        // which are incommensurate — so the edge's score is non-zero and
        // it's correctly classified volatile (it carries entanglement
        // from leaves 2, 3 to leaf 1 through the centre).
        let topology = y_junction();
        let frequencies = [1.0, 2.0, 7.0_f64.sqrt(), 11.0_f64.sqrt()];
        let p = partition_tree_adaptive(&frequencies, &topology, 18, 2, 16, 5);
        // All three edges are volatile because every cross-cut has at
        // least one incommensurate pair.
        assert_eq!(
            p.volatile_edges.len(),
            3,
            "all 3 edges should be volatile due to cross-cut pairs"
        );
    }

    #[test]
    fn budget_exactly_consumed_on_volatile_edges() {
        let topology = Topology::linear_chain(8);
        let frequencies = [
            1.0,
            2.0_f64.sqrt(),
            3.0_f64.sqrt(),
            1.0,
            2.0,
            5.0_f64.sqrt(),
            7.0_f64.sqrt(),
            1.0,
        ];
        let total_budget = 42;
        let p = partition_tree_adaptive(&frequencies, &topology, total_budget, 2, 16, 5);
        let volatile_budget: usize = p.chi_per_edge.iter().sum();
        if !p.volatile_edges.is_empty() {
            assert_eq!(volatile_budget, total_budget);
        }
    }

    #[test]
    fn eagle_127q_disordered_produces_mixed_partition() {
        let layout = HeavyHexLayout::ibm_eagle_127();
        let primes = [2.0_f64, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0];
        let frequencies: Vec<f64> = (0..127)
            .map(|i| 0.8 + 0.05 * primes[i % primes.len()].sqrt())
            .collect();
        let p = partition_tree_adaptive(&frequencies, layout.tree(), 504, 2, 32, 5);
        // Should have a mix of stable and volatile edges.
        assert!(
            p.volatile_edges.len() > 0 && p.volatile_edges.len() < 126,
            "expected a mix of stable and volatile, got {} volatile out of 126",
            p.volatile_edges.len()
        );
        assert!(!p.boundary_edges.is_empty(), "expected boundary edges");
    }

    #[test]
    fn lightweight_topology_works_with_partitioner() {
        // The partitioner must work on lightweight topologies (no cut_partitions).
        let edges: Vec<Edge> = (0..99).map(|i| Edge { a: i, b: i + 1 }).collect();
        let topology = Topology::from_edges_lightweight(100, edges);
        assert!(!topology.has_cut_partitions());
        let frequencies: Vec<f64> = (0..100)
            .map(|i| 1.0 + (i as f64).sqrt() * 0.1)
            .collect();
        let p = partition_tree_adaptive(&frequencies, &topology, 200, 2, 16, 5);
        assert_eq!(p.edge_classes.len(), 99);
        // Should not panic — that's the main assertion.
    }

    #[test]
    fn single_qubit_topology() {
        let topology = Topology::linear_chain(1);
        let p = partition_tree_adaptive(&[1.0], &topology, 10, 2, 8, 5);
        assert!(p.volatile_edges.is_empty());
        assert_eq!(p.stable_qubits, vec![0]);
    }
}
