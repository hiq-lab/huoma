//! Tree-edge χ allocator based on sin(C/2) commensurability scoring.
//!
//! This is the tree-topology generalisation of
//! [`crate::allocator::chi_allocation_sinc`] (D.4 per `TRACK_D_DESIGN.md`).
//! Given a spanning tree [`Topology`] and a per-qubit frequency vector, it
//! emits a per-edge χ allocation that exactly consumes a caller-supplied
//! total budget and allocates bond dimension in proportion to how much
//! incommensurable "weight" crosses each edge's cut partition.
//!
//! # Definition
//!
//! For each tree edge `e`, removing `e` splits the tree into two disjoint
//! vertex sets `(A_e, B_e)`. The **sin(C/2) cut score** for edge `e` is
//!
//! ```text
//! score(e)  =  Σ_{i ∈ A_e, j ∈ B_e}  |sin(C(ω_i, ω_j) / 2)|
//! ```
//!
//! where `C(ω_i, ω_j)` is the pairwise commensurability residual from
//! [`crate::channel::commensurability_residual`]. The sum is the total
//! incommensurable coupling the bond has to carry — a direct generalisation
//! of the 1D `bond_weight` on a linear chain, where the cut is a simple
//! prefix / suffix split and the sum above reduces to the same pairs the
//! 1D allocator already considers (minus the exponential distance
//! weighting, which is deliberately dropped here — the design doc spec is
//! the un-weighted sum).
//!
//! The score vector is then fed through
//! [`crate::allocator::chi_allocation_target_budget`] — the score-agnostic
//! integer water-filling primitive that all Huoma allocators share — to
//! produce the final `Vec<usize>` of per-edge χ values.
//!
//! # Cost
//!
//! O(N² · n_edges) in the worst case, i.e. O(N³) for a tree with `N - 1`
//! edges. For Eagle 127q this is ~2 × 10⁶ floating-point operations,
//! microseconds on any modern CPU, and negligible next to a single TEBD
//! layer. A radius-truncated sparse variant (tree-distance ≤ radius) is a
//! natural follow-up if the benchmark shows it matters, but at
//! `N = 127` it does not.
//!
//! # Usage
//!
//! ```no_run
//! use huoma::ttn::{HeavyHexLayout, allocator::chi_allocation_sinc_tree};
//!
//! let layout = HeavyHexLayout::ibm_eagle_127();
//! let frequencies: Vec<f64> = (0..127).map(|i| 1.0 + 0.01 * i as f64).collect();
//! let chi = chi_allocation_sinc_tree(&frequencies, layout.tree(), 126 * 16, 2, 64);
//! assert_eq!(chi.len(), 126);
//! assert_eq!(chi.iter().sum::<usize>(), 126 * 16);
//! ```

use crate::allocator::chi_allocation_target_budget;
use crate::channel::sin_c_half;
use crate::ttn::topology::{EdgeId, Topology};

/// Compute the sin(C/2) cut score for a single tree edge.
///
/// Removing the edge splits the tree into two vertex sets `(A, B)`; the
/// score is the sum of `|sin(C(ω_i, ω_j) / 2)|` over all cross-cut pairs
/// `(i, j)` with `i ∈ A`, `j ∈ B`.
///
/// Exposed as a free function so tests (and future allocators) can reuse
/// it without going through the full [`chi_allocation_sinc_tree`] path.
#[must_use]
pub fn edge_sinc_score(frequencies: &[f64], topology: &Topology, edge: EdgeId) -> f64 {
    let (a_side, b_side) = topology.cut_partition(edge);
    let mut acc = 0.0_f64;
    for &i in a_side {
        for &j in b_side {
            // Guard against out-of-range indices so the allocator still
            // produces a finite score if the caller passes a shorter
            // `frequencies` slice than the topology expects.
            let (Some(&omega_i), Some(&omega_j)) = (frequencies.get(i), frequencies.get(j)) else {
                continue;
            };
            acc += sin_c_half(omega_i, omega_j);
        }
    }
    acc
}

/// Production sin(C/2) χ allocator for tree topologies — Track D D.4.
///
/// Scores every edge of `topology` by [`edge_sinc_score`] and feeds the
/// result through the shared [`chi_allocation_target_budget`] water-filling
/// primitive, returning a per-edge χ profile of length
/// `topology.n_edges()` that exactly consumes `total_budget` subject to
/// `χ_e ∈ [chi_min, chi_max]`.
///
/// # Behaviour at the corners
///
/// - `topology.n_edges() == 0` (single-qubit topology) → empty `Vec`.
/// - `frequencies.len() != topology.n_qubits()` → the allocator uses only
///   the pairs where both endpoints are in range (so short `frequencies`
///   produce reduced scores but still a valid budget allocation).
/// - `total_budget < n_edges · chi_min` → returns all `chi_min`.
/// - `total_budget > n_edges · chi_max` → returns all `chi_max`.
/// - All-zero / non-finite scores → delegates to the underlying
///   water-filling primitive, which emits the most uniform integer
///   allocation that consumes the budget exactly.
///
/// # Relationship to the 1D allocator
///
/// On `Topology::linear_chain(n)` the cut partitions are prefix / suffix
/// splits `(0..=b, b+1..n)`, so `edge_sinc_score` reduces to
///
/// ```text
/// score(b)  =  Σ_{i ≤ b, j > b}  |sin(C(ω_i, ω_j) / 2)|
/// ```
///
/// which is the same pair set the 1D dense `ChannelMap` considers, minus
/// the exponential distance weighting. The two allocators therefore
/// produce *similar but not identical* profiles on linear chains — both
/// consume the same budget, both respect the same bounds, and both favour
/// bonds where incommensurable pairs cross the cut, but the tree version
/// does not down-weight far-apart pairs. This matches the design-doc spec
/// and is deliberately simpler than the 1D version.
#[must_use]
pub fn chi_allocation_sinc_tree(
    frequencies: &[f64],
    topology: &Topology,
    total_budget: usize,
    chi_min: usize,
    chi_max: usize,
) -> Vec<usize> {
    let n_edges = topology.n_edges();
    if n_edges == 0 {
        return Vec::new();
    }
    let scores: Vec<f64> = (0..n_edges)
        .map(|i| edge_sinc_score(frequencies, topology, EdgeId(i)))
        .collect();
    chi_allocation_target_budget(&scores, total_budget, chi_min, chi_max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ttn::topology::Edge;
    use crate::ttn::HeavyHexLayout;

    fn y_junction() -> Topology {
        // Centre 0, leaves 1, 2, 3.
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
    fn single_qubit_topology_returns_empty_vec() {
        let topology = Topology::linear_chain(1);
        let chi = chi_allocation_sinc_tree(&[1.0], &topology, 16, 2, 8);
        assert!(chi.is_empty());
    }

    #[test]
    fn budget_is_exactly_consumed_on_linear_chain() {
        let topology = Topology::linear_chain(6);
        let frequencies = [1.0, 2.0_f64.sqrt(), 3.0_f64.sqrt(), 5.0, 7.0_f64.sqrt(), 11.0];
        for budget in [16, 24, 36, 50] {
            let chi = chi_allocation_sinc_tree(&frequencies, &topology, budget, 2, 16);
            assert_eq!(chi.len(), 5);
            assert_eq!(
                chi.iter().sum::<usize>(),
                budget,
                "budget mismatch at {budget}: {chi:?}"
            );
            for &c in &chi {
                assert!((2..=16).contains(&c), "out of bounds: {c}");
            }
        }
    }

    #[test]
    fn under_budget_returns_all_chi_min() {
        let topology = Topology::linear_chain(5);
        let frequencies = [1.0, 2.0, 3.0, 5.0, 7.0];
        // 4 edges × chi_min = 8, so any budget ≤ 8 returns [2,2,2,2].
        let chi = chi_allocation_sinc_tree(&frequencies, &topology, 6, 2, 16);
        assert_eq!(chi, vec![2, 2, 2, 2]);
    }

    #[test]
    fn over_budget_returns_all_chi_max() {
        let topology = Topology::linear_chain(5);
        let frequencies = [1.0, 2.0, 3.0, 5.0, 7.0];
        // 4 edges × chi_max = 32, so any budget ≥ 32 returns [8,8,8,8].
        let chi = chi_allocation_sinc_tree(&frequencies, &topology, 100, 2, 8);
        assert_eq!(chi, vec![8, 8, 8, 8]);
    }

    #[test]
    fn uniform_frequencies_produce_near_uniform_allocation() {
        // All sites at the same frequency → every pair has C = 0 →
        // every score is zero → water-filling falls back to the uniform
        // distribution.
        let topology = Topology::linear_chain(6);
        let frequencies = vec![1.0; 6];
        let chi = chi_allocation_sinc_tree(&frequencies, &topology, 25, 2, 16);
        assert_eq!(chi.len(), 5);
        assert_eq!(chi.iter().sum::<usize>(), 25);
        let min = *chi.iter().min().unwrap();
        let max = *chi.iter().max().unwrap();
        assert!(
            max - min <= 1,
            "uniform frequencies should produce near-uniform χ, got {chi:?}"
        );
    }

    #[test]
    fn irrational_frequencies_produce_nontrivial_profile() {
        let topology = Topology::linear_chain(6);
        let frequencies = [
            1.0,
            2.0_f64.sqrt(),
            3.0_f64.sqrt(),
            5.0_f64.sqrt(),
            7.0_f64.sqrt(),
            11.0_f64.sqrt(),
        ];
        let chi = chi_allocation_sinc_tree(&frequencies, &topology, 40, 2, 16);
        assert_eq!(chi.iter().sum::<usize>(), 40);
        let min = *chi.iter().min().unwrap();
        let max = *chi.iter().max().unwrap();
        assert!(
            max > min,
            "irrational frequencies should produce non-flat χ, got {chi:?}"
        );
    }

    #[test]
    fn y_junction_center_edge_scores_equal_under_symmetry() {
        // For the Y-junction with symmetric leaf frequencies (a, b, b, b),
        // the three edges (0,1), (0,2), (0,3) should each score the same
        // cross-cut sum by symmetry. We use three identical leaf
        // frequencies so any per-edge asymmetry must come from a bug.
        let topology = y_junction();
        let frequencies = [0.5, 2.0_f64.sqrt(), 2.0_f64.sqrt(), 2.0_f64.sqrt()];
        let s0 = edge_sinc_score(&frequencies, &topology, EdgeId(0));
        let s1 = edge_sinc_score(&frequencies, &topology, EdgeId(1));
        let s2 = edge_sinc_score(&frequencies, &topology, EdgeId(2));
        assert!((s0 - s1).abs() < 1e-12, "s0 vs s1: {s0} vs {s1}");
        assert!((s1 - s2).abs() < 1e-12, "s1 vs s2: {s1} vs {s2}");
    }

    #[test]
    fn y_junction_asymmetric_frequencies_produce_asymmetric_allocation() {
        // One leaf is strongly incommensurate with the centre; the other
        // two are close. The edge to the "weird" leaf should carry more
        // χ than the two edges to the near-in-frequency leaves.
        let topology = y_junction();
        // leaf 1 is very incommensurate with centre, leaves 2 and 3 are
        // nearly commensurate with centre.
        let frequencies = [1.0, 7.0_f64.sqrt(), 2.0, 3.0];
        let chi = chi_allocation_sinc_tree(&frequencies, &topology, 24, 2, 16);
        assert_eq!(chi.len(), 3);
        assert_eq!(chi.iter().sum::<usize>(), 24);
        assert!(
            chi[0] >= chi[1] && chi[0] >= chi[2],
            "edge (0,1) to the incommensurate leaf should get ≥ χ than the others, got {chi:?}"
        );
    }

    #[test]
    fn bounds_are_respected_on_y_junction() {
        let topology = y_junction();
        let frequencies = [1.0, 7.0_f64.sqrt(), 2.0, 3.0];
        let chi = chi_allocation_sinc_tree(&frequencies, &topology, 18, 4, 8);
        for &c in &chi {
            assert!((4..=8).contains(&c), "out of bounds: {c} in {chi:?}");
        }
        assert_eq!(chi.iter().sum::<usize>(), 18);
    }

    #[test]
    fn short_frequencies_slice_does_not_panic() {
        // Caller passes fewer frequencies than the topology has qubits.
        // The allocator must still produce a valid budget allocation,
        // it just ignores the pairs outside the given range.
        let topology = Topology::linear_chain(6);
        let frequencies = [1.0, 2.0_f64.sqrt()]; // only 2 of 6
        let chi = chi_allocation_sinc_tree(&frequencies, &topology, 24, 2, 16);
        assert_eq!(chi.len(), 5);
        assert_eq!(chi.iter().sum::<usize>(), 24);
    }

    #[test]
    fn handles_nan_and_zero_frequencies() {
        // NaN frequencies push `sin_c_half` into its zero-denominator
        // fallback branch; the allocator must not blow up and must still
        // consume the budget exactly.
        let topology = Topology::linear_chain(5);
        let frequencies = [1.0, f64::NAN, 0.0, 2.0, 3.0];
        let chi = chi_allocation_sinc_tree(&frequencies, &topology, 20, 2, 16);
        assert_eq!(chi.iter().sum::<usize>(), 20);
        for &c in &chi {
            assert!((2..=16).contains(&c));
        }
    }

    #[test]
    fn edge_sinc_score_is_nonnegative_and_bounded() {
        // `sin_c_half` is defined as `|sin(C/2)|` which is in `[0, 1]` for
        // every pair. The edge score is a sum over `|A_e| × |B_e|` pairs,
        // so it must be in `[0, |A_e| × |B_e|]`. This is a structural
        // sanity check on `edge_sinc_score` independent of any specific
        // frequency vector.
        let topology = Topology::linear_chain(6);
        let frequencies = [1.0, 2.0_f64.sqrt(), 3.0_f64.sqrt(), 5.0, 7.0_f64.sqrt(), 11.0];
        for b in 0..topology.n_edges() {
            let (a_side, b_side) = topology.cut_partition(EdgeId(b));
            let max_score = (a_side.len() * b_side.len()) as f64;
            let score = edge_sinc_score(&frequencies, &topology, EdgeId(b));
            assert!(
                (0.0..=max_score + 1e-12).contains(&score),
                "edge {b} score {score} outside [0, {max_score}]"
            );
        }
    }

    #[test]
    fn chi_allocation_sinc_tree_on_eagle_127q_has_126_edges() {
        // Smoke test on the real Eagle 127q layout: allocation has the
        // right shape, exactly consumes the budget, and stays in bounds.
        let layout = HeavyHexLayout::ibm_eagle_127();
        // A deterministic "random-like" frequency vector — one per qubit.
        let frequencies: Vec<f64> = (0..127)
            .map(|i| 1.0 + (i as f64).sqrt() * 0.173)
            .collect();
        let budget = 126 * 8; // 1008
        let chi = chi_allocation_sinc_tree(&frequencies, layout.tree(), budget, 2, 32);
        assert_eq!(chi.len(), 126);
        assert_eq!(chi.iter().sum::<usize>(), budget);
        for &c in &chi {
            assert!((2..=32).contains(&c), "out of bounds: {c}");
        }
    }
}
