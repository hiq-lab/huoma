//! Per-bond χ allocators for MPS truncation.
//!
//! This module provides Huoma's production-recommended χ allocation strategy:
//! sin(C/2) commensurability scoring (from the Tilde Pattern, Hinderink 2026)
//! fed through an integer water-filling allocator that exactly consumes a
//! caller-supplied total budget.
//!
//! # Recommended path
//!
//! ```ignore
//! use huoma::allocator::chi_allocation_sinc;
//!
//! let frequencies: Vec<f64> = ...; // per-site natural frequency
//! let chi = chi_allocation_sinc(&frequencies, total_budget, chi_min, chi_max);
//! // pass `chi` to apply_two_qubit_layer_parallel as chi_per_bond
//! ```
//!
//! `chi_allocation_sinc` is O(N · radius²) (microseconds for N ≤ 200) and
//! requires no pilot run, no AD, no SVD differentiation. It is the right
//! default for any 1D MPS workload where per-site frequencies (h_x, ω_i,
//! coupling, …) are defined.
//!
//! # Custom scores
//!
//! For research workflows that compute per-bond complexity scores by some
//! other means (gradient descent, automatic differentiation, perturbative
//! analysis, …) the lower-level [`chi_allocation_target_budget`] accepts an
//! arbitrary `&[f64]` of non-negative scores and runs the same water-filling
//! step. No additional Huoma machinery is needed — the simulator does not
//! ship a finite-difference Jacobian. If you need one, compute it externally
//! and feed the resulting per-bond scores into the water-filling allocator.
//!
//! # Why not the discarded-weight Jacobian?
//!
//! Huoma originally shipped a finite-difference Jacobian of per-bond
//! cumulative discarded weight as the recommended scorer. Matched-budget
//! benchmarks against uniform-χ on disordered KIM (commit `19a5793`,
//! `PHASE7_REPORT.md`) showed it produces allocations strictly worse than
//! uniform — by 6-11× — because the discarded-weight observable censors
//! boundary bonds (a bond with one qubit on its left has Schmidt rank ≤ 2,
//! so the pilot's discarded weight there is artificially small, and the
//! resulting score clamps the bond to `chi_min`). The censoring is
//! structural, not a tuning issue, and removing the Jacobian was the right
//! call. Sin(C/2) does not have an analogous blind spot because pairwise
//! commensurability aggregation is local-frequency-driven, not pilot-driven.

use crate::channel::ChannelMap;

/// Default neighborhood radius for sparse channel-map construction.
/// Captures > 99 % of the entanglement prediction at O(N · radius²) cost.
pub const DEFAULT_SINC_RADIUS: usize = 5;

/// Allocate a per-bond χ profile that **exactly** consumes the supplied
/// `total_budget`, distributing it across bonds in proportion to a
/// non-negative per-bond `scores` vector subject to `χ_k ∈ [chi_min, chi_max]`.
///
/// This is the integer water-filling solution to
///
/// ```text
///   maximise   Σ_k score_k · log(χ_k)
///   subject to Σ_k χ_k = total_budget
///              chi_min ≤ χ_k ≤ chi_max
/// ```
///
/// which is equivalent to "make χ_k proportional to score_k", clipped to the
/// bounds. The greedy increments the bond with the largest marginal utility
/// `score_k / (χ_k + 1)` at every step until the budget is spent.
///
/// Behaviour at the corners:
/// - `total_budget < n · chi_min`: returns all `chi_min` (under-budget signal).
/// - `total_budget > n · chi_max`: returns all `chi_max` (over-budget signal).
/// - All-zero / non-finite scores: returns the most-uniform integer
///   allocation that consumes the budget exactly.
/// - Empty `scores`: returns an empty vector.
///
/// Score-agnostic: pass it sin(C/2) channel weights, hand-tuned scores,
/// or any other non-negative per-bond complexity metric.
#[must_use]
pub fn chi_allocation_target_budget(
    scores: &[f64],
    total_budget: usize,
    chi_min: usize,
    chi_max: usize,
) -> Vec<usize> {
    let n = scores.len();
    if n == 0 {
        return Vec::new();
    }
    assert!(chi_min >= 1, "chi_min must be ≥ 1, got {chi_min}");
    assert!(
        chi_min <= chi_max,
        "chi_min ({chi_min}) must be ≤ chi_max ({chi_max})"
    );

    let min_budget = n.saturating_mul(chi_min);
    let max_budget = n.saturating_mul(chi_max);
    if total_budget <= min_budget {
        return vec![chi_min; n];
    }
    if total_budget >= max_budget {
        return vec![chi_max; n];
    }

    let mut chi = vec![chi_min; n];
    let mut remaining = total_budget - min_budget;

    // Sanitize scores: NaN/negative → 0.
    let clean: Vec<f64> = scores
        .iter()
        .map(|&s| if s.is_finite() && s > 0.0 { s } else { 0.0 })
        .collect();
    let total_score: f64 = clean.iter().sum();

    if total_score <= 0.0 {
        // No discriminating signal: spread the surplus as evenly as possible.
        let base = remaining / n;
        let extra = remaining % n;
        for (k, slot) in chi.iter_mut().enumerate() {
            let add = base + usize::from(k < extra);
            *slot = (*slot + add).min(chi_max);
        }
        return chi;
    }

    // Greedy water-filling. Each iteration spends one unit of budget on
    // whichever non-saturated bond has the largest marginal utility
    // score_k / (χ_k + 1). For typical sizes (n ≤ 200, remaining ≤ a few
    // thousand) the O(remaining · n) loop is well under 1 ms.
    while remaining > 0 {
        let mut best_k: Option<usize> = None;
        let mut best_ratio = f64::NEG_INFINITY;
        for k in 0..n {
            if chi[k] >= chi_max {
                continue;
            }
            let s = clean[k];
            if s <= 0.0 {
                continue;
            }
            let ratio = s / (chi[k] as f64 + 1.0);
            if ratio > best_ratio {
                best_ratio = ratio;
                best_k = Some(k);
            }
        }
        match best_k {
            Some(k) => {
                chi[k] += 1;
                remaining -= 1;
            }
            None => {
                // All bonds with positive score are saturated; pour the
                // residual into zero-score bonds (round-robin) so the
                // total budget is still spent exactly.
                for k in 0..n {
                    if remaining == 0 {
                        break;
                    }
                    if chi[k] < chi_max {
                        chi[k] += 1;
                        remaining -= 1;
                    }
                }
                break;
            }
        }
    }

    chi
}

/// Production sin(C/2) χ allocator. Builds a sparse [`ChannelMap`] from the
/// supplied per-site `frequencies` (radius [`DEFAULT_SINC_RADIUS`]) and feeds
/// the resulting per-bond commensurability scores into the water-filling
/// allocator.
///
/// Returns a vector of length `frequencies.len() - 1` (one χ per MPS bond).
/// Empty if `frequencies.len() < 2`.
///
/// This is the recommended production allocator for 1D MPS in Huoma.
#[must_use]
pub fn chi_allocation_sinc(
    frequencies: &[f64],
    total_budget: usize,
    chi_min: usize,
    chi_max: usize,
) -> Vec<usize> {
    chi_allocation_sinc_with_radius(
        frequencies,
        DEFAULT_SINC_RADIUS,
        total_budget,
        chi_min,
        chi_max,
    )
}

/// Variant of [`chi_allocation_sinc`] with an explicit channel-map radius.
/// Larger radius captures more long-range commensurability structure at
/// O(radius²) cost; the default of [`DEFAULT_SINC_RADIUS`] is calibrated
/// for nearest-neighbour Floquet circuits.
#[must_use]
pub fn chi_allocation_sinc_with_radius(
    frequencies: &[f64],
    radius: usize,
    total_budget: usize,
    chi_min: usize,
    chi_max: usize,
) -> Vec<usize> {
    let n = frequencies.len();
    if n < 2 {
        return Vec::new();
    }
    let cm = ChannelMap::from_frequencies_sparse(frequencies, 1.0, radius);
    let scores: Vec<f64> = (0..n - 1).map(|b| cm.bond_weight(b)).collect();
    chi_allocation_target_budget(&scores, total_budget, chi_min, chi_max)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── chi_allocation_target_budget (water-filling primitive) ────────────

    #[test]
    fn target_budget_empty_input() {
        assert!(chi_allocation_target_budget(&[], 10, 2, 8).is_empty());
    }

    #[test]
    fn target_budget_under_budget_returns_chi_min() {
        let chi = chi_allocation_target_budget(&[1.0, 1.0, 1.0, 1.0], 6, 2, 8);
        assert_eq!(chi, vec![2, 2, 2, 2]);
    }

    #[test]
    fn target_budget_over_budget_returns_chi_max() {
        let chi = chi_allocation_target_budget(&[1.0, 1.0, 1.0, 1.0], 100, 2, 8);
        assert_eq!(chi, vec![8, 8, 8, 8]);
    }

    #[test]
    fn target_budget_uniform_scores_split_evenly() {
        let chi = chi_allocation_target_budget(&[1.0, 1.0, 1.0, 1.0], 20, 2, 8);
        assert_eq!(chi.iter().sum::<usize>(), 20);
        let max = *chi.iter().max().unwrap();
        let min = *chi.iter().min().unwrap();
        assert!(max - min <= 1, "uniform scores produced non-flat: {chi:?}");
    }

    #[test]
    fn target_budget_proportional_to_scores() {
        let chi = chi_allocation_target_budget(&[3.0, 1.0], 8, 1, 100);
        assert_eq!(chi.iter().sum::<usize>(), 8);
        assert!(chi[0] > chi[1], "high-score bond should get more: {chi:?}");
        assert!(chi[0] >= 2 * chi[1], "ratio too narrow: {chi:?}");
    }

    #[test]
    fn target_budget_exactly_consumes_budget() {
        let scores = vec![0.1, 0.5, 0.2, 0.9, 0.3, 0.7];
        for budget in [12, 20, 30, 40] {
            let chi = chi_allocation_target_budget(&scores, budget, 2, 16);
            assert_eq!(
                chi.iter().sum::<usize>(),
                budget,
                "budget mismatch at {budget}: {chi:?}"
            );
        }
    }

    #[test]
    fn target_budget_respects_bounds() {
        let scores = vec![0.0, 100.0, 0.0, 100.0];
        let chi = chi_allocation_target_budget(&scores, 20, 2, 8);
        assert_eq!(chi.iter().sum::<usize>(), 20);
        for &c in &chi {
            assert!((2..=8).contains(&c), "out of bounds: {c} in {chi:?}");
        }
        assert_eq!(chi, vec![2, 8, 2, 8]);
    }

    #[test]
    fn target_budget_zero_scores_distribute_evenly() {
        let chi = chi_allocation_target_budget(&[0.0, 0.0, 0.0], 12, 2, 8);
        assert_eq!(chi.iter().sum::<usize>(), 12);
        let max = *chi.iter().max().unwrap();
        let min = *chi.iter().min().unwrap();
        assert!(max - min <= 1, "zero scores produced non-flat: {chi:?}");
    }

    #[test]
    fn target_budget_handles_nan_scores() {
        let scores = vec![1.0, f64::NAN, 1.0, f64::NEG_INFINITY];
        let chi = chi_allocation_target_budget(&scores, 14, 2, 8);
        assert_eq!(chi.iter().sum::<usize>(), 14);
        assert!(chi[0] >= chi[1], "NaN bond should not exceed finite: {chi:?}");
        assert!(chi[2] >= chi[3], "neg-inf bond should not exceed finite: {chi:?}");
    }

    #[test]
    fn target_budget_saturated_high_score_bonds_overflow_into_low_score() {
        let scores = vec![100.0, 100.0, 0.0, 0.0];
        let chi = chi_allocation_target_budget(&scores, 12, 2, 4);
        assert_eq!(chi.iter().sum::<usize>(), 12);
        assert_eq!(chi[0], 4);
        assert_eq!(chi[1], 4);
        assert_eq!(chi[2] + chi[3], 4);
    }

    #[test]
    fn target_budget_matches_uniform_when_scores_flat() {
        let n = 10;
        let scores = vec![1.0; n];
        let chi_uniform = 8;
        let budget = n * chi_uniform;
        let chi = chi_allocation_target_budget(&scores, budget, 2, chi_uniform);
        assert_eq!(chi, vec![chi_uniform; n]);
    }

    // ─── chi_allocation_sinc (production sin(C/2) path) ───────────────────

    #[test]
    fn sinc_short_input_returns_empty() {
        assert!(chi_allocation_sinc(&[], 10, 2, 8).is_empty());
        assert!(chi_allocation_sinc(&[1.0], 10, 2, 8).is_empty());
    }

    #[test]
    fn sinc_returns_one_chi_per_bond() {
        let frequencies = vec![1.0, 2.0, 3.0, 5.0_f64.sqrt(), 7.0_f64.sqrt()];
        let chi = chi_allocation_sinc(&frequencies, 25, 2, 8);
        assert_eq!(chi.len(), frequencies.len() - 1);
        assert_eq!(chi.iter().sum::<usize>(), 25);
        for &c in &chi {
            assert!((2..=8).contains(&c), "out of bounds: {c}");
        }
    }

    #[test]
    fn sinc_all_commensurate_collapses_to_uniform() {
        // Integer-ratio frequencies produce uniformly low scores; the
        // allocator should fall back to even budget distribution.
        let frequencies = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let chi = chi_allocation_sinc(&frequencies, 30, 2, 8);
        assert_eq!(chi.iter().sum::<usize>(), 30);
        // No bond should be more than ±2 from the average.
        let avg = 30 / chi.len();
        for &c in &chi {
            assert!(
                c.abs_diff(avg) <= 2,
                "expected near-uniform on commensurate frequencies, got {chi:?}"
            );
        }
    }

    #[test]
    fn sinc_irrational_frequencies_produce_nontrivial_profile() {
        // Frequencies spanning genuinely different commensurability classes
        // should produce a non-flat allocation.
        let frequencies = vec![
            1.0,
            2.0_f64.sqrt(),
            3.0_f64.sqrt(),
            5.0_f64.sqrt(),
            7.0_f64.sqrt(),
            11.0_f64.sqrt(),
        ];
        let chi = chi_allocation_sinc(&frequencies, 30, 2, 12);
        assert_eq!(chi.iter().sum::<usize>(), 30);
        let max = *chi.iter().max().unwrap();
        let min = *chi.iter().min().unwrap();
        assert!(
            max > min,
            "irrational frequencies should produce non-flat allocation, got {chi:?}"
        );
    }
}
