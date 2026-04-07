//! Finite-difference Jacobian for input → bond-observable sensitivity.
//!
//! Provides a clean public API for computing
//!
//! ```text
//!   J[k, i] = ∂(observable_k) / ∂(input_i)
//! ```
//!
//! where the simulation is parametrized by an N-vector of input parameters
//! (e.g. site frequencies or gate angles) and produces an M-vector of
//! per-bond observables (e.g. discarded weight, entropy, fidelity).
//!
//! This is the **operationally faithful** finite-difference equivalent of
//! forward-mode automatic differentiation: same numerical result up to
//! O(δ²) discretization error, same asymptotic cost (O(N · primal_cost)),
//! but no symbolic differentiation through SVD.
//!
//! # Usage
//!
//! ```ignore
//! use huoma::jacobian::{InputJacobian, JacobianConfig};
//!
//! let factory = |inputs: &[f64]| {
//!     let mut mps = build_mps_from_inputs(inputs);
//!     mps
//! };
//! let observe = |mps: &Mps| -> Vec<f64> {
//!     mps.discarded_weight_per_bond.clone()
//! };
//!
//! let cfg = JacobianConfig::default();
//! let j = InputJacobian::compute(&base_inputs, factory, observe, &cfg);
//! // j.matrix[k][i] = ∂(observable_k)/∂(input_i)
//! ```
//!
//! # Cost
//!
//! For N inputs and primal cost C, the Jacobian costs `2 N C` (central
//! differences). Each row of the matrix is computed from the same M
//! observables but with different perturbed inputs.
//!
//! # Adaptive bond dimension via Jacobian
//!
//! After computing J, the per-row participation ratio
//!
//! ```text
//!   PR[k] = (Σ_i |J[k,i]|)² / Σ_i J[k,i]²
//! ```
//!
//! predicts which bonds need higher χ. Bonds with **high** PR are
//! "globally determined" — many inputs contribute roughly equally — and
//! tend to accumulate the most discarded weight. Bonds with **low** PR
//! depend on a few dominant inputs and are easier to compress.
//!
//! See `participation_ratio` and `chi_allocation_from_jacobian`.

/// Configuration for Jacobian computation.
#[derive(Debug, Clone, Copy)]
pub struct JacobianConfig {
    /// Step size for the central difference. Default: 1e-3.
    /// Sweet spot for f64 is around √eps ≈ 1.5e-8 but for noisy MPS
    /// observables a larger step (1e-3 to 1e-2) is more stable.
    pub delta: f64,
    /// If true, sample only every Nth input direction (for very large N).
    /// 1 = every input, 2 = every other, etc.
    pub stride: usize,
}

impl Default for JacobianConfig {
    fn default() -> Self {
        Self {
            delta: 1e-3,
            stride: 1,
        }
    }
}

/// Result of a Jacobian computation: a dense `Vec<Vec<f64>>` plus metadata.
#[derive(Debug, Clone)]
pub struct InputJacobian {
    /// `matrix[k][i] = ∂(observable_k)/∂(input_i)`
    /// Shape: `[n_outputs][n_inputs]`. May contain NaN if `stride > 1`
    /// and the column was skipped.
    pub matrix: Vec<Vec<f64>>,
    pub n_inputs: usize,
    pub n_outputs: usize,
    pub delta: f64,
    pub stride: usize,
}

impl InputJacobian {
    /// Compute the Jacobian via central finite differences.
    ///
    /// `factory(inputs)` builds and runs a simulation, returning the final
    /// state from which `observe` extracts an output vector.
    ///
    /// `base_inputs` is the unperturbed parameter vector. The Jacobian is
    /// evaluated at this point.
    pub fn compute<F, S, O>(
        base_inputs: &[f64],
        mut factory: F,
        mut observe: O,
        cfg: &JacobianConfig,
    ) -> Self
    where
        F: FnMut(&[f64]) -> S,
        O: FnMut(&S) -> Vec<f64>,
    {
        let n_inputs = base_inputs.len();
        // Probe to learn n_outputs
        let probe = factory(base_inputs);
        let probe_obs = observe(&probe);
        let n_outputs = probe_obs.len();

        // matrix[k][i]
        let mut matrix = vec![vec![f64::NAN; n_inputs]; n_outputs];

        let stride = cfg.stride.max(1);
        let delta = cfg.delta;
        let inv_two_delta = 1.0 / (2.0 * delta);

        let mut perturbed = base_inputs.to_vec();

        for i in (0..n_inputs).step_by(stride) {
            // +δ
            let saved = perturbed[i];
            perturbed[i] = saved + delta;
            let s_plus = factory(&perturbed);
            let obs_plus = observe(&s_plus);

            // -δ
            perturbed[i] = saved - delta;
            let s_minus = factory(&perturbed);
            let obs_minus = observe(&s_minus);

            // Restore
            perturbed[i] = saved;

            // Central difference: J[k][i] = (obs_plus[k] - obs_minus[k]) / (2δ)
            for k in 0..n_outputs {
                let plus = obs_plus.get(k).copied().unwrap_or(0.0);
                let minus = obs_minus.get(k).copied().unwrap_or(0.0);
                matrix[k][i] = (plus - minus) * inv_two_delta;
            }
        }

        Self {
            matrix,
            n_inputs,
            n_outputs,
            delta,
            stride,
        }
    }

    /// Number of evaluated input columns (n_inputs / stride, rounded up).
    #[must_use]
    pub fn n_evaluated_columns(&self) -> usize {
        (self.n_inputs + self.stride - 1) / self.stride
    }
}

/// Participation ratio of a Jacobian row: a measure of how many inputs
/// contribute roughly equally to the output. NaN entries are skipped.
///
/// ```text
///   PR = (Σ_i |J_i|)² / Σ_i J_i²
/// ```
///
/// `PR → 1` means the output is determined by ONE input.
/// `PR → N` means the output is uniformly determined by all N inputs.
#[must_use]
pub fn participation_ratio(row: &[f64]) -> f64 {
    let mut l1 = 0.0_f64;
    let mut l2_sq = 0.0_f64;
    for &v in row {
        if v.is_finite() {
            let abs = v.abs();
            l1 += abs;
            l2_sq += abs * abs;
        }
    }
    if l2_sq < 1e-30 {
        0.0
    } else {
        (l1 * l1) / l2_sq
    }
}

/// Per-bond participation ratio for the entire Jacobian.
#[must_use]
pub fn participation_ratio_profile(jacobian: &InputJacobian) -> Vec<f64> {
    jacobian
        .matrix
        .iter()
        .map(|row| participation_ratio(row))
        .collect()
}

/// L1 norm (total sensitivity) of each output to all inputs.
/// Useful as an alternative to PR — measures total magnitude rather
/// than spread.
#[must_use]
pub fn total_sensitivity_profile(jacobian: &InputJacobian) -> Vec<f64> {
    jacobian
        .matrix
        .iter()
        .map(|row| row.iter().filter(|v| v.is_finite()).map(|v| v.abs()).sum())
        .collect()
}

/// Allocate a per-bond χ budget proportional to a "complexity" score
/// derived from the Jacobian. Two strategies:
///
/// - `JacobianAllocation::ParticipationRatio`: χ_k ∝ PR_k (more inputs ⇒
///   more chi)
/// - `JacobianAllocation::TotalSensitivity`: χ_k ∝ ||J_k||_1 (larger
///   gradients ⇒ more chi)
///
/// The result is normalised so that `Σ χ_k = chi_max · n_bonds` (uniform
/// budget) but redistributed.
#[derive(Debug, Clone, Copy)]
pub enum JacobianAllocation {
    ParticipationRatio,
    TotalSensitivity,
}

#[must_use]
pub fn chi_allocation_from_jacobian(
    jacobian: &InputJacobian,
    chi_min: usize,
    chi_max: usize,
    strategy: JacobianAllocation,
) -> Vec<usize> {
    let scores: Vec<f64> = match strategy {
        JacobianAllocation::ParticipationRatio => participation_ratio_profile(jacobian),
        JacobianAllocation::TotalSensitivity => total_sensitivity_profile(jacobian),
    };

    if scores.is_empty() {
        return Vec::new();
    }

    let max_score = scores.iter().cloned().fold(0.0_f64, f64::max);
    if max_score < 1e-30 {
        return vec![chi_min; scores.len()];
    }

    let chi_range = chi_max.saturating_sub(chi_min) as f64;

    scores
        .iter()
        .map(|&s| {
            let frac = (s / max_score).sqrt();
            let chi = chi_min + (chi_range * frac) as usize;
            chi.clamp(chi_min, chi_max)
        })
        .collect()
}

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
/// which is equivalent to "make χ_k proportional to score_k", clipped to
/// the bounds. The greedy increments the bond with the largest marginal
/// utility `score_k / (χ_k + 1)` at every step until the budget is spent.
///
/// Behaviour at the corners:
/// - `total_budget < n · chi_min`: returns all `chi_min` (under-budget signal).
/// - `total_budget > n · chi_max`: returns all `chi_max` (over-budget signal).
/// - All-zero / non-finite scores: returns the most-uniform integer
///   allocation that consumes the budget exactly.
/// - Empty `scores`: returns an empty vector.
///
/// Score-agnostic: pass it Jacobian-derived scores, sin(C/2) channel weights,
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
    // score_k / (χ_k + 1). For typical Track-A sizes (n ≤ 200, remaining
    // ≤ a few thousand) the O(remaining · n) loop is well under 1 ms.
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

/// Convenience wrapper: build per-bond Jacobian scores via `strategy` and
/// then run the target-budget water-filling allocator on them.
#[must_use]
pub fn chi_allocation_from_jacobian_target_budget(
    jacobian: &InputJacobian,
    total_budget: usize,
    chi_min: usize,
    chi_max: usize,
    strategy: JacobianAllocation,
) -> Vec<usize> {
    let scores = match strategy {
        JacobianAllocation::ParticipationRatio => participation_ratio_profile(jacobian),
        JacobianAllocation::TotalSensitivity => total_sensitivity_profile(jacobian),
    };
    chi_allocation_target_budget(&scores, total_budget, chi_min, chi_max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jacobian_of_linear_function_is_identity() {
        // f(x) = x, so J = identity
        let inputs = vec![1.0, 2.0, 3.0];
        let factory = |x: &[f64]| x.to_vec();
        let observe = |s: &Vec<f64>| s.clone();
        let j = InputJacobian::compute(&inputs, factory, observe, &JacobianConfig::default());

        assert_eq!(j.n_inputs, 3);
        assert_eq!(j.n_outputs, 3);

        for k in 0..3 {
            for i in 0..3 {
                let expected = if k == i { 1.0 } else { 0.0 };
                assert!(
                    (j.matrix[k][i] - expected).abs() < 1e-6,
                    "J[{k}][{i}] = {} != {expected}",
                    j.matrix[k][i]
                );
            }
        }
    }

    #[test]
    fn jacobian_of_squared_function() {
        // f_k(x) = x_k², so J[k][i] = 2x_k δ_{ki}
        let inputs = vec![1.0, 2.0, 3.0];
        let factory = |x: &[f64]| x.iter().map(|v| v * v).collect::<Vec<f64>>();
        let observe = |s: &Vec<f64>| s.clone();
        let j = InputJacobian::compute(&inputs, factory, observe, &JacobianConfig::default());

        for k in 0..3 {
            for i in 0..3 {
                let expected = if k == i { 2.0 * inputs[k] } else { 0.0 };
                assert!(
                    (j.matrix[k][i] - expected).abs() < 1e-3,
                    "J[{k}][{i}] = {} != {expected}",
                    j.matrix[k][i]
                );
            }
        }
    }

    #[test]
    fn participation_ratio_sanity() {
        // One dominant input → PR = 1
        let row = vec![5.0, 0.0, 0.0, 0.0];
        assert!((participation_ratio(&row) - 1.0).abs() < 1e-10);

        // All equal → PR = N
        let row = vec![1.0, 1.0, 1.0, 1.0];
        assert!((participation_ratio(&row) - 4.0).abs() < 1e-10);

        // Mix
        let row = vec![3.0, 1.0];
        // l1 = 4, l2² = 10, PR = 16/10 = 1.6
        assert!((participation_ratio(&row) - 1.6).abs() < 1e-10);

        // Empty
        assert_eq!(participation_ratio(&[]), 0.0);

        // NaN ignored
        let row = vec![1.0, f64::NAN, 1.0];
        let pr = participation_ratio(&row);
        assert!((pr - 2.0).abs() < 1e-10, "got {pr}");
    }

    #[test]
    fn chi_allocation_uses_full_range() {
        // Two outputs: one local (PR≈1), one global (PR≈4)
        let j = InputJacobian {
            matrix: vec![
                vec![5.0, 0.0, 0.0, 0.0], // local
                vec![1.0, 1.0, 1.0, 1.0], // global
            ],
            n_inputs: 4,
            n_outputs: 2,
            delta: 1e-3,
            stride: 1,
        };

        let chi = chi_allocation_from_jacobian(&j, 4, 64, JacobianAllocation::ParticipationRatio);
        assert_eq!(chi.len(), 2);
        // Local bond should get less chi than global bond
        assert!(
            chi[0] < chi[1],
            "expected chi[local] < chi[global], got {chi:?}"
        );
    }

    // ─── chi_allocation_target_budget (A.1 water-filling) ──────────────────

    #[test]
    fn target_budget_empty_input() {
        assert!(chi_allocation_target_budget(&[], 10, 2, 8).is_empty());
    }

    #[test]
    fn target_budget_under_budget_returns_chi_min() {
        // 4 bonds × chi_min=2 = 8, request 6 → all chi_min.
        let chi = chi_allocation_target_budget(&[1.0, 1.0, 1.0, 1.0], 6, 2, 8);
        assert_eq!(chi, vec![2, 2, 2, 2]);
    }

    #[test]
    fn target_budget_over_budget_returns_chi_max() {
        // 4 bonds × chi_max=8 = 32, request 100 → all chi_max.
        let chi = chi_allocation_target_budget(&[1.0, 1.0, 1.0, 1.0], 100, 2, 8);
        assert_eq!(chi, vec![8, 8, 8, 8]);
    }

    #[test]
    fn target_budget_uniform_scores_split_evenly() {
        // Equal scores, budget 20 over 4 bonds → exactly 5 each.
        let chi = chi_allocation_target_budget(&[1.0, 1.0, 1.0, 1.0], 20, 2, 8);
        assert_eq!(chi.iter().sum::<usize>(), 20);
        // All bonds within ±1 of perfectly even.
        let max = *chi.iter().max().unwrap();
        let min = *chi.iter().min().unwrap();
        assert!(max - min <= 1, "uniform scores produced non-flat: {chi:?}");
    }

    #[test]
    fn target_budget_proportional_to_scores() {
        // Scores [3, 1] over 2 bonds, budget 8, [chi_min=1, chi_max=100].
        // Linear-proportional ideal: 6 above floor split 4.5 / 1.5,
        // ceiled-to-integer water-filling gives 6/2.
        let chi = chi_allocation_target_budget(&[3.0, 1.0], 8, 1, 100);
        assert_eq!(chi.iter().sum::<usize>(), 8);
        assert!(chi[0] > chi[1], "high-score bond should get more: {chi:?}");
        // Ratio sanity: bond 0 should get at least 2× bond 1.
        assert!(chi[0] >= 2 * chi[1], "ratio too narrow: {chi:?}");
    }

    #[test]
    fn target_budget_exactly_consumes_budget() {
        // Random-looking scores, no saturation: total must equal budget.
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
            assert!(c >= 2 && c <= 8, "out of bounds: {c} in {chi:?}");
        }
        // Zero-score bonds should sit at chi_min, high-score bonds at chi_max
        // (8+8+2+2 = 20 — exactly fits).
        assert_eq!(chi, vec![2, 8, 2, 8]);
    }

    #[test]
    fn target_budget_zero_scores_distribute_evenly() {
        // No discriminating signal: must still spend the budget exactly.
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
        // NaN/-inf bonds should be treated as zero-score → sit at chi_min,
        // unless saturating bonds force overflow into them.
        // Finite-score bonds (0, 2) should get at least chi_min+1.
        assert!(chi[0] >= chi[1], "NaN bond should not exceed finite: {chi:?}");
        assert!(chi[2] >= chi[3], "neg-inf bond should not exceed finite: {chi:?}");
    }

    #[test]
    fn target_budget_saturated_high_score_bonds_overflow_into_low_score() {
        // Two bonds with score 100 saturate at chi_max=4 each (=8 total),
        // remaining budget 4 must go to the zero-score bonds.
        let scores = vec![100.0, 100.0, 0.0, 0.0];
        let chi = chi_allocation_target_budget(&scores, 12, 2, 4);
        assert_eq!(chi.iter().sum::<usize>(), 12);
        assert_eq!(chi[0], 4);
        assert_eq!(chi[1], 4);
        assert_eq!(chi[2] + chi[3], 4);
    }

    #[test]
    fn target_budget_matches_uniform_when_scores_flat() {
        // The whole point of A.1: same total budget as uniform-χ → identical
        // allocation when there is no discriminating signal.
        let n = 10;
        let scores = vec![1.0; n];
        let chi_uniform = 8;
        let budget = n * chi_uniform;
        let chi = chi_allocation_target_budget(&scores, budget, 2, chi_uniform);
        assert_eq!(chi, vec![chi_uniform; n]);
    }

    #[test]
    fn jacobian_target_budget_wrapper_smoke() {
        // Same toy Jacobian as the legacy allocator test.
        let j = InputJacobian {
            matrix: vec![
                vec![5.0, 0.0, 0.0, 0.0], // local
                vec![1.0, 1.0, 1.0, 1.0], // global
            ],
            n_inputs: 4,
            n_outputs: 2,
            delta: 1e-3,
            stride: 1,
        };
        let budget = 12;
        let chi = chi_allocation_from_jacobian_target_budget(
            &j,
            budget,
            2,
            10,
            JacobianAllocation::ParticipationRatio,
        );
        assert_eq!(chi.len(), 2);
        assert_eq!(chi.iter().sum::<usize>(), budget);
        assert!(chi[1] > chi[0], "global bond (high PR) should win: {chi:?}");
    }
}
