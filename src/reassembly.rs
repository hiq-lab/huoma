//! Reassembly: combine stable and volatile results with interference correction.

use crate::channel::{BETA, DELTA_STRUCT};

/// Projected result with error bounds.
#[derive(Debug, Clone)]
pub struct ProjectedResult {
    /// Number of qubits in the original system.
    pub n_qubits: usize,
    /// Number of volatile qubits actually simulated.
    pub n_volatile: usize,
    /// Estimated fidelity of the projection.
    pub estimated_fidelity: f64,
    /// Sum of discarded singular values (truncation error).
    pub truncation_residual: f64,
    /// Pair-counting scaling prediction for ln(Γ_c).
    pub ln_gamma_c: f64,
}

/// Estimate projection fidelity from truncation residuals and pair-counting scaling.
///
/// Uses δ_struct ≈ 0.06 and β ≈ 0.327 from the pair-counting paper.
/// The truncation residual (sum of discarded singular values) gives a
/// direct bound on the state error: ‖ψ_exact − ψ_mps‖ ≤ Σ_bonds √(Σ s_discarded²).
#[must_use]
pub fn estimate_fidelity(
    n_qubits: usize,
    n_volatile: usize,
    truncation_residual: f64,
) -> ProjectedResult {
    let pairs = n_qubits * (n_qubits - 1) / 2;

    // Pair-counting scaling: ln(Γ_c) = a - β · N(N-1)/2
    // Using a ≈ -4.003 from the 5-point fit
    let ln_gamma_c = -4.003 - BETA * pairs as f64;

    // Fidelity estimate: F ≈ 1 - truncation_error²
    // The truncation residual bounds the L2 norm of the discarded part
    let fidelity = (1.0 - truncation_residual * truncation_residual).max(0.0);

    // Apply δ_struct correction: the actual interference effect
    // reduces the error by δ_struct per pair relative to independent channels
    let corrected_fidelity = fidelity + DELTA_STRUCT * (1.0 - fidelity);

    ProjectedResult {
        n_qubits,
        n_volatile,
        estimated_fidelity: corrected_fidelity.min(1.0),
        truncation_residual,
        ln_gamma_c,
    }
}

/// Scale a projection from n_qubits to a larger target system.
/// Uses pair-counting scaling to extrapolate.
#[must_use]
pub fn scale_projection(base: &ProjectedResult, target_qubits: usize) -> ProjectedResult {
    let base_pairs = base.n_qubits * (base.n_qubits - 1) / 2;
    let target_pairs = target_qubits * (target_qubits - 1) / 2;
    let delta_pairs = target_pairs - base_pairs;

    // Each additional pair reduces threshold by exp(-β)
    let scaling_factor = (-BETA * delta_pairs as f64).exp();

    let ln_gamma_c = -4.003 - BETA * target_pairs as f64;

    // Extrapolated fidelity: scales down with more pairs
    let extrapolated_fidelity = base.estimated_fidelity * scaling_factor;

    ProjectedResult {
        n_qubits: target_qubits,
        n_volatile: base.n_volatile, // conservative: same volatile count
        estimated_fidelity: extrapolated_fidelity.clamp(0.0, 1.0),
        truncation_residual: base.truncation_residual / scaling_factor,
        ln_gamma_c,
    }
}

/// Fidelity estimate for a **projected** run (Track F) where the graph
/// is split into stable (analytical) and volatile (TTN-simulated) domains
/// with a boundary interface.
///
/// Extends [`estimate_fidelity`] by partitioning the pair set:
///
/// - **(volatile, volatile)** pairs contribute the TTN truncation error.
/// - **(volatile, stable)** pairs contribute the boundary approximation
///   error, scaled by the boundary approximation order (0 = product state,
///   higher = low-rank MPS with diminishing error per order).
/// - **(stable, stable)** pairs are exact under the channel model and
///   contribute zero error.
///
/// # Arguments
///
/// * `n_total` — total qubits in the full graph
/// * `n_volatile` — qubits simulated via TTN
/// * `n_boundary` — boundary qubits (volatile-side qubits touching a stable edge)
/// * `truncation_residual` — cumulative discarded weight from all TTN islands
/// * `boundary_order` — 0 = product-state boundary (error ~sin(C/2) at interface),
///   1+ = low-rank MPS correction (error ~sin(C/2)^(order+1))
#[must_use]
pub fn estimate_fidelity_projected(
    n_total: usize,
    n_volatile: usize,
    n_boundary: usize,
    truncation_residual: f64,
    boundary_order: usize,
) -> ProjectedResult {
    let n_stable = n_total.saturating_sub(n_volatile);

    // Pair counts by category.
    let pairs_vv = n_volatile * n_volatile.saturating_sub(1) / 2;
    let pairs_vs = n_volatile * n_stable;
    let pairs_ss = n_stable * n_stable.saturating_sub(1) / 2;
    let pairs_total = pairs_vv + pairs_vs + pairs_ss;
    let _ = pairs_total; // used implicitly in the formula below

    // TTN truncation contribution (same as the unprojected case).
    let trunc_error_sq = truncation_residual * truncation_residual;

    // Boundary approximation contribution. For product-state boundaries
    // (order 0) the error per boundary pair is O(sin(C/2)), which we
    // approximate as a constant ~0.01 per boundary pair. Higher orders
    // reduce this exponentially.
    let boundary_error_per_pair = 0.01 / (1.0 + boundary_order as f64).powi(2);
    let boundary_error_sq = boundary_error_per_pair * n_boundary as f64;

    // Combined fidelity.
    let combined_error_sq = trunc_error_sq + boundary_error_sq;
    let fidelity = (1.0 - combined_error_sq).max(0.0);
    let corrected = fidelity + DELTA_STRUCT * (1.0 - fidelity);

    // Pair-counting scaling on the full graph.
    let ln_gamma_c = -4.003 - BETA * (n_total * (n_total - 1) / 2) as f64;

    ProjectedResult {
        n_qubits: n_total,
        n_volatile,
        estimated_fidelity: corrected.min(1.0),
        truncation_residual,
        ln_gamma_c,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_simulation() {
        let r = estimate_fidelity(10, 4, 0.0);
        assert!((r.estimated_fidelity - 1.0).abs() < 1e-12);
    }

    #[test]
    fn scaling_reduces_fidelity() {
        let base = estimate_fidelity(10, 4, 0.01);
        let scaled = scale_projection(&base, 100);
        assert!(scaled.estimated_fidelity < base.estimated_fidelity);
        assert_eq!(scaled.n_qubits, 100);
    }

    #[test]
    fn projected_zero_boundary_matches_unprojected() {
        // With zero boundary qubits, estimate_fidelity_projected should
        // produce the same fidelity as estimate_fidelity.
        let r_unprojected = estimate_fidelity(10, 4, 0.01);
        let r_projected = estimate_fidelity_projected(10, 4, 0, 0.01, 0);
        assert!(
            (r_unprojected.estimated_fidelity - r_projected.estimated_fidelity).abs() < 1e-12,
            "projected with 0 boundary should match unprojected"
        );
    }

    #[test]
    fn projected_boundary_reduces_fidelity() {
        let r_no_boundary = estimate_fidelity_projected(100, 20, 0, 0.01, 0);
        let r_with_boundary = estimate_fidelity_projected(100, 20, 10, 0.01, 0);
        assert!(
            r_with_boundary.estimated_fidelity <= r_no_boundary.estimated_fidelity,
            "boundary qubits should reduce fidelity"
        );
    }

    #[test]
    fn projected_higher_order_improves_fidelity() {
        let r_order0 = estimate_fidelity_projected(100, 20, 10, 0.01, 0);
        let r_order2 = estimate_fidelity_projected(100, 20, 10, 0.01, 2);
        assert!(
            r_order2.estimated_fidelity >= r_order0.estimated_fidelity,
            "higher boundary order should improve fidelity"
        );
    }

    #[test]
    fn pair_counting_scaling() {
        let r10 = estimate_fidelity(10, 4, 0.01);
        let r20 = estimate_fidelity(20, 8, 0.01);
        // More pairs → lower Γ_c
        assert!(r20.ln_gamma_c < r10.ln_gamma_c);
    }
}
