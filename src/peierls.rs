//! Peierls-substituted nearest-neighbour hopping — foundation gate
//! for Track F.3 (magnetic / Hofstadter Hamiltonians) and the audit
//! anchor for Track F.1 (complex-tensor support).
//!
//! Hamiltonian (per bond `i`):
//!
//! ```text
//! H_bond = -t_i (e^{iφ_i} σ⁺_i σ⁻_{i+1} + e^{-iφ_i} σ⁻_i σ⁺_{i+1})
//! ```
//!
//! where `σ⁺ = (X + iY)/2`, `σ⁻ = (X − iY)/2`, `t_i` is the real
//! hopping amplitude, and `φ_i` is the Peierls phase (the line
//! integral of the magnetic vector potential along the bond). This
//! is the standard tight-binding minimal-coupling substitution
//! `c†_i c_j → e^{iA_ij} c†_i c_j`.
//!
//! Particle-number conservation: the gate acts non-trivially only on
//! the (|01⟩, |10⟩) hopping subspace and is identity on (|00⟩, |11⟩).
//!
//! # Why this module exists for F.1
//!
//! Huoma's MPS / TTN tensors are already `Complex64` throughout, but
//! none of the existing gate paths (KIM RZ/ZZ, XXZ XY+ZZ-bond) force
//! genuine non-trivial complex multiplication through the SVD/QR
//! pipeline — their phase structures all reduce to real `cos(θ)` +
//! imaginary `sin(θ)` single-rotation patterns that a real-only fast
//! path could in principle fake. Peierls hopping at `φ ∉ {0, π/2}`
//! has off-diagonal entries `±i e^{±iφ} sin(t·dt)` whose real and
//! imaginary parts both come from non-trivial mixing — this is the
//! audit anchor for "Huoma's complex pipeline really does propagate
//! complex amplitudes correctly through truncation and observable
//! extraction."

use num_complex::Complex64;

use crate::error::Result;
use crate::mps::Mps;

type C = Complex64;

/// Two-qubit Peierls hopping gate `exp(-i dt H_bond)` for
/// `H_bond = -t (e^{iφ} c†_i c_{i+1} + e^{-iφ} c†_{i+1} c_i)` on the
/// basis `(|00⟩, |01⟩, |10⟩, |11⟩)`. Convention: qubit `i` (left) is
/// the most significant bit, so `|01⟩` has the particle on the right
/// qubit and `|10⟩` has it on the left. The hopping `c†_i c_{i+1}`
/// takes `|01⟩ → |10⟩` and contributes `-t e^{+iφ}` to `H[2,1]`;
/// its Hermitian conjugate contributes `-t e^{-iφ}` to `H[1,2]`.
/// The Peierls phase `φ` is the line integral of the vector potential
/// from site `i` to site `i+1`.
///
/// The gate is identity on `|00⟩` and `|11⟩` (no hopping possible
/// without a particle to hop, or with both occupied) and acts as
///
/// ```text
/// [[ cos(t·dt),                 i e^{-iφ} sin(t·dt) ],
///  [ i e^{+iφ} sin(t·dt),       cos(t·dt)           ]]
/// ```
///
/// on the `(|01⟩, |10⟩)` subspace.
///
/// At `φ = 0`: off-diagonals are `i sin(t·dt)` (purely imaginary,
/// reduces to the standard XX/YY hopping exchange).
/// At `φ = π/2`: off-diagonals are `± sin(t·dt)` (purely real, gate
/// becomes a real anti-symmetric rotation).
/// At any other `φ`: off-diagonals are genuinely complex,
/// exercising the full complex SVD/QR pipeline.
///
/// Unitarity of the 2×2 block on `(|01⟩, |10⟩)` gives
/// `U[2,1] = -conj(U[1,2])` — *not* `U[2,1] = conj(U[1,2])`. That
/// stronger relationship is what `H_bond` (the generator) satisfies,
/// not the gate itself.
#[must_use]
pub fn peierls_hopping_gate(t: f64, dt: f64, phi: f64) -> [[C; 4]; 4] {
    let tau = t * dt;
    let c = tau.cos();
    let s = tau.sin();
    // U[1,2] = i · e^{-iφ} · sin(τ); U[2,1] = i · e^{+iφ} · sin(τ).
    // Naming follows the sign of φ in the exponent.
    let off_neg = C::new(0.0, s) * C::new(phi.cos(), -phi.sin()); // i e^{-iφ} sin(τ)
    let off_pos = C::new(0.0, s) * C::new(phi.cos(), phi.sin()); //  i e^{+iφ} sin(τ)
    let zero = C::new(0.0, 0.0);
    let one = C::new(1.0, 0.0);
    let cos_c = C::new(c, 0.0);

    [
        [one, zero, zero, zero],
        [zero, cos_c, off_neg, zero],
        [zero, off_pos, cos_c, zero],
        [zero, zero, zero, one],
    ]
}

/// Apply one first-order Trotter step of Peierls-substituted
/// nearest-neighbour hopping to `mps`. Per-bond hopping amplitudes
/// `t_per_bond[i]` and Peierls phases `phi_per_bond[i]` are
/// independent; the step size `dt` is shared across all bonds.
///
/// Bonds are processed sequentially `0, 1, …, n−2`. SVD truncation
/// runs at every bond per `chi_per_bond`.
pub fn apply_peierls_step(
    mps: &mut Mps,
    t_per_bond: &[f64],
    phi_per_bond: &[f64],
    dt: f64,
    chi_per_bond: &[usize],
) -> Result<()> {
    let n = mps.n_qubits;
    assert_eq!(
        t_per_bond.len(),
        n - 1,
        "t_per_bond must have length n-1 (one per bond)"
    );
    assert_eq!(
        phi_per_bond.len(),
        n - 1,
        "phi_per_bond must have length n-1"
    );
    assert_eq!(chi_per_bond.len(), n - 1, "chi_per_bond must have length n-1");

    for q in 0..n - 1 {
        let gate = peierls_hopping_gate(t_per_bond[q], dt, phi_per_bond[q]);
        let max_chi = chi_per_bond[q];
        mps.apply_two_qubit(q, gate, max_chi)?;
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Independent reference dense statevector simulator
// ─────────────────────────────────────────────────────────────────────────────

/// Apply the 4×4 Peierls bond gate `u` to adjacent qubits `(q, q+1)`
/// of a dense statevector. Convention: qubit 0 is MSB, matching
/// huoma's `to_statevector` ordering (same as `kicked_ising` and
/// `xxz` reference simulators).
fn apply_peierls_bond_dense(psi: &mut [C], n: usize, q: usize, u: &[[C; 4]; 4]) {
    let dim = 1_usize << n;
    let bit_q = n - 1 - q;
    let bit_qp1 = n - 1 - (q + 1);
    let mask = (1_usize << bit_q) | (1_usize << bit_qp1);

    let mut visited = vec![false; dim];
    for idx in 0..dim {
        if visited[idx] {
            continue;
        }
        let base = idx & !mask;
        let i00 = base;
        let i01 = base | (1_usize << bit_qp1);
        let i10 = base | (1_usize << bit_q);
        let i11 = base | mask;

        let a00 = psi[i00];
        let a01 = psi[i01];
        let a10 = psi[i10];
        let a11 = psi[i11];

        psi[i00] = u[0][0] * a00 + u[0][1] * a01 + u[0][2] * a10 + u[0][3] * a11;
        psi[i01] = u[1][0] * a00 + u[1][1] * a01 + u[1][2] * a10 + u[1][3] * a11;
        psi[i10] = u[2][0] * a00 + u[2][1] * a01 + u[2][2] * a10 + u[2][3] * a11;
        psi[i11] = u[3][0] * a00 + u[3][1] * a01 + u[3][2] * a10 + u[3][3] * a11;

        visited[i00] = true;
        visited[i01] = true;
        visited[i10] = true;
        visited[i11] = true;
    }
}

fn measure_all_z_dense(psi: &[C], n: usize) -> Vec<f64> {
    let dim = 1_usize << n;
    let mut z = vec![0.0_f64; n];
    for q in 0..n {
        let bit = n - 1 - q;
        let mut acc = 0.0_f64;
        for (idx, amp) in psi.iter().enumerate().take(dim) {
            let p = amp.norm_sqr();
            if (idx >> bit) & 1 == 0 {
                acc += p;
            } else {
                acc -= p;
            }
        }
        z[q] = acc;
    }
    z
}

/// Run the Peierls hopping chain on a dense statevector and return
/// `⟨Z_q(t)⟩` for every qubit at every Trotter step (returned matrix
/// is `(n_steps+1) × n`, index 0 is the initial state).
///
/// `initial` is the starting computational-basis state index. Qubit 0
/// is MSB (same as `kicked_ising::reference_kim_run` /
/// `xxz::reference_xxz_run`).
#[must_use]
pub fn reference_peierls_run(
    n: usize,
    t_per_bond: &[f64],
    phi_per_bond: &[f64],
    dt: f64,
    initial: usize,
    n_steps: usize,
) -> Vec<Vec<f64>> {
    assert_eq!(t_per_bond.len(), n - 1);
    assert_eq!(phi_per_bond.len(), n - 1);
    let dim = 1_usize << n;
    assert!(initial < dim);

    let mut psi = vec![C::new(0.0, 0.0); dim];
    psi[initial] = C::new(1.0, 0.0);

    let gates: Vec<[[C; 4]; 4]> = t_per_bond
        .iter()
        .zip(phi_per_bond.iter())
        .map(|(&t, &phi)| peierls_hopping_gate(t, dt, phi))
        .collect();

    let mut history = Vec::with_capacity(n_steps + 1);
    history.push(measure_all_z_dense(&psi, n));

    for _step in 0..n_steps {
        for (q, gate) in gates.iter().enumerate() {
            apply_peierls_bond_dense(&mut psi, n, q, gate);
        }
        history.push(measure_all_z_dense(&psi, n));
    }

    history
}

/// Prepare a product-state MPS in the computational-basis state with
/// index `initial`. Convention: qubit 0 is MSB.
#[must_use]
pub fn product_state_mps(n: usize, initial: usize) -> Mps {
    // Same MSB convention as xxz::product_state_mps; duplicated here to
    // keep peierls.rs independent of the XXZ module.
    let mut mps = Mps::new(n);
    let x_gate = [
        [C::new(0.0, 0.0), C::new(1.0, 0.0)],
        [C::new(1.0, 0.0), C::new(0.0, 0.0)],
    ];
    let i_gate = [
        [C::new(1.0, 0.0), C::new(0.0, 0.0)],
        [C::new(0.0, 0.0), C::new(1.0, 0.0)],
    ];
    for q in 0..n {
        let bit = n - 1 - q;
        let one = (initial >> bit) & 1 == 1;
        if one {
            let mut layer: Vec<[[C; 2]; 2]> = (0..n).map(|_| i_gate).collect();
            layer[q] = x_gate;
            mps.apply_single_layer(&layer);
        }
    }
    mps
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Peierls hopping gate must be unitary up to FP precision
    /// across all four "interesting" phase regimes:
    /// 0 (real-XY), π/4 (mixed), π/3 (generic), π/2 (real-anti-sym),
    /// 2π/3 (generic negative).
    #[test]
    fn peierls_gate_is_unitary() {
        let cases = [
            (1.0, 0.5, 0.0),
            (0.7, 1.2, std::f64::consts::FRAC_PI_4),
            (1.3, 0.4, std::f64::consts::FRAC_PI_3),
            (0.5, 0.9, std::f64::consts::FRAC_PI_2),
            (1.0, 0.8, 2.0 * std::f64::consts::FRAC_PI_3),
        ];
        for (t, dt, phi) in cases {
            let u = peierls_hopping_gate(t, dt, phi);
            for i in 0..4 {
                for j in 0..4 {
                    let mut acc = C::new(0.0, 0.0);
                    for k in 0..4 {
                        acc += u[i][k] * u[j][k].conj();
                    }
                    let expected = if i == j {
                        C::new(1.0, 0.0)
                    } else {
                        C::new(0.0, 0.0)
                    };
                    assert!(
                        (acc - expected).norm() < 1e-13,
                        "(t,dt,φ)=({t},{dt},{phi}) (i,j)=({i},{j}): {acc:?}",
                    );
                }
            }
        }
    }

    /// At φ = 0 the off-diagonal entries must be purely imaginary
    /// `i sin(t·dt)` — the standard XX/YY hopping exchange. Pinned
    /// explicitly so any future change to the gate construction that
    /// silently drops the i factor (real-only path mistake) fails
    /// here loudly.
    #[test]
    fn phi_zero_reduces_to_standard_xy_hopping() {
        let t = 1.3;
        let dt = 0.7;
        let u = peierls_hopping_gate(t, dt, 0.0);
        let expected_off = C::new(0.0, (t * dt).sin());
        assert!((u[1][2] - expected_off).norm() < 1e-15);
        assert!((u[2][1] - expected_off).norm() < 1e-15);
        // Diagonals on the hopping subspace are cos(t·dt) real.
        assert!((u[1][1] - C::new((t * dt).cos(), 0.0)).norm() < 1e-15);
        assert!((u[2][2] - C::new((t * dt).cos(), 0.0)).norm() < 1e-15);
    }

    /// At φ = π/3 the off-diagonal entries have *both* non-zero real
    /// and imaginary parts. This is the configuration the F.1 audit
    /// anchor below uses, so we pin its structural features explicitly.
    ///
    /// Two off-diagonals are related by unitarity as
    /// `U[2,1] = -conj(U[1,2])` (not `U[2,1] = conj(U[1,2])`, which
    /// is what the *Hamiltonian* satisfies). Pinned so a future code
    /// change cannot silently shift to the Hamiltonian-style relation
    /// without breaking unitarity.
    #[test]
    fn phi_pi_third_has_genuinely_complex_off_diagonals() {
        let t = 1.0;
        let dt = 0.5;
        let phi = std::f64::consts::FRAC_PI_3;
        let u = peierls_hopping_gate(t, dt, phi);

        // u[1][2] = i e^{-iφ} sin(t·dt) = sin(t·dt) (sin φ + i cos φ)
        // u[2][1] = i e^{+iφ} sin(t·dt) = sin(t·dt) (-sin φ + i cos φ)
        let s = (t * dt).sin();
        let expected_12 = C::new(s * phi.sin(), s * phi.cos());
        let expected_21 = C::new(-s * phi.sin(), s * phi.cos());
        assert!((u[1][2] - expected_12).norm() < 1e-15);
        assert!((u[2][1] - expected_21).norm() < 1e-15);

        // Both real and imaginary parts must be substantially non-zero
        // (this is the "exercises complex mixing" property the F.1 audit
        // needs).
        assert!(u[1][2].re.abs() > 0.1 && u[1][2].im.abs() > 0.1);
        assert!(u[2][1].re.abs() > 0.1 && u[2][1].im.abs() > 0.1);

        // Unitarity of the (|01⟩, |10⟩) block: U[2,1] = -conj(U[1,2]).
        // This is *not* U[2,1] = conj(U[1,2]); that stronger relation
        // holds for the *generator* H_bond (Hermitian), not for the
        // unitary it integrates to.
        let neg_conj = -u[1][2].conj();
        assert!(
            (u[2][1] - neg_conj).norm() < 1e-15,
            "U[2,1] = {:?} should equal -conj(U[1,2]) = {:?}",
            u[2][1],
            neg_conj
        );
    }

    /// **F.1 audit anchor.** N = 6 chain, single particle initially at
    /// site 2 (state |001000⟩), 20 first-order Trotter steps of
    /// Peierls hopping with `t = 1`, `dt = 0.1`, `φ = π/3` per bond
    /// (uniform). Lossless χ = 2^3 = 8 on the chain midpoint. The
    /// MPS pipeline must reproduce the dense reference to `≤ 1e-13`
    /// max |⟨Z_i⟩| error every step.
    ///
    /// If this passes, Huoma's complex-tensor support is confirmed
    /// operational end-to-end through the SVD/QR/truncation/observable
    /// pipeline for genuinely complex-mixing gates. **This is what
    /// closes Track F.1 — no separate "complex pivot" sprint is
    /// needed.**
    #[test]
    fn peierls_step_matches_dense_lossless_at_n6() {
        let n = 6;
        let n_steps = 20;
        let dt = 0.1;
        let phi = std::f64::consts::FRAC_PI_3;
        let initial = 0b001000_usize; // particle at qubit 2

        let t_per_bond: Vec<f64> = vec![1.0; n - 1];
        let phi_per_bond: Vec<f64> = vec![phi; n - 1];

        let dense_history =
            reference_peierls_run(n, &t_per_bond, &phi_per_bond, dt, initial, n_steps);

        // Lossless χ at N=6: max bond dim at the chain midpoint with
        // local d=2 is 2^(n/2) = 8.
        let chi: Vec<usize> = vec![8; n - 1];
        let mut mps = product_state_mps(n, initial);

        // Sanity: initial MPS ⟨Z⟩ must match dense initial ⟨Z⟩.
        for q in 0..n {
            assert!(
                (mps.expectation_z(q) - dense_history[0][q]).abs() < 1e-14,
                "initial mismatch at q={q}",
            );
        }

        for step in 1..=n_steps {
            apply_peierls_step(&mut mps, &t_per_bond, &phi_per_bond, dt, &chi).unwrap();
            for q in 0..n {
                let mps_z = mps.expectation_z(q);
                let dense_z = dense_history[step][q];
                assert!(
                    (mps_z - dense_z).abs() < 1e-13,
                    "step {step}, q {q}: MPS {mps_z} vs dense {dense_z} (diff {})",
                    (mps_z - dense_z).abs(),
                );
            }
        }
    }
}
