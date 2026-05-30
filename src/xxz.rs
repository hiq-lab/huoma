//! 1D bond-disordered XXZ spin chain — circuit builder, gate kernel, and
//! reference dense simulator. Foundation of Track G (the sin(C/2) vs
//! uniform-χ shootout in the Griffiths regime, ROADMAP).
//!
//! Hamiltonian:
//!
//! ```text
//! H = Σ_i J_i (Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + Δ Sz_i Sz_{i+1})
//!   = (1/4) Σ_i J_i (σx_i σx_{i+1} + σy_i σy_{i+1} + Δ σz_i σz_{i+1})
//! ```
//!
//! - `J_i > 0` per bond; sampled from a log-uniform distribution for the
//!   Griffiths-regime experiment (`sample_bond_disorder_log_uniform`).
//! - `Δ` (anisotropy): `Δ = 0` is the XX (free-fermion) point,
//!   `Δ = 1` the SU(2)-symmetric Heisenberg point, `|Δ| > 1` the gapped
//!   Ising-like phase. We sweep across `Δ ∈ {0.5, 1.0, 1.5}` in the
//!   Track G shootout.
//!
//! Time evolution is first-order Trotter: one step applies the per-bond
//! gate `exp(-i H_bond_i dt)` sequentially for `i = 0, 1, …, N-2`. For
//! Track G this is sufficient because all three contenders (uniform-χ,
//! ε-truncated ITensor, sin(C/2)) use the same Trotter schedule — the
//! Trotter error is a common-mode signal that cancels out of the
//! shootout.
//!
//! The bond gate decomposes neatly: it is diagonal on |00⟩ and |11⟩
//! (with phase `exp(-i J·dt·Δ/4)`) and acts as a 2x2 block on the
//! (|01⟩, |10⟩) subspace (an XY-rotation by `J·dt/2` modulated by the
//! same diagonal phase). Constructed by `xxz_bond_gate(j_dt, delta)`.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::mps::Mps;

type C = Complex64;

/// Parameters for the 1D bond-disordered XXZ chain.
///
/// Per-bond `J_i` are passed separately to `apply_xxz_step`; this struct
/// holds only the homogeneous knobs.
#[derive(Debug, Clone, Copy)]
pub struct XxzParams {
    /// Anisotropy along z. `Δ = 0` → XX point, `Δ = 1` → isotropic
    /// Heisenberg, `|Δ| > 1` → Ising-like gapped phase.
    pub delta: f64,
    /// Trotter step size.
    pub dt: f64,
}

/// Two-qubit XXZ bond gate `exp(-i (J·dt/4) (XX + YY + Δ ZZ))` on the
/// basis `(|00⟩, |01⟩, |10⟩, |11⟩)`.
///
/// The matrix is block-diagonal: `(|00⟩, |11⟩)` carry a global phase
/// `exp(-i τ Δ/4)` where `τ = J·dt`, and `(|01⟩, |10⟩)` get the same
/// global phase multiplied by an `exp(-i τ/2 · σx)` rotation in that
/// 2-dimensional subspace. Off-block entries are zero.
#[must_use]
pub fn xxz_bond_gate(j_dt: f64, delta: f64) -> [[C; 4]; 4] {
    let tau = j_dt;
    let phase_diag = C::new((tau * delta / 4.0).cos(), -(tau * delta / 4.0).sin()); // exp(-i τ Δ/4)
    let phase_off = C::new((tau * delta / 4.0).cos(), (tau * delta / 4.0).sin()); //  exp(+i τ Δ/4)

    let c = (tau / 2.0).cos();
    let s = (tau / 2.0).sin();
    let m_diag = phase_off * C::new(c, 0.0); // exp(+i τ Δ/4) · cos(τ/2)
    let m_off = phase_off * C::new(0.0, -s); // exp(+i τ Δ/4) · (-i sin(τ/2))

    let zero = C::new(0.0, 0.0);
    [
        [phase_diag, zero, zero, zero],
        [zero, m_diag, m_off, zero],
        [zero, m_off, m_diag, zero],
        [zero, zero, zero, phase_diag],
    ]
}

/// Apply one Trotter step of the XXZ Hamiltonian to `mps` with per-bond
/// couplings `j_per_bond[i]` and per-bond truncation budget
/// `chi_per_bond[i]`. Step size and anisotropy come from `params`.
///
/// First-order Trotter: bonds are processed sequentially `0, 1, …, N-2`.
/// SVD truncation runs at every bond per `chi_per_bond`.
pub fn apply_xxz_step(
    mps: &mut Mps,
    j_per_bond: &[f64],
    params: XxzParams,
    chi_per_bond: &[usize],
) -> Result<()> {
    let n = mps.n_qubits;
    assert_eq!(
        j_per_bond.len(),
        n - 1,
        "j_per_bond must have length n-1 (one per bond)"
    );
    assert_eq!(
        chi_per_bond.len(),
        n - 1,
        "chi_per_bond must have length n-1"
    );

    for q in 0..n - 1 {
        let gate = xxz_bond_gate(j_per_bond[q] * params.dt, params.delta);
        let max_chi = chi_per_bond[q];
        mps.apply_two_qubit(q, gate, max_chi)?;
    }

    Ok(())
}

/// Sample `n_bonds` couplings `J_i` from a log-uniform distribution on
/// `[j_min, j_max]`. Used to generate the bond-disorder realisations for
/// the Track G Griffiths-regime shootout — strong disorder is the
/// regime where rare regions of anomalously small `J` dominate the slow
/// dynamics (Aramthottil et al., PRL 133, 196302, 2024).
///
/// `seed` is the PRNG seed for deterministic reproducibility across
/// runs. Uses a self-contained splitmix64 generator so we do not pull
/// in a `rand` dep just for this.
#[must_use]
pub fn sample_bond_disorder_log_uniform(
    n_bonds: usize,
    j_min: f64,
    j_max: f64,
    seed: u64,
) -> Vec<f64> {
    assert!(j_min > 0.0, "log-uniform requires j_min > 0");
    assert!(j_max > j_min, "j_max must be > j_min");

    let log_min = j_min.ln();
    let log_max = j_max.ln();
    let span = log_max - log_min;

    let mut state = seed;
    (0..n_bonds)
        .map(|_| {
            // splitmix64 step → uniform u64 → uniform f64 in [0, 1)
            state = state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            let u = (z >> 11) as f64 / (1_u64 << 53) as f64;
            (log_min + u * span).exp()
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// ITensor cross-reference manifest (Track G.2)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for an ITensor cross-reference run. Serialised to a
/// `*.manifest.json` file consumed by `external/itensor_ref/xxz_griffiths.jl`,
/// and produces an `*.itensor.json` next to the manifest.
///
/// The schema is the load-bearing contract between Huoma and the Julia
/// reference runner — both sides must agree on every field name and
/// convention (especially `initial_index` MSB-ordering and the
/// `j_per_bond` bond direction).
///
/// See `external/itensor_ref/README.md` for the matching Julia loader
/// (`load_manifest`) and the output schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItensorXxzManifest {
    /// Chain length.
    pub n: usize,
    /// XXZ anisotropy along z.
    pub delta: f64,
    /// Trotter step size.
    pub dt: f64,
    /// Number of Trotter steps.
    pub n_steps: usize,
    /// Computational-basis initial-state index. MSB = qubit 0 (Huoma
    /// convention). E.g. for N = 4 the Néel state |0101⟩ is
    /// `0b0101 = 5`.
    pub initial_index: u64,
    /// Per-bond couplings `J_i`, length `n - 1`. `j_per_bond[i]` is the
    /// coupling on the bond between qubits `i` and `i + 1`.
    pub j_per_bond: Vec<f64>,
    /// ITensor `maxdim` bond-dim cap.
    pub chi_cap: usize,
    /// ITensor `cutoff` discarded-weight threshold. Pass `0.0` to
    /// disable cutoff-truncation (rely on `chi_cap` alone).
    pub epsilon: f64,
}

impl ItensorXxzManifest {
    /// Serialise the manifest to pretty-printed JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("ItensorXxzManifest serialisation is infallible")
    }

    /// Parse a manifest from JSON.
    pub fn from_json(s: &str) -> std::result::Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Independent reference dense statevector simulator
// ─────────────────────────────────────────────────────────────────────────────

/// Apply the 4×4 XXZ bond gate `u` to adjacent qubits `(q, q+1)` of a
/// dense statevector. Convention: qubit 0 is the MSB, matching huoma's
/// `to_statevector` ordering and the dense simulator used by
/// `kicked_ising::reference_kim_run`.
///
/// Acts on the 4-dim subspace indexed by the two qubit bits and leaves
/// the remaining `2^(n-2)` amplitudes untouched per outer index.
fn apply_xxz_bond_dense(psi: &mut [C], n: usize, q: usize, u: &[[C; 4]; 4]) {
    let dim = 1_usize << n;
    let bit_q = n - 1 - q;
    let bit_qp1 = n - 1 - (q + 1);
    let mask = (1_usize << bit_q) | (1_usize << bit_qp1);

    // Iterate over outer index (the other n-2 qubits). For each, the
    // 4 amplitudes (00, 01, 10, 11) on (q, q+1) form a contiguous 4-vec
    // in the (q, q+1) subspace.
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

/// Compute `⟨Z_q⟩` for every qubit from a dense (normalised) statevector.
/// Convention: qubit 0 is MSB. Same as `kicked_ising::measure_all_z`.
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

/// Run the bond-disordered XXZ chain on a dense statevector and return
/// `⟨Z_q(t)⟩` for every qubit at every Trotter step (returned matrix is
/// `(n_steps+1) × n`, index 0 is the initial state).
///
/// `initial` is the starting computational-basis state index (e.g. for
/// the Néel state at N = 4 use `0b0101 = 5`; for the all-zero state
/// use `0`). Independent of the MPS code path — ground truth for
/// validation at small N.
///
/// **Only feasible for `n ≤ ~24`** (`2^24` complex amplitudes ≈ 256 MB).
#[must_use]
pub fn reference_xxz_run(
    n: usize,
    j_per_bond: &[f64],
    params: XxzParams,
    initial: usize,
    n_steps: usize,
) -> Vec<Vec<f64>> {
    assert_eq!(j_per_bond.len(), n - 1);
    let dim = 1_usize << n;
    assert!(initial < dim, "initial state index out of range");

    let mut psi = vec![C::new(0.0, 0.0); dim];
    psi[initial] = C::new(1.0, 0.0);

    let gates: Vec<[[C; 4]; 4]> = j_per_bond
        .iter()
        .map(|&j| xxz_bond_gate(j * params.dt, params.delta))
        .collect();

    let mut history = Vec::with_capacity(n_steps + 1);
    history.push(measure_all_z_dense(&psi, n));

    for _step in 0..n_steps {
        for (q, gate) in gates.iter().enumerate() {
            apply_xxz_bond_dense(&mut psi, n, q, gate);
        }
        history.push(measure_all_z_dense(&psi, n));
    }

    history
}

/// Prepare a product-state MPS in the computational-basis state with
/// index `initial`. Useful as the starting state for XXZ TEBD runs to
/// match the dense-reference initial condition exactly.
///
/// Convention: qubit 0 is MSB. So `initial = 0` is `|0…0⟩` and the Néel
/// state on N = 4 (`|0101⟩`) is `initial = 0b0101 = 5`.
#[must_use]
pub fn product_state_mps(n: usize, initial: usize) -> Mps {
    let mut mps = Mps::new(n);
    for q in 0..n {
        let bit = n - 1 - q;
        let one = (initial >> bit) & 1 == 1;
        if one {
            let x = [
                [C::new(0.0, 0.0), C::new(1.0, 0.0)],
                [C::new(1.0, 0.0), C::new(0.0, 0.0)],
            ];
            mps.apply_single_layer(&{
                let mut layer: Vec<[[C; 2]; 2]> = (0..n)
                    .map(|_| {
                        [
                            [C::new(1.0, 0.0), C::new(0.0, 0.0)],
                            [C::new(0.0, 0.0), C::new(1.0, 0.0)],
                        ]
                    })
                    .collect();
                layer[q] = x;
                layer
            });
        }
    }
    mps
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The XXZ bond gate must be unitary up to FP precision.
    #[test]
    fn bond_gate_is_unitary() {
        let cases = [
            (0.3, 0.0),  // XX point, mild evolution
            (0.5, 1.0),  // Heisenberg
            (1.2, 1.5),  // Ising-like, deeper rotation
            (0.0, 0.7),  // identity case (no time)
            (-0.4, 2.0), // negative coupling
        ];
        for (j_dt, delta) in cases {
            let u = xxz_bond_gate(j_dt, delta);
            // U U† = I → for every (i, j), Σ_k u[i][k] · conj(u[j][k]) = δ_ij
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
                        "(i,j)=({i},{j}) (j_dt,Δ)=({j_dt},{delta}): got {acc:?}",
                    );
                }
            }
        }
    }

    /// At `j_dt = 0` the gate must be the identity for every `Δ`.
    #[test]
    fn bond_gate_zero_time_is_identity() {
        for delta in [-1.0, 0.0, 0.5, 1.0, 2.0] {
            let u = xxz_bond_gate(0.0, delta);
            for i in 0..4 {
                for j in 0..4 {
                    let expected = if i == j {
                        C::new(1.0, 0.0)
                    } else {
                        C::new(0.0, 0.0)
                    };
                    assert!(
                        (u[i][j] - expected).norm() < 1e-15,
                        "Δ={delta} (i,j)=({i},{j}): got {:?}",
                        u[i][j]
                    );
                }
            }
        }
    }

    /// Bond-disorder sampling is deterministic across seeds and lies in
    /// the requested interval.
    #[test]
    fn bond_disorder_log_uniform_is_deterministic_and_in_range() {
        let n_bonds = 256;
        let (j_min, j_max) = (0.1, 10.0);
        let a = sample_bond_disorder_log_uniform(n_bonds, j_min, j_max, 0x4242);
        let b = sample_bond_disorder_log_uniform(n_bonds, j_min, j_max, 0x4242);
        let c = sample_bond_disorder_log_uniform(n_bonds, j_min, j_max, 0x4243);

        assert_eq!(a, b, "same seed must produce same sample");
        assert_ne!(a, c, "different seed must produce different sample");
        for (i, &j) in a.iter().enumerate() {
            assert!(j_min <= j && j <= j_max, "j[{i}] = {j} out of [{j_min}, {j_max}]");
        }

        // Distribution sanity: in log-space the mean should sit near the
        // midpoint of [log(j_min), log(j_max)].
        let log_mean = a.iter().map(|j| j.ln()).sum::<f64>() / a.len() as f64;
        let expected_log_mean = (j_min.ln() + j_max.ln()) / 2.0;
        assert!(
            (log_mean - expected_log_mean).abs() < 0.2,
            "log-mean {log_mean} too far from {expected_log_mean}",
        );
    }

    /// The ITensor manifest must round-trip through JSON without loss
    /// of any field. This is the cross-language contract — if it
    /// breaks, the Julia loader breaks silently.
    #[test]
    fn itensor_manifest_round_trips() {
        let original = ItensorXxzManifest {
            n: 32,
            delta: 1.0,
            dt: 0.1,
            n_steps: 50,
            initial_index: 0b1010_1010_1010_1010_1010_1010_1010_1010,
            j_per_bond: vec![0.5, 1.2, 0.3, 1.7, 0.9],
            chi_cap: 64,
            epsilon: 1e-12,
        };
        let json = original.to_json();
        let parsed = ItensorXxzManifest::from_json(&json).expect("round-trip parse must succeed");
        assert_eq!(parsed.n, original.n);
        assert_eq!(parsed.delta, original.delta);
        assert_eq!(parsed.dt, original.dt);
        assert_eq!(parsed.n_steps, original.n_steps);
        assert_eq!(parsed.initial_index, original.initial_index);
        assert_eq!(parsed.j_per_bond, original.j_per_bond);
        assert_eq!(parsed.chi_cap, original.chi_cap);
        assert_eq!(parsed.epsilon, original.epsilon);

        // Field names must match what the Julia loader expects. If a
        // future refactor renames a struct field (e.g. via #[serde(rename)])
        // this assert fails loudly.
        for needed in [
            "\"n\"",
            "\"delta\"",
            "\"dt\"",
            "\"n_steps\"",
            "\"initial_index\"",
            "\"j_per_bond\"",
            "\"chi_cap\"",
            "\"epsilon\"",
        ] {
            assert!(
                json.contains(needed),
                "manifest JSON is missing field {needed}; Julia loader will break:\n{json}",
            );
        }
    }

    /// Lossless χ at N = 10: MPS evolution under `apply_xxz_step` must
    /// match the dense reference to FP precision over 50 Trotter steps.
    /// This is the Track-G load-bearing anchor — every shootout claim
    /// rests on this dense-comparable correctness.
    #[test]
    fn apply_xxz_step_matches_dense_lossless_at_n10() {
        let n = 10;
        let n_steps = 50;
        let params = XxzParams {
            delta: 1.0, // Heisenberg
            dt: 0.1,
        };

        // Deterministic disorder realisation
        let j_per_bond = sample_bond_disorder_log_uniform(n - 1, 0.5, 2.0, 0xA11CE);

        // Néel initial state |0101010101⟩ (10 qubits) — non-trivial
        // entanglement growth under XXZ. Qubit 0 is MSB.
        // bits: 0,1,0,1,0,1,0,1,0,1 → index = 0b0101010101 = 341
        let initial = 0b01_0101_0101;

        // Reference: dense statevector
        let dense_history = reference_xxz_run(n, &j_per_bond, params, initial, n_steps);

        // MPS path: lossless χ. Max bond at the chain midpoint with
        // d = 2 local dim is 2^(n/2) = 32, so χ = 32 is lossless.
        let chi: Vec<usize> = vec![32; n - 1];
        let mut mps = product_state_mps(n, initial);

        // Sanity: initial MPS ⟨Z⟩ must match dense initial ⟨Z⟩
        for q in 0..n {
            assert!(
                (mps.expectation_z(q) - dense_history[0][q]).abs() < 1e-14,
                "initial mismatch at q={q}",
            );
        }

        // Step-by-step comparison
        for step in 1..=n_steps {
            apply_xxz_step(&mut mps, &j_per_bond, params, &chi).unwrap();
            for q in 0..n {
                let mps_z = mps.expectation_z(q);
                let dense_z = dense_history[step][q];
                assert!(
                    (mps_z - dense_z).abs() < 1e-12,
                    "step {step}, q {q}: MPS {mps_z} vs dense {dense_z} (diff {})",
                    (mps_z - dense_z).abs(),
                );
            }
        }
    }
}
