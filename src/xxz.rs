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
// Per-site natural frequencies for bond-disordered XXZ (Track G feeder)
// ─────────────────────────────────────────────────────────────────────────────

/// Per-site natural frequencies for the bond-disordered XXZ chain,
/// constructed so the existing `huoma::chi_allocation_sinc` allocator
/// applies without modification.
///
/// Site `i` has frequency `ω_i = √(|J_{i-1}| · |J_i|)` — the geometric
/// mean of its two adjacent bond couplings. Boundary sites take the
/// single touching bond: `ω_0 = |J_0|`, `ω_{n-1} = |J_{n-2}|`.
///
/// # Provenance: geometric mean from Dasgupta–Ma RG
///
/// In the strong-disorder regime, the Dasgupta–Ma real-space
/// renormalisation group (which generates Griffiths physics in
/// 1D bond-disordered chains) eliminates the strongest bond `J_max` at
/// each step by forming a singlet across it. The two adjacent sites
/// are removed from the chain and the next-nearest neighbours pick up
/// an effective second-order coupling
///
/// ```text
/// J_eff = J_left · J_right / J_max
/// ```
///
/// — a strictly multiplicative composition. The natural energy scale
/// at site `i` *before* any RG step is taken is therefore the
/// geometric mean of its two adjacent bond couplings, not their
/// arithmetic mean and not the smaller of the two.
///
/// # Honest framing: this is the sin(C/2) negative control, not the
///   recommended Griffiths allocator
///
/// sin(C/2) was derived from KAM physics for *driven* systems (KIM,
/// QKR): given two oscillation frequencies, it measures whether they
/// are in a low-order rational resonance (stable, integrable) or
/// off-resonance (chaotic, requiring high χ). Bond-disordered XXZ
/// has *no drive*. The relevant physics is **scale hierarchy** of
/// couplings under RG, not frequency commensurability.
///
/// Concretely: with a random bond-coupling distribution, adjacent
/// `ω_i / ω_{i+1}` ratios occasionally land near low-order rationals
/// by chance — sin(C/2) then reports "commensurate" and the allocator
/// withholds χ from those bonds. This is **physically wrong** for
/// Griffiths: those bonds are not in any drive resonance, the
/// rational coincidence is statistical noise.
///
/// `xxz_site_frequencies` exists so the G.3 shootout can exhibit this
/// failure mode of sin(C/2) directly and contrast it against the
/// scale-hierarchy allocator `xxz_griffiths_bond_scores`. **Use
/// `xxz_griffiths_bond_scores` for production allocation on
/// bond-disordered XXZ; use `xxz_site_frequencies` only as the
/// sin(C/2) negative control.**
///
/// # Dimensions
///
/// `J` is energy (= frequency with ℏ = 1), so `ω_i = √(J · J)` is a
/// frequency. Identical units to the KIM use of `chi_allocation_sinc`.
///
/// # References
///
/// - Dasgupta & Ma, Phys. Rev. B 22, 1305 (1980)
/// - D. S. Fisher, Phys. Rev. B 50, 3799 (1994)
/// - Refael & Moore, Phys. Rev. Lett. 93, 260602 (2004)
/// - Aramthottil et al., PRL 133, 196302 (2024) — XXZ Griffiths regime
/// - Hinderink 2026, "The Tilde Pattern" — sin(C/2) derivation and
///   its driven-system scope
///
/// # Panics
///
/// If `j_per_bond.is_empty()`.
#[must_use]
pub fn xxz_site_frequencies(j_per_bond: &[f64]) -> Vec<f64> {
    let n_bonds = j_per_bond.len();
    assert!(
        n_bonds > 0,
        "xxz_site_frequencies requires at least one bond"
    );
    let n_sites = n_bonds + 1;
    let abs_j: Vec<f64> = j_per_bond.iter().map(|j| j.abs()).collect();

    let mut omega = vec![0.0_f64; n_sites];
    omega[0] = abs_j[0];
    omega[n_sites - 1] = abs_j[n_bonds - 1];
    for i in 1..n_sites - 1 {
        omega[i] = (abs_j[i - 1] * abs_j[i]).sqrt();
    }
    omega
}

/// Per-bond scores for the bond-disordered XXZ chain, suitable as
/// input to `huoma::chi_allocation_target_budget`. Returns one score
/// per bond, equal to `|J_i|`.
///
/// # First-principles derivation
///
/// In finite-time TEBD evolution from a product initial state under
/// XXZ, the bipartite entanglement entropy across bond `i` grows as
///
/// ```text
/// S_i(t) ≈ min(log 2, |J_i| · t)
/// ```
///
/// — linearly until saturation at the single-singlet bound. The MPS
/// bond dimension required to represent this entropy faithfully is
/// approximately `χ_i ~ exp(S_i)`, which is monotone increasing in
/// `|J_i|` at any fixed time. Strong bonds form singlets first and
/// carry maximal local entanglement; weak bonds remain near-product
/// and carry little.
///
/// Dasgupta–Ma RG (Dasgupta & Ma 1980; Fisher 1994) confirms this:
/// the strongest bond is eliminated first by singlet formation, and
/// rare weak bonds in a sea of strong bonds (Griffiths regions) carry
/// suppressed entanglement because correlations cannot pass through
/// them. The right adaptive-χ strategy is therefore **give χ to the
/// strong bonds, save it at the weak ones** — the opposite of the
/// common Griffiths-as-"rare-weak-bond" framing (which describes
/// *transport*, not *bipartite entanglement*).
///
/// # Relation to sin(C/2)
///
/// sin(C/2) does not apply here. It measures KAM resonance of
/// *driven* oscillators; bond-disordered XXZ has no drive, no KAM
/// torus structure, no commensurability physics to filter against.
/// The Tilde Pattern's per-bond water-filling architecture is reused
/// unchanged via `chi_allocation_target_budget`, but the *score*
/// `|J_i|` comes from RG / TEBD-entropy first-principles, not from
/// sin(C/2). See `xxz_site_frequencies` for the sin(C/2)
/// negative-control variant.
///
/// # References
///
/// - Dasgupta & Ma, Phys. Rev. B 22, 1305 (1980)
/// - D. S. Fisher, Phys. Rev. B 50, 3799 (1994)
/// - Aramthottil et al., PRL 133, 196302 (2024)
/// - Calabrese & Cardy, J. Stat. Mech. P04010 (2005) — TEBD entropy
///   growth bounds
///
/// # Panics
///
/// If `j_per_bond.is_empty()`.
#[must_use]
pub fn xxz_griffiths_bond_scores(j_per_bond: &[f64]) -> Vec<f64> {
    assert!(
        !j_per_bond.is_empty(),
        "xxz_griffiths_bond_scores requires at least one bond"
    );
    j_per_bond.iter().map(|j| j.abs()).collect()
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

    /// Uniform J → all per-site geometric-mean frequencies equal →
    /// sin(C/2) on that input ≡ 0 → the derived χ allocation
    /// degenerates to uniform-χ. This is the "no signal" sanity
    /// behaviour and the only case in which `xxz_site_frequencies` is
    /// guaranteed to interact correctly with sin(C/2) — see the doc
    /// comment for why disordered J fragility makes this extractor a
    /// negative control rather than a production allocator.
    #[test]
    fn uniform_j_yields_uniform_frequencies_and_uniform_chi() {
        use crate::allocator::chi_allocation_sinc;

        let n_bonds = 31;
        let j_uniform = vec![1.5; n_bonds];
        let omega = xxz_site_frequencies(&j_uniform);

        for (i, &w) in omega.iter().enumerate() {
            assert!(
                (w - 1.5).abs() < 1e-15,
                "site {i} ω = {w} != 1.5 (uniform J case)",
            );
        }

        let total_budget = n_bonds * 8;
        let chi = chi_allocation_sinc(&omega, total_budget, 2, 16);
        for (i, &c) in chi.iter().enumerate() {
            assert_eq!(
                c, 8,
                "bond {i} got χ = {c} (uniform J, budget {total_budget}, expected 8)",
            );
        }
    }

    /// The geometric-mean construction must reduce correctly to the
    /// single-bond touching coupling at both boundaries.
    #[test]
    fn boundary_frequencies_use_single_touching_bond() {
        let j = vec![0.3, 0.7, 1.2, 2.5];
        let omega = xxz_site_frequencies(&j);
        assert_eq!(omega.len(), 5);
        assert!((omega[0] - 0.3).abs() < 1e-15);
        assert!((omega[4] - 2.5).abs() < 1e-15);
        assert!((omega[1] - (0.3_f64 * 0.7).sqrt()).abs() < 1e-15);
        assert!((omega[2] - (0.7_f64 * 1.2).sqrt()).abs() < 1e-15);
        assert!((omega[3] - (1.2_f64 * 2.5).sqrt()).abs() < 1e-15);
    }

    /// Documents the sin(C/2)-on-bond-disorder fragility that
    /// motivated the split between `xxz_site_frequencies` (negative
    /// control) and `xxz_griffiths_bond_scores` (production
    /// allocator). With `J_weak = 0.01 = 1/100`, the ratio of
    /// neighbouring per-site frequencies lands exactly on an integer
    /// (a low-order rational p/1), and sin(C/2) reports "perfect
    /// commensurability" — withholding χ from the rare-region bonds
    /// instead of routing more to them. This is a statistical
    /// coincidence in the disorder, not a real KAM resonance: the
    /// physics has no drive, no torus structure. Test confirms the
    /// failure mode is reproducible so we can point at it in the G.4
    /// verdict.
    #[test]
    fn sinc2_on_bond_disorder_is_fragile_to_integer_ratios() {
        use crate::allocator::chi_allocation_sinc;

        let n_bonds = 21;
        let weak_idx = 10;
        let mut j = vec![1.0; n_bonds];
        j[weak_idx] = 0.01; // J_weak / J_typical = 1/100 — exact integer ratio

        let omega = xxz_site_frequencies(&j);
        let total_budget = n_bonds * 8;
        let chi = chi_allocation_sinc(&omega, total_budget, 2, 16);

        // sin(C/2) sees ω-ratios of 10 (or 1/10) as "perfectly
        // commensurate" with p/q = 10/1, scores them zero, and
        // delivers a uniform-χ allocation. The Griffiths physics is
        // invisible to this allocator.
        let typical_chi = chi[2];
        for k in 0..n_bonds {
            assert_eq!(
                chi[k], typical_chi,
                "bond {k} got χ = {} ≠ typical {typical_chi}; \
                 sin(C/2) was supposed to fail uniformly on this \
                 integer-ratio bond-disorder case",
                chi[k]
            );
        }
    }

    /// The first-principles Griffiths score must (a) reproduce uniform
    /// χ on uniform J, and (b) on a chain with one anomalously *strong*
    /// bond, route more χ to that strong bond (because that is where a
    /// singlet forms and bipartite entropy saturates fastest under
    /// TEBD). This is the load-bearing physical behaviour the
    /// production allocator on bond-disorder XXZ must reproduce.
    #[test]
    fn griffiths_score_redirects_chi_to_strong_bonds() {
        use crate::allocator::chi_allocation_target_budget;

        // (a) uniform J ⇒ uniform scores ⇒ uniform χ
        {
            let n_bonds = 21;
            let j_uniform = vec![1.0; n_bonds];
            let scores = xxz_griffiths_bond_scores(&j_uniform);
            let total_budget = n_bonds * 8;
            let chi = chi_allocation_target_budget(&scores, total_budget, 2, 16);
            for (k, &c) in chi.iter().enumerate() {
                assert_eq!(
                    c, 8,
                    "uniform-J bond {k} got χ = {c} ≠ 8 (expected uniform allocation)",
                );
            }
        }

        // (b) one anomalously strong bond ⇒ that bond gets more χ
        {
            let n_bonds = 21;
            let strong_idx = 10;
            let mut j = vec![1.0; n_bonds];
            j[strong_idx] = 5.0; // 5× stronger than the rest

            let scores = xxz_griffiths_bond_scores(&j);
            assert_eq!(scores[strong_idx], 5.0);
            for (k, &s) in scores.iter().enumerate() {
                if k != strong_idx {
                    assert_eq!(s, 1.0, "uniform-region bond {k} score = {s} ≠ 1.0");
                }
            }

            let total_budget = n_bonds * 8;
            let chi = chi_allocation_target_budget(&scores, total_budget, 2, 16);

            // The strong bond must get strictly more χ than every
            // bond in the uniform region.
            let typical_chi = chi[2];
            assert!(
                chi[strong_idx] > typical_chi,
                "strong bond {strong_idx} got χ = {} ≤ typical χ = {typical_chi}; \
                 Griffiths score did not redirect to the strong bond",
                chi[strong_idx]
            );

            // Budget exactly consumed.
            let consumed: usize = chi.iter().sum();
            assert_eq!(consumed, total_budget, "budget not exactly consumed");
        }
    }

    /// A Griffiths regime with one anomalously *weak* bond is the
    /// dual case: the production allocator should *save* χ at that
    /// bond (because no singlet forms across it and bipartite entropy
    /// stays low), not waste budget on it. This contradicts the
    /// common "weak bonds = bottleneck, give them more χ" intuition
    /// (which is true for transport observables, not for bipartite
    /// entanglement); the test pins the correct TEBD-entropy
    /// behaviour explicitly so the intuition mistake is caught at
    /// CI time.
    #[test]
    fn griffiths_score_saves_chi_at_weak_bonds() {
        use crate::allocator::chi_allocation_target_budget;

        let n_bonds = 21;
        let weak_idx = 10;
        let mut j = vec![1.0; n_bonds];
        j[weak_idx] = 0.01; // 100× weaker — the original "rare weak bond" case

        let scores = xxz_griffiths_bond_scores(&j);
        assert_eq!(scores[weak_idx], 0.01);

        let total_budget = n_bonds * 8;
        let chi = chi_allocation_target_budget(&scores, total_budget, 2, 16);

        // The weak bond should get strictly fewer χ than typical:
        // it's the place where bipartite entropy stays smallest under
        // TEBD evolution from a product state, so allocating extra χ
        // there is wasted budget.
        let typical_chi = chi[2];
        assert!(
            chi[weak_idx] < typical_chi,
            "weak bond {weak_idx} got χ = {} ≥ typical χ = {typical_chi}; \
             Griffiths score wrongly inflated χ at the weak bond",
            chi[weak_idx]
        );

        // Budget exactly consumed.
        let consumed: usize = chi.iter().sum();
        assert_eq!(consumed, total_budget, "budget not exactly consumed");
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
