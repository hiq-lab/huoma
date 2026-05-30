# Track F — Non-Euclidean Topologies and Magnetic Hamiltonians

**Status**: Design doc, not yet committed to the roadmap. This is the
strategic-options paper for the next architecture extension after Track D
(heavy-hex / IQM topology mappings) lands.

Track D taught us that Huoma's TTN architecture is genuinely
topology-agnostic — heavy-hex was a special case of "tree-decomposable
coupling graph with adaptive per-edge χ via sin(C/2)". Track F is the
question: *what's the next worthwhile generalisation?*

The answer this doc argues for: **complex-valued tensors + magnetic
Hamiltonians on arbitrary tilings, including non-Euclidean ones**. This
unlocks a real commercial-chemistry application (heavy-atom/lanthanide
magnetic Hamiltonians) and, as a side-effect, makes Huoma the only
TN simulator that scales to million-vertex hyperbolic lattices — a
fast-growing experimental physics target.

## Why this is the right next investment

Track D made Huoma a **topology engine**. Today the engine is restricted
in two ways that we did not previously recognise as restrictions:

1. **All tensors are `f64`.** Real-valued. This is fine for kicked-Ising
   benchmarks but fails the moment any Hamiltonian carries a phase —
   magnetic fields (Peierls substitution), spin-orbit coupling, gauge
   theories, or any complex-energy non-Hermitian regularisation.
2. **All shipped layouts are Euclidean.** Heavy-hex, IQM grids,
   1D chains. The TTN abstraction does not require this — the data
   structure works on any tree-decomposable graph. We just haven't
   exercised it.

Both restrictions are removable with focused engineering. Together they
unlock a class of Hamiltonians that the field is actively interested in
*right now*:

- **Lanthanide and actinide chemistry** (5f electrons + spin-orbit + B-field)
  — single-molecule magnets, MRI contrast agents (Gd-DOTA next-gen),
  Lu-177 / Bi-213 / Ac-225 radionuclide therapeutics, photocatalysis
- **Circuit-QED hyperbolic lattices** (Houck/Princeton, Schuster/Chicago,
  Roushan/Google) — superconducting transmons in {p,q}-tilings, no
  existing TN simulator scales past ~10K sites
- **Holographic codes / AdS-CFT condensed matter** — Pastawski-Yoshida-
  Harlow-Preskill 2015 codes are literally hyperbolic TNs by construction
- **Hyperbolic surface codes for QEC** — better rate-vs-distance than
  planar surface codes, currently being prototyped on superconducting QPUs

The commercial story sits in the first bullet. The other three are
reputational/academic payoff that comes "for free" once the engine is
generalised.

## North star for Track F

> A million-vertex magnetic-Hamiltonian simulator on arbitrary
> tree-decomposable tilings, Euclidean or hyperbolic, with
> sin(C/2)-driven adaptive χ per edge, validated against published
> experimental benchmarks in lanthanide spin chemistry and circuit-QED
> hyperbolic lattices, at floating-point precision against independent
> dense references in the small-N limit.

The "tilings, Euclidean or hyperbolic" part is the architectural
generalisation. The "magnetic Hamiltonian" part is the new physics
content. The "lanthanide + circuit-QED" part is what gives Track F
external validation independent of Riemann-Hypothesis-style speculation.

## Phases

### F.1 — Complex-valued tensor pivot ✅ closed via audit (VQ-139, 2026-05-30)

**Original framing (stale, kept for history):** Huoma's tensors were
assumed `f64`, requiring a ~2-week mechanical pivot to a generic
`Scalar` trait over `f64 ⊕ Complex<f64>`.

**Actual state on audit:** Huoma is already `Complex64` throughout —
`src/mps.rs`, `src/ttn/mod.rs`, `src/ttn/contraction.rs`,
`src/ttn/gauge.rs`, `src/kicked_ising.rs`, `src/xxz.rs`. faer SVD/QR
paths go through `faer::c64`. Hermitian inner products use `.conj()`
correctly in 22 places. The "all tensors f64" claim went stale around
Phase 6 or 7. KIM and XXZ already produce genuinely complex MPS states
with dense-validated anchors. There is no real-only fast path to
maintain, no `Scalar` trait to introduce, no mechanical pivot to
execute.

**What was missing and is now added (1-day audit instead of 2 weeks):**
a smoke-test that drives a gate whose off-diagonal entries have *both*
non-trivial real and imaginary parts (KIM RZ, ZZ, and XXZ bond gates
all reduce to structured `cos(θ) + i sin(θ)` patterns that a real-only
path could in principle fake). The Peierls hopping gate at
`φ ∉ {0, π/2}` exercises this — off-diagonals are
`±i e^{±iφ} sin(t·dt)` with both real and imaginary parts non-zero
for generic `φ`.

**Audit deliverable (committed as part of VQ-139):**

- `src/peierls.rs` — `peierls_hopping_gate(t, dt, phi)`,
  `apply_peierls_step`, `reference_peierls_run`, `product_state_mps`
- `peierls::tests::peierls_step_matches_dense_lossless_at_n6` —
  N=6 chain, single particle initially at qubit 2, 20 first-order
  Trotter steps with `t = 1, dt = 0.1, φ = π/3` per bond, lossless
  χ = 8. MPS reproduces dense to `max |⟨Z_i⟩| err ≤ 1e-13` every
  step. **This is the F.1 audit anchor.**
- Three supporting unit tests:
  `peierls_gate_is_unitary` (5 (t,dt,φ) cases including the
  edge-case angles 0 and π/2), `phi_zero_reduces_to_standard_xy_hopping`
  (pins the standard-XY limit), and
  `phi_pi_third_has_genuinely_complex_off_diagonals` (pins the
  non-trivial complex structure + unitarity relationship
  `U[2,1] = -conj(U[1,2])`).

Peierls hopping is also the foundational gate for F.3 (Hofstadter /
magnetic Hamiltonians), so the audit doubles as the F.3 entry point.

### F.2 — Hyperbolic layout generator

**Goal**: `HyperbolicLayout::tiling(p, q, radius)` produces a
tree-decomposable tiling of a hyperbolic surface, analogous to
`HeavyHexLayout::ibm_eagle_127()`.

**Tasks**:
- Implement the Fuchsian-group word-generator for {p,q} tilings (Poincaré
  disc model). Standard algorithm: breadth-first growth from a central
  cell, applying generators of the fundamental group at each step.
- Cayley-graph structure: assign each vertex a unique word in the
  generators; edges correspond to single-generator transitions.
- Tree-decomposition: hyperbolic tilings with p ≥ 5 have exponential
  growth, so a spanning tree is trivially constructed by retaining only
  outward-going edges. The "non-tree" edges become "boundary identifications"
  — handled by the same swap-network machinery Track D built for
  heavy-hex.
- Tilings to support initially: {3,7}, {4,5}, {5,4}, {7,3}. The first
  three are what circuit-QED experiments use.

**Cost**: ~3-4 weeks. The Fuchsian-group word generator is the only
genuinely new mathematical code; the rest re-uses Track D infrastructure.

**Acceptance**: visualisation script `examples/hyperbolic_tiling.rs`
renders the {7,3} tiling out to radius 8 (vertices ≈ 3500), and
`partition::partition_adaptive` reports a sensible stable/volatile split
under uniform random frequencies.

### F.3 — Magnetic hopping Hamiltonian (Peierls substitution)

**Goal**: an edge operator that implements `exp(-i H Δt)` for the
magnetic Laplacian `H_B = -t Σ_⟨i,j⟩ exp(i φ_ij) c†_i c_j + h.c.`,
where `φ_ij` are Peierls phases derived from a chosen gauge of the
magnetic vector potential.

**Tasks**:
- `MagneticEdgeOp { hopping: f64, phase: f64 }` — applies a 2×2
  unitary to the bond Hilbert space
- Gauge fixing: Landau gauge on Euclidean lattices, hyperbolic Landau
  gauge (constant B perpendicular to the surface) on hyperbolic tilings
- Validation: small-cell magnetic Aharonov-Bohm flux quantisation
  reproduces the analytic spectrum (Hofstadter butterfly on square
  lattice, hyperbolic Hofstadter on {7,3} — both have known analytic
  references in the condensed-matter literature)

**Cost**: ~2-3 weeks. Most of the work is gauge bookkeeping; the
unitary application itself is straightforward.

**Acceptance**: square-lattice Hofstadter butterfly reproduced for
flux ratios p/q with q ≤ 13 at N = 16 sites, agreeing with dense
diagonalisation to 1e-12.

### F.4 — Spin-orbit coupling extension

**Goal**: per-site spin degrees of freedom coupled to local magnetic
moments and to neighbouring sites via a Pauli-matrix-valued hopping.
This is the lift from "lattice model with magnetic flux" to "5f-electron
chemistry on irregular ligand sphere".

**Tasks**:
- `Site::with_spin_dim(d)` — local Hilbert space dimension 2 (s),
  4 (s+p), or higher (5f manifold up to dimension 14)
- Hopping matrix becomes `d × d` instead of scalar
- Coupling to external B-field via Zeeman term
- Spin-orbit `λ L·S` as on-site operator

**Cost**: ~6-8 weeks. This is where research uncertainty enters. Spin-
orbit coupling in tensor-network frameworks is 2024-2025 state-of-the-art.
We may discover that the tree-decomposition assumption breaks for
strongly entangled spin-orbital states, in which case χ blows up and
the Track F commercial story degrades.

**Acceptance**: a Gd³⁺ in an octahedral ligand field (the simplest
realistic lanthanide case) at modest active space (J = 7/2 manifold)
reproduces the analytic Zeeman splitting under a uniform B-field to
within 0.1 mEV, at compute cost no worse than 10× the dense reference
for N ≤ 8 ligands.

### F.5 — Application benchmark: Lanthanide spin chemistry

**Goal**: a published-quality benchmark on a real lanthanide system,
something like Gd-DOTA or Lu-DOTA-TATE (the latter is the Lu-177
radionuclide-therapy ligand-conjugate molecule actively used in clinic).

**Tasks**:
- Choose a benchmark system with experimental EPR / NMR shielding /
  Zeeman-splitting data in the literature
- Construct the ligand-sphere topology (typically 8-coordinate octahedral
  for Ln³⁺) — small enough to validate, real enough to be publishable
- Build the magnetic + spin-orbit Hamiltonian
- Compute the eigenvalue spectrum on Huoma vs the dense reference vs
  the experimental value
- Sweep external B-field and reproduce the EPR / NMR signature

**Cost**: ~4-6 weeks. The science here is interesting but not novel —
others have done this with MOLCAS / OpenMolcas. The Huoma contribution
is *that we can do it with sin(C/2)-allocated χ on irregular topologies*,
which is the differentiator.

**Acceptance**: Huoma reproduces a published experimental Zeeman splitting
or EPR g-tensor to within 5 % accuracy on a 5f-electron lanthanide
complex, with a methodology paper draft ready.

### F.6 — Application benchmark: Circuit-QED hyperbolic lattice

**Goal**: reproduce a published experimental observation of single-particle
dynamics on a hyperbolic circuit-QED lattice, e.g. Kollár et al. (Nature
2019, 571, 45) or Lenggenhager et al. (Nat. Commun. 2022).

**Tasks**:
- Identify a clean experimental dataset with simulator-comparable
  observables (Wannier-Stark localisation, magnetic-flux-tuned spectra,
  edge-mode transport)
- Build the equivalent magnetic Hamiltonian on the matching {p,q} tiling
- Validate Huoma's prediction against the experimental data

**Cost**: ~3-4 weeks. Parallelisable with F.5 — different team if we
ever have one, otherwise sequential.

**Acceptance**: Huoma's simulated observable matches the experimental
result to within experimental error bars.

### F.7 — Optional: Selberg-spectrum reputational benchmark

**Goal**: compute the first 100 eigenvalues of the Laplacian on a
modular surface (e.g. PSL(2,ℤ)\\ℍ, fundamental domain truncated and
discretised on a {3,7} tiling). Compare to published Selberg-spectrum
tables.

**Why it's optional**: zero commercial relevance. Pure mathematical /
reputational. Useful as the methodological bridge if we ever choose to
chase Riemann-Hypothesis-adjacent work, but should not gate F.1–F.6.

**Cost**: ~2-3 weeks if F.4 lands; the Laplacian is a degenerate case
of the magnetic Laplacian (zero B-field, scalar spin).

## Hard decision points

| Phase | If we hit this, kill the track | If we hit this, continue |
|---|---|---|
| F.1 | ~~Complex pivot breaks more than 5 of the 178 existing tests irreversibly~~ ✅ Closed via audit 2026-05-30: tensors were already Complex64, no pivot needed. Peierls smoke-test at φ = π/3, N = 6, lossless χ = 8, 20 Trotter steps reproduces dense to ≤ 1e-13. | n/a — anchor pinned in `peierls::tests::peierls_step_matches_dense_lossless_at_n6` |
| F.2 | Fuchsian-group generator produces topologically broken tilings (wrong vertex degrees, missing edges) for {7,3} at radius ≥ 5 | Visualisation matches known {7,3}-tiling figures from the literature |
| F.3 | Hofstadter butterfly disagrees with dense reference at p/q = 1/2 (the most-stress-tested case) | Reproduces to 1e-12 |
| F.4 | χ required for a Gd³⁺ ligand sphere grows faster than exponentially with ligand-sphere size | χ scales sub-exponentially, ideally polynomially |
| F.5 | No useful agreement with experimental EPR data on any lanthanide we try | Even one clean published-data match closes the application story |
| F.6 | Circuit-QED experiments require open-system / dissipation modelling that we explicitly ruled out | Coherent-dynamics-only experiments are sufficient |

## Strategic positioning

Track F is **not** a near-term priority. It is the right *next* track
after the current commitments (Atlas v1 publication, IQM Award, Cryo-PUF
verwertung) are landed. Realistic earliest start: Q4/2026 or Q1/2027.

Three reasons to plan it now even if we don't start it now:

1. **Architectural decisions in Track D land in production this quarter.**
   We should know what generalisations are realistic before we commit to
   API stability that would later constrain F.
2. **The lanthanide-pharma application** is not on Valiant's current radar
   but the Theranostics market (Lu-177, Ac-225) is growing rapidly and
   will be a real commercial conversation by 2027–2028. Having a
   working magnetic-Hamiltonian engine ready by then is a multi-year
   lead-time investment, not a one-quarter pivot.
3. **The IQM Resonance hardware envelope** is the binding constraint on
   the Atlas Track. Track F gives Huoma a path to scientific relevance
   *independent* of how QPU hardware scales over the next 24 months —
   even if Resonance plateaus at Sirius-class fidelity, Track F's
   classical hyperbolic-lattice and lanthanide-chemistry niches remain
   computationally interesting on their own.

## Cost summary

| Phase | Weeks | Risk | Notes |
|---|---|---|---|
| ~~F.1~~ | ~~2~~ | ~~low~~ | ✅ Closed via 1-day audit 2026-05-30 (VQ-139). Tensors were already Complex64; no pivot needed. |
| F.2 | 3–4 | low | New maths code, well-understood |
| F.3 | 2–3 | medium | Gauge bookkeeping non-trivial |
| F.4 | 6–8 | **high** | Where research uncertainty enters |
| F.5 | 4–6 | medium | Application + paper |
| F.6 | 3–4 | medium | Parallel with F.5 |
| F.7 | 2–3 | low | Optional, pure science |
| **Total F.1–F.5 (baseline)** | **17–23 weeks** | — | ~5 months focused work |
| **Total F.1–F.7 (full track)** | **22–30 weeks** | — | ~6–7 months focused work |

## Track F is not Track D-bis

Track D added topology generalisation but stayed inside real-valued
Euclidean lattices. Track F adds two orthogonal axes — complex tensors
and non-Euclidean geometry — *and* it carries the first scientifically
novel content beyond benchmark-reproduction. The risk profile is higher
than D's; the payoff profile is also higher, because the application
domain (lanthanide chemistry under magnetic fields) is one where almost
nobody else has scaled the tensor-network approach.

If Huoma is to be more than "the heavy-hex simulator with sin(C/2)
allocator", Track F is the path.
