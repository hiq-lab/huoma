# VQ-136 / Track G — Verdict

> **Status: closed as position statement, 2026-05-30.**
> Subset of the originally-planned shootout was run. The remaining
> three-way empirical shootout (G.3 in the ticket) is deferred as
> future work; the conceptual question Track G existed to answer is
> already settled by first principles plus the G.1/G.2 work below.

**Ticket:** valiant-ops VQ-136 · **Author:** Daniel Hinderink
**Hardware:** local MacBook · **Date:** 2026-05-30

---

## Headline

Track G existed to close the one open question Phase 7 left behind:
*for which class of 1D models does sin(C/2) provably beat uniform-χ
at matched budget?* Phase 7 ruled this out for clean and
weakly-disordered KIM (Dalzell–Brandão puts uniform-χ structurally
near-optimal there). The natural next candidate, surfaced by
Aramthottil et al. (PRL 133, 196302, 2024), was bond-disordered XXZ
in the Griffiths regime.

The answer, after building the XXZ engine (G.1), the ITensor
cross-reference (G.2), and the score-design pass, is:

> **sin(C/2) is not the right tool for bond-disordered XXZ.** It
> was derived from KAM resonance physics for *driven* systems
> (KIM, QKR); bond-disordered XXZ has no drive and no KAM torus
> structure. The right tool is a different per-bond score on the
> same water-filling primitive.

That answer reframes the underlying Track A question. The allocator
is best understood as three layers:

```
Layer 1: water-filling primitive (universal)
         huoma::chi_allocation_target_budget
            ↑ takes any non-negative per-bond score vector
            ↑ greedy water-filling, exact-budget consumption
            ↑ score-agnostic by design

Layer 2: per-bond score (regime-specific)

Layer 3: physical-regime feeder
         Driven systems        : sin(C/2) on per-site drive
                                 frequencies (Phase 7 production
                                 path, KIM)
         Static-coupling
         disorder (Griffiths)  : |J_i| on bond couplings (this
                                 report, XXZ)
```

Layer 1 carries through unchanged. Layer 2 is what swaps when the
physical regime swaps. This is what Track G found.

---

## What was built, with anchors

### G.1 — XXZ gate set and dense reference (`src/xxz.rs`)

- `XxzParams { delta, dt }`
- `xxz_bond_gate(j_dt, delta) → [[C; 4]; 4]` — explicit 4×4 unitary
  `exp(-i (J·dt/4) (σx σx + σy σy + Δ σz σz))`, block-diagonal in
  the (|00⟩, |11⟩) and (|01⟩, |10⟩) subspaces
- `apply_xxz_step(mps, j_per_bond, params, chi_per_bond)` — first-order
  Trotter sweep, sequential bonds 0..n−2
- `sample_bond_disorder_log_uniform(n_bonds, j_min, j_max, seed)` —
  reproducible bond-disorder sampling via self-contained splitmix64
- `reference_xxz_run(...)` — independent dense statevector reference
- `product_state_mps(n, initial)` — MSB-convention product state

**Validation:** `apply_xxz_step_matches_dense_lossless_at_n10` —
50 Trotter steps, N=10 Heisenberg (Δ=1), bond-disorder J ∈ [0.5, 2.0]
with fixed seed, Néel initial, lossless χ=32, MPS reproduces dense
reference to **max |⟨Z_i⟩| error ≤ 1e-12** every step. **This is the
load-bearing anchor.** Anything Track G claims sits on top of it.

### G.2 — ITensor cross-reference (`external/itensor_ref/xxz_griffiths.jl`)

- Julia 1.12.6 + ITensors v0.9.30 + ITensorMPS v0.4.1 (committed
  `Project.toml` + `Manifest.toml`)
- Reads a JSON manifest written by the Rust side
  (`ItensorXxzManifest` in `src/xxz.rs`, round-trip-tested)
- First-order Trotter XXZ via `apply()` with combined `maxdim` +
  `cutoff`; matches Huoma's bond order and gate convention
- Outputs a sibling JSON with per-step `⟨σz⟩` history (×2 from
  ITensor's S=1/2 convention to Huoma's σz=±1) and per-step linkdims

**Cross-check anchor** (`results/VQ-136/g2_smoke/`): lossless N=10
Heisenberg uniform-J, 10 Trotter steps, Néel initial. Element-wise
max |Huoma − ITensor| = **1.8 × 10⁻¹⁰** on final ⟨σz⟩. Two
structurally independent TEBD implementations agree to 10 decimal
digits — the ITensor reference is trustworthy as the second leg of
any future shootout.

### G.2.5 — Score design (the load-bearing analytical step)

Two feeders for the water-filling primitive, with explicit
delineation of which physical regime each applies to:

**`xxz_griffiths_bond_scores(j_per_bond) → Vec<f64>`** — the
production allocator for bond-disorder XXZ. Returns `|J_i|` per
bond. Fed into `chi_allocation_target_budget`.

Derivation: in finite-time TEBD from a product state, bipartite
entanglement entropy across bond i grows as

```
S_i(t) ≈ min(log 2, |J_i| · t)
```

— linearly until saturation at the single-singlet bound. MPS bond
dimension required at bond i scales like `χ_i ~ exp(S_i)`, monotone
increasing in `|J_i|`. Strong bonds form singlets fastest and carry
the highest local entanglement; weak bonds carry suppressed
entanglement because correlations cannot pass through them.
Dasgupta–Ma real-space RG (1980, Fisher 1994) confirms: the
strongest bond is eliminated first by singlet formation. **Strong
bonds get more χ; weak bonds get less.**

This contradicts the common "Griffiths = weak bonds are
bottlenecks, give them more χ" reading. That intuition is correct
for *transport observables* but inverted for *bipartite
entanglement*. Pinned by
`griffiths_score_saves_chi_at_weak_bonds`.

**`xxz_site_frequencies(j_per_bond) → Vec<f64>`** — sin(C/2)
**negative control**, not for production. Returns the geometric
mean ω_i = √(|J_{i−1}| · |J_i|) of adjacent bond couplings (the
quantity that flows under Dasgupta–Ma RG). Intended to be fed into
`chi_allocation_sinc` for the explicit purpose of exhibiting
sin(C/2)'s failure mode on static-disorder physics.

Failure mode is concrete:
`sinc2_on_bond_disorder_is_fragile_to_integer_ratios` —
J_weak = 0.01 = 1/100 produces an exact integer ω-ratio across the
chain; sin(C/2) reports "perfect commensurability" and delivers a
uniform-χ allocation, withholding χ from the rare-region bonds.
This is **statistical coincidence in the disorder values**, not a
real KAM resonance: the physics has no drive, no torus, nothing
to be commensurate *with*. sin(C/2) is brittle to integer
coincidences in coupling-disorder distributions.

---

## Verdict against the Track G ticket's three outcomes

The VQ-136 ticket listed three possible G.4 verdicts:

| Outcome | Settled? |
|---|---|
| (a) sin(C/2) wins measurably in the Griffiths regime → methods note | **No.** sin(C/2) does not transfer to static-disorder physics; the question is mis-framed. |
| (b) sin(C/2) ties uniform → close Track A for good | **No.** sin(C/2) does not even tie cleanly — it is brittle to integer coincidences in disorder values. |
| (c) sin(C/2) fails because frequency-channel structure does not map onto bond disorder → propose a sin(C/2) variant that consumes bond-disorder directly | **Yes — and this is the verdict.** The "variant" turned out to be a *different score family entirely* (`|J_i|`), motivated by RG + TEBD entropy rather than KAM resonance, sharing only the water-filling primitive with sin(C/2). |

Track G closes on (c).

---

## What this means for Track A

Track A was closed by Phase 7 with the production allocator
`huoma::chi_allocation_sinc`. The open question Phase 7 left was
whether sin(C/2) had headroom beyond clean KIM. The answer is
**no** in the form the question was asked, and **yes for the
underlying architecture** when re-framed:

- sin(C/2) is the right *score* for driven systems with per-site
  frequency channels (KIM, QKR). That use case is unchanged and
  production.
- The water-filling *architecture* is universal across any 1D model
  for which one can construct a physically motivated per-bond
  score.
- Static-coupling-disorder needs a different score: `|J_i|` for
  XXZ-class Hamiltonians, derivable from RG + TEBD entropy.

The strictly stronger Track A statement is therefore not "sin(C/2)
is universal" — it is **"per-bond water-filling on a
regime-specific score is universal, and Huoma supplies the
infrastructure for both layers."**

This is the position Track A now rests on.

---

## What was *not* done — G.3 deferred

The originally-planned G.3 three-way (now four-way, with the
negative control) shootout was not run:

- N ∈ {32, 64, 128} × 5 disorder strengths × 10 realisations ×
  {uniform, `|J_i|`, sin(C/2) on `ω_i`, ITensor-ε}
- Metrics: max |⟨Sz_i⟩| error vs lossless reference, max bipartite
  entropy error, wall time

**Why deferred:** the conceptual finding Track G existed to produce
is already settled. G.3 would convert that finding into a measured
table — necessary for a publication, but not necessary for the
architectural verdict Track G owes the ROADMAP. Spending 3–5 days
on the shootout instead of starting Track F (Complex pivot →
hyperbolic layouts → Lanthanide / Riemann paths) trades strategic
progress for empirical fidelity that no immediate caller needs.

**When G.3 should be revisited:** when a methods-note submission to
Phys. Rev. B (or similar) is concretely on the calendar, or when a
customer/collaboration specifically needs the magnitude numbers.
The G.1 + G.2 infrastructure is the heavy lift; G.3 from here is
~3–5 days of harness + runs + analysis on top.

**What G.3 could change about this verdict:**

1. If `|J_i|` only marginally beats uniform (5–15 %) — the verdict
   stands but is "theoretically motivated, empirically modest."
2. If ITensor ε-truncation wipes both static allocators (uniform
   and `|J_i|`) — the verdict shifts: Huoma should adopt
   ε-truncation as the production default for bond-disorder
   workloads, and the static per-bond χ-precompute is the wrong
   abstraction. This would be the strongest possible negative
   finding for the entire sin(C/2)-water-filling thesis on
   bond-disorder. *It is the outcome that would most justify
   running G.3.*

These are honest risks, recorded here so the future caller (us or
someone else) does not have to re-derive them.

---

## Anchors and reproducibility

| Anchor | Where | Tolerance |
|---|---|---|
| XXZ MPS vs dense at N=10, lossless χ, 50 steps | `tests::apply_xxz_step_matches_dense_lossless_at_n10` | max ⟨Z⟩ err ≤ 1e-12 |
| Huoma vs ITensor at N=10 lossless, uniform J, Heisenberg | `results/VQ-136/g2_smoke/` | max abs diff ≈ 1.8e-10 |
| ItensorXxzManifest JSON round-trip + field-name pinning | `tests::itensor_manifest_round_trips` | bit-equal |
| `xxz_griffiths_bond_scores` redirects to strong bonds | `tests::griffiths_score_redirects_chi_to_strong_bonds` | chi[strong] > typical |
| `xxz_griffiths_bond_scores` saves at weak bonds | `tests::griffiths_score_saves_chi_at_weak_bonds` | chi[weak] < typical |
| sin(C/2) fragility on integer-ratio disorder | `tests::sinc2_on_bond_disorder_is_fragile_to_integer_ratios` | uniform allocation despite J_weak = 0.01 |

All ten `xxz::tests::*` green on `cargo test --release --lib xxz::`,
2026-05-30.

---

## References

- Dasgupta & Ma, *Phys. Rev. B* **22**, 1305 (1980) — strong-disorder RG
- D. S. Fisher, *Phys. Rev. B* **50**, 3799 (1994) — random AFM Heisenberg
- Refael & Moore, *Phys. Rev. Lett.* **93**, 260602 (2004) — disorder
  & entanglement
- Calabrese & Cardy, *J. Stat. Mech.* P04010 (2005) — TEBD entropy
  growth bounds
- Aramthottil, Bardarson, Pollmann, *PRL* **133**, 196302 (2024) — XXZ
  Griffiths regime
- Dalzell & Brandão, *Quantum* **3**, 187 (2019) — uniform-χ near-optimality
  for clean 1D systems
- Hinderink 2026, "The Tilde Pattern" — sin(C/2) derivation, scope, and
  driven-system applicability
- ROADMAP.md (this repo), Track A history (Phase 7),
  `docs/history/PHASE7_REPORT.md`
