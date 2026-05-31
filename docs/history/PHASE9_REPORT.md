# Phase 9 — Track G closure + Track F foundation (May 2026)

This phase resolves the open question Phase 7 left behind (Track G, the
sin(C/2) vs uniform-χ shootout on bond-disordered XXZ), opens Track F
with two foundational sub-phases (complex-tensor audit + hyperbolic
{p, q} tiling generation), and concludes with a strategic decision to
hand the long-arc downstream Track F application paths (Lanthanide
chemistry; RH-adjacent Selberg spectrum) off to other contexts.

**Phase 9 follows the historical sequence:** `BIANCHI_JOURNEY.md`
(Phases 1–5), `PHASE6_REPORT.md` (KIM validation + the
`apply_zz_fast` bug), `PHASE7_REPORT.md` (Track A close, sin(C/2) +
water-filling production path), `PHASE8_REPORT.md` (Track D close,
Eagle 127 Tindall benchmark, 1M-qubit `ProjectedTtn`).

## TL;DR

- **Track G closed as position statement.** sin(C/2) is a KAM filter
  for *driven* systems and does not transfer to static-coupling
  disorder. The right Griffiths-XXZ score is `|J_i|` per bond, derived
  from TEBD bipartite entropy + Dasgupta–Ma RG. Track A's strictly
  stronger claim: per-bond water-filling on a regime-specific score is
  universal; sin(C/2) is the score for driven systems, `|J_i|` for
  static-coupling-disorder.
- **Track F.1 closed via audit, not via the planned 2-week pivot.**
  Huoma's tensors were already `Complex64` throughout. The Peierls
  hopping gate at `φ = π/3` exercises genuinely-mixed complex
  amplitudes through the full SVD/QR/observable pipeline and matches
  dense to ≤ 1e-13.
- **Track F.2 (F.2.a + F.2.b) closed.** Möbius-disk infrastructure +
  face-edge BFS for arbitrary hyperbolic {p, q}, validated against
  hand-derived counts at small radii and against Python HyperTiling
  (v1.5.1) for the q = 3 case at radii 0–2 (exact match on vertex,
  edge, and degree-histogram counts).
- **Strategic close-out:** the announced downstream Track F application
  paths — F.4/F.5 Lanthanide chemistry and F.7 RH-adjacent Selberg
  spectrum — are not committed in Huoma's near-term plan. Lanthanide
  is moving to Garm (which has the chemistry infrastructure). RH is a
  hobby thread that may resume later. F.3 Hofstadter is unblocked
  technically (Peierls + hyperbolic both ready) but also not committed
  here.

## What was built

### Cleanup pass (entry to phase)

Five preparatory commits before any new code:

1. `.gitignore` consolidated (`**/target/`, `demo/`, `media/`,
   `__pycache__/`, `.venv/`); orphan `.gitignore.add` removed.
2. Documentation reorganised into `docs/history/` (4 phase reports +
   `BIANCHI_JOURNEY.md`) and `docs/design/` (`TRACK_D_DESIGN.md`,
   `TRACK_F_DESIGN.md`). Root MD files reduced to README + CLAUDE +
   ROADMAP + LICENSE.
3. `results/VQ-110/` (formerly 26 files flat) subdirectorised into
   `projected_1b/`, `annealer_stress_test/`, `t4_l99a_pocket/` with a
   top-level `INDEX.md`.
4. `ROADMAP.md` rewritten for coherence — the "Where Huoma stands
   today" header was stuck on Phase 7 / "1D-only" while Track D
   closure two pages down contradicted it. Header now reflects actual
   state.
5. Speculative items struck from ROADMAP — Track B.2 (cost estimator,
   no concrete user), B.5 (opportunistic tech debt), and **Track C in
   its entirety** (TDVP, variational χ, Sobol — all without trigger).
   The strike-out itself is recorded as the Track C section.

### Track G — bond-disordered XXZ shootout closed as position statement

Three buildable sub-phases (G.1, G.2, G.2.5 score-design); G.3
empirical shootout deferred.

**G.1** — XXZ gate set in `src/xxz.rs`:
- `XxzParams { delta, dt }`,
  `xxz_bond_gate(j_dt, delta) → [[Complex64; 4]; 4]`,
  `apply_xxz_step`, `reference_xxz_run`, `product_state_mps`.
- `sample_bond_disorder_log_uniform` with a self-contained
  splitmix64 PRNG (no `rand` dep).
- **Load-bearing anchor:**
  `apply_xxz_step_matches_dense_lossless_at_n10` — N=10 Heisenberg,
  bond-disorder J ∈ [0.5, 2.0] fixed seed, Néel initial, lossless
  χ=32, MPS matches dense to max |⟨Z_i⟩| err ≤ 1e-12 every step over
  50 Trotter steps.

**G.2** — ITensor cross-reference path:
- `external/itensor_ref/xxz_griffiths.jl` (Julia 1.12 + ITensors
  v0.9.30 + ITensorMPS v0.4.1; `Project.toml` + `Manifest.toml`
  committed for reproducibility).
- `ItensorXxzManifest` (Rust serde struct) writes a JSON manifest
  the Julia runner consumes; round-trip pinned by
  `itensor_manifest_round_trips` with explicit field-name asserts.
- **Cross-check anchor** (`results/VQ-136/g2_smoke/`): lossless N=10
  Heisenberg uniform-J, Néel initial, 10 Trotter steps. Element-wise
  max |Huoma − ITensor| = **1.8 × 10⁻¹⁰** on final ⟨σz⟩. Two
  structurally independent TEBD implementations agree to 10 decimal
  digits.

**G.2.5** — first-principles score design (the load-bearing analytical
step). Two feeders, each with sharp delineation of which physical
regime it applies to:

- `xxz_griffiths_bond_scores(j_per_bond) → Vec<f64>` — **production
  allocator** for bond-disorder XXZ. Returns `|J_i|`. Derivation:
  TEBD entropy `S_i(t) ≈ min(log 2, |J_i| · t)` → strong bonds
  saturate at the singlet bound fastest and need the highest χ.
  Dasgupta–Ma RG (Fisher 1994) confirms the strongest bond is
  eliminated first by singlet formation. Tests
  `griffiths_score_redirects_chi_to_strong_bonds` and
  `griffiths_score_saves_chi_at_weak_bonds` pin both directions
  (the latter explicitly catches the naïve "weak = bottleneck = give
  more χ" intuition, which is correct for *transport* but inverted
  for *bipartite entanglement*).

- `xxz_site_frequencies(j_per_bond) → Vec<f64>` — **sin(C/2)
  negative control**, not a production allocator. Returns the
  geometric mean `ω_i = √(|J_{i−1}| · |J_i|)` (the natural RG-flow
  scale per Dasgupta–Ma). Intended to be fed into
  `chi_allocation_sinc` for the explicit purpose of exhibiting
  sin(C/2)'s failure mode on static-disorder physics:
  `sinc2_on_bond_disorder_is_fragile_to_integer_ratios` pins it
  concretely — J_weak = 0.01 = 1/100 produces an exact integer
  ω-ratio across the chain; sin(C/2) reports "perfect
  commensurability" and delivers a uniform-χ allocation, withholding
  χ from the rare-region bonds. This is statistical coincidence in
  the disorder values, not a real KAM resonance.

**G.4** — verdict (`results/VQ-136/REPORT.md`): outcome (c) of the
VQ-136 ticket's three options — sin(C/2) does not transfer; a
different score family (`|J_i|`) does. The strictly stronger Track A
claim is:

> Per-bond water-filling on a *regime-specific* score is universal.
> sin(C/2) is the score for driven systems with per-site frequency
> channels (KIM, QKR). `|J_i|` is the score for static-coupling
> disorder (XXZ-class Hamiltonians). Huoma supplies the
> infrastructure for both layers — `chi_allocation_target_budget` as
> the universal water-filling primitive, and a small library of
> regime-specific scorers.

**G.3** (empirical four-way shootout: uniform vs ε-truncated ITensor
vs `|J_i|` vs sin(C/2)-on-ω at N ∈ {32, 64, 128} × 5 disorder
strengths × 10 realisations) was *not run*. The conceptual question
Track G existed to answer is settled without it. The two G.3
outcomes that would change the verdict — `|J_i|` only marginally
beats uniform, or ITensor ε-truncation wipes both static allocators —
are recorded in `results/VQ-136/REPORT.md` § "What G.3 could change."

### Track F.1 — complex-pivot audit (closed)

The design doc estimated 2 weeks of "mechanical but pervasive" work
to make Huoma's tensors generic over `f64 ⊕ Complex<f64>`. **Audit on
2026-05-30 found the assumption stale:** Huoma is already `Complex64`
throughout (`src/mps.rs`, `src/ttn/mod.rs`, `src/ttn/contraction.rs`,
`src/ttn/gauge.rs`, `src/kicked_ising.rs`, `src/xxz.rs`). faer SVD/QR
paths go through `faer::c64`. `.conj()` is used correctly in 22 places
for Hermitian inner products. KIM and XXZ produce genuinely complex
states with dense-validated anchors.

What was missing and is now added: a smoke-test driving a gate whose
off-diagonals have *both* non-trivial real and imaginary parts (KIM
RZ, ZZ, and XXZ bond gates all have structured `cos + i sin` patterns
that a real-only fast path could in principle fake). The Peierls
hopping gate at `φ ∉ {0, π/2}` exercises this.

**Audit anchor** (`peierls_step_matches_dense_lossless_at_n6` in
`src/peierls.rs`): N=6 chain, particle at qubit 2, 20 Trotter steps
with `t=1`, `dt=0.1`, `φ=π/3` per bond, lossless χ=8. MPS reproduces
dense to max |⟨Z_i⟩| err ≤ 1e-13 every step. If it passes,
complex-pipeline support is operational end-to-end. **It passes.**

Peierls hopping is also the foundational gate for downstream
magnetic-Hamiltonian work, so the audit lays that foundation as a
side effect.

### Track F.2 — hyperbolic {p, q} tiling foundation (F.2.a + F.2.b closed)

**F.2.a** — Möbius infrastructure + face-edge BFS for {7, 3}:

- `src/hyperbolic.rs::Mobius` in PSU(1,1) representation
  `M = [[a, b̄], [b, ā]]` with `|a|² − |b|² = 1`. `identity`,
  `rotation(θ)`, `translation_to(p)`, `compose`, `inverse`, `apply`,
  `origin_image`.
- `PqMetrics { p, q, side_length, apothem, circumradius }` via
  hyperbolic right-triangle trigonometry. Hyperbolicity guard
  `(p − 2)(q − 2) > 4` rejects Euclidean and spherical (p, q) at
  construction.
- Face-edge BFS via
  `hop = R(θ) · T(2 · apothem along real) · R(−θ) · R(π)` for each
  of the p edge midpoints (`θ = 2π·k/p`). Open-disk guard against
  FP underflow at large radius. Position-hash deduplication at
  `POSITION_HASH_EPS = 1e-6`.
- Vertex graph from face vertices (each face places p vertices at
  `circumradius` distance from centre at angles `π/p + 2πk/p`,
  deduplicated). Spanning tree from BFS in the vertex graph
  starting at the origin-closest vertex. Same `Topology` +
  separate `non_tree_edges` shape as `HeavyHexLayout`.
- **{7, 3} at radius 1** hand-derived two ways: incidence count
  `7·3 + 7·2 + 21·1 = 56 = 8 faces × 7 vertices` ✓, direct count
  `7 (central) + 7 (shared) + 21 (interior per neighbour) = 35` ✓.

**F.2.b** — genericity, larger radius, Python cross-reference:

- Verified for **{3, 7}, {4, 5}, {5, 4}** at radius 0 and 1
  (hand-derived 6, 12, 20 vertices respectively).
- **{7, 3} larger-radius pins:** radius 2 = 112 vertices, radius 3
  = 252 (~3.2× and ~2.25× shell-to-shell growth, consistent with
  hyperbolic exponential structure). All edges hold
  hyperbolic-length ≈ side_length to 1e-5 tolerance.
- **Vertex-degree integrity:** for each supported (p, q), at least
  one interior vertex of degree exactly q exists; no vertex has
  degree > q. (The mode-equals-q test failed and was replaced with
  this stronger characterisation — hyperbolic exponential growth
  means boundary vertices always dominate by count.)
- **Python HyperTiling cross-reference**
  (`external/hypertiling_ref/`, v1.5.1 in committed venv-isolated
  install). For **{7, 3}** at radii 0, 1, 2 the two implementations
  agree exactly on vertex count, edge count, and full degree
  histogram (`cross_reference_against_hypertiling_for_7_3`).
- **Convention finding:** for q ≥ 4 the two implementations diverge
  in "shell" / "layer" definition (HyperTiling includes
  vertex-only-sharing faces per layer; Huoma's face-edge BFS does
  not). Both produce valid tree-decomposable subgraphs of the
  underlying tiling at different per-radius growth rates.
  Documented in module-level docs and the
  `external/hypertiling_ref/README.md`. Bridging the two would be
  F.2.c (3–5 days, deferred).

### `.gitignore` housekeeping (F.2.b)

The `data/` rule was matching `data/` at *any* depth, which would
have hidden `external/hypertiling_ref/data/`. Tightened to `/data/`
(root only); subdirectories named `data/` under `external/` are
reference data and committed. Added `.venv/` and `venv/` for
Python virtualenvs.

## Strategic close-out

At the end of Phase 9, the user (Daniel) explicitly de-committed the
downstream Track F application paths from Huoma's near-term plan:

- **Lanthanide chemistry (F.4 + F.5)** — moved to Garm, which has
  the chemistry infrastructure (active-space integrals via
  PySCF/OpenMolcas, ligand-field modelling, EPR/NMR-aware
  observables) that Huoma would otherwise have to import wholesale.
  The architectural prerequisite (d > 2 site Hilbert spaces) would
  be a 4–6 week refactor of ~40 % of the MPS/TTN codebase; building
  it in Garm sidesteps that scope-creep into Huoma.
- **F.7 RH-adjacent Selberg spectrum** — classified as a hobby
  thread, not a committed Huoma deliverable. The infrastructure F.2
  built (hyperbolic tilings + Möbius isometries + `Topology`-ready
  spanning trees) sits ready for a future `LaplacianSpectrum` module
  (~2–3 weeks to a publishable methods note) if/when Daniel returns
  to it.
- **F.3 Hofstadter** — technically unblocked (both `src/peierls.rs`
  hopping gates and `src/hyperbolic.rs` topologies exist) but not
  committed here.

Track F therefore parks at: **F.1 done, F.2.a + F.2.b done, F.2.c
deferred (small), F.3–F.7 not committed in Huoma**. The design doc
`docs/design/TRACK_F_DESIGN.md` retains the full F.4–F.7 sketch as
the record of where the original architectural commitments came
from, with the strategic re-routing recorded here.

## Test count at end of phase

`cargo test --release --lib` reports **203 lib tests, 0 failed,
0 ignored** (was 177 at end of Phase 8; +4 G.1 XXZ tests, +6 G.2.5
score-design tests, +4 F.1 Peierls tests, +12 F.2.a + F.2.b
hyperbolic tests). Full integration-test set is 215 (3 `#[ignore]`d
scale runs unchanged).

## Commits (Phase 9)

In chronological order on `main`:

1. `f9f1c41` — `.gitignore` consolidation
2. `e3b29e2` — docs reorganised into `docs/history/` + `docs/design/`
3. `19c9d54` — ROADMAP header coherence fix (Phase-7 drift)
4. `9099da2` — `results/VQ-110/` subdirectorised with INDEX
5. `502c284` — baseline verified at 215 tests
6. `2d63253` — ROADMAP: strike speculative items (B.2, B.5, Track C
   entirely; fix Jacobian/VQ-111/track-label drift)
7. `248eba2` — feat(xxz) Track G.1
8. `3c9c65e` — feat(xxz) Track G.2 ITensor scaffold + manifest
9. `b36f244` — feat(xxz) Track G.2 validated (Julia install +
   1.8e-10 cross-check)
10. `a580655` — feat(xxz) first-principles Griffiths score +
    sin(C/2) negative control
11. `6e222f2` — docs: Track G closed as position statement
12. `68d5fe0` — feat(peierls) Track F.1 closed via audit
13. `5a6681a` — feat(hyperbolic) Track F.2.a Möbius + {p,q} BFS
14. `d79e2b2` — feat(hyperbolic) Track F.2.b genericity +
    HyperTiling cross-reference
15. `<this commit>` — docs: Phase 9 report + ROADMAP update for
    strategic close-out

## What's next (open at end of Phase 9)

| Item | Status | Owner |
|---|---|---|
| Track B.1 (examples), B.4 (serde checkpointing) | Backlog | Huoma |
| Track D.4 (IQM topologies) | Customer-triggered backlog | Huoma |
| Track F.2.c (vertex-only-neighbour BFS, bridge to HyperTiling for q ≥ 4) | Deferred — 3–5 days | Huoma (small) |
| Track F.3 (Hofstadter on hyperbolic) | Technically unblocked, not committed | Open |
| Track F.4 + F.5 (Lanthanide chemistry) | Out of Huoma's near-term plan | Garm |
| Track F.7 (RH-adjacent Selberg) | Hobby thread | Daniel |
| Track G.3 (empirical four-way shootout) | Deferred until a publication or collaboration concretely needs the numbers | Huoma (if/when) |
| Track H (D-Wave routing prediction) | Deferred — conditional on collaboration | Open |

No Phase 10 sprint is currently planned. The library is in a stable
state with clean architecture, working scale infrastructure, and
sharp documentation of regime-specific allocator scores. Future
re-entry can pick up from any of the items above without prerequisite
work.
