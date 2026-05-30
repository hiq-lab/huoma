# Huoma Roadmap

This document is the **forward-looking plan** for Huoma. The historical
journey lives in `docs/history/BIANCHI_JOURNEY.md` (Phases 1–5), `docs/history/PHASE6_REPORT.md`
(KIM validation + the `apply_zz_fast` bug discovery), and
`docs/history/PHASE7_REPORT.md` (matched-budget allocator + sin(C/2) reframe + Track A
verdict).

## North star

Huoma should be the simulator that makes **commensurability and per-bond
sensitivity into first-class scheduling primitives** for large-N quantum
dynamics, on the topologies that the simulator's tensor structure can
honestly represent.

Concretely: the right benchmark for Huoma is "given a circuit of N ≥ 10³
qubits with broken translation invariance (disorder, defects, frequency
hierarchies, boundary structure), produce observables at lower compute cost
than uniform-χ MPS at the same fidelity, with **truncation error counted,
not estimated**."

The wrong benchmarks for Huoma — and we have already learned this the
hard way — are anything that needs (a) open-system / Lindblad dynamics,
(b) translation-invariant 1D chains where uniform-χ is already optimal,
or (c) couplings on graphs that are not tree-decomposable.

## Where Huoma stands today (May 2026, after Phase 8 + scale sprint)

✅ **Validated**:
- 1D MPS at FP precision vs dense statevector: N = 12 (2.6e-15) and
  N = 24 (7.4e-16); 6.7× speedup at N = 24.
- Aer statevector ground truth: F = 1.000000, TVD = 0 at χ = 8 up to
  N = 28 (`accuracy::accuracy_vs_aer`).
- TTN backend on IBM Eagle 127q heavy-hex: depth-1 ⟨Z₆₂⟩ exact vs
  Tindall et al. PRX Quantum 5 010308 (2024) (`ttn_tindall_127`).
- ProjectedTtn: 10⁶ qubits 1D chain in 5.8 s on laptop; 10⁹ qubits 2D
  heavy-hex in ~31 min on Mac Studio M4 Ultra (`projected_ttn_scale`,
  `results/VQ-110/projected_1b/REPORT.md`).
- Adiabatic-ramp engine: 1M chain in 18 min, 30K 2D heavy-hex grid with
  non-tree edges in 150 min — with three FP-precision dense anchors at
  N = 12 chain, N = 15 binary tree, N = 19 heavy-hex grid.
- 215 tests total across lib + integration binaries (3 `#[ignore]`d
  scale runs), all green on 2026-05-29.

✅ **Production allocator path** (1D):
- `huoma::chi_allocation_sinc(frequencies, total_budget, chi_min, chi_max)`
  is the recommended one-call entry point for adaptive χ on any 1D MPS
  workload with per-site frequencies. O(N · radius²), microseconds at
  N ≤ 200, no pilot, no censoring.
- `huoma::chi_allocation_target_budget(scores, total_budget, chi_min,
  chi_max)` is the score-agnostic water-filling primitive.
- Both live in `src/allocator.rs` and are re-exported at crate root.
- Tree-edge analogue `chi_allocation_sinc_tree` in `src/ttn/allocator.rs`.

✅ **Infrastructure in place**:
- MPS backend: `expectation_z`, `expectation_z_all` (O(N · χ⁴) sweep),
  `norm_squared`, `canonicalize_left_and_normalize` (load-bearing at 10⁶).
- TTN backend: `Ttn` with gauge tracking, two-site contraction + SVD on
  arbitrary tree edges, swap-network for non-tree edges,
  `canonicalize_and_normalize` (load-bearing at 30K 2D, canon-every-step
  cadence at that scale).
- Topologies: `HeavyHexLayout::ibm_eagle_127()` (hard-coded golden) +
  `HeavyHexLayout::grid(R, B)` (parametric).
- Projected path: `ProjectedTtn` with `partition_tree_adaptive` +
  `extract_volatile_islands` + `BoundaryTensor` analytical ⟨Z⟩ on stable
  qubits — million-vertex scaling for commensurate hosts.
- sin(C/2) channel map (`ChannelMap::from_frequencies_sparse`) and
  partitioning (`partition::partition_adaptive` + tree analogue).
- Discarded-weight tracker on every bond + `TruncationMode::DiscardedWeight`.
- Independent dense reference simulator for KIM (homogeneous + disordered)
  and topology-agnostic statevector reference for TTN tests.
- Bianchi-violation diagnostic for gauge consistency.
- Reproducible deterministic test infrastructure (fixed PRNG seeds).

⚠️ **Honest limitations**:
- sin(C/2) is *competitive with* uniform-χ on weakly-disordered 1D KIM
  at matched budget (~5–30 % off in either direction depending on N),
  not strictly better. Per Dalzell–Brandão (Quantum 2019) this is the
  predicted ceiling for any allocator on clean / weakly-disordered 1D
  systems — uniform-χ is structurally near-optimal there.
- Whether sin(C/2) provably beats uniform on *strongly* disordered
  (Griffiths regime) 1D systems is the open question for Track G; it
  is in flight, not resolved.
- sin(C/2) presupposes a per-site frequency channel. Hamiltonians with
  only couplings and no frequency structure (standard annealing problems,
  most chemistry problem Hamiltonians) cannot use the sin(C/2) allocator —
  they use Huoma as a generic TTN simulator. This is a deliberate scope
  limit, not a missing feature.
- 2D coverage is restricted to **tree-decomposable** topologies (heavy-hex,
  general trees, sparse-defect projected hosts). Dense 2D lattices where
  perimeter-law entanglement grows faster than χ can absorb are out of
  reach at our χ budget; the 9.5K → 30K 2D adiabatic ramp work shows
  where the practical ceiling sits at χ = 8 on `grid(R, B)`.

🚫 **Removed in Phase 7**:
- The finite-difference Jacobian module and all its allocators. The
  discarded-weight observable censors boundary bonds at low chi_min,
  producing matched-budget allocations 6–11× worse than uniform on
  disordered KIM. Documented in `docs/history/PHASE7_REPORT.md` and
  commit `19a5793`.
- The previous claim that sin(C/2) "doesn't predict per-bond discarded
  weight on QKR" was correct as a Spearman result on one benchmark
  family but does not survive as a general statement. As a matched-budget
  allocator on disordered KIM, sin(C/2) is the safe default. See
  Phase 7 report.

🚫 **Permanently out of scope** (Track E):
- TJM / open-system simulation. Treats noise as Lindblad bath, exactly
  the wrong abstraction for systems where noise is information-bearing
  (cryocooler PUF, levitated-NV torsional coupling). Closed-system only.
- Anything that requires materialising a dense statevector at large N.
  If you need that, use Aer.
- Becoming a compiler. Huoma executes circuits, it does not compile them.
- Python bindings inside this repo. `python/huoma-py/` is a separate
  wrapper crate that takes Huoma as a dependency; no PyO3 inside the
  core library.

## Roadmap

**Updated 2026-05-28.** The roadmap reflects the actual state of the code:
Track A closed (Phase 7), Track B partially advanced by the April-2026
scale sprint, Track C unchanged, Track D fully landed (Phase 8 + Eagle
127 Tindall benchmark + 1M-qubit `ProjectedTtn`), Track E unchanged.
Two new tracks are added: **Track F** (non-Euclidean topologies + magnetic
Hamiltonians, design doc only — see `docs/design/TRACK_F_DESIGN.md`) and **Track G**
(bond-disordered XXZ in the Griffiths regime — the missing sin(C/2)
validation experiment, the next active work item). A previously-active
"closed-system adiabatic-ramp + annealer-routing-prediction" thread is
reframed as **Track H — deferred**: the engine evidence (1M chain,
9.5K 2D grid) lives on as scale infrastructure, but the predictive
D-Wave-routing programme it was framed against was never built and is
out of scope for now.

---

### Track A — Closed (Phase 7, commits `19a5793`, `811cb15`, `ce488e0`)

**Outcome**: the production allocator path is `huoma::chi_allocation_sinc`,
built on the score-agnostic `chi_allocation_target_budget` water-filling
primitive. The finite-difference Jacobian story is retired.

#### A.1 — Target-budget allocator ✅ done

Delivered as `chi_allocation_target_budget(scores, total_budget, chi_min,
chi_max)` in `src/allocator.rs`. Greedy integer water-filling, exact
budget consumption, 11 corner-case unit tests. Score-agnostic by design —
accepts any non-negative per-bond score vector. Commit `19a5793`.

#### A.2 — Score shootout ⚠ obviated

PR vs. total-sensitivity were both benchmarked at matched budget on
Stage F. Both produce allocations 6–11× worse than uniform-χ because the
discarded-weight observable they share has a structural boundary blind
spot (a bond with one qubit on its left has Schmidt rank ≤ 2, so the
pilot's discarded weight there is artificially small, and the resulting
score clamps the bond to chi_min). max-row-norm and effective-rank are
derived from the same observable and inherit the same failure mode.
Extending the shootout to more scores on the same observable was not
worth the effort. See `docs/history/PHASE7_REPORT.md` § "The boundary blind spot".

#### A.3 — Better benchmarks ⚠ obviated → reopened as Track G

The literature scan during Phase 7 confirmed that for clean /
weakly-disordered 1D systems, Dalzell–Brandão (Quantum 2019) implies
uniform-χ is structurally near-optimal — *no* allocator can give more
than constant-factor improvements there. The only published opening for
adaptive χ in 1D is bond-disordered XXZ in the Griffiths regime
(Aramthottil et al., PRL 133, 196302, 2024). That experiment is now
Track G.

#### A.4 — Answered

In its actual form ("does the Jacobian allocator beat uniform on 1D"):
**no**, and the structural reason is now documented. In its intended
form ("is there a production-quality per-bond χ allocator for 1D
Huoma"): **yes**, sin(C/2) via water-filling, which has been Huoma's
foundational thesis since the original Tilde Pattern paper. The Phase 5e
detour into ML-style sensitivity analysis is closed.

#### Stage F numbers from Phase 7

For reference, the matched-budget shootout on disordered self-dual KIM
(`stage_f_disordered_sinc_vs_uniform`):

| N  | strategy            | max ⟨Z⟩ err | budget | build time |
|----|---------------------|-------------|--------|------------|
| 14 | uniform χ=8         | 6.49e-2     | 104    | —          |
| 14 | sinc2 matched       | 6.78e-2     | 104    | 0.01 ms    |
| 14 | (jacobian PR/L1)    | 7.27e-1     | 104    | 32 ms      |
| 50 | uniform χ=8         | 1.02e-1     | 392    | —          |
| 50 | sinc2 matched       | 1.36e-1     | 392    | 0.06 ms    |
| 50 | (jacobian PR/L1)    | 6-7e-1      | 392    | 227 ms     |

Jacobian rows are recorded here for reference only — the implementation
is gone as of commit `ce488e0`.

---

### Track B — Production hardening

Items that earn their keep on the library surface. Two done (B.3, B.6),
two remain on the backlog (B.1, B.4). Speculative items were struck
2026-05-30 — opportunistic tech-debt cleanup (former B.5) happens during
the next touch of the affected file rather than as a roadmap line item,
and a generic cost estimator (former B.2) is on hold until a concrete
user surfaces.

#### B.1 — Examples

Currently the only entry point is the test suite. Add two runnable
examples in `examples/`:
- `examples/kim_basic.rs` — minimal kicked Ising on N = 100
- `examples/disordered_quench.rs` — disordered KIM with measurement at
  multiple time slices

#### B.3 — Streaming / chunked observables ✅ done

Delivered as `Mps::expectation_z_all() -> Vec<f64>` in commit `29642db`.
O(N · χ⁴) total (shared left/right env builds, parallelised over q)
replacing the O(N² · χ⁴) naïve loop. `Ttn::expectation_z_all` delegates
to this fast path on `Backend::Linear`. Unit test
`expectation_z_all_matches_naive_loop` pins agreement with the slow-but-
trusted O(N² · χ⁴) reference.

#### B.4 — Serialisation

Save and restore `Mps` / `Ttn` to/from disk so long-running benchmarks
can checkpoint and resume. Concrete trigger: the 30K 2D adiabatic ramp
hit `SvdFailed(0)` at 102 min wall (`canon_every=5` attempt) — losing
102 minutes of CPU time to a crash that could have been recovered from
a checkpoint at min 90 is the kind of pain that justifies this. Use
`serde` (already a dep) with the `bincode` format, gated behind a
feature flag.

#### B.6 — Canonical-form stability primitives ✅ done

Two new primitives proven load-bearing at million-scale in the April-2026
scale sprint:

- `Mps::canonicalize_left_and_normalize` (commit `ccb479f`) — left-to-right
  SVD sweep, no truncation, `U` left-isometric, `S V†` folded right,
  rightmost site divided by its Frobenius norm. O(N · χ³). Required
  because the gate-and-truncate pipeline absorbs `sqrt(S)` on both sides
  after every bond, leaving the MPS in no canonical form; cumulative FP
  drift overflows env contractions at N ≳ 10⁵ × 50 steps and breaks
  `faer` SVD mid-ramp at N = 10⁶.
- `Ttn::canonicalize_and_normalize` (commit `4584311`) — tree analogue via
  `gauge::canonicalize_to(sites, topology, root)` leaves-to-root QR sweep.
  Same role at scale on the 2D path.

Both pinned by unit tests
(`canonicalize_left_and_normalize_preserves_expectation`, equivalent for
tree). Without them no million-scale run is numerically defensible.

**Cadence finding (2D, May 2026)**: the canonicalize cadence does not
scale linearly with `N`. On the 2D `grid(R, B)` path, `canon_every=5`
ran successfully up to 19K qubits but failed at 30K with `SvdFailed(0)`
partway through the ramp — accumulated truncation drove a local Θ
matrix singular faster than the every-5-step sweep could clean it up.
`canon_every=1` (every ramp step) ran 30K to completion with norm² =
1.000000 and discarded weight in line with the 19K extrapolation. The
wall-time penalty is small (~50 × ~170 ms canon at 30K, < 6 % of the
149-min ramp). Implied design rule: at 2D scale, canon-every-step is
the default; the 1D path can stay on every-5-step because chain
area-law keeps truncation pressure bounded. Logs in
`results/VQ-110/annealer_stress_test/adiabatic_2d_30k_attempt_failed.log`
and `results/VQ-110/annealer_stress_test/adiabatic_2d_30k_run.log`.

---

### Track C — struck 2026-05-30

Was: TDVP-instead-of-TEBD, variational χ allocation, Sobol indices. All
three were speculative research items without a concrete user or
blocker. TDVP and variational-χ were "could open new use cases" without
any circuit class demanding them. Sobol was already tried in Phase 5f
with mediocre correlation (Spearman 0.65–0.71) and its revisit-trigger
(Track A.2 showing PR as bottleneck) is dead because A.2 is obviated.
Huoma is a library and a paper pipeline, not a research lab — these
items either earn their place via a concrete trigger or stay struck.

---

### Track D — Tree-Tensor-Network generalisation ✅ closed

Closed by Phase 8 (`docs/history/PHASE8_REPORT.md`). The full TTN backend lives in
`src/ttn/` (5,200+ lines across 13 files). All sub-items delivered:

- **D.1** — `Ttn` data structure, gauge tracking, two-site contraction +
  bipartition SVD on arbitrary tree edges, discarded-weight tracker per
  edge (`src/ttn/mod.rs`, `gauge.rs`, `contraction.rs`).
- **D.2** — `HeavyHexLayout::ibm_eagle_127()` spanning-tree decomposition
  with golden file (`src/ttn/heavy_hex.rs`, `tests/golden/ibm_eagle_127.json`).
- **D.3** — Tindall et al. (PRX Quantum 5, 010308, 2024) benchmark
  reproduced: depth-1 ⟨Z₆₂⟩ exact to FP precision (`tests/ttn_tindall_127.rs`).
  Depth-5/10/20 trajectory differs as expected at χ = 8 (truncation
  artefact, not a bug).
- **D.4** — IQM topology mappings not built. The Eagle 127 path covers the
  load-bearing case; the IQM grids would extend coverage but are not
  required for Huoma's production story now that `HeavyHexLayout::grid(R, B)`
  generalises beyond the hard-coded Eagle layout.
- **D.5** — `ProjectedTtn` with `partition_tree_adaptive` +
  `extract_volatile_islands` + `BoundaryTensor` analytical ⟨Z⟩ on stable
  qubits. Scales to 1M qubits in 5.8 s and 1B qubits in ~31 min on
  Mac Studio M4 Ultra (`tests/projected_ttn_scale.rs`, `results/VQ-110/projected_1b/REPORT.md`).

What remains under Track D as a follow-on item rather than a Phase 9
sprint: D.4 IQM mappings if/when a customer needs Garnet/Emerald/Crystal
specifically.

---

### Track E — Things explicitly *not* on the roadmap

Recording these so we don't re-litigate them every quarter:

- **TJM / Lindblad / open-system**: scope-rejected. See docs/history/PHASE6_REPORT.md and
  the previous conversation log. The right tools are symbolic Pauli
  noise propagation and channel-folded fidelity tracking, not stochastic
  state unravelling.
- **Dense statevector beyond N = 28**: not Huoma's job. Use Aer or
  cuStateVec if you need that.
- **GPU**: `faer` already uses SIMD on CPU. GPU for tensor contractions
  would be a separate project (cuTensorNet etc) and the gain on χ ≤ 64
  is probably modest. Re-evaluate only if Track D produces large-χ TTN
  benchmarks where it would obviously help.
- **Python bindings**: Huoma is a Rust library. Anyone who wants to call
  it from Python can write a wrapper crate; we will not maintain bindings
  in this repo.
- **Compiler / routing / basis translation**: out of scope. Huoma takes
  circuits, not source.

---

### Track F — Non-Euclidean topologies and magnetic Hamiltonians (design only)

Design doc lives at `docs/design/TRACK_F_DESIGN.md`. Realistic
earliest start for the bulk of the track: Q4/2026 or Q1/2027.

1. **Complex-valued tensors** (F.1) — **✅ closed via audit
   2026-05-30 (VQ-139).** The original 2-week-mechanical-pivot
   estimate was based on a stale assumption; tensors were already
   `Complex64` throughout. The Peierls hopping smoke-test in
   `src/peierls.rs` anchors complex-pipeline correctness end-to-end
   (N=6, lossless, φ = π/3, max ⟨Z⟩ err ≤ 1e-13 over 20 Trotter
   steps). No remaining pivot work.
2. **Hyperbolic layouts** (F.2) — Fuchsian-group word generator for
   {p, q} tilings on the Poincaré disc. ~3-4 weeks. Independent of
   the Peierls foundation, but the natural pairing for F.3 below.

Downstream phases (F.3 Peierls/Hofstadter, F.4 spin-orbit, F.5 lanthanide
benchmark, F.6 circuit-QED hyperbolic, F.7 Selberg / Riemann-adjacent)
depend on the F.1 + F.2 foundation. F.4 is the high-risk research item
(tensor-network spin-orbit is 2024-2025 state-of-the-art). F.7 is the
RH-adjacent option: Laplace spectrum on a {3,7}-discretised modular
surface, statistics compared to Selberg-spectrum tables and Montgomery
GUE — methodologically a small extension of F.3, scientifically a quantum
chaos / spectral geometry contribution rather than a number-theory one.

Honest framing: an RH-flavoured F.1 → F.2 → F.3 → F.7 sub-path is ~9-12
weeks total and produces empirical Selberg-spectrum data on
million-vertex hyperbolic tilings that nobody else has, but does not
constitute work on RH itself.

---

### Track G — Bond-disordered XXZ in the Griffiths regime ✅ closed (position statement)

Closed 2026-05-30 as a position statement based on first principles
plus the G.1 + G.2 + score-design work. The originally-planned G.3
empirical shootout is deferred — the conceptual question Track G
existed to answer is settled without it.

**Verdict (full discussion in `results/VQ-136/REPORT.md`):** sin(C/2)
is the right *score* for driven systems with per-site frequency
channels (KIM, QKR — Phase 7 production path). It does *not* transfer
to static-coupling-disorder physics: bond-disordered XXZ has no drive,
no KAM torus structure, and even with the geometrically correct site
frequencies `ω_i = √(|J_{i-1}| · |J_i|)` it is brittle to
integer-coincidence in disorder values (`J_weak = 0.01 = 1/100`
produces an exact integer ω-ratio across the chain and is reported as
"perfectly commensurate").

The right Griffiths-XXZ score is `|J_i|` per bond, fed into
`chi_allocation_target_budget`. Derivation: TEBD bipartite entropy
across bond i grows as `S_i(t) ≈ min(log 2, |J_i| · t)`, so strong
bonds form singlets fastest and need the highest χ. Dasgupta–Ma RG
confirms (the strongest bond is eliminated first by singlet
formation). This contradicts the common "weak bonds = transport
bottleneck = give them more χ" intuition — which is correct for
transport but inverted for bipartite entanglement.

The strictly stronger Track A statement is therefore: **per-bond
water-filling on a regime-specific score is universal; Huoma supplies
the infrastructure for both layers.** sin(C/2) is the score for
driven systems; `|J_i|` is the score for static-coupling-disorder.

**Delivered:**

- **G.1** — XXZ gate set in `src/xxz.rs` (`apply_xxz_step`,
  `xxz_bond_gate`, `sample_bond_disorder_log_uniform`,
  `reference_xxz_run`, `product_state_mps`) + dense anchor at N=10
  lossless to max ⟨Z⟩ err ≤ 1e-12 over 50 Trotter steps.
- **G.2** — ITensor reference runner
  `external/itensor_ref/xxz_griffiths.jl`, Julia 1.12 + ITensors v0.9
  + ITensorMPS v0.4 (committed `Project.toml` + `Manifest.toml`),
  cross-checked against Huoma's dense at lossless N=10 to 1.8e-10
  element-wise. Manifest contract pinned by
  `ItensorXxzManifest::round_trip_test`.
- **G.2.5** — `xxz_griffiths_bond_scores` (production allocator
  score, `|J_i|`) and `xxz_site_frequencies` (sin(C/2) negative
  control, geometric mean), with tests pinning both the correct
  Griffiths behaviour and the sin(C/2) failure mode explicitly.

**Deferred (G.3):** the four-way shootout at N ∈ {32, 64, 128} × 5
disorder strengths × 10 realisations. Required for a Phys. Rev. B
methods-note submission but not for the architectural verdict.
~3–5 days of harness + analysis on top of the existing G.1/G.2
infrastructure; revisit when a publication or collaboration concretely
asks for the numbers. The two G.3 outcomes that would *change* the
verdict are recorded in `results/VQ-136/REPORT.md` § "What G.3 could
change."

**Tracked in valiant-ops as VQ-136 (done).**

---

### Track H — Annealer routing prediction (deferred)

The April-2026 adiabatic-ramp sprint built the engine (1M chain, 30K 2D
heavy-hex grid with non-tree edges, `canonicalize_and_normalize`) but
not the programme it was framed against: Pegasus/Zephyr topology
generators, a routing-variation sweep, and the back-projection onto
next-generation D-Wave architecture. The engine evidence lives at
`results/VQ-110/annealer_stress_test/`.

Deferred, not killed: Huoma's sin(C/2) primitive presupposes a
frequency channel; an annealing problem Hamiltonian is real-valued
couplings only, so this track would use Huoma as a generic TTN
simulator, not as the sin(C/2)-structured one. Returning to it is
conditional on (a) a concrete D-Wave-adjacent collaboration that
justifies an application sprint, or (b) Track G finding that the
allocator story extends into annealer-relevant graph structure.

---

## Decision points

1. ✅ **Resolved (Phase 7)**: does any sensitivity-based allocator clearly
   beat uniform-χ on a well-defined 1D benchmark at matched budget? **No**,
   in the form the question was asked, and the right *production* answer
   was sin(C/2) via water-filling all along. See `docs/history/PHASE7_REPORT.md`.

2. ✅ **Resolved (Phase 8)**: does the TTN generalisation reproduce
   published heavy-hex benchmarks? **Yes** — Tindall ⟨Z₆₂⟩ at depth 1
   exact to FP precision, full Eagle 127q pipeline at 760 ms / 20 steps,
   `ProjectedTtn` carries 10⁹ qubits in ~31 min. See `docs/history/PHASE8_REPORT.md`
   and `results/VQ-110/projected_1b/REPORT.md`.

3. ✅ **Resolved (Track G, position statement 2026-05-30)**: is there a
   1D regime where sin(C/2) provably beats uniform-χ at matched budget?
   **No** for sin(C/2) specifically — it is a KAM filter, applies only
   to driven systems. **Yes** for the underlying water-filling
   architecture, with a regime-specific score (`|J_i|` for static
   bond-coupling-disorder, derived from RG + TEBD entropy). See
   `results/VQ-136/REPORT.md`. The empirical magnitude (G.3) is
   deferred until a publication or collaboration calls for it.

4. **Open (Track F)**: when do we commit to the complex-tensor +
   hyperbolic-layout pivot? Currently sketched only. F.1 + F.2 are the
   gating prerequisite for everything downstream including the RH-adjacent
   sub-path. Q4/2026 or Q1/2027 by default; pull forward if a customer
   conversation or paper deadline justifies it. **With Track G closed,
   this is now the next active planning conversation.**

---

## Tracking

Active work items live as `VQ-XXX` tickets in valiant-ops
(`~/Projects/valiant-ops/board.yaml`). Track G sits there as VQ-136.
There is no per-track GitHub-issue labelling scheme in this repo and
none planned — valiant-ops is the single source of cross-project task
state.

Historical work-streams under `docs/history/` are **append-only**:

- `BIANCHI_JOURNEY.md` — Phases 1–5
- `PHASE6_REPORT.md` — KIM validation + the `apply_zz_fast` bug
- `PHASE7_REPORT.md` — Track A close, sin(C/2) + water-filling
  production path
- `PHASE8_REPORT.md` — Track D close, Eagle 127 Tindall benchmark

This `ROADMAP.md` is the only living planning document — update it when
tracks complete, decisions are made, or scope changes.
