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

The wrong benchmarks for Huoma — and we have already learned this the hard
way — are anything that needs (a) 2D physical topology beyond 1D
nearest-neighbour, (b) open-system / Lindblad dynamics, or (c) translation-
invariant chains where uniform-χ is already optimal.

## Where Huoma stands today (April 2026, after Phase 7)

✅ **Validated**:
- 1D MPS evolution at floating-point precision against an independent
  dense statevector reference at N = 12 (2.6e-15) and N = 24 (7.4e-16)
- 6.8× speedup over dense statevector at N = 24 with FP-limit observable
  agreement
- 49 lib tests + 4 KIM stages (A, B, D, F), all green

✅ **Production allocator path**:
- `huoma::chi_allocation_sinc(frequencies, total_budget, chi_min, chi_max)`
  is the recommended one-call entry point for adaptive χ on any 1D MPS
  workload with per-site frequencies. O(N · radius²), microseconds at
  N ≤ 200, no pilot, no censoring.
- `huoma::chi_allocation_target_budget(scores, total_budget, chi_min,
  chi_max)` is the score-agnostic water-filling primitive for users who
  compute their own per-bond complexity scores by other means.
- Both live in `src/allocator.rs` and are re-exported at crate root.

✅ **Infrastructure in place**:
- MPS-native `expectation_z` and `norm_squared` (work for any N)
- Discarded-weight tracker on every bond + `TruncationMode::DiscardedWeight`
- sin(C/2) channel map (`ChannelMap::from_frequencies_sparse`) and
  partitioning (`partition::partition_adaptive`)
- Independent dense reference simulator for KIM (homogeneous + disordered)
- Reproducible deterministic test infrastructure (fixed PRNG seeds)
- Bianchi-violation diagnostic for gauge consistency

⚠️ **Honest limitations**:
- 1D-only. No 2D topology support (heavy-hex, square lattice). **This is
  the live constraint** that gates the strategic case for Track D.
- sin(C/2) is *competitive with* uniform-χ on disordered KIM at matched
  budget (~5–30 % off in either direction depending on N), not strictly
  better. Per Dalzell–Brandão (Quantum 2019) this is the predicted
  ceiling for any allocator on clean / weakly-disordered 1D systems —
  uniform-χ is structurally near-optimal there.
- Whether sin(C/2) can beat uniform on *strongly* disordered (Griffiths
  regime) 1D systems is untested. The roadmap does not currently plan to
  test it; see Track A's "what closed Phase 7" section.

🚫 **Removed in Phase 7**:
- The finite-difference Jacobian module and all its allocators. The
  discarded-weight observable censors boundary bonds at low chi_min,
  producing matched-budget allocations 6–11× worse than uniform on
  disordered KIM. Documented in `docs/history/PHASE7_REPORT.md` and commit `19a5793`.
- The previous claim that sin(C/2) "doesn't predict per-bond discarded
  weight on QKR" was correct as a Spearman result on one benchmark
  family but does not survive as a general statement. As a matched-budget
  allocator on disordered KIM, sin(C/2) is the safe default. See
  Phase 7 report.

🚫 **Permanently out of scope**:
- TJM / open-system simulation. Treats noise as Lindblad bath, exactly
  the wrong abstraction for systems where noise is information-bearing
  (cryocooler PUF, levitated-NV torsional coupling). Closed-system only.
- Anything that requires materialising a dense statevector at large N.
  If you need that, use Aer.
- Becoming a compiler. Huoma executes circuits, it does not compile them.

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

#### A.3 — Better benchmarks ⚠ obviated

The literature scan during Phase 7 (recorded in session notes) confirmed
that for clean / weakly-disordered 1D systems, Dalzell–Brandão (Quantum
2019) implies uniform-χ is structurally near-optimal — *no* allocator
can give more than constant-factor improvements there. The only published
opening for adaptive χ in 1D is bond-disordered XXZ in the Griffiths
regime (Aramthottil et al., PRL 133, 196302, 2024). Adding more KIM
benchmarks is not the experiment that resolves Track A — running Huoma
against ITensor's ε-truncation on bond-disordered XXZ would be, but that
is independently more interesting as a Track B / C item than as a
gating Track A test.

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

Independent of whether the Jacobian wins, the simulator infrastructure has
real value and should be production-ready.

#### B.1 — Doc tests + examples

Currently the only entry point is the test suite. Add at least three
runnable examples in `examples/`:
- `examples/kim_basic.rs` — minimal kicked Ising on N = 100
- `examples/jacobian_alloc.rs` — build a Jacobian, allocate χ, run a
  production circuit, compute observables
- `examples/disordered_quench.rs` — disordered KIM with measurement at
  multiple time slices

#### B.2 — Real cost estimator

`reassembly::estimate_fidelity` and `bench::PipelineResult` already track
some cost metrics. Build a public `huoma::cost::estimate(...)` that takes
a circuit + a χ allocation and returns:
- expected wall time (calibrated against benchmarks)
- expected memory peak
- expected total discarded weight (if a Jacobian / pilot run is supplied)

**Deliverable**: `src/cost.rs` + tests + integration with the existing
benchmarks.

#### B.3 — Streaming / chunked observables ✅ done

Delivered as `Mps::expectation_z_all() -> Vec<f64>` in commit `29642db`.
O(N · χ⁴) total (shared left/right env builds, parallelised over q)
replacing the O(N² · χ⁴) naïve loop. `Ttn::expectation_z_all` delegates
to this fast path on `Backend::Linear`. Unit test
`expectation_z_all_matches_naive_loop` pins agreement with the slow-but-
trusted O(N² · χ⁴) reference.

#### B.4 — Serialisation

Save and restore `Mps` to/from disk so long-running benchmarks can
checkpoint. Use `serde` (already a dep) with the `bincode` format, gated
behind a feature flag.

#### B.5 — Lift the Bianchi feature flag and the dead-code allowances

`bench.rs` still has `#[allow(clippy::needless_return,
clippy::type_complexity, clippy::manual_clamp)]`. Some of these are real
issues that should be fixed in the code; others are spurious and should be
narrowed. Same for several `derive_default_impl` warnings on
`TruncationMode`. Tech debt, low priority but worth a one-shot pass.

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
`results/VQ-110/adiabatic_2d_30k_attempt_failed.log` and
`adiabatic_2d_30k_run.log`.

---

### Track C — Algorithmic experiments (no longer gated)

Track A is closed but did not produce evidence that adaptive χ on 1D MPS
delivers more than constant-factor wins over uniform-χ in any regime
Huoma currently tests. Track C ideas remain individually interesting but
none are *required* for Huoma's production story; they are research
investments that compete with Track D for time. Pursue selectively.

#### C.1 — TDVP instead of TEBD

The current Trotterised evolution accumulates error from finite step
size. Time-Dependent Variational Principle (TDVP) is parameter-free and
works directly on the MPS manifold. For Floquet circuits TEBD is still
preferred (the gates are explicit), but for arbitrary time-dependent
Hamiltonians TDVP could open new use cases.

Reference: Haegeman, Lubich, Oseledets, Vandereycken, Verstraete (2016).

#### C.2 — Variational χ allocation

Instead of building a Jacobian on a pilot run and then committing to a χ
profile, optimise the profile *during* the production run by minimising a
combination of (estimated) discarded weight and total budget. This would
make the allocator adaptive to the actual circuit being executed, not just
the pilot.

#### C.3 — Sobol indices instead of finite differences

Replace the central-difference Jacobian with proper Sobol sensitivity
indices using a Saltelli sampling scheme. More expensive per pilot run but
gives variance-decomposition information that the FD Jacobian does not.

This was tried in Phase 5f and found correlation 0.65–0.71 with measured
discarded weight. Worth revisiting if A.2 shows that PR is the bottleneck.

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
  Mac Studio M4 Ultra (`tests/projected_ttn_scale.rs`, `results/VQ-110/REPORT.md`).

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

Design doc lives at `docs/design/TRACK_F_DESIGN.md`. **No code written.** Realistic
earliest start: Q4/2026 or Q1/2027. Two orthogonal generalisations:

1. **Complex-valued tensors** (F.1) — `Scalar` trait over `f64` ⊕
   `Complex<f64>`, all `gemm`/SVD/QR through `faer`'s complex paths.
   ~2 weeks mechanical work, prerequisite for everything else in F.
2. **Hyperbolic layouts** (F.2) — Fuchsian-group word generator for
   {p, q} tilings on the Poincaré disc. ~3-4 weeks.

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

### Track G — Bond-disordered XXZ in the Griffiths regime (active)

**The next planning conversation.** This is the validation experiment
that closes the one open question Phase 7 left behind: *for which class
of 1D models does sin(C/2) provably beat uniform-χ at matched budget?*
Phase 7 ruled it out for clean / weakly-disordered KIM (Dalzell–Brandão
puts uniform-χ near-optimal there). The published opening is bond-
disordered XXZ in the Griffiths regime (Aramthottil et al., PRL 133,
196302 (2024)) — rare regions of anomalously small `J_i` dominate the
slow dynamics, and a per-bond allocator has a real opportunity if it can
identify them.

**Sub-items**:

- **G.1** — XXZ gate set in `src/kicked_ising.rs` (or new
  `src/models/xxz.rs`): `H = Σ J_i (Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + Δ Sz_i Sz_{i+1})`
  with sampling primitives for the standard bond-disorder distributions.
- **G.2** — ITensor reference path. `external/itensor_ref/` already
  carries the Julia/ITensor scaffold from Track D; extend with an
  ε-truncation runner that matches a configurable total-χ budget.
- **G.3** — Shootout. At matched total budget per system size, three
  allocators: uniform, ε-truncated ITensor reference, sin(C/2) +
  water-filling. Metrics: max ⟨S^z⟩ error, max bipartite entanglement
  entropy error, total wall time. Disorder strength swept across weak →
  Griffiths regime.
- **G.4** — Verdict. Either (a) sin(C/2) beats uniform in a measurable
  regime → publishable methods note, Track A's open question definitively
  answered, or (b) sin(C/2) is competitive but not better → Track A
  closes for good with that statement, no further allocator work.

**Cost**: 1-2 weeks focused. Risk: medium — it's possible sin(C/2)'s
frequency-channel structure does not map cleanly onto bond-disorder
(disorder is in couplings, not local frequencies), in which case the
right control is a sin(C/2) variant that consumes the bond-disorder
distribution directly. That would itself be a real finding.

**Tracked in valiant-ops as VQ-111.**

---

### Track H — Annealer routing prediction (deferred)

The originally-framed motivation behind the closed-system adiabatic-ramp
runs (`results/VQ-110/ANNEALER_THREAD.md`, `ANNEALER_THREAD_2D.md`) was
to scale the simulator to qubit counts beyond current D-Wave Pegasus /
Zephyr hardware, observe routing pathologies at scale, and project back
onto next-generation D-Wave architecture decisions. The April-2026
sprint did not build that programme; it built the *engine* (validated 1M
chain, 9.5K 2D heavy-hex grid with non-tree edges, `canonicalize_and_normalize`
primitives) but not Pegasus/Zephyr topology generators, not a routing
variation sweep, and not the back-projection.

What is missing for Track H to be a real programme:

1. `PegasusLayout::generation(m)` matching the D-Wave Boothby et al. 2020
   spec, validated bit-stable against the `dwave-system` Python SDK.
2. `ZephyrLayout::generation(m)` for Advantage2-class topologies.
3. Routing-variation framework: fixed Ising instance, varied
   swap-network orderings and χ allocations, per-edge discarded-weight
   diagnostics as TSV + heatmap.
4. Scaling sweep across Pegasus(15), (16), (20) and Zephyr(4), (6) —
   pattern extraction by edge class.
5. A predictive write-up that says "at Pegasus(M=X), expect
   bottleneck-class Y to dominate at fraction Z."

**Why this is deferred, not killed**: Huoma's distinguishing primitive is
sin(C/2) commensurability on frequency channels. An annealing problem
Hamiltonian is real-valued couplings, not a frequency channel — the
sin(C/2) machinery does not apply. Track H would use Huoma as a generic
TTN simulator, not as the sin(C/2)-structured one. That is a fine
application project but it does not advance Huoma as a library. Returning
to it is conditional on (a) a concrete D-Wave-adjacent collaboration that
justifies an application sprint, or (b) Track G finding that the
allocator story has further headroom in a regime that overlaps with
annealer-relevant graph structure.

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
   and `results/VQ-110/REPORT.md`.

3. **Open (Track G)**: is there a 1D regime where sin(C/2) provably beats
   uniform-χ at matched budget? Bond-disordered XXZ in the Griffiths
   regime is the published opening (Aramthottil et al., 2024). Either
   answers definitively or extends the allocator. This is the next
   planning conversation.

4. **Open (Track F)**: when do we commit to the complex-tensor +
   hyperbolic-layout pivot? Currently sketched only. F.1 + F.2 are the
   gating prerequisite for everything downstream including the RH-adjacent
   sub-path. Q4/2026 or Q1/2027 by default; pull forward if a customer
   conversation or paper deadline justifies it.

---

## Tracking

Tasks for each track will live as GitHub issues in this repo with the
labels `track-a`, `track-b`, `track-c`, `track-d`. The current state of
each track is summarised in the GitHub Project board (to be created
alongside the first round of issues).

The two design documents `docs/history/BIANCHI_JOURNEY.md` and `docs/history/PHASE6_REPORT.md` are
**append-only history**, not living roadmap documents. This ROADMAP file
is the only living planning document — update it when tracks complete,
when decisions are made, or when scope changes.
