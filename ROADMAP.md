# Huoma Roadmap

This document is the **forward-looking plan** for Huoma. The historical
journey lives in `BIANCHI_JOURNEY.md` (Phases 1–5), `PHASE6_REPORT.md`
(KIM validation + the `apply_zz_fast` bug discovery), and
`PHASE7_REPORT.md` (matched-budget allocator + sin(C/2) reframe + Track A
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
  disordered KIM. Documented in `PHASE7_REPORT.md` and commit `19a5793`.
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

The roadmap is organised in four tracks. Track A is **closed** by Phase 7
(`PHASE7_REPORT.md`); Track B is the production-hardening backlog; Track C
contains research ideas that may or may not get funded; Track D is the
next strategic commitment.

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
worth the effort. See `PHASE7_REPORT.md` § "The boundary blind spot".

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

#### B.3 — Streaming / chunked observables

Currently `expectation_z` builds two left environments end-to-end per
qubit, so computing all N expectation values is O(N² · χ³). For large N
this dominates wall time. A single sweep that emits all N expectation
values in one pass is O(N · χ³) — same as one current call.

**Deliverable**: `Mps::expectation_z_all() -> Vec<f64>` that does the
single-sweep variant.

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

### Track D — Tree-Tensor-Network generalisation (the next strategic commitment)

**No longer gated by Track A.** This is the largest single investment
that would change what Huoma is. It is months of work with real research
risk. The strategic case is straightforward: a simulator that cannot
execute on the actual target backends (IBM Eagle/Heron heavy-hex,
IQM Garnet/Emerald grids) is not a useful tool, regardless of how
well-validated its 1D allocator is. Track A confirmed that the 1D
allocator story is *finished* — sin(C/2) via water-filling is the right
production default and there is no further headroom there — so the
months of Track D work do not have to carry the weight of an unresolved
1D question alongside them.

Track D is the next planning conversation. A separate design doc will
break it down properly; the bullet points below are the existing Phase 6
sketch, kept as a starting point.

#### D.1 — TTN data structure and basic operations

Generalise `Mps` from a 1D chain to a tree. Each site has at most three
neighbours (left, right, vertical). The "left environment" sweep that
underpins everything in `mps.rs` becomes a depth-first traversal of the
tree. Basic operations:
- Apply single-qubit gate at a leaf
- Apply two-qubit gate on adjacent leaves (same parent)
- Compute expectation value
- SVD truncate at any internal edge
- Discarded-weight tracking per edge

**Cost estimate**: ~2000 lines of new Rust, ~3 weeks of focused work for
the basic data structure + the seven validation tests that match what
`mps.rs` already has for the 1D case.

#### D.2 — Heavy-hex topology mapping

IBM Eagle (127q) and Heron (156q) are heavy-hex graphs. Map heavy-hex onto
a TTN by tree-decomposing the heavy qubits. Most heavy-hex edges are local
in the tree decomposition; the few that aren't get handled via the same
SVD-based gate application as 1D.

**Deliverable**: `src/topology/heavy_hex.rs` with the IBM Eagle 127q
topology hardcoded as a tree, plus a heavy_hex test in `tests/`.

#### D.3 — Tindall benchmark

Once D.1 and D.2 are in place, run the Tindall et al. (PRX Quantum 5,
010308, 2024) 127q kicked Ising benchmark on the same observables they
report (single-qubit ⟨Z⟩ at depth 5, 10, 20 with J·dt = 0.5, h·dt = 0.4).

**Success criterion**: ⟨Z⟩ within 1 % of Tindall's belief-propagation
result, at compute cost competitive with their reported numbers. **Or**:
demonstrate clearly *where* Huoma's TTN approach differs and why
(probably: Jacobian-driven adaptive bond instead of belief propagation).

This is the long-arc project that, if successful, makes Huoma a
first-class citizen in the IBM-superconducting-platform benchmark
landscape. If unsuccessful, the negative result is itself publishable
("here is a class of TN methods that does not extend to heavy-hex
benchmarks competitively") and Huoma falls back to its 1D niche.

#### D.4 — IQM topology mappings

Similar to D.2 for IQM Garnet (20q), Emerald (54q), and Crystal — all 2D
grids, more "naturally TTN-able" than heavy-hex. Likely cheaper than D.2
because the tree decomposition is more natural.

---

### Track E — Things explicitly *not* on the roadmap

Recording these so we don't re-litigate them every quarter:

- **TJM / Lindblad / open-system**: scope-rejected. See PHASE6_REPORT and
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

## Decision points

1. ✅ **Resolved (Phase 7)**: does any sensitivity-based allocator clearly
   beat uniform-χ on a well-defined 1D benchmark at matched budget? **No**,
   in the form the question was asked, and the right *production* answer
   was sin(C/2) via water-filling all along. See `PHASE7_REPORT.md`.

2. **Before starting Track D**: is the heavy-hex / 2D extension a
   research project we want to invest months in? The Phase 7 framing
   ("a simulator that can't simulate most of our target backends isn't
   worth the effort") is a strong yes by default; the open sub-question
   is *how much* of Track D to scope into the first deliverable
   (just heavy-hex topology + N=127 Tindall benchmark vs. also IQM grids
   vs. also a new χ allocator extension to the tree case). This is the
   next planning conversation.

---

## Tracking

Tasks for each track will live as GitHub issues in this repo with the
labels `track-a`, `track-b`, `track-c`, `track-d`. The current state of
each track is summarised in the GitHub Project board (to be created
alongside the first round of issues).

The two design documents `BIANCHI_JOURNEY.md` and `PHASE6_REPORT.md` are
**append-only history**, not living roadmap documents. This ROADMAP file
is the only living planning document — update it when tracks complete,
when decisions are made, or when scope changes.
