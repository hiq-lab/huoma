# Huoma Roadmap

This document is the **forward-looking plan** for Huoma. The historical
journey lives in `BIANCHI_JOURNEY.md` (Phases 1–5) and `PHASE6_REPORT.md`
(KIM validation + the `apply_zz_fast` bug discovery).

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

## Where Huoma stands today (April 2026)

✅ **Validated**:
- 1D MPS evolution at floating-point precision against an independent
  dense statevector reference at N = 12 (2.6e-15) and N = 24 (7.4e-16)
- 6.8× speedup over dense statevector at N = 24 with FP-limit observable
  agreement
- 39 lib tests + 6 KIM stages, all green

✅ **Infrastructure in place**:
- MPS-native `expectation_z` and `norm_squared` (work for any N)
- Discarded-weight tracker on every bond
- Finite-difference Jacobian engine with PR / total-sensitivity allocators
- Independent dense reference simulator for KIM (homogeneous + disordered)
- Reproducible deterministic test infrastructure (fixed PRNG seeds)
- Bianchi-violation diagnostic for gauge consistency

⚠️ **Honest limitations**:
- The Jacobian-PR allocator does not yet beat uniform-χ at matched total
  budget. It produces sensible differentiated profiles on disordered
  models but the wins are proportional speedup at smaller budget, not
  better accuracy at the same budget.
- 1D-only. No 2D topology support (heavy-hex, square lattice).
- No site-disordered models beyond `h_x` disorder. No frequency-hierarchy
  models. No boundary-effect studies.
- The `channel.rs` commensurability filter (`sin(C/2)`) is wired in but
  empirically does not predict discarded weight on QKR / Bethe-ansatz
  circuits — see `BIANCHI_JOURNEY.md`.

🚫 **Permanently out of scope**:
- TJM / open-system simulation. Treats noise as Lindblad bath, exactly
  the wrong abstraction for systems where noise is information-bearing
  (cryocooler PUF, levitated-NV torsional coupling). Closed-system only.
- Anything that requires materialising a dense statevector at large N.
  If you need that, use Aer.
- Becoming a compiler. Huoma executes circuits, it does not compile them.

## Roadmap

The roadmap is organised in four tracks. Tracks A–B are concrete and
near-term. Tracks C–D are strategic bets that depend on the answer to one
unsettled question (Track A.4).

---

### Track A — Make the existing pipeline pull its weight

**Goal**: turn the Jacobian allocator from "produces sensible profiles"
into "wins on accuracy at matched budget" on at least one well-defined
benchmark family.

#### A.1 — Target-budget allocator (small)

The current `chi_allocation_from_jacobian` heuristic maps PR scores to χ
via `sqrt(s/max_s)` clamped to `[chi_min, chi_max]`. Total budget is an
emergent property, not a constraint. Compare-at-matched-budget is therefore
not directly achievable.

**Deliverable**: a `chi_allocation_target_budget(jacobian, total_budget,
chi_min, chi_max)` function that solves the constrained allocation problem
exactly. Use water-filling on the score distribution.

**Test**: re-run KIM Stage F (disordered) with budget exactly matching
uniform-χ, check whether the Jacobian-allocated run beats uniform on max
⟨Z⟩ error.

#### A.2 — Score functions beyond participation ratio

PR is a spread metric. It is the right thing if "this bond depends on many
inputs ⇒ allocate more χ" is a good heuristic. It is not obviously the
right thing for entanglement growth.

Test alternatives, all derivable from the same Jacobian matrix:
- Total sensitivity `Σ_i |J_{ki}|²` (already implemented but never
  benchmarked head-to-head against PR)
- Maximum-row-norm: `max_i |J_{ki}|`
- Effective rank: numerical rank of the row, e.g. `(Σ σ)² / Σ σ²` of the
  row's singular value spectrum (PR-style but based on a singular value
  decomposition of the row instead of L1/L2)

**Deliverable**: shootout test in `tests/jacobian_score_shootout.rs` that
runs all four scoring functions on the disordered KIM and reports
accuracy-vs-budget Pareto curves.

#### A.3 — Better benchmarks where the Jacobian *can* win

The disordered self-dual KIM has only one axis of inhomogeneity (h_x), and
the underlying physics localises strongly at large disorder, suppressing
entanglement everywhere. So the dynamic range of "where do bonds need
more χ" is small.

Better targets:
- **Frequency-hierarchy KIM**: per-site ω_i drawn from a distribution
  with two well-separated peaks (slow + fast sites). The bonds between
  fast-fast pairs need more χ than slow-slow pairs.
- **Domain-wall melting in disordered XXZ**: published reference results
  (Jepsen et al., Nature 588, 2020; many MPS follow-ups) and the spatial
  structure varies dramatically with time.
- **Quench across a defect**: a single anomalous site in an otherwise
  uniform chain. The entanglement front bends around the defect.

**Deliverable**: at least one of the above as a new validation stage in
`tests/`. Use the same statevector-reference structure as `kim_validation`.

#### A.4 — The unsettled question: does *any* per-bond sensitivity beat uniform on 1D?

This is the question that decides Tracks C and D. If the answer is **yes**,
Huoma has a real algorithmic story and TTN generalisation (Track D) is
worth the months of work. If the answer is **no** — i.e. uniform-χ is
provably near-optimal for any 1D MPS at the level of accuracy we care
about — then Huoma's value is in the *infrastructure* (validation,
discarded-weight tracking, MPS-native observables), not in the allocator,
and the right move is to harden the existing tools rather than build new
ones.

**A.4 is the gating question for everything below.** Resolve it via the
A.1 + A.2 + A.3 work.

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

### Track C — Algorithmic experiments (gated by A.4)

These are research ideas that are worth trying *only if* Track A
demonstrates that per-bond adaptive χ has real value on 1D.

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

### Track D — Tree-Tensor-Network generalisation (the strategic bet)

**Gated by A.4 and a deliberate strategic decision.** This is the largest
single investment that would change what Huoma is. It is months of work
with real research risk.

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

The roadmap has two explicit decision points where strategic input is
required, not just engineering execution:

1. **After Track A (estimated 2–3 weeks)**: does any sensitivity-based
   allocator clearly beat uniform-χ on at least one well-defined 1D
   benchmark family at matched budget? If yes → Track D becomes worth
   the investment. If no → harden Track B and treat Huoma as
   high-quality 1D MPS infrastructure with Jacobian as a useful but not
   game-changing diagnostic.

2. **Before starting Track D**: is the heavy-hex / 2D extension a
   research project we want to invest months in, or is the right
   strategic move to keep Huoma 1D-focused and let other tools handle
   2D? This depends on (a) the answer to decision 1, (b) external
   factors like collaboration interest and benchmark visibility.

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
