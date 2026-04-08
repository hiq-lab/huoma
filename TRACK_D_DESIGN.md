## Track D — Heavy-hex Huoma via Tree Tensor Networks

This is the planning document for Track D: extending Huoma from a 1D MPS
simulator to a tree tensor network simulator capable of executing the
IBM Eagle 127q heavy-hex topology and reproducing the Tindall et al.
(PRX Quantum 5, 010308, 2024) kicked-Ising benchmark.

The forward roadmap entry is in `ROADMAP.md` § Track D. Decisions taken
in the planning conversation that pin the shape of this document:

1. **First deliverable**: the Tindall N=127 benchmark. No incremental
   small-heavy-hex milestone — we go straight at the published number.
2. **Code layout**: `src/ttn.rs` is a *new parallel type* alongside
   `src/mps.rs`. The 49 green tests around `Mps` are not at risk.
3. **Long-range gates**: swap network on the tree path, the same way
   1D MPS handles non-adjacent gates. Cluster-update / multi-edge
   contraction is deferred until/unless the benchmark forces it.
4. **Off-ramp**: none. Track D is the strategic commitment. Negative
   results ship as a paper, not as a quiet retreat.
5. **Allocator scope**: `chi_allocation_sinc` ports to tree edges as
   part of the first push. The Tindall comparison is published with
   *both* uniform-χ and sin(C/2) profiles at matched tree budget.
6. **Validation gate before trusting Tindall**: small heavy-hex
   subgraph (N≈12–16) vs the dense statevector reference simulator,
   to FP precision, the same way 1D was validated in Phase 6.
7. **First implementation milestone in this conversation**: D.1 stub
   through 1D regression — a `Ttn` type that can represent a linear
   chain and reproduce one `Mps` correctness test bit-for-bit.

Track D inherits Huoma's hard scope rules: closed-system only, no
Lindblad, no dense statevector beyond N=28, no GPU, no Python bindings,
no compiler. The TTN extends *what topologies Huoma can execute*. It
does not soften any of the rejections in `ROADMAP.md` § Track E.

---

### What "done" looks like for Track D

A single end-to-end run, reproducible from the test suite, that:

1. Builds the IBM Eagle 127q heavy-hex topology from a hard-coded
   adjacency list.
2. Initialises a `Ttn` for that topology in the `|0…0⟩` product state.
3. Applies five layers of the Tindall kicked-Ising circuit (J·dt = 0.5,
   h·dt = 0.4) using uniform-χ and using `chi_allocation_sinc` over the
   per-edge sin(C/2) scores at matched edge budget.
4. Reports single-qubit ⟨Z⟩ at depth 5, 10, 20 for every qubit.
5. Compares to Tindall's belief-propagation numbers element-wise.

Success: ⟨Z⟩ within 1 % of Tindall's published values for both the
uniform and sin(C/2) profiles, at compute cost competitive with their
reported wall time. Either a clear win for sin(C/2), a clear tie, or a
clear, structurally-explained loss is publishable.

The publishable outcome is *not* gated on beating Tindall. It is gated
on (a) the small-heavy-hex validation passing against dense, and (b)
the 127q numbers being honest — i.e. truncation error counted not
estimated, exactly the discipline `mps.rs` enforces today.

---

### Architecture

#### The new type

```text
src/ttn.rs

  Ttn {
      sites:       Vec<TtnSite>,        // one per qubit
      topology:    Topology,            // immutable, owns adjacency
      truncation:  TruncationMode,
      discarded:   Vec<f64>,            // one per edge, indexed by EdgeId
  }

  TtnSite {
      // Rank-(d+1) tensor: one physical leg σ ∈ {0,1} plus
      // d virtual legs, one per incident edge in the tree.
      // Stored flat as Vec<C> with explicit shape and a small
      // edge → axis index map.
      data:       Vec<C>,
      shape:      Vec<usize>,           // [bond_0, bond_1, …, 2]
      edges:      Vec<EdgeId>,          // tensor leg ↔ edge
      lambda:     HashMap<EdgeId, Vec<f64>>,  // SVs on each incident edge
  }

  Topology {
      n_qubits:   usize,
      edges:      Vec<Edge>,            // (a, b)
      neighbours: Vec<Vec<EdgeId>>,     // n_qubits → incident edges
      // Tree invariant: |edges| = n_qubits - 1, no cycles.
      // Validated at construction.
  }
```

`Ttn` is intentionally *not* a `TensorNetwork` trait implementation in
the first push. Sharing a trait with `Mps` is on the table for a later
refactor once both types have been hammered into shape; for now the two
modules are independent and the algorithms (allocator, observables,
truncation) live in their own module each. If duplication becomes
painful later, that's a Track B follow-up, not a blocker for Track D.

#### Topology invariants

`Topology` enforces, at construction:

- `|edges| = n_qubits - 1` (tree, not graph)
- No isolated vertices
- No cycles (DFS from any vertex visits every other vertex exactly once)
- `EdgeId` is a `usize` that indexes into `edges`; stable for the
  lifetime of the topology so that `chi_per_edge: &[usize]` indexing
  works the same way `chi_per_bond` works for `Mps` today.

`Topology::linear_chain(n)` gives the degenerate-tree special case used
for the 1D regression milestone.

#### Heavy-hex mapping

IBM Eagle is a graph, not a tree. The heavy-hex graph has one cycle
per hexagonal plaquette (8 cycles for a 127q Eagle, give or take). The
tree decomposition strategy:

- Pick a spanning tree of the heavy-hex graph (BFS from qubit 0 is
  fine — the choice does not affect correctness, only the constant
  factor on swap depth for the gates that fall off the tree).
- Edges in the spanning tree become `Topology` edges.
- Non-tree edges (one per plaquette) are recorded in a separate
  `non_tree_edges: Vec<(usize, usize)>` field on the heavy-hex
  topology constructor.
- Two-qubit gates on non-tree edges are executed via the swap-network
  path through the spanning tree, exactly the same code path as 1D
  long-range gates use today. The swap depth for a non-tree edge is
  bounded by the diameter of the spanning tree, which for Eagle's
  spanning tree is ≤ ~20.

This is the cheapest possible heavy-hex mapping. Cluster-update over
non-tree edges is the obvious next step if the swap-network truncation
error is too high; it lives in the deferred-decisions list at the end
of this document.

#### Gate application

Three cases, in order of cost:

1. **Single-qubit gate.** Contract the 2×2 unitary into the physical
   axis of one site tensor. O(d · χ^d) where d is the site's degree
   and χ is the largest incident bond. Identical in structure to
   `apply_single_to_site` in `mps.rs`.

2. **Two-qubit gate on adjacent sites** (sites that share an edge).
   This is the tree analogue of `Mps::apply_two_qubit`:
   - Contract the two sites and the edge tensor into a single
     rank-(d_a + d_b) tensor.
     - Apply the 4×4 unitary to the two physical legs.
     - Reshape so that the legs incident on site a and the legs incident
       on site b are grouped on opposite sides.
     - SVD with bond truncation governed by `chi_per_edge[edge_id]`
       and the shared `TruncationMode`.
     - Accumulate the discarded weight into `discarded[edge_id]`.
   - Cost: O(χ^(d_a + d_b)) — exponential in degree, which is *the*
     reason heavy-hex is structurally harder than 1D. Tindall's
     spanning tree has max degree 3; 1D MPS has max degree 2.

3. **Two-qubit gate on non-adjacent sites** (sites connected only via
   the tree path, including all non-tree heavy-hex edges):
   - Walk the unique tree path between the two sites.
   - SWAP along the path until the two qubits are adjacent.
   - Apply the two-qubit gate with case (2).
   - SWAP back along the same path.
   - Each SWAP is itself a case-(2) gate application and contributes
     truncation error to every bond on the path. The path length for a
     heavy-hex non-tree edge is ≤ ~20 in the worst case for Eagle.

The swap network is the bluntest possible answer here. We pick it for
the first push because it reuses the case-(2) machinery, has no new
correctness traps, and gives an honest baseline cost number against
which any later cluster-update optimisation can be measured.

#### Observables

`Ttn::expectation_z(target)` is a depth-first traversal of the tree
that builds the contracted environment from the leaves inward, applies
σ_z at the target site, and contracts back out. This is the tree
analogue of the left-environment sweep that `Mps::expectation_z` does
today and has the same O(N · χ^d_max) cost.

`Ttn::expectation_z_all() -> Vec<f64>` is the single-traversal variant
(Track B B.3 for `Mps`, designed-in from day one for `Ttn` because
the Tindall benchmark needs all 127 ⟨Z⟩ values per depth).

`Ttn::norm_squared()` is the same traversal with the identity at the
target. Required because SVD truncation in mixed canonical form does
not preserve the 2-norm exactly, just like in `Mps`.

#### Truncation accounting

The honest discarded-weight tracker from `Mps` carries over directly:
one `f64` per edge, accumulated on every SVD. The total truncation
error in 2-norm² is the sum across edges. This is the discipline that
makes the Tindall comparison meaningful — Tindall's belief-propagation
result has its own approximation budget, ours has the SVD-truncation
budget, and Track D's correctness story is "we counted ours."

---

### Allocator port

`chi_allocation_target_budget` is already score-agnostic and operates
on a flat `&[f64]` of per-bond scores. It ports to tree edges with
literally zero changes — the only difference is that the score vector
is now indexed by `EdgeId` instead of by 1D bond index. Reuse it as-is.

`chi_allocation_sinc` is the part that needs rethinking. The current
1D implementation builds a sparse `ChannelMap` over the linear chain
and reads off `bond_weight(b)` per bond. For a tree, "per-bond
commensurability" needs a clean definition because:

- A bond in 1D separates the chain into two halves; the sin(C/2)
  weight on that bond aggregates the pairwise commensurability across
  the cut.
- A tree edge separates the tree into two subtrees. The natural
  generalisation is: the sin(C/2) weight on edge `e` is the sum of
  pairwise sin(C/2) values over all pairs (i, j) where i and j sit on
  opposite sides of the cut induced by removing `e`.

That sum is well-defined, monotone in the radius (you can truncate to
local neighbourhoods the same way the 1D `bond_weight` does), and
collapses to the existing 1D definition when the tree is a chain. This
is the only non-trivial mathematical content the allocator needs, and
it lives behind a single function:

```text
huoma::ttn::allocator::chi_allocation_sinc_tree(
    frequencies: &[f64],
    topology:    &Topology,
    total_budget: usize,
    chi_min: usize,
    chi_max: usize,
) -> Vec<usize>   // length = topology.edges.len()
```

The implementation is:

1. For each edge `e`, compute the cut partition `(A_e, B_e)` once at
   topology construction (cached on `Topology`).
2. For each edge `e`, score `e` as Σ_{i ∈ A_e, j ∈ B_e} sin(C(ω_i, ω_j) / 2),
   optionally truncated to pairs within tree-distance ≤ radius.
3. Hand the score vector to `chi_allocation_target_budget`.

Cost: O(N² · log) without the radius truncation (fine for N=127),
O(N · radius²) with it. Both are negligible compared to one TEBD layer.

The Tindall publication uses both this and uniform-χ at matched total
edge budget. That comparison is exactly the Phase 7 Stage F shootout
shape, just on a tree.

---

### Validation strategy

Four layers, in order:

1. **1D regression.** A `Ttn` built on `Topology::linear_chain(n)`
   reproduces a known `Mps` test (the `kim_validation` self-dual
   small-N case is a good candidate) bit-for-bit at FP precision. If
   this fails, the contraction / SVD machinery is wrong and topology
   is irrelevant. **This is the first milestone, scaffolded in this
   conversation.**

2. **Small heavy-hex vs dense.** Pick the smallest non-trivial heavy-hex
   subgraph that fits the existing dense reference simulator's N≤16
   ceiling — a single hexagonal plaquette with its flag qubits, around
   12–16 qubits depending on how it's chosen. Run a short kicked-Ising
   circuit on it, compare every single-qubit ⟨Z⟩ to the dense
   statevector result to FP precision (~1e-14). This is the analogue
   of the Phase 6 N=12 / N=24 dense-vs-MPS validation that put the
   1D path on a firm footing.

3. **Tindall N=127 benchmark vs the published numbers.** Once 1 and 2
   are green, run the actual benchmark. Depth-5 / 10 / 20 single-qubit
   ⟨Z⟩ for every qubit, compared to Tindall et al.'s belief-propagation
   values element-wise.

4. **ITensor TTN cross-reference at N=127.** Independent re-run of the
   same Tindall circuit with ITensors.jl + ITensorNetworks.jl on the
   same heavy-hex spanning tree at the same matched edge budget. This
   is the only independent numerical reference available at N=127 —
   dense does not reach past N≈28. See the "ITensor cross-reference"
   section below for the reproducibility pipeline; it is a hard
   dependency of D.5, not a nice-to-have.

The dense reference for layer 2 already exists in `kicked_ising::reference_kim_run`
for the 1D case. It needs a heavy-hex variant — one extra function in
`kicked_ising.rs` that takes a `Topology` and applies the same gates
in the same order. Cost: ~80 lines.

#### ITensor cross-reference (layer 4)

**Motivation.** Layers 1–3 anchor the correctness story at small and
medium N, and layer 3 pins Huoma's N=127 result to Tindall's published
numbers. But Tindall's reference is belief propagation with its own
approximation budget (loop corrections, message-passing tolerance),
not an independent TN simulator with a comparable truncation budget.
A second TTN that counts its truncation error the same way Huoma does
— and that is implemented in a completely different codebase — is the
only way to distinguish "Huoma and Tindall disagree by X because
Huoma's allocator is wrong" from "Huoma and Tindall disagree by X
because belief propagation and TTN truncation are structurally
different approximations of the same state." ITensor plays that role.

**What ITensor is used for here (and what it is not).** ITensor is
our *second opinion* at N=127, not the ground truth. No TTN simulator
at that size has a dense reference, ours included. The comparison
asks: given the same heavy-hex spanning tree, the same KIM parameters,
the same total edge budget, and the same discarded-weight truncation
mode, do two independent implementations agree on ⟨Z⟩ at depth 5, 10,
20 to within a few times their individual truncation noise floors?
If yes, both are trustworthy in that regime. If no, at least one of
the two has a structural problem worth finding.

**Reproducibility pipeline.** One Julia script lives outside the Rust
workspace, under `external/itensor_ref/`:

- Pinned versions of Julia + ITensors.jl + ITensorNetworks.jl declared
  in a `Project.toml` / `Manifest.toml` pair committed to the repo.
- `kim_heavy_hex.jl` takes the same inputs Huoma's D.5 test takes:
  `topology.json` (edge list for the 127q spanning tree + the
  non-tree edges), `params.json` (J·dt, h·dt, depth list), and a
  `budget.json` (total edge budget, chi_min, chi_max, truncation
  epsilon).
- Output: a single `z_expectations.csv` with columns
  `qubit, depth, z, discarded_weight_total` — the same schema Huoma
  will emit.

**Hard inputs shared with Huoma.** The same `topology.json` is
consumed by both sides: `Topology::ibm_eagle_127().to_json()` writes
it, and the Julia script reads it. This guarantees that "the same
spanning tree" is not an aspirational statement — it is a literal
`sha256` match on the file both simulators read.

**Rust-side cross-check.** `tests/ttn_tindall_127.rs` has three
independent assertions:

1. Huoma vs Tindall's published numbers element-wise.
2. Huoma vs ITensor CSV element-wise.
3. ITensor CSV vs Tindall's published numbers element-wise (purely
   a sanity check on the pipeline — ITensor is expected to agree
   with itself over time, so this is guarded by a coarser tolerance).

The Huoma-vs-ITensor assertion is the one that carries the weight.
Tolerances for (2) will be set by the *larger* of each side's reported
discarded-weight floor plus a small multiplicative margin — not by a
hard-coded epsilon. Truncation error on both sides is *counted*, so
the comparison tolerance is a function of the truncation budget both
simulators were given, not a hand-tuned constant. This keeps the
check honest across budget sweeps.

**Pre-D.5 work this creates.**

- `Topology::ibm_eagle_127` ships with a `to_json()` method and a
  golden-file test that hashes the JSON against a committed fixture.
- `external/itensor_ref/` is bootstrapped alongside D.5 with its
  Julia manifest + runnable script + README describing the Julia
  version and how to invoke it offline.
- CI stays Rust-only for speed; the ITensor run is executed on a
  developer machine before each D.5 tag and its output committed as
  a versioned CSV in `external/itensor_ref/data/`. The Rust test
  reads the committed CSV and does not invoke Julia itself, so the
  Rust CI does not grow a Julia dependency.
- If and only if the Huoma-vs-ITensor assertion fails, we regenerate
  the CSV (not before), to prevent silent drift of the reference.

**Scope discipline.** ITensor is used *only* for (a) the D.5 Tindall
cross-reference described above, and (b) the Track B follow-up item
in the next sub-section. It is not used for D.1/D.2/D.3 validation,
not used for the chi allocator ports, and not used as a development
aid (we don't tune Huoma against ITensor output — that would defeat
the point of an independent reference). The dense statevector path
remains the sole reference for everything below N=16.

**Track B follow-up — 1D MPS cross-check.** Independent of Track D,
a one-shot comparison of Huoma's 1D MPS vs ITensor MPS on a fixed KIM
benchmark is worth adding as a Track B item (not a blocker). Our 1D
path is already validated against the dense reference at 2.6e-15 at
N=12 and 7.4e-16 at N=24, so ITensor there is a second opinion on a
path we already trust to floating point precision — low urgency, but
nice to have in the ROADMAP section listing production-hardening
items. A short follow-up issue `track-b: itensor-mps-crosscheck-1d`
is enough to pin the intent.

---

### Milestones

#### D.1 — Linear-tree `Ttn` (this conversation)

- New file `src/ttn.rs` with `Ttn`, `TtnSite`, `Topology`, `EdgeId`.
- `Topology::linear_chain(n)` constructor.
- `Ttn::new(topology)` initialises the product state.
- `Ttn::apply_single`, `Ttn::apply_two_qubit_adjacent`,
  `Ttn::expectation_z`, `Ttn::norm_squared` — minimum surface area
  required to drive a 1D regression test.
- One regression test in `tests/ttn_linear_regression.rs` that runs
  the same KIM evolution through both `Mps` and `Ttn::linear_chain`
  and asserts ⟨Z⟩ agreement to ~1e-13.

This milestone is **scope-locked at 1D**. No heavy-hex code, no
allocator port, no swap network. The point is to prove the engine
contracts and truncates correctly on the topology Huoma already
trusts. Everything that follows builds on a green test here.

#### D.2 — General tree topology

- `Topology::from_edges(n, edges)` general-tree constructor with the
  cycle / connectivity / leaf-count validation listed above.
- `Topology::cut_partition(edge_id)` — the partition pair used by the
  allocator and by the tree-environment contractions.
- `Ttn::apply_two_qubit_adjacent` generalised from the linear case to
  arbitrary site degree (≤ 3 in practice for heavy-hex).
- Test: a Y-junction TTN (3-leaf, 1-internal-vertex, N=4) reproduces
  the dense statevector after a few CX layers.

#### D.3 — Heavy-hex topology + small-heavy-hex validation

- `src/topology/heavy_hex.rs` with `Topology::ibm_eagle_127()` and
  the spanning-tree decomposition. Non-tree edges stored separately.
- `kicked_ising::reference_kim_run_heavy_hex(topology, params, steps)`
  in the dense-reference module.
- Swap-network long-range gate via `Ttn::apply_two_qubit_via_path(a, b, u)`.
- Test: small heavy-hex subgraph (N≈12–16) Ttn vs dense statevector
  agreement at FP precision after 5 KIM layers.

#### D.4 — sin(C/2) tree allocator

- `chi_allocation_sinc_tree` per the formula above.
- Cached `cut_partition` on `Topology`.
- Tests mirroring `allocator.rs`'s 11 corner-case suite, on linear
  trees (regression) and on a small Y-junction (sanity).

#### D.5 — Tindall N=127 benchmark (incl. ITensor cross-reference)

- `tests/ttn_tindall_127.rs` runs the full benchmark with both
  `chi_allocation_sinc_tree` and uniform-χ at matched edge budget and
  records depth-5/10/20 ⟨Z⟩ for all 127 qubits.
- Three independent assertions (see "ITensor cross-reference"):
  1. Huoma vs Tindall's published numbers element-wise.
  2. Huoma vs committed ITensor CSV element-wise, tolerance derived
     from the larger discarded-weight floor of the two runs.
  3. Pipeline sanity check: ITensor CSV vs Tindall numbers element-
     wise at a looser tolerance.
- `Topology::ibm_eagle_127().to_json()` + golden-file test hashing
  the JSON against a committed fixture, so the ITensor script reads
  the *same* topology bytes Huoma executes on.
- `external/itensor_ref/` bootstrap: pinned Julia `Project.toml` /
  `Manifest.toml`, `kim_heavy_hex.jl` runner, committed
  `data/z_expectations.csv` reference, README describing the offline
  regeneration workflow. CI remains Rust-only.
- `examples/tindall_eagle.rs` — a runnable demo, the public face of
  the milestone.
- `PHASE8_REPORT.md` — the historical record. ROADMAP and BIANCHI
  history files remain append-only as before.

---

### Things explicitly deferred

- **Cluster-update / multi-edge contraction** for non-tree heavy-hex
  edges. Considered if and only if the swap-network noise floor
  blocks the 1 % Tindall agreement. Otherwise it lives in Track C.
- **TensorNetwork trait** unifying `Mps` and `Ttn`. Refactor it later
  if duplication crosses ~300 lines and both modules have stabilised.
- **TTN-native TDVP** (Track C C.1). Track D ships TEBD-on-tree.
- **IQM grids** (Track D D.4 in `ROADMAP.md`). Cheaper than heavy-hex
  but not on the critical path for Tindall. Add after D.5 if there's
  appetite.
- **`expectation_z_all`** for `Mps` (Track B B.3). The single-sweep
  variant is built into `Ttn` from day one because the Tindall
  benchmark forces it; the `Mps` retrofit is independent and stays
  in Track B.
- **Sharing the Bianchi diagnostic** between `Mps` and `Ttn`. Skipped
  unless a specific reason emerges. The Bianchi story is currently
  feature-flagged and is not required for the Tindall correctness
  argument; honest discarded weight is.

---

### Open questions for later sessions

These are not blockers for D.1 but should be answered before D.3
ships:

1. Which spanning tree of Eagle do we pick? Any DFS / BFS choice is
   valid; the choice affects the swap-network depth distribution but
   not the correctness. Worth a one-paragraph justification in
   PHASE8_REPORT when the time comes.
2. How are Tindall's reference numbers obtained — do they ship them
   as a CSV or do we have to extract them from the paper figures? The
   benchmark assertion's tightness depends on this answer.
3. The dense reference for layer-2 validation needs ~16 qubits worth
   of statevector. The current `kicked_ising::reference_kim_run` uses
   the same dense path as the Phase 6 validation; a one-line check
   that it actually scales to 16 qubits in ≤ a few seconds is owed.
4. Allocator score function: cap the radius for the tree score the
   same way the 1D radius is capped (`DEFAULT_SINC_RADIUS = 5`)? The
   tree distance metric is well-defined, so the same cap applies, but
   for N=127 even the uncapped O(N²) variant is fast enough that the
   choice is mostly a matter of taste / consistency with the 1D path.
5. ITensor cross-reference pinning: which specific version of Julia,
   ITensors.jl, and ITensorNetworks.jl do we commit to? The
   comparison is only meaningful if both sides are reproducible; a
   `Manifest.toml` with an exact pin is the mechanism, but we still
   need to pick the pin. Default: latest stable at the time D.5 opens,
   bumped deliberately in follow-up PRs rather than floating.
6. ITensor regeneration policy: committed CSV is the norm, but if
   Tindall ship an updated benchmark or ITensorNetworks changes its
   default gauge / truncation, do we regenerate silently or require a
   PR with a matching PHASE8_REPORT update? Default: the latter. The
   reference CSV lives under version control for exactly this reason.
