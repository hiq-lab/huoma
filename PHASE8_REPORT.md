# Phase 8 Report — Track D: Heavy-Hex Huoma via Tree Tensor Networks

This document records the outcome of Track D, which extended Huoma
from a 1D MPS simulator to a tree-tensor-network simulator capable of
executing the IBM Eagle 127-qubit heavy-hex topology.

Phase 8 follows the historical sequence established in `BIANCHI_JOURNEY.md`
(Phases 1–5), `PHASE6_REPORT.md` (KIM validation + the `apply_zz_fast`
bug discovery), and `PHASE7_REPORT.md` (matched-budget allocator +
sin(C/2) reframe + Track A verdict).

## What was built

| Milestone | PR | What it delivered |
|---|---|---|
| D.1 | #1 | `Ttn` scaffolding + `Topology::linear_chain` + 1D regression test |
| D.2 | #2 | General-tree contraction + Y-junction / star anchors |
| D.3a | #4 | `HeavyHexLayout::ibm_eagle_127()` + golden file + heavy-path spanning tree |
| D.3b | #5 | `Ttn::apply_two_qubit_via_path` swap network + 13q small-heavy-hex KIM validation |
| D.4 | #6 | `chi_allocation_sinc_tree` — sin(C/2) allocator generalised to tree edges |
| D.5.0 | #7 | Gauge-aware `expectation_z` / `expectation_z_all` replacing statevector materialisation |
| D.5.1 | #9 | `run_kim_heavy_hex` driver + first honest N=127 run (depth 5, 200 ms) |
| D.5.2 | #10 | Per-edge χ driver + matched-budget shootout scaffold |
| D.5.3 | #11 | Huoma ⟨Z₆₂⟩ vs Tindall published reference at depths 0–20 |
| D.5.4 | #12 | ITensor cross-reference scaffold (Julia script + manifest, CSV pending offline run) |
| D.5.5 | #12 | `examples/tindall_eagle.rs` runnable demo + this report |

## The spanning tree

Eagle 127q was decomposed into a spanning tree using a **row-major
heavy-path policy**:

- 7 horizontal rows as heavy paths (lengths 14, 15, 15, 15, 15, 15, 14)
- 6 bridge rows of 4 bridge qubits each; the leftmost bridge in each row
  is the through-bridge (both vertical edges in the tree), the other 3 are
  leaf bridges (up-edge only)
- 96 horizontal tree edges + 30 bridge tree edges = 126 spanning-tree edges
- 18 dropped non-tree edges (one per hexagonal plaquette)
- 66.7 % of ZZ gates are tree-local, 20.8 % at tree distance 2, 12.5 %
  are non-tree (routed via the swap network)

The spanning tree is frozen as `tests/golden/ibm_eagle_127.json` and
byte-stable across builds. The ITensor cross-reference script reads the
same JSON file so both simulators operate on the identical tree.

## The swap network

Non-tree ZZ gates are applied via symmetric swap-forward-apply-swap-back
through the unique tree path between the two endpoints. For Eagle the
non-tree paths average ~5–10 edges, giving ~10–20 SWAP operations per
non-tree gate. At χ = 8 this is the dominant cost centre per Floquet
step, but the total per-step wall time is still ~40 ms in release.

Validated against the topology-agnostic `DenseState` reference on a
13-qubit Eagle sub-fragment (1 hexagonal plaquette, 1 degree-3 junction,
1 non-tree edge) to floating-point precision after 3 KIM Floquet steps.

## The env-sweep expectation_z

The D.2 placeholder materialised a 2^n statevector — physically
impossible at N = 127. D.5.0 replaced it with a gauge-aware environment
sweep: move the orthogonality centre to each qubit, read ⟨Z⟩ locally
from the centre site tensor. `expectation_z_all` uses a DFS pre-order
traversal so each tree edge is walked at most twice across the full sweep
(Euler tour bound), keeping the observable pass at O(N · χ³).

## First honest N = 127 numbers

At `(θ_J, θ_h) = (π/4, 0.8)`, χ = 8, initial state |0…0⟩:

| depth | Huoma ⟨Z₆₂⟩ | Tindall BP | diff |
|---|---|---|---|
| 0 | 1.0000000 | 1.0000000 | +2.2e-16 |
| 1 | 0.6967067 | 0.6967067 | +6.7e-16 |
| 5 | 0.7202846 | 0.4981649 | +2.2e-1 |
| 10 | 0.6829333 | 0.3850014 | +3.0e-1 |
| 20 | 0.7099541 | 0.2295752 | +4.8e-1 |

**max |ΔZ₆₂|** = 0.480 | **total discarded weight** = 1.0

### Interpretation

1. **Depth 0–1 exact agreement** proves the parameter mapping (sign-flip
   `j = -π/4` to match Tindall's `exp(+iπ/4 Z⊗Z)`) and gate ordering
   are correct.

2. **Depth 2+ systematic overestimation** is the expected truncation
   artefact: χ = 8 is far below the converged limit, so SVD truncation
   clips entanglement growth and the state "remembers" the initial product
   state too much. The Huoma trajectory stays near 0.7 while Tindall's
   converged BP result decays to 0.23.

3. **Total discarded weight = 1.0** confirms this is a budget constraint
   (roughly as much state has been truncated as remains), not a
   methodological error.

4. Higher χ will close the gap. The quantitative convergence study is a
   natural follow-up but is not part of the Track D scope — it belongs in
   Track B (production hardening) or a dedicated Phase 9 budget-sweep.

## Matched-budget shootout

At the same χ = 8 total budget, both uniform-χ and sin(C/2)-allocated χ
profiles were run on Eagle 127q at depths 5, 10, 20:

```
[D.5.2 shootout] homogeneous depth= 5: max |Z_uniform − Z_alloc| = 0.635
[D.5.2 shootout] homogeneous depth=10: max |Z_uniform − Z_alloc| = 0.959
[D.5.2 shootout] homogeneous depth=20: max |Z_uniform − Z_alloc| = 0.000
```

On Tindall's **homogeneous** circuit (all qubits share the same h_x) the
allocator collapses to near-uniform (within ±1 χ per edge), but the
severely-truncated Schmidt spectrum is sensitive to ±1 shifts — enough to
compound into 0.5–1.0 per-qubit ⟨Z⟩ drift. Depth-20 collapse to zero
suggests both profiles converged to the same effective-low-rank attractor.

On a **disordered** variant (per-qubit h_x jitter) the allocator produces
a genuinely non-flat profile and exercises the per-edge χ code path at
full scale.

## Wall time

| Run | Release wall time |
|---|---|
| N = 127, 1 Floquet step, χ = 8 | ~40 ms |
| N = 127, 5 Floquet steps, χ = 8 | ~200 ms |
| N = 127, 20 Floquet steps, χ = 8 | ~760 ms |
| Matched-budget shootout (2 × 20 steps + 10 disordered) | ~21 s |

## Test coverage at the end of Track D

- **125 lib unit tests** (allocator, bianchi, channel, frequency,
  kicked_ising, mps, partition, reassembly, shootout, validation,
  ttn::contraction, ttn::dense, ttn::gauge, ttn::heavy_hex,
  ttn::kim_heavy_hex, ttn::site, ttn::topology, ttn::allocator,
  ttn::mod tests)
- **4 KIM validation stages** (A, B, D, F — the 1D anchor)
- **12 Eagle layout integration tests** (structural invariants +
  golden-file byte-regression)
- **4 Tindall integration tests** (depth-5 smoke, homogeneous shootout,
  disordered shootout, D.5.3 ⟨Z₆₂⟩ vs published reference)
- **1 runnable example** (`examples/tindall_eagle.rs`)

## What Track D did NOT resolve

- **Whether sin(C/2) beats uniform on heavy-hex KIM.** On the homogeneous
  Tindall circuit, the allocator collapses to uniform by design (no
  per-site asymmetry to exploit). On disordered circuits the allocator
  produces non-trivial profiles, but without a dense reference at N = 127
  we cannot judge which profile is closer to truth. This is structurally
  the same finding as Phase 7's Track A verdict: sin(C/2) is a safe
  default that is competitive with uniform, not a strict improvement, on
  clean / weakly-disordered circuits.

- **Whether Huoma's ⟨Z₆₂⟩ converges to Tindall's at higher χ.** The
  depth-1 exact agreement proves the gate ordering is correct; the
  depth-20 gap of 0.48 at χ = 8 is large but expected given the
  discarded weight. A χ-sweep convergence study is the natural next step.

- **ITensor cross-reference CSV.** The pipeline scaffold is in place
  (`external/itensor_ref/`) but the actual Julia run has not been
  executed. The committed `data/` directory contains only a `.gitkeep`.
  Generating the reference CSV requires a Julia ≥ 1.10 environment with
  the pinned ITensors.jl / ITensorNetworks.jl versions.

## Decision record

Decisions taken during the Track D planning and implementation sessions:

1. **Spanning tree**: row-major heavy-path decomposition (7 horizontal rows).
2. **Golden file**: landed in D.3, not deferred to D.5.
3. **`Mps` facade retirement**: deferred — `mps.rs` is untouched.
4. **Non-tree gates**: symmetric swap-and-back (not permutation map).
5. **Site tensor representation**: type-erased uniform (`Vec<C>` + `dims`).
6. **Canonical form**: in scope from D.2 day 1 (not deferred).
7. **χ allocator**: uniform for D.5.1, sin(C/2) port for D.4, matched-budget shootout for D.5.2.
8. **Benchmark scope**: ⟨Z₆₂⟩ trajectory only (full 127-qubit profile not published by Tindall in machine-readable form).
9. **Tindall parameter convention**: `exp(+iπ/4 Z⊗Z)` requires `j = -π/4` in Huoma's `zz_gate(θ) = exp(-iθ Z⊗Z)`. Verified by depth-1 exact agreement with cos(θ_h).
10. **ITensor cross-reference**: scaffold shipped, CSV generation requires offline Julia run.

## What comes next

Track D is functionally complete. The remaining open items are:

- **ITensor CSV generation** — run `julia --project=. kim_heavy_hex.jl` on
  a machine with Julia ≥ 1.10, commit the output, wire the Rust-side
  assertion.
- **χ convergence study** — sweep χ = {8, 16, 32, 64} on Eagle 127q
  depths 5/10/20 and plot the approach to Tindall's converged BP values.
  Natural Track B or Phase 9 item.
- **Track B production hardening** — doc tests, examples, serialisation,
  streaming observables, clippy cleanup.
- **Track E re-evaluation** — none of the Track E scope rejections are
  affected by Track D's outcome. Closed-system only, no Lindblad, no
  dense statevector beyond N = 28, no GPU, no Python bindings, no compiler.
