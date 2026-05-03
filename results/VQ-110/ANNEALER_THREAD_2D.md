# VQ-110 — 2D heavy-hex closed-system adiabatic ramp

**Date:** 2026-05-03
**Hardware:** Mac Studio M4 Ultra, 128 GB unified memory, macOS 26.3
**Tests:** `tests/adiabatic_ramp_2d_scale.rs::adiabatic_ramp_2d_grid_*_completes`

Annealer-thread chapter 2 — the unprojected-2D headline.
[Chapter 1](ANNEALER_THREAD.md) carried the same closed-system adiabatic
ramp on a 1D chain at 10⁶ qubits. This chapter pushes the same primitive
into a true 2D heavy-hex topology with non-tree edges, where 2D
entanglement obeys *perimeter* law not chain area-law and the same χ=8
that worked on a 1M chain saturates at much smaller N.

---

## What this run shows

A regular heavy-hex grid (`HeavyHexLayout::grid(R, B)`: R data rows of
width 2B+1 with B bridges in every bridge row) carrying the same
adiabatic ramp `H(s) = (1−s)·H_X + s·H_problem` as the chain headline,
with the time-varying schedule routed through *both* the tree-edge
fast path and the swap-network non-tree-edge path every step.

The lattice is structurally closer to D-Wave's Pegasus/Zephyr graphs
post-embedding than the 1D chain is — qubits are degree-3 (data) or
degree-2 (bridges), the spanning tree carries `2RB + (R-1)(B+1)` edges,
and `(R−1)(B−1)` non-tree edges are routed via swap-network through
the unique tree path between their endpoints.

---

## Sized series

| Size | Lattice | N | Tree / non-tree edges | χ | canon | Steps | Wall | Per step | norm² | max\|⟨Z⟩\| | Discarded |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1K | grid(20, 20) | 1,200 | 1199 / 361 | 16 | every 5 | 50 | 500 s | 10.0 s | 1.000000 | 0.227 | 8.50 |
| 4K | grid(45, 30) | 4,065 | 4064 / 1276 | 8 | every 5 | 50 | 367 s | 7.35 s | 1.000000 | 0.437 | 9.29 |
| 9.5K | grid(63, 50) | 9,463 | 9462 / 3038 | 8 | every 5 | 50 | 1361 s (22.7 min) | 27.2 s | 1.000000 | 0.734 | 9.64 |
| 19K | grid(80, 80) | 19,200 | 19199 / 6241 | 8 | every step | 50 | 4439 s (74.0 min) | 88.8 s | 1.000000 | 0.737 | 46.4 |
| **30K** | grid(100, 100) | 30,000 | 29999 / 9801 | 8 | **every step** | 50 | **8967 s** (149.5 min) | 179.3 s | 1.000000 | 0.797 | 47.0 |
| 30K (first attempt) | grid(100, 100) | 30,000 | 29999 / 9801 | 8 | every 5 | 50 | _failed at 102 min_ | _SvdFailed mid-ramp_ | — | — | — |

Bulk physics observation: `mean⟨Z⟩` stays in the `-0.05 … -0.10` band
across all five successful sizes — the bulk is in the same ramp regime
as the chain, just with 2D-flavour entanglement structure underneath.
`max|⟨Z⟩|` grows with N from 1K to 9.5K (0.227 → 0.437 → 0.734),
saturates around 19K (0.737), then nudges up at 30K (0.797). The
slow growth past χ=8 saturation reflects the larger lattice having
more boundary length where the schedule can pull amplitude away from
|+⟩, even when per-bond truncation is binding.

Cumulative discarded weight is **essentially flat** between 19K (46.4)
and 30K (47.0) — the canonicalize-every-step cadence keeps the
truncation regime stable across the size scan rather than letting
errors compound. Per-effective-bond cost holds remarkably constant:
9.5K = 89 µs, 19K = 87 µs, 30K = 90 µs (179.3 s/step ÷ ~2M effective
bonds at 30K). The linear-in-effective-bonds extrapolation predicts
to within ~3%.

The 30K **first attempt** with `canon_every=5` hit `SvdFailed(0)`
partway through the ramp at 102 min wall — accumulated truncation
made a local Θ matrix numerically singular faster than the
every-5-steps canonicalize-and-normalize sweep could clean up. The
**successful retry** with `canon_every=1` (canonicalize every ramp
step) ran to completion in 149.5 min on the same χ=8, with norm²
1.000000 exact and discarded weight in line with 19K. Going forward,
the right trade-off at 2D scale is canonicalize-every-step; the
wall-time penalty is small (50 × ~170 ms canon at 30K ≈ 8.5 s extra
over a 149 min ramp).

`norm² = 1.000000` exactly across all five successful sizes — the
`Ttn::canonicalize_and_normalize` sweep keeps the gauge state stable
and the env contraction bounded, same role it played for the chain
at 1M.

---

## Validation chain

The 2D headline rides on three smaller-scale anchors:

1. **Closed-form correctness, N = 12 chain, lossless χ.** Chain ramp
   validated against `DenseState`,
   `src/ttn/kim_heavy_hex.rs::adiabatic_ramp_chain_matches_dense_lossless`,
   max ⟨Z⟩ err = 1.1e-14.
2. **Closed-form correctness, N = 15 binary tree, lossless χ.** Forces
   `Backend::Tree`, `adiabatic_ramp_binary_tree_matches_dense_lossless`,
   max err = 2.9e-14.
3. **Closed-form correctness, N = 19 heavy-hex grid with non-tree
   edges, lossless χ.** `adiabatic_ramp_heavy_hex_grid_matches_dense_lossless`
   on `grid(3, 2)` with 2 non-tree edges, max ⟨Z⟩ err = 2.94e-13.
   This is the new anchor — same primitive that drives the 2D scale
   runs.

Plus the structural side: `HeavyHexLayout::grid` has 8 unit tests
covering qubit-count formula, tree edge count = N−1, non-tree count =
(R−1)(B−1), through-bridge degree-2, leaf-bridge degree-1, heavy-path
partition, JSON round-trip, and the IBM Eagle 127 byte-stable golden
unchanged.

---

## What had to land first

### `HeavyHexLayout::grid(rows, bridges_per_row)` (commit `c152e0b`)

The Eagle 127 layout is hard-coded; scaling beyond 127 needs a
parametric generator. `grid(R, B)` builds R data rows of width 2B+1
with R−1 bridge rows of B bridges each, using the same spanning-tree
convention as Eagle (horizontal + every "up" + leftmost "down"). The
generator is honestly named `grid` not `heavy_hex` because it lacks
the alternating stagger of true IBM heavy-hex — topologically
equivalent at the gate-application level, but rectangular not hex.

### `Ttn::canonicalize_and_normalize` (commit `4584311`)

Tree analog of `Mps::canonicalize_left_and_normalize`. Underlying
primitive `gauge::canonicalize_to(sites, topology, root)` does a
leaves-to-root QR sweep restoring full canonical form on every site,
O(N · χ³). Without this, FP drift in repeated SVD/QR makes the OC's
Frobenius norm wander, env contractions overflow at scale, and `faer`
SVDs eventually abort mid-ramp on numerically singular Θ matrices —
exactly what we hit on the chain at 1M. The 2D runs canonicalize every
5 ramp steps + once before measurement; cost is sub-second at 1K and
1K at 5K (negligible vs the multi-second-per-step ramp).

### Heavy-hex grid lossless anchor (commit `d0c440a`)

`adiabatic_ramp_heavy_hex_grid_matches_dense_lossless` on a 19-qubit
`grid(3, 2)` motif with 2 non-tree edges, validated against
`DenseState` to FP precision (max ⟨Z⟩ err = 2.94e-13) over 50 ramp
steps. The chain anchor only exercises the 1D fast path; the
binary-tree anchor only exercises `Backend::Tree` without non-tree
edges. This third anchor closes the gap: the actual 2D-annealer case
where `apply_kim_step_heavy_hex` routes both the tree-edge ZZ layer
and the non-tree-edge swap-network ZZ layer through the same gauge
state every step.

---

## Where this lives in the bigger picture

| Path | Topology | Headline | Physics |
|---|---|---|---|
| `Mps + apply_kim_step` | 1D chain | 1M qubits, 18 min | chain area-law, χ=8 holds |
| `Ttn + apply_kim_step_heavy_hex` (this report) | 2D heavy-hex grid | 30K qubits, 150 min | perimeter-law, χ=8 with canon-every-step |
| `ProjectedTtn + analytical bulk` (VQ-110 ch.0) | 2D heavy-hex with sparse defects | 1B qubits, 31 min | sparse defects in clean host |

These are *different abstractions for different physics regimes*, not
competing speed claims:

- **1D chain** — the trivially-tractable case, useful as the
  "everything works at 1M" stress test for the MPS primitives.
- **2D unprojected** — the load-bearing 2D headline. Every qubit is
  tracked explicitly through the schedule, no analytic projection.
  Honest scope is "10⁴ qubits unprojected" not "10⁶+", because the
  perimeter-law entanglement growth saturates χ=8–16 at much smaller
  N than chain area-law does.
- **Projected** — for "sparse defects in clean host" workloads. Bulk
  is analytic; only volatile islands carry full TTN. Lets the 1B
  qubit headline land but is the wrong tool when *every* qubit has a
  non-trivial trajectory.

For the annealer thread the 2D unprojected path is the most directly
relevant: D-Wave problems live on 2D-ish graphs (Pegasus, Zephyr) and
don't have commensurate-host structure to project against. The 30K
unprojected 2D headline is therefore roughly **43× a top-end annealer's
post-embedding logical capacity** (D-Wave Advantage6 today after
minor-embedding ≈ 700 logical variables on dense problems; Advantage2
within the year ≈ 1000), under closed-system unitary at bounded χ.

---

## Honest scope

**This is *not*:**

- A claim that 2D heavy-hex ferromagnetic Ising is hard. The state
  stays near-product for the first many steps and the entanglement
  growth stays bounded for a 50-step ramp at ferromagnetic `J = 1`.
- A claim of beating D-Wave on D-Wave's actual workloads. Real
  annealers exploit thermal fluctuations; closed-system unitary is a
  different physical regime.
- An adiabatic ground-state finder. T = `n_steps · dt` = 5.0 is far
  below the adiabatic limit T ~ 1/Δ_min² for the 2D TFIM gap, so the
  final ⟨Z⟩ values reflect a finite-time ramp, not the problem
  Hamiltonian's ground state. The point is *capability at scale*.
- An IBM-Eagle-style stagger lattice. The `grid(R, B)` layout is a
  regular brick pattern; it's topologically a heavy-hex (degree-3
  data, degree-2 bridges, same spanning-tree machinery) but lacks the
  alternating stagger of true IBM heavy-hex. The Eagle 127 path is
  unchanged and byte-stable.

**What this *is*:** closed-system unitary evolution on a 2D
tree-decomposable Ising graph at 10⁴ scale, with the gate primitives
validated end-to-end at small N against `DenseState`, with the
canonicalize-and-normalize primitive proven necessary at scale on the
chain (1M) and now load-bearing on the 2D path. Roughly an order of
magnitude beyond current annealer hardware's logical capacity.

---

## Reproducibility

```bash
# χ-scaling spike (~3 s)
cargo test --release --test adiabatic_ramp_2d_scale \
    adiabatic_ramp_2d_chi_scaling_spike -- --ignored --nocapture

# 1K calibration (~8 min)
cargo test --release --test adiabatic_ramp_2d_scale \
    adiabatic_ramp_2d_grid_1k_completes -- --ignored --nocapture

# 4K mid-scale (~6 min)
cargo test --release --test adiabatic_ramp_2d_scale \
    adiabatic_ramp_2d_grid_5k_completes -- --ignored --nocapture

# 9.5K mid-scale (~23 min)
cargo test --release --test adiabatic_ramp_2d_scale \
    adiabatic_ramp_2d_grid_10k_completes -- --ignored --nocapture

# 19K mid-stretch (~74 min, canon every step)
cargo test --release --test adiabatic_ramp_2d_scale \
    adiabatic_ramp_2d_grid_19k_completes -- --ignored --nocapture

# 30K headline (~150 min, canon every step)
cargo test --release --test adiabatic_ramp_2d_scale \
    adiabatic_ramp_2d_grid_30k_completes -- --ignored --nocapture
```

All `#[ignore]`-gated. Resident memory peaks at well under 1 GB even
for 10K (the limiting factor is per-step wall-clock, not memory, since
χ=8–16 keeps each site tensor small).

Raw stdout: [`adiabatic_2d_1k_run.log`](adiabatic_2d_1k_run.log),
[`adiabatic_2d_5k_run.log`](adiabatic_2d_5k_run.log),
[`adiabatic_2d_10k_run.log`](adiabatic_2d_10k_run.log),
[`adiabatic_2d_19k_run.log`](adiabatic_2d_19k_run.log),
[`adiabatic_2d_30k_run.log`](adiabatic_2d_30k_run.log) (canon-every-step, success),
[`adiabatic_2d_30k_attempt_failed.log`](adiabatic_2d_30k_attempt_failed.log) (canon_every=5, failed).
