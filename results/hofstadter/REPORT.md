# Track F.3 — Hofstadter butterflies (M0/M1 baseline, NOT yet a bulk hyperbolic result)

**Date:** 2026-05-31
**Hardware:** local MacBook
**Code:** `src/magnetic.rs`, `examples/hofstadter_butterfly.rs`

> **Status: in progress.** This is the **M0 baseline** of the
> hyperbolic-Hofstadter pipeline (see the production plan): a
> validated square-lattice butterfly plus a *pre-PBC* hyperbolic
> demonstration. **The hyperbolic `{7,3}` result below is open-boundary
> (OBC) and is NOT a valid bulk spectrum** — on a hyperbolic lattice
> the boundary is an O(1) fraction of all sites, so a finite open
> patch is dominated by edge modes. It is kept here as the **contrast
> baseline for validation gate 3** (PBC-vs-OBC: boundary modes must
> vanish as genus grows). The bulk result requires the closed-surface
> (genus-g, periodic) construction in milestones M1–M3, which is the
> work in progress.
>
> The established hyperbolic-Hofstadter physics being reproduced (not
> claimed as novel) is Stegmaier, Upreti, Thomale & Boettcher, PRL
> **128**, 166402 (2022). No "first"/novelty claims; single-particle
> tight-binding only (`N` = vertex count, not Hilbert-space dimension).

Spectral mode (not time evolution): build the Hermitian magnetic
Hamiltonian `H_ij = -t·exp(iφ_ij)`, diagonalise via
`faer::self_adjoint_eigen`, sweep flux `Φ/Φ₀ ∈ [0, 1]`. Gauge is
selectable (`Gauge::EmbeddingLandau` / `EmbeddingSymmetric`); the
spectrum is gauge-invariant on open patches (unit-tested to 1e-9),
the property the closed-surface construction (M2) will rely on.

## Square lattice 8×8 — canonical validation (Hofstadter 1976)

`results/hofstadter/square_8x8.{csv,png}` — N = 64, 501 flux steps,
32,064 (flux, E) points, 170 ms wall.

Reproduces the three defining features of the textbook butterfly:

| Feature | Expected | Measured |
|---|---|---|
| E → −E symmetry at every flux (bipartite lattice) | exact | **max asymmetry 0.00e0** across all 501 flux values |
| Zero-flux bandwidth (8×8 OBC, → ±4 in thermodynamic limit) | ≈ ±3.76 | **[−3.7588, +3.7588]** |
| Central gap opening at Φ = ½ | nonzero gap at E = 0 | **gap 0.982 between E = ±0.491** |
| Zero-flux spectrum vs analytic `−2t[cos(πn/9)+cos(πm/9)]` | exact | **≤ 1e-10** (unit test) |
| Flux periodicity `E(Φ) = E(Φ+1)` | exact | **≤ 1e-9** (unit test) |

The self-similar fractal structure is visible in the PNG; the
numerical anchors above are the load-bearing validation.

## Hyperbolic {7,3} tiling, face-radius 2 — OBC contrast baseline (NOT bulk)

`results/hofstadter/hyperbolic_7_3_r2.{csv,png}` — N = 112, 501 flux
steps, 56,112 points, 677 ms wall.

A magnetic-flux spectrum on a negatively-curved lattice — the kind of
result the hyperbolic-Hofstadter literature (Stegmaier et al. 2022;
Lenggenhager et al., Nat. Commun. 13, 2022) targets, here produced
from Huoma's pure-Rust {p,q} tiling generator.

| Feature | Expected | Measured |
|---|---|---|
| Non-bipartite spectrum (odd heptagonal cycles forbid E → −E symmetry) | asymmetric | **max asymmetry 0.41** (clearly nonzero) |
| Zero-flux bandwidth | asymmetric about 0 | **[−2.787, +2.509]** |

The asymmetry is the physical fingerprint of the odd cycles in the
{7,3} vertex graph — a square lattice (all even cycles) cannot produce
it, and the contrast between the two figures is the cleanest
demonstration that the hyperbolic geometry is being represented
correctly.

## Honest scope

- This is a **single-particle tight-binding spectrum**, not a
  many-body simulation. The magnetic Hamiltonian is an N×N matrix
  diagonalised densely; N here is the vertex count (64, 112), not a
  Hilbert-space dimension. No MPS/TTN truncation is involved.
- The gauge is Landau on the **embedding chart** (uniform B in
  Euclidean disk coordinates), which is the standard first-order
  convention in the hyperbolic-Hofstadter literature, not the
  hyperbolic-area-form "uniform hyperbolic B field." A future
  refinement could use the area form `dx∧dy/(1−x²−y²)²`.
- The hyperbolic lattice is the finite face-radius-2 truncation
  (112 vertices) with open boundary; boundary states are present and
  not separated from bulk.

## Reproduce

```sh
cargo run --release --example hofstadter_butterfly
cd external/hofstadter_ref
.venv/bin/python plot_butterfly.py \
    ../../results/hofstadter/square_8x8.csv \
    ../../results/hofstadter/square_8x8.png
.venv/bin/python plot_butterfly.py \
    ../../results/hofstadter/hyperbolic_7_3_r2.csv \
    ../../results/hofstadter/hyperbolic_7_3_r2.png
```
