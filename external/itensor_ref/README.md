# ITensor Cross-Reference for the Tindall N=127 Benchmark

This directory contains the Julia/ITensor pipeline that produces an
independent numerical reference for the Tindall kicked-Ising benchmark
on IBM Eagle 127q. The pipeline is executed **offline** on a developer
machine — Huoma's CI stays Rust-only.

## Purpose

The ITensor run is Huoma's **second opinion** at N=127. No TTN simulator
at that size has a dense-statevector reference; the comparison asks
whether two independent implementations (Huoma in Rust, ITensor in Julia)
agree on ⟨Z⟩ at depths 5/10/20 to within a few times their individual
truncation noise floors. See `TRACK_D_DESIGN.md` § "ITensor cross-reference"
for the full rationale.

## Prerequisites

- Julia ≥ 1.10
- The `Project.toml` in this directory pins the required packages

## Usage

```sh
cd external/itensor_ref

# First time: instantiate the Julia environment.
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run with default parameters (χ=8, 20 steps, depths 5/10/20).
julia --project=. kim_heavy_hex.jl

# Override parameters via environment variables.
CHI_MAX=16 N_STEPS=20 DEPTHS=5,10,20 julia --project=. kim_heavy_hex.jl
```

Output: `data/z_expectations.csv` with columns `qubit,depth,z`.

## Topology

The script reads `tests/golden/ibm_eagle_127.json` — the same golden file
Huoma's Rust tests use. This guarantees both simulators operate on the
**exact same** spanning tree (the row-major heavy-path decomposition
from PR #4).

## Regeneration policy

The committed `data/z_expectations.csv` is the norm. Regenerate only when:

1. The spanning tree changes (golden file updated), OR
2. ITensor package versions are bumped, OR
3. The Huoma-vs-ITensor assertion fails

In all three cases, regeneration requires a PR with a matching update to
`PHASE8_REPORT.md` explaining the change. Silent regeneration is not
permitted — the reference CSV is under version control for exactly this
reason.

## Huoma-side consumption

`tests/ttn_tindall_127.rs` reads `data/z_expectations.csv` and compares
Huoma's per-qubit ⟨Z⟩ element-wise, with a tolerance derived from the
larger of each side's reported discarded-weight floor. The Rust test
does **not** invoke Julia — it reads the committed CSV directly.
