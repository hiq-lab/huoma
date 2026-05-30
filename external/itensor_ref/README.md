# ITensor Cross-References for Huoma

This directory contains Julia/ITensor pipelines that produce
independent numerical references for Huoma benchmarks. All scripts
execute **offline** on a developer machine — Huoma's CI stays
Rust-only.

Two reference runners live here:

| Script                | Benchmark                                | Track |
|-----------------------|------------------------------------------|-------|
| `kim_heavy_hex.jl`    | Tindall N=127 kicked Ising on IBM Eagle  | D.5   |
| `xxz_griffiths.jl`    | Bond-disordered XXZ chain (Griffiths)    | G.2   |

## `kim_heavy_hex.jl` — Tindall N=127 benchmark

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

---

## `xxz_griffiths.jl` — Track G bond-disordered XXZ reference

Independent TEBD reference for the Track G shootout (sin(C/2) vs
uniform-χ vs ε-truncated ITensor at matched budget). One run per
(N, Δ, disorder-realisation, budget-mode) tuple. Driven by a JSON
manifest the Rust shootout harness writes; outputs a JSON next to the
manifest.

### Manifest schema

The Rust driver writes a `*.manifest.json` file like:

```json
{
  "n":             32,
  "delta":         1.0,
  "dt":            0.1,
  "n_steps":       50,
  "initial_index": 1431655765,
  "j_per_bond":    [0.5234, 1.1023, ...],
  "chi_cap":       32,
  "epsilon":       1e-12
}
```

- `initial_index` is the computational-basis state index, **MSB =
  qubit 0** (Huoma convention).
- `j_per_bond[i]` is the coupling on the bond between qubits `i` and
  `i+1`, length `n-1`.
- `chi_cap` is the ITensor `maxdim` bond-dim cap.
- `epsilon` is the ITensor `cutoff` discarded-weight threshold; pass
  `0.0` to disable cutoff-truncation (rely on `chi_cap` alone).

### Output schema

A sibling JSON `*.itensor.json`:

```json
{
  "schema_version":     1,
  "inputs":             { ...echoed manifest... },
  "history":            [[z_0, z_1, ...], ...],   // (n_steps+1) × n
  "linkdims_per_step":  [[1, 2, 4, ...], ...],    // (n_steps+1) × (n-1)
  "wall_seconds":       12.34
}
```

`history[t][q]` is `⟨σ_z⟩_q` at Trotter step `t` (in the **σ_z
convention ±1**, matching Huoma — ITensor's internal `Sz = ±0.5` is
already scaled inside the script). `linkdims_per_step[t]` is the
bond-dim profile after step `t`.

### Usage

```sh
cd external/itensor_ref
julia --project=. -e 'using Pkg; Pkg.instantiate()'   # first time
julia --project=. xxz_griffiths.jl path/to/run.manifest.json
```

The output JSON is written next to the manifest (suffix
`.manifest.json` → `.itensor.json`). The Rust-side G.3 shootout harness
will write the manifest, spawn this Julia subprocess, and read back
the JSON.

### Regeneration policy

Unlike the Tindall reference, the XXZ outputs are **transient** — one
per shootout configuration. They live under
`results/VQ-136/itensor_runs/` rather than under `external/itensor_ref/data/`,
and are regenerated whenever the disorder seed, system size, or
truncation budget changes. The committed artefact is the Julia script
itself, not its outputs.

### Validation

The script is structurally cross-checked against Huoma's
`src/xxz.rs::reference_xxz_run` at small `N` (≤ 10) lossless: both
must reproduce the same per-step ⟨σ_z⟩ trajectory to FP precision for
a given disorder realisation. This anchor is enforced by the G.3
harness on the Rust side; running the Julia script standalone on a
lossless N=10 manifest is the smoke test.
