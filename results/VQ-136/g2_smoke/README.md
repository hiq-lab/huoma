# G.2 smoke test — Huoma `reference_xxz_run` vs ITensor `xxz_griffiths.jl`

**Date:** 2026-05-30
**Hardware:** local MacBook
**Software:** Julia 1.12.6, ITensors v0.9.30, ITensorMPS v0.4.1

Lossless cross-check at N=10, Heisenberg (Δ=1), uniform J=1, dt=0.1,
10 Trotter steps, Néel initial state |0101010101⟩. Both
implementations use first-order Trotter with sequential bonds 0..N-2.
No truncation — Huoma uses lossless χ implicitly via dense statevector
evolution; ITensor uses `maxdim=32, cutoff=1e-14` (effectively
lossless at this size).

**Artefacts:**

- `test_xxz.manifest.json` — input parameters (Huoma writes, Julia reads)
- `test_xxz.itensor.json` — Julia output (per-step ⟨σz⟩ + linkdims)

**Result:**

```
Step 10 (final), ⟨σz⟩ per qubit:

  qubit |  ITensor       |  Huoma         |  |diff|
  ------|----------------|----------------|---------
    0   |  0.58014632    |  0.58014632    |  1.0e-13
    1   | -0.22465865    | -0.22465865    |  1.0e-12
    2   |  0.28358830    |  0.28358830    |  1.8e-11
    3   | -0.27844270    | -0.27844270    |  6.3e-11
    4   |  0.27872208    |  0.27872208    |  1.4e-10
    5   | -0.27871168    | -0.27871168    |  1.8e-10
    6   |  0.27865567    |  0.27865567    |  1.5e-10
    7   | -0.28108776    | -0.28108776    |  7.8e-11
    8   |  0.23124547    |  0.23124547    |  2.2e-11
    9   | -0.58945705    | -0.58945705    |  7.2e-12
```

Max |diff| ≈ **1.8e-10** — well below any plausible truncation noise
floor and consistent with FP-rounding drift between two BLAS-path
implementations of mathematically identical algorithms. Initial state
matches exactly (`±1` Néel pattern in both).

**G.2 anchor: passed.** The ITensor reference is structurally
trustworthy as the second leg of the G.3 three-way shootout.

**Reproduce:**

```bash
# Huoma side
cargo run --release --example xxz_cross_check

# ITensor side
julia --project=external/itensor_ref \
      external/itensor_ref/xxz_griffiths.jl \
      results/VQ-136/g2_smoke/test_xxz.manifest.json
```
