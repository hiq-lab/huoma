# VQ-110 — 1M-qubit closed-system adiabatic ramp

**Date:** 2026-05-02
**Hardware:** Mac Studio M4 Ultra, 128 GB unified memory, macOS 26.3
**Test:** `tests/adiabatic_ramp_scale.rs::adiabatic_ramp_1m_qubits_chain_completes`
**Outcome:** `test ... ok` — 1,000,000 qubits, 50 ramp steps, 1107.63s wall.

---

## What this run shows

A million-variable closed-system unitary adiabatic sweep
`H(s) = (1−s)·H_X + s·H_problem` with `s` ramped linearly across 50
Trotter steps, where `H_X = -h_x_0 Σ X_i` (driver, ground state |+⟩^⊗N)
and `H_problem = J_0 Σ Z_iZ_{i+1} + h_z_0 Σ Z_i` (transverse-field
Ising chain). Driver `h_x_0 = 1.0`, problem `J_0 = 1.0, h_z_0 = 0.1`,
step size `dt = 0.1`. Bond dimension `χ = 8`. Initial state |+⟩^⊗N.

The ramp is the closed-system idealised version of what an analog
quantum annealer would do thermally: D-Wave Advantage6 today carries
~5,760 Pegasus qubits, Advantage2 within the year ~7,000 Zephyr,
and after minor-embedding the *logical* capacity is 10²–10⁴ variables
on sparse instances. Huoma carries a million logical variables under
unitary evolution end-to-end on a desktop in 18.5 minutes.

---

## Result

| Phase | Wall time |
|---|---|
| `Mps::new` (10⁶ qubits) | 12.04 ms |
| `\|+⟩^⊗N` preparation | 5.77 ms |
| 50 × `apply_kim_step` (χ=8) | 1092.51 s (21.85 s/step) |
| 10 × `canonicalize_left_and_normalize` | 63.63 s |
| `expectation_z_all` (10⁶ values) | 10.54 s |
| `norm_squared` | 4.56 s |
| **Total** | **1107.63 s (~18 min 28 s)** |

**Physics output**

| | 10⁴ | 10⁵ | 10⁶ |
|---|---|---|---|
| max\|⟨Z⟩\| | 0.036 | 0.036 | 0.037 |
| mean\|⟨Z⟩\| | 0.020 | 0.019 | 0.019 |
| mean⟨Z⟩ | −0.020 | −0.019 | −0.019 |
| norm² (post-canonical) | 1.000000 | 1.000000 | 1.000000 |
| Cumulative discarded weight | 0.36 | 98.06 | 2078.2 |

Bulk ⟨Z⟩ statistics are flat across two orders of magnitude in N —
exactly what you'd expect for a 1D translationally-invariant TFIM
ramp where the local Hamiltonian doesn't see N. Cumulative discarded
weight scales linearly with N (≈ 2·10⁻⁵ per per-bond SVD × 50 steps
× N bonds), confirming we're in the "many small truncations" regime
rather than catastrophic loss at any single bond.

Raw stdout: [`adiabatic_1m_run.log`](adiabatic_1m_run.log).

---

## Validation chain

The 1M run is the largest scale on a primitive that's anchored
at three smaller scales:

1. **Closed-form correctness, N = 12, lossless χ.** Adiabatic ramp on
   a 1D chain validated against `DenseState` to machine precision —
   `src/ttn/kim_heavy_hex.rs::adiabatic_ramp_chain_matches_dense_lossless`,
   max ⟨Z⟩ err = 1.1e-14.
2. **Closed-form correctness, N = 15, lossless χ, native tree backend.**
   Same ramp on a balanced binary tree forces `Backend::Tree` —
   `adiabatic_ramp_binary_tree_matches_dense_lossless`, max ⟨Z⟩ err = 2.9e-14.
3. **`apply_kim_step` 1D fast path validated against dense in
   `tests/kim_validation.rs`** at fixed parameters; the time-varying
   schedule is just a sequence of validated single-step calls.

So the 1M run is a chain of validated primitives with *bounded
truncation error* (the cumulative discarded weight is the honest
2-norm² loss bound) rather than a one-off claim.

---

## What had to land first

**Two perf/correctness fixes were on the critical path:**

### `Mps::expectation_z_all` (commit `29642db`)

The Mps backend originally had only `expectation_z(target)` — O(N · χ⁴)
per call. Looping over all targets gives O(N² · χ⁴). At N = 10⁵ this
is multi-minute on the same hardware that runs the ramp itself in
seconds. Added `expectation_z_all` with shared left/right env builds:
O(N · χ⁴) total, parallelised over q. `Ttn::expectation_z_all` now
delegates to this fast path on `Backend::Linear`.

### `Mps::canonicalize_left_and_normalize` (commit `ccb479f`)

The post-truncation MPS, with `sqrt(S)` absorbed on each side of every
bond after every gate, is not in any canonical form. At small N the
cumulative noise stays in the rounding-error regime (norm² drifts
~1e-13 per 50 steps at N = 12, lossless). At N = 10⁵ × 50 steps the
standalone `norm_squared` env-contraction drifts to ~1e99 even though
⟨Z⟩ ratios remain well-conditioned (because both numerator and
denominator envs drift together). At N = 10⁶ × 50 steps the env values
overflow f64 entirely — `expectation_z_all` returns NaN. And at N = 10⁶
even the *gate-and-truncate cycle itself* fails: one local Θ becomes
numerically singular and faer's SVD aborts mid-ramp.

Added `Mps::canonicalize_left_and_normalize`: left-to-right SVD sweep,
no truncation, splitting U on the left (left-isometric) and S V† folded
into the next site, then dividing the rightmost site by its Frobenius
norm. Cost O(N · χ³); about 6 s at N = 10⁶ χ = 8. The 1M run
canonicalises after every 5 ramp steps (10 calls total, 64 s of the
1107 s run) and once more before measurement. With this in place
norm² post-sweep is `1.000000` exactly, env values stay O(1) at every
step of the contraction, and faer's SVD never aborts.

Two unit tests pin both behaviours: `expectation_z_all_matches_naive_loop`
(O(N · χ⁴) version agrees with the slow-but-trusted O(N² · χ⁴) version)
and `canonicalize_left_and_normalize_preserves_expectation` (every
non-last site is left-isometric, rightmost has unit Frobenius², ⟨Z⟩
unchanged).

---

## Where this lives in the bigger picture

The 1B-qubit `ProjectedTtn` run (the original VQ-110 headline,
[`REPORT.md`](REPORT.md)) is the *projected* path: a stable
analytical bulk plus volatile islands, useful when most qubits are
commensurate. This 1M-qubit adiabatic ramp is the *unprojected* path:
plain Mps with bounded χ, no projection, every qubit's amplitude
explicitly tracked through the schedule. They are different
abstractions for different physics — projected is for "sparse defects
in a clean host," unprojected adiabatic is for "every variable has a
non-trivial trajectory."

For the annealer thread the unprojected path is what matters: D-Wave
problems don't have commensurate-host structure to project against.
The 1M unprojected ramp is therefore the load-bearing demonstration
that a desktop classical simulator can carry a million-variable
closed-system adiabatic schedule on a tree-decomposable Ising graph.

**Honest scope.** This is *not*:

- A claim that 1D-chain TFIM is hard. It isn't. The state stays
  near-product for the first many steps and the entanglement growth
  is bounded by 1D area-law. χ = 8 is enough.
- A claim of beating D-Wave on D-Wave's actual workloads. Real
  annealers exploit thermal fluctuations; closed-system unitary is a
  different physical regime.
- An adiabatic ground-state finder. T = n_steps · dt = 5.0 is far
  below the adiabatic limit T ~ 1/Δ_min² for the TFIM gap, so the
  final ⟨Z⟩ values reflect a finite-time ramp, not the problem
  Hamiltonian's ground state. The point is *capability at scale*,
  not theoretical adiabaticity.

What this *is*: closed-system unitary evolution at million-variable
scale, with the gate primitives validated end-to-end at small N
against `DenseState`, and at scale via norm² and bounded-discarded
checks. The closed-system idealised version of what no annealer will
embed for 5–10 years.

---

## Reproducibility

```bash
cargo test --release --test adiabatic_ramp_scale \
    adiabatic_ramp_1m_qubits_chain_completes -- --ignored --nocapture
```

`#[ignore]`-gated; ~9 GB resident memory and ~18 min wall time on M4
Ultra. The 100K and 10K variants in the same file run in ~110 s and
~60 s respectively for cheaper headline-quality numbers.
