# Huoma

**Commensurability-guided tree-tensor-network quantum simulator.**

Huoma simulates unitary quantum dynamics on arbitrary tree topologies —
including IBM Eagle/Heron heavy-hex — with adaptive per-edge bond dimension
driven by the sin(C/2) commensurability filter (Hinderink 2026, "The Tilde
Pattern"). It partitions the coupling graph into stable (analytically
solvable) and volatile (must-simulate) regions, builds a Tree Tensor Network
over the volatile islands, and presents unified observables over the entire
graph. This lets it scale to **one million qubits in ~5 seconds** on
circuits where most of the graph is commensurate.

## Status

- **Exact match against Qiskit Aer** at χ = 8: F = 1.000000, TVD = 0.000
  on VQE-like circuits up to N = 28 (268M amplitudes). Regenerate with
  `python experiments/generate_aer_ground_truth.py`.
- **IBM Eagle 127q heavy-hex** running end-to-end: 20 Floquet steps in
  760 ms at χ = 8, depth-1 ⟨Z₆₂⟩ matches Tindall et al. (PRX Quantum 5,
  010308, 2024) to floating-point precision. See
  [PHASE8_REPORT.md](PHASE8_REPORT.md).
- **1D MPS** validated at FP precision against independent dense statevector
  at N = 12 (2.6e-15) and N = 24 (7.4e-16), 6.7× faster than dense at
  N = 24. See [PHASE6_REPORT.md](PHASE6_REPORT.md).
- **1M-qubit projected TTN** via commensurability partitioning, 3 Floquet
  steps in 5.8 s. See `tests/projected_ttn_scale.rs`.
- **178 tests, all green.** Standalone Rust crate, no workspace
  dependencies. Builds with stable Rust ≥ 1.75.

## What Huoma is

- A **tree-tensor-network** simulator for arbitrary tree topologies with
  adaptive per-edge χ allocation via sin(C/2) commensurability scoring
- A **heavy-hex topology engine** with IBM Eagle 127q layout, spanning-tree
  decomposition, and swap-network routing for non-tree gates
- A **commensurability partitioner** that classifies edges as stable (KAM
  tori, analytically representable) vs volatile (must simulate), enabling
  million-qubit scaling by only simulating the volatile islands
- A **1D MPS backend** (the original core) with discarded-weight tracking,
  Bianchi-violation diagnostics, and a water-filling budget allocator
- A reference 1D Floquet kicked-Ising model with an independent dense
  statevector simulator for ground-truth comparisons

## What Huoma is not

- **Not an open-system / Lindblad solver.** Closed-system unitary evolution
  only. No TJM, no MPDO — see ROADMAP for the reasoning.
- **Not a compiler.** It executes circuits, it does not compile them.
- **Not a GPU simulator.** CPU with SIMD via `faer`. Re-evaluate if TTN
  benchmarks at large χ demand it.

## Quickstart

### 1D kicked Ising (the simple path)

```rust
use huoma::kicked_ising::{apply_kim_step, KimParams};
use huoma::mps::Mps;

let n = 50;
let params = KimParams::self_dual();           // J = h_x = π/4, h_z = 0
let chi_per_bond = vec![16; n - 1];

let mut mps = Mps::new(n);
for _ in 0..10 {
    apply_kim_step(&mut mps, params, &chi_per_bond).unwrap();
}
println!("⟨Z_25⟩ = {}", mps.expectation_z(25));
```

### IBM Eagle 127q heavy-hex

```rust
use huoma::ttn::heavy_hex::HeavyHexLayout;
use huoma::ttn::kim_heavy_hex::run_kim_heavy_hex;
use huoma::kicked_ising::KimParams;

let layout = HeavyHexLayout::ibm_eagle_127();
let params = KimParams { j: -std::f64::consts::FRAC_PI_4, h_x: 0.8, h_z: 0.0, dt: 1.0 };
let chi = 8;
let n_steps = 5;

let (mut ttn, history) = run_kim_heavy_hex(&layout, params, chi, n_steps);
println!("⟨Z_62⟩ at depth {n_steps} = {}", history.last().unwrap()[62]);
```

### Adaptive χ allocation

```rust
use huoma::chi_allocation_sinc;

// Per-site natural frequencies (h_x, ω_i, coupling rates, ...)
let frequencies: Vec<f64> = vec![/* ... */];
let total_budget = 500;
let chi = chi_allocation_sinc(&frequencies, total_budget, 2, 16);
// chi[k] = bond dimension for bond k, Σ chi[k] = total_budget exactly
```

## Building

```bash
cargo build --release
cargo test  --release                                          # 177 tests
cargo test  --release --test kim_validation -- --nocapture     # 1D anchor (4 stages)
cargo test  --release --test ttn_tindall_127 -- --nocapture    # Eagle 127q benchmark
cargo test  --release --test projected_ttn_scale -- --nocapture # 100K + 1M scale test
cargo run   --release --example tindall_eagle                  # runnable demo
```

## Crate layout

```
src/
├── lib.rs
├── allocator.rs                    # sin(C/2) + water-filling χ allocator (1D production path)
├── mps.rs                          # 1D MPS backend — balanced canonical, expectation_z, norm_squared
├── channel.rs                      # sin(C/2) commensurability filter + channel map
├── frequency.rs                    # frequency extraction from circuits
├── partition.rs                    # stable / volatile bond classification (1D)
├── reassembly.rs                   # fidelity tracking
├── kicked_ising.rs                 # 1D Floquet KIM model + dense reference simulator
├── bianchi.rs                      # Bianchi-violation diagnostic (gauge consistency check)
├── error.rs
└── ttn/
    ├── mod.rs                      # Ttn struct — tree-tensor-network state vector
    ├── topology.rs                 # general tree graph — edges, neighbours, paths, cut partitions
    ├── heavy_hex.rs                # HeavyHexLayout — IBM Eagle 127q spanning tree + non-tree edges
    ├── site.rs                     # TtnSite — type-erased multi-axis tensor
    ├── contraction.rs              # two-site merge + bipartition SVD on arbitrary tree edges
    ├── gauge.rs                    # orthogonality-center tracking + QR sweeps
    ├── kim_heavy_hex.rs            # heavy-hex KIM Floquet driver (uniform + per-edge χ)
    ├── allocator.rs                # chi_allocation_sinc_tree — sin(C/2) for tree edges
    ├── partition.rs                # partition_tree_adaptive — stable/volatile edge classification
    ├── subtree.rs                  # extract_volatile_islands — connected-component extraction
    ├── boundary.rs                 # BoundaryTensor — analytical ⟨Z⟩ for stable qubits
    ├── projected.rs                # ProjectedTtn — million-qubit scaling via partitioning
    └── dense.rs                    # topology-agnostic statevector reference (test-only)

tests/
├── kim_validation.rs               # 1D anchor: 4 KIM stages (A, B, D, F)
├── ttn_eagle_heavy_hex.rs          # 12 Eagle structural invariant + golden-file tests
├── ttn_tindall_127.rs              # 4 Tindall benchmark tests (smoke, shootout, ⟨Z₆₂⟩ reference)
└── projected_ttn_scale.rs          # 100K + 1M qubit scale smoke tests

examples/
└── tindall_eagle.rs                # runnable Eagle 127q demo

tests/golden/
└── ibm_eagle_127.json              # byte-stable spanning-tree golden file

experiments/
├── generate_aer_ground_truth.py    # Qiskit Aer statevector reference generator
└── angles_*.npy                    # deterministic circuit angles (committed)
```

## Validation evidence

| Test | What it checks | Result |
|---|---|---|
| `accuracy::accuracy_vs_aer` | N=14/18/24/28 vs Qiskit Aer statevector | F = 1.000000, TVD = 0.000 at χ = 8 |
| `kim_validation::stage_a` | N=12 χ=64 vs dense statevector | 2.6e-15 max ⟨Z⟩ error |
| `kim_validation::stage_b` | N=12 χ-sweep, fidelity vs budget | documented |
| `kim_validation::stage_d` | N=24 χ=256 vs 16M-amp dense statevector | 7.4e-16 max ⟨Z⟩ error, 6.7× speedup |
| `kim_validation::stage_f` | N=14/50 disordered KIM: uniform vs sin(C/2) | sin(C/2) ties uniform within 5–30 % at 0.01 ms build cost |
| `ttn_tindall_127::depth_1` | N=127 Eagle ⟨Z₆₂⟩ at depth 1 | exact match to Tindall BP reference |
| `ttn_tindall_127::depth_5_20` | N=127 Eagle ⟨Z₆₂⟩ trajectory | χ=8 overestimates (expected), correct trend |
| `ttn_eagle_heavy_hex::*` | Eagle spanning tree structural invariants | 12 assertions + golden-file byte match |
| `projected_ttn_scale::million` | 1M qubits, 3 Floquet steps | 5.2 s, all ⟨Z⟩ finite and bounded |
| `bench::bench_scaling` | 1D scaling 50 → 100,000 q | < 10 min wall |

## History

Huoma originated as the `arvak-proj` crate inside the Arvak compiler
project and graduated to its own repository in April 2026.

- [BIANCHI_JOURNEY.md](BIANCHI_JOURNEY.md) — Phases 1–5: the failed
  Bianchi-projection truncation correction, what was tried, what worked,
  and the lessons learned.
- [PHASE6_REPORT.md](PHASE6_REPORT.md) — Phase 6: KIM validation +
  discovery of the `apply_zz_fast` corruption bug.
- [PHASE7_REPORT.md](PHASE7_REPORT.md) — Phase 7: matched-budget
  water-filling allocator + boundary blind spot in the discarded-weight
  Jacobian + sin(C/2) restored as the production allocator. Track A closed.
- [PHASE8_REPORT.md](PHASE8_REPORT.md) — Phase 8: Track D complete.
  TTN generalisation, IBM Eagle 127q heavy-hex, swap network, Tindall
  benchmark, sin(C/2) allocator for trees.
- [TRACK_D_DESIGN.md](TRACK_D_DESIGN.md) — Track D design doc (milestones
  D.1–D.5).
- [ROADMAP.md](ROADMAP.md) — forward-looking plan (Tracks A–E + decision
  points).

## Name

Huoma (火马) — "fire horse." Old Chinese name for the comet that the West
calls Halley's Comet. Fast, periodic, returns from far away with a
predictable signature. Fits a Floquet simulator that exploits commensurate
return times.

## License

LGPL-3.0-or-later. See [LICENSE](LICENSE).
