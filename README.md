# Huoma

**Commensurability-guided MPS quantum simulator with Jacobian-driven adaptive bond truncation.**

Huoma is a 1D Matrix Product State simulator for unitary quantum dynamics that
allocates bond dimension adaptively from the structure of the circuit instead
of using a uniform `χ_max` budget. It targets large-N (10³ – 10⁵+ qubits)
Floquet and Trotterised circuits where translation invariance is broken — by
disorder, defects, boundary conditions, or frequency hierarchies — and where
uniform bond allocation wastes compute on regions of the chain that do not
need it.

The simulator originated as `arvak-proj` inside the Arvak compiler project
and graduated to its own repository in April 2026 once the core ideas
(commensurability filter, finite-difference Jacobian sensitivity, balanced
canonical form with discarded-weight tracking) had stabilised.

## Status

- **Validated** at floating-point precision against an independent dense
  statevector reference at N = 12 (full χ) and N = 24 (χ = 256). On N = 24
  Huoma is **6.8× faster** than the dense statevector with FP-limit
  agreement on observables. See [PHASE6_REPORT.md](PHASE6_REPORT.md).
- **39 lib tests + 6 KIM validation stages, all green.** Total release-mode
  test runtime ≈ 4 minutes.
- **Standalone Rust crate**, no workspace dependencies. Builds with stable
  Rust ≥ 1.75.

## What Huoma is

- A 1D MPS simulator with adaptive **per-bond** χ allocation
- A finite-difference Jacobian engine that measures, for any user-supplied
  observable, how each input variable contributes to truncation error at
  each bond — and turns that into a per-bond χ profile
- A discarded-weight tracker on every bond, so truncation error is not
  estimated, it is **counted**
- A commensurability analyser (`channel.rs`) that classifies bonds as
  *stable* (analytically representable) vs *volatile* (must simulate) using
  the `sin(C/2)` filter from the underlying frequency structure
- A reference 1D Floquet kicked-Ising model with an independent dense
  statevector simulator for ground-truth comparisons

## What Huoma is not

- **Not a 2D simulator.** It does not handle heavy-hex, square lattice, or
  any topology beyond 1D nearest-neighbour. A Tree-Tensor-Network
  generalisation that would unlock 2D is on the roadmap but not committed.
- **Not an open-system / Lindblad solver.** No TJM, no MPDO. Closed-system
  unitary evolution only. Open-system support is explicitly out of scope —
  see ROADMAP for the reasoning.
- **Not a compiler.** It executes circuits, it does not compile them. Pair
  with an external compiler if you need routing, basis translation, or
  noise-aware optimisation.

## Quickstart

```rust
use huoma::kicked_ising::{apply_kim_step, KimParams};
use huoma::mps::Mps;

let n = 50;
let n_steps = 10;
let params = KimParams::self_dual();           // J = h_x = π/4, h_z = 0
let chi_per_bond = vec![16; n - 1];

let mut mps = Mps::new(n);
for _ in 0..n_steps {
    apply_kim_step(&mut mps, params, &chi_per_bond).unwrap();
}

// MPS-native expectation value, works at any N
let z_central = mps.expectation_z(n / 2);
println!("⟨Z_{}⟩ = {z_central}", n / 2);
```

## Building

```bash
cargo build --release
cargo test  --release             # 39 lib tests
cargo test  --release --test kim_validation -- --nocapture  # 6 stages
```

## Crate layout

```
src/
├── lib.rs
├── mps.rs                          # core MPS, balanced canonical, expectation_z, norm_squared
├── channel.rs                      # sin(C/2) commensurability filter
├── frequency.rs                    # frequency extraction from circuits
├── partition.rs                    # stable / volatile bond classification
├── reassembly.rs                   # fidelity tracking
├── kicked_ising.rs                 # 1D Floquet KIM model + dense reference simulator
├── finite_difference_jacobian.rs   # FD Jacobian + per-bond χ allocator
├── bianchi.rs                      # Bianchi-violation diagnostic (gauge consistency check)
└── error.rs

tests/
└── kim_validation.rs               # 6-stage Phase 6 validation suite
```

## Validation evidence

| Test | What it checks | Result |
|---|---|---|
| `kim_validation::stage_a` | N=12 χ=64 vs dense statevector | 2.6e-15 max ⟨Z⟩ error |
| `kim_validation::stage_b` | N=12 χ-sweep, monotone behavior | documented |
| `kim_validation::stage_c` | Jacobian vs uniform on homogeneous KIM | honest negative result |
| `kim_validation::stage_d` | N=24 χ=256 vs 16M-amp dense statevector | 7.4e-16 max ⟨Z⟩ error, 6.8× speedup |
| `kim_validation::stage_e` | N=50 negative control (homogeneous) | documented |
| `kim_validation::stage_f` | N=14 + N=50 disordered KIM | Jacobian produces sensible profiles |
| `bench::bench_50q_pipeline` | 50q QKR end-to-end | < 60 s wall |
| `bench::bench_scaling` | scaling 50 → 100 000 q | < 10 min wall to 100 000 q |
| `bench::validate_jacobian_module` | Jacobian API matches reference FD | exact agreement |

## History

Huoma originated as the `arvak-proj` crate inside the Arvak compiler
project. Two design documents preserve the journey:

- [BIANCHI_JOURNEY.md](BIANCHI_JOURNEY.md) — full record of Phases 1–5
  including the failed Bianchi-projection truncation correction, what was
  tried, what worked, what was kept, and the lessons learned. **Required
  reading** before suggesting "let me add a projection step that fixes the
  truncation error" — that path has been thoroughly explored.
- [PHASE6_REPORT.md](PHASE6_REPORT.md) — full record of the Phase 6 KIM
  validation including the discovery of the `apply_zz_fast` corruption
  bug that had been silently producing wrong ZZ results since codebase
  inception.

## Name

Huoma (火马) — "fire horse." Old Chinese name for the comet that the West
calls Halley's Comet. Fast, periodic, returns from far away with a
predictable signature. Fits a Floquet simulator that exploits commensurate
return times.

## License

MIT OR Apache-2.0
