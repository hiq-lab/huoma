# CLAUDE.md — Huoma

Working notes for Claude Code (and any future contributor) operating in this
repo. The README is the public-facing description; this file is the
operating manual.

---

## Definition of Done

**"Done" = output shown, not "code looks right".** Every change must produce
visible evidence of correct behaviour:

| Change type | Proof of done |
|---|---|
| New gate / circuit primitive | `cargo test --release` green, plus a numerical check against the dense reference at small N |
| TTN topology change | The relevant `tests/ttn_*` binary green, plus structural assertions on edge counts / degree |
| Allocator change | A `kim_validation::stage_*` run with measured ⟨Z⟩ error printed (`-- --nocapture`) |
| Performance change | Wall-time numbers from a release run, not "should be faster" |
| Scale claim (1M, 1B, …) | Actual run output captured to a log, not extrapolation |

Forbidden completion phrasing: "compiles", "tests pass" (without naming
which), "should work", "implemented". Always show the number.

---

## Architecture map

Huoma has three execution paths, deliberately separated:

```
1D MPS  ──────────►  src/mps.rs              (the original core, well-tested)
                     src/allocator.rs        (sin(C/2) + water-filling)

TTN     ──────────►  src/ttn/mod.rs          (Ttn struct, gauge, contractions)
                     src/ttn/heavy_hex.rs    (IBM Eagle 127q layout)
                     src/ttn/kim_heavy_hex.rs (heavy-hex Floquet driver)

Projected TTN  ───►  src/ttn/projected.rs    (million-qubit scaling)
                     src/ttn/partition.rs    (stable / volatile classification)
                     src/ttn/boundary.rs     (analytical ⟨Z⟩ on stable bonds)
```

**Decision rule for which path to use:**
- `N ≤ ~100` and 1D nearest-neighbour: **MPS**.
- Heavy-hex / arbitrary tree, `N ≤ ~10⁴`: **TTN**.
- `N ≥ 10⁵`, mostly commensurate: **ProjectedTtn**.

The dense reference (`src/kicked_ising.rs`, `src/ttn/dense.rs`) exists only
for validation at `N ≤ 28`. Never use it as a production path.

---

## Build & test

```bash
cargo build --release
cargo test  --release                                          # 178 tests
cargo test  --release --test kim_validation -- --nocapture     # 1D anchor (4 stages)
cargo test  --release --test ttn_tindall_127 -- --nocapture    # Eagle 127q benchmark
cargo test  --release --test projected_ttn_scale -- --nocapture # 100K + 1M scale
cargo run   --release --example tindall_eagle                  # runnable demo
```

**Ignored / on-demand:**

```bash
# 1B-qubit run on Mac Studio M4 Ultra, 128 GB — VQ-110 in valiant-ops
cargo test --release --test projected_ttn_scale \
    projected_ttn_1b_qubits_completes -- --ignored --nocapture
```

`#[ignore]` is the right gate for any test that needs > 1 min wall, > 8 GB
RSS, or external data. Add a reason string: `#[ignore = "…"]`.

---

## Validation anchors

These are the load-bearing tests. If any of them goes red, stop and
investigate — do not "fix" them by relaxing tolerances.

| Test | Anchor | Tolerance |
|---|---|---|
| `accuracy::accuracy_vs_aer` | Qiskit Aer statevector at N = 14/18/24/28 | F = 1.000000, TVD = 0 at χ = 8 |
| `kim_validation::stage_a` | Dense statevector at N = 12, χ = 64 | max ⟨Z⟩ err ≤ 1e-14 |
| `kim_validation::stage_d` | Dense statevector at N = 24, χ = 256 | max ⟨Z⟩ err ≤ 1e-15 |
| `ttn_tindall_127::depth_1` | Tindall et al. PRX Quantum 5 010308 (2024) | exact at FP precision |
| `ttn_eagle_heavy_hex::*` | Eagle 127q golden file | byte-identical to `tests/golden/ibm_eagle_127.json` |
| `projected_ttn_scale::*` | 100K + 1M qubit pipeline | all ⟨Z⟩ finite, in `[-1, 1]` |

The Aer ground-truth files are *not* checked in (`.gitignore` excludes
`data/`). Regenerate via `python experiments/generate_aer_ground_truth.py`
when needed.

---

## Hard constraints

1. **Standalone crate.** No workspace dependencies. Huoma is consumed by
   downstream projects (Arvak, Garm) but does not import from them. Python
   bindings live in a *separate* wrapper crate, not here — see VQ-109 D1.
2. **Closed-system unitary only.** No Lindblad, no TJM, no MPDO. If a
   request implies open-system dynamics, reject it and refer to ROADMAP
   Track E.
3. **No GPU.** CPU + SIMD via `faer`. Re-evaluate only if a Track D
   benchmark at large χ obviously demands it.
4. **No compiler.** Huoma executes circuits, it does not compile them.
   Routing, basis translation, and gate decomposition belong upstream.
5. **No dense statevector beyond N = 28.** Use Aer if you need that — it's
   not Huoma's job.
6. **`stable` Rust ≥ 1.75.** No nightly features.

---

## Out of scope (do not propose, do not implement)

| Topic | Why | Reference |
|---|---|---|
| TJM / Lindblad / open-system | Wrong abstraction — noise can be information-bearing | ROADMAP Track E |
| GPU tensor contractions | `faer` SIMD already saturates CPU at χ ≤ 64 | ROADMAP Track E |
| Python bindings inside this repo | Wrapper crate is the contract | README, ROADMAP Track E |
| Compiler / routing | Huoma takes circuits, not source | ROADMAP Track E |
| 2D physical lattices beyond TTN-decomposable | Tree structure is the load-bearing assumption | ROADMAP §North star |

---

## Data and external references

- `data/` — gitignored. Internal data, never commit.
- `experiments/angles_*.npy` — committed deterministic circuit angles.
- `experiments/aer_probs_*.npy` — *not* committed, regenerate locally.
- `external/tindall_ref/` — published reference data, committed.
- `external/itensor_ref/` — Julia/ITensor cross-checks, committed.
- `tests/golden/ibm_eagle_127.json` — byte-stable spanning-tree golden.

When adding a new external reference, put the *generator* under `external/`
or `experiments/` and commit it. Never commit the generated artefact unless
it is small (`< 100 KB`) and load-bearing for tests.

---

## Commit and branch hygiene

- Commits follow conventional-commit style: `feat:`, `fix:`, `docs:`,
  `perf:`, `chore:`, `test:`, `refactor:`.
- One logical change per commit. If a commit message needs "and", split it.
- Never auto-commit. Daniel reviews numbers before anything lands on `main`.
- Scale runs (1M+) commit on a feature branch (`huoma/scale-*`) with a
  results report under `results/VQ-XXX/REPORT.md`, then PR.
- LGPL-3.0-or-later. Do not add files under incompatible licences.

---

## Operational tickets

Cross-repo work is tracked in `~/Projects/valiant-ops/board.yaml` as `VQ-…`
tickets. Currently relevant:

- **VQ-109** — GlueQM-α, depends on a Huoma↔OpenMM PyO3 bridge in a *separate*
  crate (`huoma-py`).
- **VQ-110** — 1B-qubit ProjectedTtn run on Mac Studio M4 Ultra.

When a ticket touches Huoma, link the resulting commit SHA back into the
ticket's `result:` field.
