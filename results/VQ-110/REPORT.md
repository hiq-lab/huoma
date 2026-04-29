# VQ-110 — 1B-qubit ProjectedTtn scale run

**Date:** 2026-04-29
**Hardware:** Mac Studio M4 Ultra, 128 GB unified memory, macOS 26.3
**Test:** `tests/projected_ttn_scale.rs::projected_ttn_1b_qubits_completes`
**Outcome:** `test ... ok` — 1,000,000,000 qubits, 3 Floquet steps, 228.91s wall.

---

## Result

| Phase | Wall time |
|---|---|
| Topology generation (10⁹ nodes) | 105.11 s |
| `partition_tree_adaptive` | 99.50 s |
| `ProjectedTtn::new` (build) | 23.23 s |
| 3 × `apply_floquet_step` | 196.67 µs |
| `expectation_z_all` (10⁹ values) | 637.52 ms |
| **Total** | **228.91 s (~3 min 49 s)** |

**Physics output:** 7 volatile qubits out of 10⁹ (10⁻⁶ %), 1 island,
discarded weight = 0.0, all ⟨Z⟩ values finite and in `[-1, 1]`. Volatile
structure (6 volatile edges, 7 volatile qubits) matches the 1M-qubit
reference exactly — same incommensurability physics, scaled to N = 10⁹.

Peak resident set: ~89.5 GB (steady; no swap, no compressor pressure
once the run was given exclusive use of memory).

Raw stdout: [`run.log`](run.log).

---

## What had to be fixed first

The first attempt at this run (2026-04-28) stalled silently after
the topology phase. A second attempt on 2026-04-29 ran for 11 hours of
wall time at 1500 % CPU and never emitted the partition-phase marker.

`sample(1)` against the live process showed every rayon worker stuck
on the same call chain:

```
huoma::ttn::partition::partition_tree_adaptive
  → rayon par_iter
    → huoma::ttn::allocator::edge_sinc_score_local
      → huoma::ttn::allocator::bfs_local
        → _xzm_malloc_large_huge
          → xzm_segment_group_alloc_chunk     (mmap, global lock)
```

`bfs_local` was allocating `vec![false; topology.n_qubits()]` on every
call — a **1 GB zero-initialised heap allocation per call** at
N = 10⁹. With ~10⁹ edges scored and 15 rayon workers contending for
the system allocator's `xzm_segment_group_alloc_chunk` (which routes
allocations of this size through `mmap` under a global lock), the
partition phase was bound by allocator serialisation, not by useful
compute.

The 1M-qubit test never surfaced this for three reasons:

1. At N = 10⁶ the per-call allocation is 1 MB — a size class served
   from per-thread caches with no global lock. The allocator-path
   crossover into `xzm_malloc_large_huge` only happens above ~MB-scale
   chunks.
2. The hidden cost is O(N) per call. At small N it disappears in
   normal partition runtime; at N = 10⁹ it is the dominant cost by
   ~six orders of magnitude.
3. The 1M test only emits an aggregate phase timing
   (`[1M] partition: ..., {:?}`); there is no per-edge breakdown that
   would have flagged a fixed O(N) constant inside an
   O(radius²)-per-edge contract.

### The fix

`src/ttn/allocator.rs::bfs_local` now uses `std::collections::HashSet`
for the visited set. Membership and insertion stay O(1) amortised, but
the data structure scales with the **actual reach of the BFS**
(O(degree<sup>radius</sup>), small) rather than with `topology.n_qubits()`. The
N-sized bitmap is gone; the per-call allocation footprint drops from
N bytes to ~kilobytes.

The `edge_sinc_score_local` doc-comment already specified
"O(radius²) per edge rather than O(N²)" — `bfs_local` was silently
violating that contract.

### Speed-up

| | Before fix | After fix |
|---|---|---|
| `partition_tree_adaptive`, N = 10⁹ | > 11 h, never observed to complete | **99.50 s** |
| `partition_tree_adaptive`, N = 10⁶ | ~120 ms | ~120 ms (unchanged at this scale) |

≥ 400 × at N = 10⁹; the fix is essentially free at N = 10⁶ (the
HashSet cost there is comparable to the bitmap zero).

---

## Validation

All load-bearing tests stay green with the patched `bfs_local`:

- `cargo test --release --lib` — 153 / 153 passed (317.68 s)
- `cargo test --release --test kim_validation` — 4 / 4 passed
  (stage_a, stage_b, stage_d, stage_f, 123.41 s)
- `cargo test --release --test ttn_tindall_127` — 4 / 4 passed
  (depth_5, z62 vs published, two shootouts, 26.07 s)
- `cargo test --release --test projected_ttn_scale projected_ttn_1m_qubits_completes`
  — passes in 0.22 s with identical volatile-edge / volatile-qubit counts as
  pre-fix.

---

## Reproducibility

```bash
cargo test --release --test projected_ttn_scale \
    projected_ttn_1b_qubits_completes -- --ignored --nocapture
```

The test is `#[ignore]`-gated; it consumes ~90 GB of resident memory
and ~4 minutes of wall time on M4 Ultra. Closing other large memory
consumers (Docker Desktop's Linux VM in particular) is recommended.
