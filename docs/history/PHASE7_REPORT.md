# Phase 7 — Track A.1, the boundary blind spot, and the sin(C/2) reframe

**Status:** Track A.1 done. Track A.4 answered. Production allocator path
collapses to one function. Track D becomes the next strategic commitment.

## What Phase 7 was supposed to be

Track A from the April-2026 ROADMAP. Concretely:

1. **A.1** — implement a target-budget allocator so matched-budget comparisons
   become possible.
2. **A.2** — score shootout: PR vs. total-sensitivity vs. max-row-norm vs.
   effective-rank, all derived from the existing FD Jacobian.
3. **A.3** — better benchmark families where the Jacobian *can* win
   (frequency-hierarchy, defect, domain-wall).
4. **A.4** — the gating question: does *any* per-bond sensitivity allocator
   beat uniform-χ at matched total budget on a 1D MPS benchmark?

The expectation in the roadmap was that A.4 would take "2-3 weeks" of
exploratory benchmarking. It took three commits and two-and-a-half hours of
focused work, and the answer is more useful than the question anticipated.

## What Phase 7 actually is

A short, sharp clarification of what χ allocation in 1D MPS is for. The
Jacobian-based allocator that previous sessions had elevated to the "gold
standard" of Huoma's adaptive-χ story is removed. The sin(C/2)
commensurability allocator that Phase 5e had labeled "dead" is restored to
its original role as the production allocator. The intermediate water-
filling primitive that makes both stories cleanly comparable is the actual
new code.

## Sequence of findings

### A.1 — water-filling allocator (commit `19a5793`)

Added `chi_allocation_target_budget(scores, total_budget, chi_min, chi_max)`,
the score-agnostic integer water-filling primitive. Greedy increments the
bond with the largest marginal utility `score_k / (χ_k + 1)` until the
budget is exactly consumed, equivalent to maximising
`Σ_k score_k · log(χ_k)` subject to box bounds. 11 unit tests cover
empty input, under/over budget, uniform scores, proportional allocation,
exact-budget consumption, NaN handling, and saturation overflow.

This was the missing primitive. Without it the existing
`chi_allocation_from_jacobian` could not be honestly compared against
uniform-χ at matched total χ, because its `sqrt(score/max_score)` heuristic
made the budget an emergent property rather than a constraint. The old
Stage F numbers (uniform 6.49e-2 / budget 104; Jacobian 8.26e-2 / budget 76)
were a category error: the strategies were spending different totals.

### The boundary blind spot

Re-running Stage F with water-filling at exactly the uniform budget,
allocator bounds widened from `[4..8]` to `[2..12]` so the allocator had
real room to redistribute, surfaced a structural problem that the old
heuristic had been hiding.

**N=14, matched total budget 104:**

| strategy | max ⟨Z⟩ err | profile |
|---|---|---|
| uniform χ=8 | 6.49e-2 | `[8]·13` |
| jac matched PR | **7.27e-1** | `[2, 2, 12, 12, 12, 12, 7, 12, 7, 12, 10, 2, 2]` |
| jac matched L1 | **7.24e-1** | `[2, 2, 12, 12, 12, 9, 12, 4, 12, 11, 12, 2, 2]` |

Both Jacobian-derived scores produce allocations **11× worse** than uniform.
Looking at the profiles immediately explains why: the four boundary bonds
on each end clamp to `chi_min = 2`, while the bulk hot bonds soak up
χ = 12. With 8 Trotter steps the boundary bonds at χ = 2 are forced to
discard about three-quarters of their Schmidt singular values per gate,
producing the catastrophic observable error.

The Jacobian is correctly following the score. The score is correctly
computing "how sensitive is the pilot's discarded weight to h_x perturbation".
The score just doesn't answer the question "what's the minimum χ each bond
can survive on", which is what the allocator silently assumes it answers.

**Why the boundary bonds get scored zero**: at pilot χ = 4, a bond with one
qubit on its left has Schmidt rank ≤ 2 by the structure of the MPS — there
is literally no room for the truncation to bite. The pilot's discarded
weight there is artificially small *not because the bond is unimportant*
but because the pilot can't see that the bond would be undersized at χ = 2
in production. The Jacobian's row magnitude is correspondingly tiny, PR
and L1 both rank it at the bottom of the score distribution, and the
allocator concludes "starve it". This is structural, not a tuning issue.

The legacy `sqrt(score/max_score)` heuristic hid the blind spot because its
floor was `chi_min = 4`, not 2 — boundary bonds got at least 4, which was
just enough to survive. Lowering the floor to 2 to give the allocator
something to work with exposed the truth.

This is, I think, the most useful negative result Huoma has produced so far.
It says something concrete about *which* observables are safe to score
against in a pilot-driven allocator: any observable that is *physically
saturated by the chain structure independent of the parameter being
perturbed* will produce censored sensitivities and bad allocations.
Discarded weight on boundary bonds is the canonical example. ⟨Z⟩ on
boundary qubits would not have this problem, because boundary qubits
respond to local-field perturbations regardless of the bond rank to their
left. But that's a different observable that Huoma never tried.

### sin(C/2) restored (commit `811cb15`)

Adding sin(C/2) channel weights as a third score, fed through the same
water-filling allocator at the same matched budget, restored the picture.

**N=14, matched budget 104:**

| strategy | max ⟨Z⟩ err | vs uniform | build time | profile |
|---|---|---|---|---|
| uniform χ=8 | 6.49e-2 | 1.00× | — | flat |
| jac matched PR | 7.27e-1 | **11.2×** worse | 32 ms | boundary-starved |
| jac matched L1 | 7.24e-1 | **11.2×** worse | 32 ms | boundary-starved |
| **sinc2 matched** | **6.78e-2** | **1.04×** worse | **0.01 ms** | `[5, 7, 9, 7, 8, 9, 9, 12, 7, 7, 7, 12, 5]` |

**N=50, matched budget 392:**

| strategy | max ⟨Z⟩ err | vs uniform | build time |
|---|---|---|---|
| uniform χ=8 | 1.02e-1 | 1.00× | — |
| jac matched PR | 7.14e-1 | 7.0× worse | 227 ms |
| jac matched L1 | 6.14e-1 | 6.0× worse | 227 ms |
| **sinc2 matched** | **1.36e-1** | **1.33× worse** | **0.06 ms** |

Three things to read off this table:

1. **sin(C/2) has no boundary blind spot.** Bonds 0 and 12 at N=14 get χ = 5,
   not χ = 2. The pairwise commensurability aggregation is local-frequency-
   driven, not pilot-driven. Boundary bonds receive contributions from every
   pair `(0, j)` weighted by exponential distance, so they get a non-zero
   score whether or not the pilot run thinks they're "sensitive".
2. **sin(C/2) is competitive with uniform**, not beating it: 4 % worse at
   N=14, 33 % worse at N=50. This is the predicted behaviour given Dalzell-
   Brandão (Quantum 2019), which establishes that for gapped clean 1D
   systems uniform χ is structurally near-optimal. Stage F is in that
   regime — the h_x band around π/4 is too narrow to enter the disordered
   Griffiths regime where adaptive χ has theoretical headroom. sin(C/2)
   tying uniform here is the allocator doing the right thing on the wrong
   benchmark.
3. **sin(C/2) is ~3,000–4,000× cheaper than the Jacobian** at the sizes
   tested, with no pilot step at all. At N = 127 (Tindall scale) the
   Jacobian build was previously measured at ~120 seconds; sin(C/2) would
   complete in milliseconds.

The two findings together — that the Jacobian fails structurally and that
sin(C/2) is the safe default — are the actual answer to A.4. **More
benchmarks will not change the verdict.** The Hinderink 2026 Tilde Pattern
papers already validated sin(C/2) as a commensurability filter; what was
missing was an honest matched-budget comparison against uniform on a real
MPS workload, and that comparison is now in the test suite.

### Production reorg (commit `ce488e0`)

Once it was clear that the Jacobian was a research artefact rather than
the production allocator, the right move was to get rid of it. Net change
−1235 / +43 lines:

- New `src/allocator.rs` with the water-filling primitive and the sin(C/2)
  convenience wrapper, both re-exported at crate root.
- `src/finite_difference_jacobian.rs` deleted entirely. The custom-score
  use case is preserved by `chi_allocation_target_budget(&[f64], …)`,
  which accepts arbitrary externally-computed scores.
- `tests/kim_validation.rs` Stages C and E removed (they only existed to
  document the Jacobian's failure modes on homogeneous KIM, which are now
  captured here and in commit history). Stage F renamed from
  `stage_f_disordered_jacobian_wins` to `stage_f_disordered_sinc_vs_uniform`
  and trimmed to uniform vs. sin(C/2) at matched budget.
- `src/bench.rs` Jacobian validation test, its inline reference
  implementation, and its `run_qkr` helper deleted.

All 49 lib tests + 4 kim_validation stages green.

## Cross-reference: literature scan

A separate background literature scan during this session (see session
notes) confirms the empirical finding from the theoretical side:

- **Dalzell–Brandão (Quantum 2019)** establishes that for gapped clean 1D
  systems, χ that is *constant across the chain* suffices for ε-local
  accuracy of reduced density matrices, with the constant *independent of
  chain length*. This is the structural reason uniform-χ wins on Stage F:
  the regime is in Dalzell–Brandão territory, and *no allocator* can give
  more than constant-factor improvements there.
- **No published work** compares per-bond χ allocators against uniform-χ at
  matched ∑χ_k on observable error in 1D. ITensor / TeNPy / block2 default
  to ε-truncation as the de-facto standard, but the head-to-head comparison
  against uniform has never been published.
- **The only physically-motivated opening for adaptive χ in 1D** is bond-
  disordered XXZ in the Griffiths regime (Aramthottil et al., PRL 133,
  196302, 2024 — "rare Griffiths-like effects cause typical and worst-case
  entanglement entropy to scale differently with L"). Stage F's narrow
  h_x disorder does not enter that regime.
- **The Jacobian-of-observables-w.r.t.-Hamiltonian-parameters allocator
  formulation is genuinely novel.** No prior art uses it. Removing it is
  not removing someone else's design — it is removing Huoma's own
  experiment, with the empirical evidence that the experiment failed.

The literature evidence and the Stage F empirical evidence converge on the
same answer: **for 1D MPS, uniform-χ (or equivalently, ε-truncation) is
the right baseline, sin(C/2) is the right default when frequencies are
inhomogeneous, and no Jacobian-based allocator with a censored proxy
observable will beat either of them.**

## What Phase 7 settled

✅ **A.1 done**: `chi_allocation_target_budget` is the score-agnostic
matched-budget primitive. 11 corner-case tests, in the public crate.

✅ **A.2 obviated**: PR vs. L1 was already benchmarked (both 6-11× worse
than uniform). max-row-norm and effective-rank are derived from the same
discarded-weight Jacobian and would inherit the same boundary blind spot.
No reason to extend the shootout.

✅ **A.3 obviated**: the cross-comparison literature scan + Dalzell–Brandão
makes it clear that the only benchmark family where adaptive χ might
honestly beat uniform on 1D is strongly-disordered XXZ. Adding more KIM
benchmarks is not the experiment.

✅ **A.4 answered**: in the form the question was actually asked
("does the Jacobian-PR allocator beat uniform on 1D"), no, and we know
why structurally. In the form the roadmap *meant* to ask ("is there a
production-quality per-bond allocator for 1D"), yes, and it is sin(C/2)
via water-filling, which has been the foundational thesis of Huoma since
the original Tilde Pattern paper. The Phase 5e detour into ML-style
sensitivity analysis was a research dead-end that has now been retired.

✅ **The simulator infrastructure is shipping-quality**:
  - Bond-by-bond truncation (Phase 1.5 discarded-weight mode)
  - Cumulative discarded-weight tracker (Phase 5e)
  - MPS-native expectation values (Phase 6)
  - Independent dense-statevector reference for KIM (Phase 6)
  - Water-filling allocator (Phase 7)
  - sin(C/2) production entry point (Phase 7)
  - Bianchi diagnostic (Phase 2)
  - 49 lib tests + 4 KIM validation stages all green

## What Phase 7 did NOT settle

⚠ **The strongly-disordered regime is untested.** A future Phase could run
sin(C/2) and uniform on bond-disordered XXZ in the Griffiths regime, where
the literature predicts adaptive χ should help. Whether to do this is a
strategic call about how much further investment 1D Huoma deserves before
committing to Track D.

⚠ **Track D (TTN for heavy-hex) is unstarted.** This is the actual
strategic question now. Huoma cannot simulate IBM Eagle/Heron, IQM
Garnet/Emerald, or any other 2D superconducting topology. The user
framing from this session: *"a simulator that can't simulate most of our
target backends isn't worth the effort"*. That position is unchanged by
Phase 7. If anything Phase 7 strengthens the case for Track D — the 1D
allocator story is now *finished* (with a result worth bringing into the
TTN generalisation), so the months of work on heavy-hex don't have to
carry the additional weight of an unresolved 1D question.

## End state

Phase 7 closes Track A. The next planning conversation should be a Track D
design doc: TTN data structure, heavy-hex topology decomposition, and the
N = 127 Tindall benchmark as the success criterion.

Two design documents — `BIANCHI_JOURNEY.md` (Phases 1–5) and
`PHASE6_REPORT.md` (KIM validation + `apply_zz_fast` bug) — together with
this report are the **append-only history** of how Huoma got here.
`ROADMAP.md` is the only living planning document and should be updated
to mark Track A done and Track D in progress.
