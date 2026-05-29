# VQ-110 — Results index

Three distinct work streams ran under the VQ-110 ticket. They are
organised into sibling subdirectories so each can be read on its own
terms.

## [`projected_1b/`](projected_1b/) — the original VQ-110 headline

The 1-billion-qubit `ProjectedTtn` run on Mac Studio M4 Ultra. Stable
analytical bulk plus volatile islands, 3 Floquet steps in ~31 min
(wall) at ~4 GB peak RSS.

- [`REPORT.md`](projected_1b/REPORT.md) — Hardware string, toolchain,
  per-phase timing table, comparison to the paper-draft projection.
- [`run.log`](projected_1b/run.log) — Raw stdout of the test run.

## [`annealer_stress_test/`](annealer_stress_test/) — engine stress-test at scale (Track H, closed)

Closed-system adiabatic-ramp engine stress-test. Originally framed as
an annealer-routing-prediction programme; that framing did not survive
review (the underlying engine is not sin(C/2)-structured for couplings-
only annealing problems, and the `grid(R, B)` layout is not Pegasus or
Zephyr — see ROADMAP Track H for the deferral rationale).

What stays load-bearing from this stream is the numerical infrastructure
(`canonicalize_left_and_normalize`, `Ttn::canonicalize_and_normalize`,
`expectation_z_all`) and the stability evidence at 1M chain and 30K 2D
heavy-hex grid.

- [`ANNEALER_THREAD.md`](annealer_stress_test/ANNEALER_THREAD.md) —
  1D chain at 1M qubits. Status banner closes the thread.
- [`ANNEALER_THREAD_2D.md`](annealer_stress_test/ANNEALER_THREAD_2D.md) —
  2D heavy-hex grid up to 30K qubits with non-tree edges via swap-network.
  Includes the canon-every-step cadence finding at 30K (`canon_every=5`
  failed mid-ramp with `SvdFailed`).
- 8 raw stdout logs (`adiabatic_1m_run.log`, `adiabatic_2d_*.log`).

## [`t4_l99a_pocket/`](t4_l99a_pocket/) — pocket+ligand side quest (closed)

Orthogonal side quest: ran the adiabatic-ramp engine on a real
T4-lysozyme L99A pocket topology from 1L63, three ligands (BNZ, IND,
N4B) in three coupling-mode variants each (all-edges, single-closest,
decoy with `j_lig=0`). The decoy control is methodologically broken
(pocket topology differs between ligands, so the decoy does not isolate
ligand-coupling contribution as intended). Treat the data here as engine
demonstration, not as binding-affinity evidence.

- `inputs/T4_L99A_{BNZ,IND,N4B}.pocket.json` — pocket topology + ligand
  coupling edges.
- `realistic_{BNZ,IND,N4B}_{all,one,decoy}.tsv` — per-step discarded
  weight per edge across the three ligands × three modes.
- `pocket_ligand_kollektiv.tsv`, `pocket_ligand_perturbativ.tsv` —
  earlier toy-system comparison runs (5-edge collective vs 1-edge
  perturbative).
