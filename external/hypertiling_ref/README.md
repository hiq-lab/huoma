# HyperTiling Cross-Reference for Huoma's `{p, q}` Tiling Generator

Independent reference for the regular hyperbolic-tiling combinatorial
invariants produced by `src/hyperbolic.rs::HyperbolicLayout::pq_tiling`,
generated offline using the [HyperTiling](https://pypi.org/project/hypertiling/)
Python package (v1.5.1). Same offline-execution pattern as
`external/itensor_ref/` — Huoma's CI stays Rust-only.

## What is committed

| Path | What |
|---|---|
| `generate_pq_tiling.py` | Script that runs HyperTiling and emits JSON |
| `data/*.hypertiling.json` | Pre-generated golden references for {p,q,r} tuples Huoma tests against |
| `README.md` | This file |

Not committed (gitignored): `.venv/` (the Python virtualenv with
HyperTiling installed).

## Convention note (Huoma vs HyperTiling)

The two implementations use **different definitions** of "BFS shell":

- Huoma's `pq_tiling(p, q, max_face_shells = R)`: enumerate faces by
  **face-edge BFS** — each shell adds the `p` edge-neighbours of the
  previous shell.
- HyperTiling's `HyperbolicTiling(p, q, n)`: enumerate by "layer",
  including *all* faces sharing a vertex with the previous layer.

For `q = 3`, vertex-sharing implies edge-sharing (each vertex meets
exactly three faces, two of which are reached via edges from the
third), so the two conventions coincide and the combinatorial
counts match exactly. For `q ≥ 4`, HyperTiling's per-layer
enumeration is a strict superset of Huoma's per-radius
enumeration. Both are valid tree-decomposable subgraphs of the
underlying {p, q} tiling; they have different per-radius growth
rates.

The Rust-side cross-reference test
(`hyperbolic::tests::cross_reference_against_hypertiling_for_7_3`)
only compares {7, 3} cases. Extending Huoma to vertex-only-neighbour
BFS so the conventions agree across all (p, q) is future work (F.2.c
in the design doc).

## Argument mapping

- HyperTiling `n` parameter = number of layers including the central
  tile.
- Huoma `max_face_shells` R = number of face-edge BFS shells *after*
  the central tile.
- Bridge: `HyperTiling n = Huoma R + 1`.

The script takes Huoma's `R` directly to keep call sites self-
documenting.

## Output schema (`data/{p}_{q}_r{R}.hypertiling.json`)

```json
{
  "schema_version": 1,
  "inputs": {
    "p": 7,
    "q": 3,
    "hypertiling_n": 3,
    "huoma_max_face_shells": 2
  },
  "n_polygons":   29,
  "n_vertices":   112,
  "n_edges":      140,
  "degree_histogram": {"2": 56, "3": 56},
  "tool_version": "1.5.1"
}
```

`degree_histogram` is `{vertex_degree → count}` over all enumerated
vertices, where degree is the number of distinct tiling-edge endpoints
at that vertex (boundary vertices have degree < q in finite-radius
enumerations).

## Usage

Install (one-time, ~30 s):

```sh
cd external/hypertiling_ref
python3 -m venv .venv
.venv/bin/pip install hypertiling
```

Generate / regenerate a reference:

```sh
.venv/bin/python generate_pq_tiling.py <p> <q> <huoma_max_face_shells>

# Examples
.venv/bin/python generate_pq_tiling.py 7 3 0   # central heptagon only
.venv/bin/python generate_pq_tiling.py 7 3 1   # central + 7 neighbours
.venv/bin/python generate_pq_tiling.py 7 3 2   # + face-shell 2
```

## Regeneration policy

The committed `data/*.hypertiling.json` files are the norm. Regenerate
only when:

1. HyperTiling's package behaviour changes in a way that affects
   combinatorial counts (very unlikely — HyperTiling's API is stable),
   **or**
2. New (p, q, R) cases are added to Huoma's cross-reference test.

In both cases, regeneration requires a PR with the updated JSON file(s)
committed alongside the Rust-side test changes. The cross-reference is
under version control specifically so its evolution is visible.
