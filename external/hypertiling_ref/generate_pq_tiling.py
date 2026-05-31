#!/usr/bin/env python3
"""
Generate combinatorial invariants for a regular {p, q} hyperbolic tiling
using HyperTiling, for cross-reference against Huoma's pure-Rust
src/hyperbolic.rs implementation.

Like external/itensor_ref/, this is executed OFFLINE on a developer
machine and the output JSON is committed under data/ as a regression
golden. Huoma's Rust side does not invoke Python at test time.

Usage:
    cd external/hypertiling_ref
    .venv/bin/python generate_pq_tiling.py 7 3 2   # {7,3} face-radius 2

Argument convention bridge:
    - Huoma's `max_face_shells` = R → R+1 layers including central
    - HyperTiling's `n` parameter = layers including central
    - So HyperTiling n = Huoma R + 1.

Output: JSON under data/{p}_{q}_r{R}.hypertiling.json with:
    - schema_version
    - inputs: {p, q, hypertiling_n, huoma_max_face_shells}
    - n_polygons:   number of {p}-gon faces enumerated
    - n_vertices:   number of unique tile vertices after de-duplication
    - degree_histogram: {degree: count}, sorted ascending
    - tool_version: HyperTiling version string
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import hypertiling


# Position-hash tolerance for deduplicating vertices.  Match Huoma's
# `POSITION_HASH_EPS` so a 1-to-1 comparison is meaningful.
EPS = 1e-6


def position_key(z: complex) -> tuple[int, int]:
    """Quantise a Poincaré-disk coordinate to a (i, j) key."""
    return (round(z.real / EPS), round(z.imag / EPS))


def main() -> int:
    if len(sys.argv) != 4:
        print(f"usage: {sys.argv[0]} <p> <q> <huoma_max_face_shells>", file=sys.stderr)
        return 1
    p, q, r = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    if (p - 2) * (q - 2) <= 4:
        print(f"{{p={p}, q={q}}} is not hyperbolic", file=sys.stderr)
        return 1
    if r < 0:
        print("huoma_max_face_shells must be ≥ 0", file=sys.stderr)
        return 1

    # HyperTiling counts layers including the central tile; Huoma counts
    # shells starting at 0 (central only). HyperTiling n = Huoma r + 1.
    n_layers = r + 1

    tiling = hypertiling.HyperbolicTiling(p, q, n_layers)
    n_polygons = len(tiling)

    # Collect vertices of every polygon, deduplicated by position hash.
    vertex_index: dict[tuple[int, int], int] = {}
    polygon_vertex_indices: list[list[int]] = []
    for poly_idx in range(n_polygons):
        verts = tiling.get_vertices(poly_idx)  # ndarray (p,) of complex
        idxs: list[int] = []
        for z_arr in verts:
            z = complex(z_arr)
            key = position_key(z)
            if key not in vertex_index:
                vertex_index[key] = len(vertex_index)
            idxs.append(vertex_index[key])
        polygon_vertex_indices.append(idxs)
    n_vertices = len(vertex_index)

    # Degree distribution from face boundaries (each consecutive vertex
    # pair around a face is a tiling edge).
    edge_set: set[tuple[int, int]] = set()
    for idxs in polygon_vertex_indices:
        m = len(idxs)
        for k in range(m):
            u, v = idxs[k], idxs[(k + 1) % m]
            if u == v:
                continue
            edge_set.add((min(u, v), max(u, v)))
    degree = [0] * n_vertices
    for u, v in edge_set:
        degree[u] += 1
        degree[v] += 1
    histogram = dict(sorted(Counter(degree).items()))

    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{p}_{q}_r{r}.hypertiling.json"
    output = {
        "schema_version": 1,
        "inputs": {
            "p": p,
            "q": q,
            "hypertiling_n": n_layers,
            "huoma_max_face_shells": r,
        },
        "n_polygons": n_polygons,
        "n_vertices": n_vertices,
        "n_edges": len(edge_set),
        "degree_histogram": {str(k): v for k, v in histogram.items()},
        "tool_version": getattr(hypertiling, "__version__", "unknown"),
    }
    out_path.write_text(json.dumps(output, indent=2))
    print(f"wrote {out_path}")
    print(f"  n_polygons={n_polygons}  n_vertices={n_vertices}  n_edges={len(edge_set)}")
    print(f"  degree_histogram={histogram}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
