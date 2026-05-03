# huoma-py

Python bindings for [huoma](https://github.com/hiq-lab/huoma) — TTN/MPS quantum
simulator with sin(C/2) commensurability partitioning.

## Status

**v0.1 — VQ-109 D1 scaffold.** Minimal MPS bindings sufficient for VQE-loop
prototyping. Not published on PyPI; not pushed to crates.io. Local-build
only via `maturin develop`.

## Build

Requires Rust ≥ 1.75, Python ≥ 3.10, maturin ≥ 1.5.

```bash
cd ~/Projects/Huoma/python/huoma-py
python -m venv .venv && source .venv/bin/activate
pip install maturin pytest numpy
maturin develop --release
pytest tests/ -v
```

## API surface (v0.1)

```python
import huoma_py
import numpy as np

mps = huoma_py.Mps(n_qubits=8)
mps.apply_single(q=0, u=hadamard_2x2)
mps.apply_two_qubit(q=0, u=cnot_4x4, max_bond=8)
e = mps.expectation_z(target=0)
sv = mps.to_statevector()              # 2^n complex doubles, n ≲ 28 only
chi = huoma_py.chi_allocation_sinc(frequencies, total_budget=64, chi_min=2, chi_max=16)
```

## Out-of-scope for v0.1

Documented as TODO in `src/lib.rs`:

- `Ttn` and tree topologies (heavy-hex, custom trees) — VQE production runs
  use 1D MPS; tree-topology bindings land if/when needed.
- `expectation_pauli_string(spec)` — needs an `expectation_z_string` primitive
  in huoma core (multi-qubit Z product expectation via single MPS sweep).
  Without it, every Pauli-string in a chemistry Hamiltonian costs a full
  apply-rotation + apply-Pauli + inner-product round-trip from Python, which
  defeats VQE convergence latency targets.

## Why bindings instead of a separate Rust binary

VQE chemistry workflows live in Python (PySCF, OpenFermion, ansatz libraries,
optimizer ecosystem). Re-implementing that stack in Rust is out of scope and
unnecessary. huoma-py is the bridge: gate-application engine in Rust, VQE
loop and chemistry in Python.

## License

LGPL-3.0-or-later, same as huoma.
