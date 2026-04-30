"""Tests for expectation_z_string and expectation_pauli_string.

These methods are the primitive that the qmmm-fep VQE loop calls per
Pauli term in the chemistry Hamiltonian. Correctness against a dense
brute-force reference is the V1/V2 path we're protecting.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

import huoma_py


# ---- gate definitions (column-major-friendly numpy 2D arrays) ----


def _h() -> np.ndarray:
    s = 1.0 / np.sqrt(2.0)
    return np.array([[s, s], [s, -s]], dtype=np.complex128)


def _x() -> np.ndarray:
    return np.array([[0, 1], [1, 0]], dtype=np.complex128)


def _y() -> np.ndarray:
    return np.array([[0, -1j], [1j, 0]], dtype=np.complex128)


def _z() -> np.ndarray:
    return np.array([[1, 0], [0, -1]], dtype=np.complex128)


def _ry(theta: float) -> np.ndarray:
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _cnot() -> np.ndarray:
    return np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=np.complex128,
    )


# ---- dense reference ----


def _pauli_op_dense(spec: str) -> np.ndarray:
    """Build the full 2^n × 2^n Pauli operator matrix from spec."""
    table = {
        "I": np.eye(2, dtype=np.complex128),
        "X": _x(),
        "Y": _y(),
        "Z": _z(),
    }
    op = np.array([[1.0]], dtype=np.complex128)
    for c in spec:
        op = np.kron(op, table[c])
    return op


def _expectation_dense(sv: np.ndarray, spec: str) -> float:
    op = _pauli_op_dense(spec)
    val = sv.conj() @ op @ sv
    norm = sv.conj() @ sv
    return float((val / norm).real)


# ---- tests ----


class TestExpectationZString:
    def test_single_position_matches_z(self) -> None:
        mps = huoma_py.Mps(4)
        mps.apply_single(1, _h())
        for q in range(4):
            single = mps.expectation_z(q)
            multi = mps.expectation_z_string([q])
            assert abs(single - multi) < 1e-12, f"site {q}: single={single}, multi={multi}"

    def test_bell_zz_is_one(self) -> None:
        mps = huoma_py.Mps(2)
        mps.apply_single(0, _h())
        mps.apply_two_qubit(0, _cnot(), max_bond=4)
        zz = mps.expectation_z_string([0, 1])
        assert abs(zz - 1.0) < 1e-10

    def test_duplicates_cancel(self) -> None:
        mps = huoma_py.Mps(3)
        mps.apply_single(0, _h())
        # Same position twice ⇒ identity ⇒ ⟨I⟩ = 1.
        val = mps.expectation_z_string([1, 1])
        assert abs(val - 1.0) < 1e-12

    def test_oob_position_rejected(self) -> None:
        mps = huoma_py.Mps(3)
        with pytest.raises(ValueError):
            mps.expectation_z_string([5])


class TestExpectationPauliString:
    def test_bell_state_pauli_strings(self) -> None:
        mps = huoma_py.Mps(2)
        mps.apply_single(0, _h())
        mps.apply_two_qubit(0, _cnot(), max_bond=4)
        sv = mps.to_statevector()
        for spec in ["II", "ZZ", "XX", "YY", "ZI", "IZ", "XZ", "ZX", "XY", "YX"]:
            mps_val = mps.expectation_pauli_string(spec)
            dense_val = _expectation_dense(sv, spec)
            assert abs(mps_val - dense_val) < 1e-10, f"spec={spec}: mps={mps_val}, dense={dense_val}"

    def test_random_circuit_pauli_strings(self) -> None:
        n = 4
        mps = huoma_py.Mps(n)
        mps.apply_single(0, _h())
        mps.apply_single(2, _h())
        mps.apply_two_qubit(0, _cnot(), max_bond=8)
        mps.apply_two_qubit(1, _cnot(), max_bond=8)
        mps.apply_two_qubit(2, _cnot(), max_bond=8)
        mps.apply_single(1, _ry(0.7))
        mps.apply_single(3, _ry(1.3))
        sv = mps.to_statevector()

        specs = ["IIII", "ZZZZ", "XYZI", "IZIZ", "XXXX", "YYYY", "ZIXY", "XIIY", "ZIYI", "IIIX"]
        for spec in specs:
            mps_val = mps.expectation_pauli_string(spec)
            dense_val = _expectation_dense(sv, spec)
            assert abs(mps_val - dense_val) < 1e-9, f"spec={spec}: mps={mps_val}, dense={dense_val}"

    def test_invalid_length(self) -> None:
        mps = huoma_py.Mps(3)
        with pytest.raises(ValueError):
            mps.expectation_pauli_string("XX")

    def test_invalid_char(self) -> None:
        mps = huoma_py.Mps(3)
        with pytest.raises(ValueError):
            mps.expectation_pauli_string("XQZ")

    def test_lowercase_accepted(self) -> None:
        mps = huoma_py.Mps(2)
        mps.apply_single(0, _h())
        mps.apply_two_qubit(0, _cnot(), max_bond=4)
        upper = mps.expectation_pauli_string("XX")
        lower = mps.expectation_pauli_string("xx")
        assert abs(upper - lower) < 1e-12


class TestVQEStyleHamiltonian:
    """Simulate what a chemistry-Hamiltonian VQE step actually does:
    sum many Pauli-string expectations against weighted coefficients,
    timing target ≤ 200 µs per Pauli on a 10-qubit ansatz (V1).
    """

    def _build_ansatz(self, n: int) -> "huoma_py.Mps":
        mps = huoma_py.Mps(n)
        # Brick-pattern ansatz: HE-style, one of the simplest VQE families.
        for q in range(n):
            mps.apply_single(q, _ry(0.3 + 0.1 * q))
        for q in range(0, n - 1, 2):
            mps.apply_two_qubit(q, _cnot(), max_bond=16)
        for q in range(n):
            mps.apply_single(q, _ry(0.5 + 0.05 * q))
        for q in range(1, n - 1, 2):
            mps.apply_two_qubit(q, _cnot(), max_bond=16)
        return mps

    def _random_pauli_strings(self, n: int, count: int, seed: int = 42) -> List[str]:
        rng = np.random.default_rng(seed)
        return [
            "".join(rng.choice(list("IXYZ"), size=n).tolist()) for _ in range(count)
        ]

    def test_hamiltonian_matches_dense(self) -> None:
        n = 6
        mps = self._build_ansatz(n)
        sv = mps.to_statevector()

        rng = np.random.default_rng(42)
        terms = self._random_pauli_strings(n, count=50, seed=42)
        coeffs = rng.normal(size=len(terms))

        mps_energy = sum(c * mps.expectation_pauli_string(s) for c, s in zip(coeffs, terms))
        dense_energy = sum(c * _expectation_dense(sv, s) for c, s in zip(coeffs, terms))
        assert abs(mps_energy - dense_energy) < 1e-7, f"mps={mps_energy}, dense={dense_energy}"

    def test_v1_throughput_benchmark(self) -> None:
        """V1 from VQ-109 plan: ≤ 200 µs per Pauli string at 10 qubits.

        Soft check — fails informatively if regressed beyond 5×. Real V1
        is on the production VQE step shape; this is the early-warning
        sanity bench.
        """
        import time

        n = 10
        mps = self._build_ansatz(n)
        terms = self._random_pauli_strings(n, count=200, seed=42)

        # Warm cache.
        mps.expectation_pauli_string(terms[0])

        start = time.perf_counter()
        for s in terms:
            mps.expectation_pauli_string(s)
        elapsed = time.perf_counter() - start
        per_pauli_us = (elapsed / len(terms)) * 1e6
        # Target: ≤ 200 µs. Soft cap at 1000 µs to flag obvious regression.
        assert per_pauli_us < 1000.0, (
            f"per-Pauli latency {per_pauli_us:.1f} µs exceeds soft cap 1000 µs "
            f"(target 200 µs). Possible regression."
        )
        # Print for visibility (pytest -s or -v shows it).
        print(f"\n[V1 bench] 10q, 200 Pauli strings: {per_pauli_us:.1f} µs/Pauli")
