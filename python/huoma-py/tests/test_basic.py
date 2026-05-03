"""Basic sanity tests for huoma_py bindings.

Run: pytest -v from huoma-py/ after `maturin develop --release`.
"""

from __future__ import annotations

import numpy as np
import pytest

import huoma_py


def _pauli_x() -> np.ndarray:
    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)


def _pauli_z() -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)


def _hadamard() -> np.ndarray:
    h = 1.0 / np.sqrt(2.0)
    return np.array([[h, h], [h, -h]], dtype=np.complex128)


def _cnot() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.complex128,
    )


class TestMpsConstruction:
    def test_constructor(self) -> None:
        mps = huoma_py.Mps(8)
        assert mps.n_qubits == 8

    def test_zero_qubits_rejected(self) -> None:
        with pytest.raises(ValueError):
            huoma_py.Mps(0)

    def test_initial_state_is_all_zero(self) -> None:
        mps = huoma_py.Mps(4)
        # |0⟩ has ⟨Z⟩ = +1
        for q in range(4):
            assert mps.expectation_z(q) == pytest.approx(1.0)
        assert mps.norm_squared() == pytest.approx(1.0)


class TestSingleQubitGates:
    def test_x_flips_z_expectation(self) -> None:
        mps = huoma_py.Mps(3)
        mps.apply_single(1, _pauli_x())
        assert mps.expectation_z(0) == pytest.approx(1.0)
        assert mps.expectation_z(1) == pytest.approx(-1.0)
        assert mps.expectation_z(2) == pytest.approx(1.0)

    def test_hadamard_zeroes_z(self) -> None:
        mps = huoma_py.Mps(2)
        mps.apply_single(0, _hadamard())
        # |+⟩ has ⟨Z⟩ = 0
        assert mps.expectation_z(0) == pytest.approx(0.0, abs=1e-10)

    def test_oob_qubit_index(self) -> None:
        mps = huoma_py.Mps(3)
        with pytest.raises(ValueError):
            mps.apply_single(5, _pauli_x())

    def test_wrong_shape_rejected(self) -> None:
        mps = huoma_py.Mps(3)
        bad = np.eye(3, dtype=np.complex128)
        with pytest.raises(ValueError):
            mps.apply_single(0, bad)


class TestTwoQubitGates:
    def test_bell_state_via_h_cnot(self) -> None:
        mps = huoma_py.Mps(2)
        mps.apply_single(0, _hadamard())
        mps.apply_two_qubit(0, _cnot(), max_bond=4)
        # Bell state (|00⟩ + |11⟩)/√2: ⟨Z₀⟩ = ⟨Z₁⟩ = 0, ⟨Z₀Z₁⟩ = 1
        assert mps.expectation_z(0) == pytest.approx(0.0, abs=1e-10)
        assert mps.expectation_z(1) == pytest.approx(0.0, abs=1e-10)
        assert mps.norm_squared() == pytest.approx(1.0, abs=1e-10)

    def test_bond_dims_grow(self) -> None:
        mps = huoma_py.Mps(4)
        # Start with χ ≈ 1 everywhere
        bd_initial = mps.bond_dims()
        assert all(b == 1 for b in bd_initial)
        # Entangle adjacent pairs
        mps.apply_single(0, _hadamard())
        mps.apply_two_qubit(0, _cnot(), max_bond=4)
        mps.apply_single(2, _hadamard())
        mps.apply_two_qubit(2, _cnot(), max_bond=4)
        bd = mps.bond_dims()
        # Expect non-trivial entanglement on bond 0 and bond 2
        assert bd[0] >= 2
        assert bd[2] >= 2


class TestStatevectorReturn:
    def test_zero_state_statevector(self) -> None:
        mps = huoma_py.Mps(3)
        sv = mps.to_statevector()
        assert sv.shape == (8,)
        assert sv[0] == pytest.approx(1.0 + 0j)
        for i in range(1, 8):
            assert sv[i] == pytest.approx(0.0 + 0j)

    def test_bell_statevector(self) -> None:
        mps = huoma_py.Mps(2)
        mps.apply_single(0, _hadamard())
        mps.apply_two_qubit(0, _cnot(), max_bond=4)
        sv = mps.to_statevector()
        h = 1.0 / np.sqrt(2.0)
        # Bell state: |00⟩ + |11⟩ both have amplitude 1/√2
        assert sv[0] == pytest.approx(h + 0j, abs=1e-10)
        assert sv[3] == pytest.approx(h + 0j, abs=1e-10)
        assert sv[1] == pytest.approx(0.0 + 0j, abs=1e-10)
        assert sv[2] == pytest.approx(0.0 + 0j, abs=1e-10)


class TestChiAllocation:
    def test_uniform_frequencies(self) -> None:
        # Uniform commensurate frequencies → χ should be near chi_min everywhere
        freqs = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        chi = huoma_py.chi_allocation_sinc(freqs, total_budget=16, chi_min=2, chi_max=8)
        assert len(chi) == 4  # n - 1 bonds
        assert sum(chi) <= 16
        for c in chi:
            assert 2 <= c <= 8

    def test_invalid_chi_min_rejected(self) -> None:
        freqs = np.array([1.0, 2.0], dtype=np.float64)
        with pytest.raises(ValueError):
            huoma_py.chi_allocation_sinc(freqs, total_budget=4, chi_min=0, chi_max=4)


class TestRepr:
    def test_repr_contains_n_qubits(self) -> None:
        mps = huoma_py.Mps(5)
        r = repr(mps)
        assert "n_qubits=5" in r
