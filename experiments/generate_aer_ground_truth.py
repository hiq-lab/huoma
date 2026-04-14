"""
Generate Aer statevector ground truth for Huoma's accuracy.rs validation.

Builds the same VQE-like UCCSD circuit that accuracy.rs implements in MPS:
- HF init: X on first N/2 qubits
- 3 layers of nearest-neighbour CNOT+RY+CNOT + single-qubit RZ
- seed=42 random angles (identical to accuracy.rs and sim_shootout.py)

Outputs:
- experiments/aer_probs_{N}q.npy — probability distribution (|psi|^2)
- experiments/angles_{N}q.npy — the random angles used (so Huoma can reproduce)

Qiskit Aer convention: qubit 0 is the LEAST significant bit in the
statevector index (reversed from Huoma's MSB-first convention). The .npy
files store probabilities in Aer's native (LSB) order; accuracy.rs handles
the bit-reversal on the Rust side.
"""

import numpy as np
from pathlib import Path
from time import perf_counter

def build_circuit_and_angles(n_qubits, n_layers=3, seed=42):
    """Build the Qiskit circuit and collect all angles for the .npy export."""
    from qiskit import QuantumCircuit

    rng = np.random.RandomState(seed)
    qc = QuantumCircuit(n_qubits)
    angles = []

    # HF init
    for i in range(n_qubits // 2):
        qc.x(i)

    for layer in range(n_layers):
        # Even entangling pairs
        for i in range(0, n_qubits - 1, 2):
            theta = rng.uniform(-1.0, 1.0)
            angles.append(theta)
            qc.cx(i, i + 1)
            qc.ry(theta, i + 1)
            qc.cx(i, i + 1)
        # Odd entangling pairs
        for i in range(1, n_qubits - 1, 2):
            theta = rng.uniform(-1.0, 1.0)
            angles.append(theta)
            qc.cx(i, i + 1)
            qc.ry(theta, i + 1)
            qc.cx(i, i + 1)
        # Single-qubit RZ
        for i in range(n_qubits):
            phi = rng.uniform(-np.pi, np.pi)
            angles.append(phi)
            qc.rz(phi, i)

    return qc, np.array(angles, dtype=np.float64)


def run_aer(qc, n_qubits):
    """Run on Aer statevector simulator, return probability distribution."""
    from qiskit_aer import AerSimulator

    qc_copy = qc.copy()
    qc_copy.save_statevector()

    sim = AerSimulator(method='statevector')
    t0 = perf_counter()
    result = sim.run(qc_copy).result()
    dt = perf_counter() - t0

    sv = np.array(result.get_statevector(qc_copy))
    probs = np.abs(sv) ** 2

    return probs, dt


def main():
    out_dir = Path(__file__).parent
    qubit_sizes = [14, 18, 24, 28]

    print(f"Generating Aer ground truth in {out_dir}/")
    print(f"{'N':>4} | {'dim':>12} | {'time':>10} | {'sum(p)':>12} | output")
    print("-" * 70)

    for n in qubit_sizes:
        dim = 2 ** n
        mem_gb = dim * 16 / 1024**3  # complex128
        if mem_gb > 8:
            print(f"  {n:>2} | {dim:>12} | {'SKIP':>10} | {'':>12} | needs {mem_gb:.1f} GB")
            continue

        qc, angles = build_circuit_and_angles(n)
        probs, dt = run_aer(qc, n)

        assert probs.shape == (dim,), f"expected {dim}, got {probs.shape}"
        assert abs(probs.sum() - 1.0) < 1e-10, f"probs sum to {probs.sum()}"

        probs_path = out_dir / f"aer_probs_{n}q.npy"
        angles_path = out_dir / f"angles_{n}q.npy"
        np.save(probs_path, probs)
        np.save(angles_path, angles)

        print(f"  {n:>2} | {dim:>12} | {dt:>9.2f}s | {probs.sum():>12.10f} | {probs_path.name}, {angles_path.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
