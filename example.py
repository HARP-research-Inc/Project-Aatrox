import numpy as np
from scipy.optimize import minimize
from Quantum_Circuit import QuantumCircuit

if __name__ == "__main__":
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    def build_operator(num_qubits, target_qubit, operator):
        """Expands a single-qubit operator to act on the full quantum state."""
        full_op = 1
        for i in range(num_qubits):
            full_op = np.kron(full_op, operator if i == target_qubit else I)
        return full_op

    def apply_qaoa_layer(qc, gamma, beta, cost_hamiltonian):
        # Phase separation (U_C)
        for (i, j, w) in cost_hamiltonian:
            qc.apply_cnot(i, j)
            qc.apply_rz(2 * gamma * w, j)
            qc.apply_cnot(i, j)

        # Mixing (U_B)
        for qubit in range(qc.num_qubits):
            qc.apply_rx(2 * beta, qubit)

    def qaoa_expectation(params, qc, cost_hamiltonian):
        gamma, beta = params
        qc = QuantumCircuit(qc.num_qubits)  # Reset circuit
        for qubit in range(qc.num_qubits):
            qc.apply_rx(np.pi / 2, qubit)  # Hadamard equivalent
        apply_qaoa_layer(qc, gamma, beta, cost_hamiltonian)

        # Compute energy expectation value
        expectation = 0
        for (i, j, w) in cost_hamiltonian:
            expectation += w * qc.state.conj().T @ build_operator(qc.num_qubits, i, X) @ qc.state
        return -np.real(expectation)

    # === MAIN FUNCTION ===
    def run_qaoa(num_qubits, cost_hamiltonian):
        qc = QuantumCircuit(num_qubits)
        
        # Initial params for gamma and beta
        initial_params = [np.pi / 4, np.pi / 4]
        
        # Classical optimizer to minimize the energy
        result = minimize(qaoa_expectation, initial_params, args=(qc, cost_hamiltonian), method='COBYLA')
        
        # Final optimized parameters
        optimal_gamma, optimal_beta = result.x
        print(f"Optimal γ: {optimal_gamma:.4f}, Optimal β: {optimal_beta:.4f}")
        
        # Final QAOA Circuit with optimal parameters
        qc = QuantumCircuit(num_qubits)
        for qubit in range(qc.num_qubits):
            qc.apply_rx(np.pi / 2, qubit)  # Initialize to |+⟩
        apply_qaoa_layer(qc, optimal_gamma, optimal_beta, cost_hamiltonian)

        # Measure results
        results = qc.measure(shots=1000)
        print("Final Measurement Results:", results)

    # === SAMPLE QUBO (Max-Cut Example) ===
    cost_hamiltonian = [
        (0, 1, 1),  # Edge between qubit 0 and 1 with weight 1
        (1, 2, 1),  # Edge between qubit 1 and 2 with weight 1
        (0, 2, 1)   # Edge between qubit 0 and 2 with weight 1
    ]

    run_qaoa(num_qubits=3, cost_hamiltonian=cost_hamiltonian)