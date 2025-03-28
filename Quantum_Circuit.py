import numpy as np

# Identity matrix (I)
I = np.array([[1, 0], [0, 1]], dtype=complex)

class QuantumCircuit:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        # Initialize the quantum state to |0⟩ (all zeros)
        self.state = np.zeros(2**num_qubits, dtype=complex)
        self.state[0] = 1  # Set the |00...0⟩ state to 1

    def print_state(self):
        print("Current Quantum State:")
        for i in range(len(self.state)):
            print(f"Qubit{i//2+1}: {self.state[i]}")

    def apply_gate(self, gate, target_qubit):
        """Applies a single-qubit gate to the quantum state."""
        gate_full = 1
        for i in range(self.num_qubits):
            gate_full = np.kron(gate_full, gate if i == target_qubit else I)
        self.state = np.dot(gate_full, self.state)

    def apply_cnot(self, control_qubit, target_qubit):
        dim = 2**self.num_qubits
        cnot = np.eye(dim, dtype=complex)
        for i in range(dim):
            if (i >> control_qubit) & 1:
                target_flip = i ^ (1 << target_qubit)
                cnot[[i, target_flip]] = cnot[[target_flip, i]]
        self.state = np.dot(cnot, self.state)

    def apply_rx(self, theta, target_qubit):
        """Applies the RX(θ) gate to a qubit."""
        rx = np.array([
            [np.cos(theta / 2), -1j * np.sin(theta / 2)],
            [-1j * np.sin(theta / 2), np.cos(theta / 2)]
        ], dtype=complex)
        self.apply_gate(rx, target_qubit)

    def apply_rz(self, theta, target_qubit):
        """Applies the RZ(θ) gate to a qubit."""
        rz = np.array([
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(1j * theta / 2)]
        ], dtype=complex)
        self.apply_gate(rz, target_qubit)

    def measure(self, shots: int):
        """Measures the quantum state and returns the result."""
        probabilities = dict()
        for state in range(len(self.state)):
            probabilities[bin(state)[2:]] = 0

        for _ in range(shots):
            probability = np.abs(self.state)**2
            result = np.random.choice(len(self.state), p=probability)
            probabilities[bin(result)[2:]] += 1
        
        most_likely_state = 0
        max_count = 0
        for key in probabilities:
            if max_count < probabilities[key]:
                most_likely_state = key
                max_count = probabilities[key]
            print(f"State {key}: {probabilities[key]}")
        return most_likely_state