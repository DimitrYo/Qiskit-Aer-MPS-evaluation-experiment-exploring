"""
Quantum Circuits Module - Class-Based Implementation

This module provides quantum circuit classes for benchmarking different quantum simulators.
Supports both Qiskit and CUDA-Q implementations with properties and native printing.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod

# Qiskit imports
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    QuantumCircuit = None

# CUDA-Q imports
try:
    import cudaq
    CUDAQ_AVAILABLE = True
except ImportError:
    CUDAQ_AVAILABLE = False
    cudaq = None


# =====================================
# BASE CLASSES
# =====================================

class BaseQuantumCircuit(ABC):
    """Base class for all quantum circuits."""
    
    def __init__(self, num_qubits: int, circuit_type: str, **kwargs):
        self.num_qubits = num_qubits
        self.circuit_type = circuit_type
        self.parameters = kwargs
        self._circuit = None
        self._kernel = None
        
    @property
    @abstractmethod
    def size(self) -> int:
        """Return the number of gates in the circuit."""
        pass
    
    @property
    @abstractmethod
    def depth(self) -> int:
        """Return the depth of the circuit."""
        pass
    
    @property
    @abstractmethod
    def gate_counts(self) -> Dict[str, int]:
        """Return dictionary of gate types and their counts."""
        pass
    
    @abstractmethod
    def get_qiskit_circuit(self) -> Optional[QuantumCircuit]:
        """Return Qiskit circuit representation."""
        pass
    
    @abstractmethod
    def get_cudaq_kernel(self):
        """Return CUDA-Q kernel representation."""
        pass
    
    def print_topology(self, framework: str = 'both'):
        """Print circuit topology using native framework functions."""
        if framework.lower() in ['qiskit', 'both'] and QISKIT_AVAILABLE:
            circuit = self.get_qiskit_circuit()
            if circuit is not None:
                print(f"\n🔗 {self.circuit_type.upper()} Circuit (Qiskit) - {self.num_qubits} qubits:")
                print(f"   Size: {circuit.size()}")
                print(f"   Depth: {circuit.depth()}")
                print(f"   Classical bits: {circuit.num_clbits}")
                
                # Use Qiskit's built-in circuit drawing if possible
                try:
                    print("   Circuit diagram:")
                    print(circuit.draw(output='text', fold=-1))
                except Exception:
                    print("   (Circuit diagram not available)")
                
        if framework.lower() in ['cudaq', 'both'] and CUDAQ_AVAILABLE:
            kernel = self.get_cudaq_kernel()
            if kernel is not None:
                print(f"\n🔗 {self.circuit_type.upper()} Kernel (CUDA-Q) - {self.num_qubits} qubits:")
                print(f"   Estimated size: {self.size}")
                print(f"   Estimated depth: {self.depth}")
                
                # Use CUDA-Q's built-in kernel printing if available
                try:
                    print("   Kernel structure:")
                    print(kernel)
                except Exception:
                    print("   (Kernel structure not directly printable)")
    
    def summary(self) -> str:
        """Return a brief summary of the circuit."""
        return f"{self.circuit_type.upper()}({self.num_qubits}q, {self.size}g, depth={self.depth})"


# =====================================
# SPECIFIC CIRCUIT CLASSES
# =====================================

class QFTCircuit(BaseQuantumCircuit):
    """Quantum Fourier Transform circuit."""
    
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits, 'qft')
        
    @property
    def size(self) -> int:
        """QFT has n H gates + sum of controlled phase gates + swaps."""
        n = self.num_qubits
        h_gates = n
        cp_gates = sum(range(n))  # 0 + 1 + 2 + ... + (n-1)
        swap_gates = n // 2
        return h_gates + cp_gates + swap_gates
    
    @property
    def depth(self) -> int:
        """QFT depth is approximately n."""
        return self.num_qubits
    
    @property
    def gate_counts(self) -> Dict[str, int]:
        """Return gate count breakdown for QFT."""
        n = self.num_qubits
        return {
            'h': n,
            'cp': sum(range(n)),
            'swap': n // 2
        }
    
    def get_qiskit_circuit(self) -> Optional[QuantumCircuit]:
        """Create Qiskit QFT circuit."""
        if not QISKIT_AVAILABLE:
            return None
            
        if self._circuit is None:
            qc = QuantumCircuit(self.num_qubits)
            for j in range(self.num_qubits):
                qc.h(j)
                for k in range(j + 1, self.num_qubits):
                    qc.cp(np.pi / 2**(k - j), k, j)
            for j in range(self.num_qubits // 2):
                qc.swap(j, self.num_qubits - 1 - j)
            self._circuit = qc
        return self._circuit
    
    def get_cudaq_kernel(self):
        """Create CUDA-Q QFT kernel using RZ gates instead of problematic CP gates."""
        if not CUDAQ_AVAILABLE:
            return None
            
        if self._kernel is None:
            kernel = cudaq.make_kernel()
            qubits = kernel.qalloc(self.num_qubits)
            
            # QFT implementation using RZ gates to avoid CP gate issues
            for j in range(self.num_qubits):
                kernel.h(qubits[j])
                for k in range(j + 1, self.num_qubits):
                    # Implement controlled phase using CNOT and RZ gates
                    # This is equivalent to CP(angle) gate but avoids the problematic CP method
                    angle = np.pi / (2**(k - j))
                    kernel.cx(qubits[k], qubits[j])
                    kernel.rz(angle, qubits[j])
                    kernel.cx(qubits[k], qubits[j])
            
            # Swap qubits to reverse the order
            for j in range(self.num_qubits // 2):
                kernel.swap(qubits[j], qubits[self.num_qubits - 1 - j])
            
            kernel.mz(qubits)
            self._kernel = kernel
        return self._kernel


class GHZCircuit(BaseQuantumCircuit):
    """Greenberger-Horne-Zeilinger state circuit."""
    
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits, 'ghz')
        
    @property
    def size(self) -> int:
        """GHZ has 1 H gate + (n-1) CNOT gates."""
        return 1 + (self.num_qubits - 1)
    
    @property
    def depth(self) -> int:
        """GHZ depth is 2 (H then CNOT chain)."""
        return 2
    
    @property
    def gate_counts(self) -> Dict[str, int]:
        """Return gate count breakdown for GHZ."""
        return {
            'h': 1,
            'cx': self.num_qubits - 1
        }
    
    def get_qiskit_circuit(self) -> Optional[QuantumCircuit]:
        """Create Qiskit GHZ circuit."""
        if not QISKIT_AVAILABLE:
            return None
            
        if self._circuit is None:
            qc = QuantumCircuit(self.num_qubits)
            qc.h(0)  # Hadamard on first qubit
            for i in range(self.num_qubits - 1):
                qc.cx(i, i + 1)  # Chain of CNOT gates
            self._circuit = qc
        return self._circuit
    
    def get_cudaq_kernel(self):
        """Create CUDA-Q GHZ kernel."""
        if not CUDAQ_AVAILABLE:
            return None
            
        if self._kernel is None:
            kernel = cudaq.make_kernel()
            qubits = kernel.qalloc(self.num_qubits)
            
            kernel.h(qubits[0])  # Hadamard on first qubit
            for i in range(self.num_qubits - 1):
                kernel.cx(qubits[i], qubits[i + 1])  # Chain of CNOT gates
            
            kernel.mz(qubits)
            self._kernel = kernel
        return self._kernel


class VQECircuit(BaseQuantumCircuit):
    """Variational Quantum Eigensolver ansatz circuit."""
    
    def __init__(self, num_qubits: int, depth: int = 2, params: Optional[List[float]] = None):
        super().__init__(num_qubits, 'vqe', depth=depth, params=params)
        self.depth_param = depth
        self.params = params or np.random.uniform(0, 2*np.pi, 2 * num_qubits * depth)
        
    @property
    def size(self) -> int:
        """VQE has 2*n*depth rotation gates + (depth-1)*n entangling gates."""
        rotation_gates = 2 * self.num_qubits * self.depth_param
        entangling_gates = max(0, (self.depth_param - 1) * self.num_qubits)
        return rotation_gates + entangling_gates
    
    @property
    def depth(self) -> int:
        """VQE depth is approximately 3 * depth_param."""
        return 3 * self.depth_param
    
    @property
    def gate_counts(self) -> Dict[str, int]:
        """Return gate count breakdown for VQE."""
        return {
            'ry': self.num_qubits * self.depth_param,
            'rz': self.num_qubits * self.depth_param,
            'cx': max(0, (self.depth_param - 1) * self.num_qubits)
        }
    
    def get_qiskit_circuit(self) -> Optional[QuantumCircuit]:
        """Create Qiskit VQE circuit."""
        if not QISKIT_AVAILABLE:
            return None
            
        if self._circuit is None:
            qc = QuantumCircuit(self.num_qubits)
            
            param_idx = 0
            for layer in range(self.depth_param):
                # Parameterized single-qubit rotations
                for qubit in range(self.num_qubits):
                    qc.ry(self.params[param_idx], qubit)
                    param_idx += 1
                    qc.rz(self.params[param_idx], qubit)
                    param_idx += 1
                
                # Entangling layer (except for the last layer)
                if layer < self.depth_param - 1:
                    for qubit in range(self.num_qubits - 1):
                        qc.cx(qubit, qubit + 1)
                    # Add circular entanglement for better connectivity
                    if self.num_qubits > 2:
                        qc.cx(self.num_qubits - 1, 0)
            
            self._circuit = qc
        return self._circuit
    
    def get_cudaq_kernel(self):
        """Create CUDA-Q VQE kernel."""
        if not CUDAQ_AVAILABLE:
            return None
            
        if self._kernel is None:
            kernel = cudaq.make_kernel()
            qubits = kernel.qalloc(self.num_qubits)
            
            param_idx = 0
            for layer in range(self.depth_param):
                # Parameterized single-qubit rotations
                for qubit_idx in range(self.num_qubits):
                    kernel.ry(self.params[param_idx], qubits[qubit_idx])
                    param_idx += 1
                    kernel.rz(self.params[param_idx], qubits[qubit_idx])
                    param_idx += 1
                
                # Entangling layer
                if layer < self.depth_param - 1:
                    for qubit_idx in range(self.num_qubits - 1):
                        kernel.cx(qubits[qubit_idx], qubits[qubit_idx + 1])
                    if self.num_qubits > 2:
                        kernel.cx(qubits[self.num_qubits - 1], qubits[0])
            
            kernel.mz(qubits)
            self._kernel = kernel
        return self._kernel


class QAOACircuit(BaseQuantumCircuit):
    """Quantum Approximate Optimization Algorithm circuit."""
    
    def __init__(self, num_qubits: int, p: int = 2, gamma: Optional[List[float]] = None, 
                 beta: Optional[List[float]] = None):
        super().__init__(num_qubits, 'qaoa', p=p, gamma=gamma, beta=beta)
        self.p = p
        self.gamma = gamma or np.random.uniform(0, 2*np.pi, p)
        self.beta = beta or np.random.uniform(0, np.pi, p)
        
    @property
    def size(self) -> int:
        """QAOA has n H gates + p*(2*n ZZ + n RX) gates."""
        h_gates = self.num_qubits
        zz_gates = self.p * self.num_qubits * 2  # Each ZZ implemented as CX-RZ-CX
        rx_gates = self.p * self.num_qubits
        return h_gates + zz_gates + rx_gates
    
    @property
    def depth(self) -> int:
        """QAOA depth is approximately 1 + p * 3."""
        return 1 + self.p * 3
    
    @property
    def gate_counts(self) -> Dict[str, int]:
        """Return gate count breakdown for QAOA."""
        return {
            'h': self.num_qubits,
            'rzz': self.p * self.num_qubits,  # Simplified representation
            'rx': self.p * self.num_qubits,
            'cx': self.p * self.num_qubits * 2  # For implementing RZZ gates
        }
    
    def get_qiskit_circuit(self) -> Optional[QuantumCircuit]:
        """Create Qiskit QAOA circuit."""
        if not QISKIT_AVAILABLE:
            return None
            
        if self._circuit is None:
            qc = QuantumCircuit(self.num_qubits)
            
            # Initial superposition state
            for qubit in range(self.num_qubits):
                qc.h(qubit)
            
            for layer in range(self.p):
                # Cost Hamiltonian (ZZ interactions for MaxCut)
                for i in range(self.num_qubits - 1):
                    qc.rzz(2 * self.gamma[layer], i, i + 1)
                # Add circular coupling for better connectivity
                if self.num_qubits > 2:
                    qc.rzz(2 * self.gamma[layer], self.num_qubits - 1, 0)
                
                # Mixer Hamiltonian (X rotations)
                for qubit in range(self.num_qubits):
                    qc.rx(2 * self.beta[layer], qubit)
            
            self._circuit = qc
        return self._circuit
    
    def get_cudaq_kernel(self):
        """Create CUDA-Q QAOA kernel."""
        if not CUDAQ_AVAILABLE:
            return None
            
        if self._kernel is None:
            kernel = cudaq.make_kernel()
            qubits = kernel.qalloc(self.num_qubits)
            
            # Initial superposition state
            for qubit_idx in range(self.num_qubits):
                kernel.h(qubits[qubit_idx])
            
            for layer in range(self.p):
                # Cost Hamiltonian (ZZ interactions)
                for i in range(self.num_qubits - 1):
                    kernel.cx(qubits[i], qubits[i + 1])
                    kernel.rz(2 * self.gamma[layer], qubits[i + 1])
                    kernel.cx(qubits[i], qubits[i + 1])
                
                # Circular coupling
                if self.num_qubits > 2:
                    kernel.cx(qubits[self.num_qubits - 1], qubits[0])
                    kernel.rz(2 * self.gamma[layer], qubits[0])
                    kernel.cx(qubits[self.num_qubits - 1], qubits[0])
                
                # Mixer Hamiltonian
                for qubit_idx in range(self.num_qubits):
                    kernel.rx(2 * self.beta[layer], qubits[qubit_idx])
            
            kernel.mz(qubits)
            self._kernel = kernel
        return self._kernel


class RandomCircuit(BaseQuantumCircuit):
    """Random quantum circuit for benchmarking."""
    
    def __init__(self, num_qubits: int, depth: int = 10, seed: Optional[int] = None):
        super().__init__(num_qubits, 'random', depth=depth, seed=seed)
        self.depth_param = depth
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            
    @property
    def size(self) -> int:
        """Random circuit size is approximately the specified depth."""
        return self.depth_param
    
    @property
    def depth(self) -> int:
        """Return the specified depth."""
        return self.depth_param
    
    @property
    def gate_counts(self) -> Dict[str, int]:
        """Return estimated gate counts for random circuit."""
        # Rough estimate based on gate probabilities
        single_qubit_prob = 0.7
        estimated_single = int(self.depth_param * single_qubit_prob)
        estimated_two_qubit = self.depth_param - estimated_single
        
        return {
            'single_qubit': estimated_single,
            'two_qubit': estimated_two_qubit,
            'total': self.depth_param
        }
    
    def get_qiskit_circuit(self) -> Optional[QuantumCircuit]:
        """Create random Qiskit circuit."""
        if not QISKIT_AVAILABLE:
            return None
            
        if self._circuit is None:
            if self.seed is not None:
                np.random.seed(self.seed)
                
            qc = QuantumCircuit(self.num_qubits)
            
            single_qubit_gates = ['h', 'x', 'y', 'z', 'rx', 'ry', 'rz']
            two_qubit_gates = ['cx', 'cz', 'swap']
            
            for _ in range(self.depth_param):
                # Choose random gate type
                if np.random.random() < 0.7:  # 70% single-qubit gates
                    gate = np.random.choice(single_qubit_gates)
                    qubit = np.random.randint(self.num_qubits)
                    
                    if gate == 'h':
                        qc.h(qubit)
                    elif gate == 'x':
                        qc.x(qubit)
                    elif gate == 'y':
                        qc.y(qubit)
                    elif gate == 'z':
                        qc.z(qubit)
                    elif gate in ['rx', 'ry', 'rz']:
                        angle = np.random.uniform(0, 2*np.pi)
                        if gate == 'rx':
                            qc.rx(angle, qubit)
                        elif gate == 'ry':
                            qc.ry(angle, qubit)
                        elif gate == 'rz':
                            qc.rz(angle, qubit)
                else:  # 30% two-qubit gates
                    if self.num_qubits > 1:
                        gate = np.random.choice(two_qubit_gates)
                        qubits = np.random.choice(self.num_qubits, 2, replace=False)
                        
                        if gate == 'cx':
                            qc.cx(qubits[0], qubits[1])
                        elif gate == 'cz':
                            qc.cz(qubits[0], qubits[1])
                        elif gate == 'swap':
                            qc.swap(qubits[0], qubits[1])
            
            self._circuit = qc
        return self._circuit
    
    def get_cudaq_kernel(self):
        """Create random CUDA-Q kernel."""
        if not CUDAQ_AVAILABLE:
            return None
            
        if self._kernel is None:
            if self.seed is not None:
                np.random.seed(self.seed)
                
            kernel = cudaq.make_kernel()
            qubits = kernel.qalloc(self.num_qubits)
            
            for _ in range(self.depth_param):
                if np.random.random() < 0.7:  # Single-qubit gates
                    qubit_idx = np.random.randint(self.num_qubits)
                    gate_type = np.random.randint(4)
                    
                    if gate_type == 0:
                        kernel.h(qubits[qubit_idx])
                    elif gate_type == 1:
                        kernel.x(qubits[qubit_idx])
                    elif gate_type == 2:
                        angle = np.random.uniform(0, 2*np.pi)
                        kernel.ry(angle, qubits[qubit_idx])
                    elif gate_type == 3:
                        angle = np.random.uniform(0, 2*np.pi)
                        kernel.rz(angle, qubits[qubit_idx])
                else:  # Two-qubit gates
                    if self.num_qubits > 1:
                        qubit_indices = np.random.choice(self.num_qubits, 2, replace=False)
                        kernel.cx(qubits[qubit_indices[0]], qubits[qubit_indices[1]])
            
            kernel.mz(qubits)
            self._kernel = kernel
        return self._kernel


class GroverSearchCircuit(BaseQuantumCircuit):
    """Grover's Search Algorithm circuit with Phase Oracle and Diffuser."""

    def __init__(self, num_qubits: int, target_state: Optional[str] = None):
        super().__init__(num_qubits, 'grover', target_state=target_state)
        self.target_state = target_state or ('1' * num_qubits)

    @property
    def size(self) -> int:
        return self.num_qubits * 4 + 6

    @property
    def depth(self) -> int:
        return 6

    @property
    def gate_counts(self) -> Dict[str, int]:
        return {'h': self.num_qubits * 2, 'x': self.num_qubits, 'mcx': 2}

    def get_qiskit_circuit(self) -> Optional[QuantumCircuit]:
        if not QISKIT_AVAILABLE:
            return None
        if self._circuit is None:
            qc = QuantumCircuit(self.num_qubits)
            for i in range(self.num_qubits):
                qc.h(i)
            # Oracle
            for i, bit in enumerate(self.target_state):
                if bit == '0':
                    qc.x(i)
            if self.num_qubits > 1:
                qc.h(self.num_qubits - 1)
                qc.mcx(list(range(self.num_qubits - 1)), self.num_qubits - 1)
                qc.h(self.num_qubits - 1)
            for i, bit in enumerate(self.target_state):
                if bit == '0':
                    qc.x(i)
            # Diffuser
            for i in range(self.num_qubits):
                qc.h(i)
                qc.x(i)
            if self.num_qubits > 1:
                qc.h(self.num_qubits - 1)
                qc.mcx(list(range(self.num_qubits - 1)), self.num_qubits - 1)
                qc.h(self.num_qubits - 1)
            for i in range(self.num_qubits):
                qc.x(i)
                qc.h(i)
            from qiskit.compiler import transpile
            self._circuit = transpile(qc, basis_gates=['h', 'x', 'cx', 'rz', 'cp'])
        return self._circuit

    def get_cudaq_kernel(self):
        return None


class BernsteinVaziraniCircuit(BaseQuantumCircuit):
    """Bernstein-Vazirani Algorithm circuit for secret bitstring estimation."""

    def __init__(self, num_qubits: int, secret_string: Optional[str] = None):
        super().__init__(num_qubits, 'bv', secret_string=secret_string)
        self.secret_string = secret_string or ('1' * (num_qubits - 1))

    @property
    def size(self) -> int:
        return self.num_qubits * 2 + len(self.secret_string)

    @property
    def depth(self) -> int:
        return 3

    @property
    def gate_counts(self) -> Dict[str, int]:
        return {'h': self.num_qubits * 2, 'cx': self.secret_string.count('1')}

    def get_qiskit_circuit(self) -> Optional[QuantumCircuit]:
        if not QISKIT_AVAILABLE:
            return None
        if self._circuit is None:
            n = self.num_qubits
            qc = QuantumCircuit(n)
            qc.x(n - 1)
            for i in range(n):
                qc.h(i)
            for i, bit in enumerate(self.secret_string[:n-1]):
                if bit == '1':
                    qc.cx(i, n - 1)
            for i in range(n):
                qc.h(i)
            self._circuit = qc
        return self._circuit

    def get_cudaq_kernel(self):
        return None


# =====================================
# CIRCUIT FACTORY
# =====================================

class QuantumCircuitFactory:
    """Factory class for creating quantum circuits."""
    
    @staticmethod
    def create_circuit(circuit_type: str, num_qubits: int, **kwargs) -> BaseQuantumCircuit:
        """
        Create a quantum circuit of the specified type.
        
        Args:
            circuit_type: Type of circuit ('qft', 'ghz', 'vqe', 'qaoa', 'random', 'grover', 'bv')
            num_qubits: Number of qubits
            **kwargs: Additional parameters for specific circuits
        
        Returns:
            BaseQuantumCircuit: The requested circuit
        """
        circuit_type = circuit_type.lower()
        
        if circuit_type == 'qft':
            return QFTCircuit(num_qubits)
        elif circuit_type == 'ghz':
            return GHZCircuit(num_qubits)
        elif circuit_type == 'vqe':
            return VQECircuit(num_qubits, **kwargs)
        elif circuit_type == 'qaoa':
            return QAOACircuit(num_qubits, **kwargs)
        elif circuit_type == 'random':
            return RandomCircuit(num_qubits, **kwargs)
        elif circuit_type == 'grover':
            return GroverSearchCircuit(num_qubits, **kwargs)
        elif circuit_type == 'bv':
            return BernsteinVaziraniCircuit(num_qubits, **kwargs)
        else:
            raise ValueError(f"Unknown circuit type: {circuit_type}. "
                           f"Available: qft, ghz, vqe, qaoa, random, grover, bv")


# =====================================
# COMPATIBILITY FUNCTIONS
# =====================================

def get_qiskit_circuit(circuit_type: str, num_qubits: int, **kwargs) -> Optional[QuantumCircuit]:
    """
    Compatibility function to get a Qiskit circuit.
    
    Args:
        circuit_type: Type of circuit
        num_qubits: Number of qubits
        **kwargs: Additional parameters
    
    Returns:
        QuantumCircuit: The requested circuit
    """
    circuit = QuantumCircuitFactory.create_circuit(circuit_type, num_qubits, **kwargs)
    return circuit.get_qiskit_circuit()


def get_cudaq_kernel(circuit_type: str, num_qubits: int, **kwargs):
    """
    Compatibility function to get a CUDA-Q kernel.
    
    Args:
        circuit_type: Type of circuit
        num_qubits: Number of qubits
        **kwargs: Additional parameters
    
    Returns:
        CUDA-Q kernel: The requested kernel
    """
    circuit = QuantumCircuitFactory.create_circuit(circuit_type, num_qubits, **kwargs)
    return circuit.get_cudaq_kernel()


def print_circuit_topology(circuit_or_kernel, circuit_type: str, num_qubits: int, 
                          circuit_name: str = "Circuit", framework: str = "both"):
    """
    Compatibility function for printing circuit topology.
    Uses the new class-based approach internally.
    """
    # Create circuit object for topology printing
    circuit_obj = QuantumCircuitFactory.create_circuit(circuit_type, num_qubits)
    
    print(f"\n🔗 {circuit_name}:")
    print(f"   Summary: {circuit_obj.summary()}")
    
    # Use native framework printing
    circuit_obj.print_topology(framework)


def get_circuit_summary(circuit_type: str, num_qubits: int, **kwargs) -> str:
    """
    Get a brief summary string of circuit characteristics.
    
    Args:
        circuit_type: Type of circuit
        num_qubits: Number of qubits
        **kwargs: Additional parameters
    
    Returns:
        Summary string
    """
    circuit = QuantumCircuitFactory.create_circuit(circuit_type, num_qubits, **kwargs)
    return circuit.summary()


# =====================================
# CIRCUIT REGISTRY
# =====================================

AVAILABLE_CIRCUITS = {
    'qft': QFTCircuit,
    'ghz': GHZCircuit,
    'vqe': VQECircuit,
    'qaoa': QAOACircuit,
    'random': RandomCircuit,
    'grover': GroverSearchCircuit,
    'bv': BernsteinVaziraniCircuit
}


def list_available_circuits() -> List[str]:
    """Return list of available circuit types."""
    return list(AVAILABLE_CIRCUITS.keys())


def get_circuit_info(circuit_type: str) -> Dict[str, Any]:
    """
    Get information about a specific circuit type.
    
    Args:
        circuit_type: Type of circuit
    
    Returns:
        Dictionary with circuit information
    """
    if circuit_type not in AVAILABLE_CIRCUITS:
        raise ValueError(f"Unknown circuit type: {circuit_type}")
    
    circuit_class = AVAILABLE_CIRCUITS[circuit_type]
    
    # Create a small example to get properties
    example = circuit_class(4) if circuit_type != 'vqe' else circuit_class(4, depth=1)
    
    return {
        'type': circuit_type,
        'class': circuit_class.__name__,
        'description': circuit_class.__doc__ or "No description available",
        'example_size': example.size,
        'example_depth': example.depth,
        'gate_counts': example.gate_counts,
        'summary': example.summary()
    }
