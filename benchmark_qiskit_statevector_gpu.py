import time
import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from quantum_circuits import QuantumCircuitFactory
from timing_utils import PrecisionTimer, TimingResult, time_quantum_benchmark

# Benchmark metadata
BENCHMARK_NAME = 'qiskit_statevector_gpu'
BENCHMARK_DESCRIPTION = 'Qiskit Aer - Statevector (GPU)'

def benchmark_qiskit_statevector_gpu(qubit_range, shots=1024, print_topology=False, circuit_type='qft',
                                   include_init_time=False):
    """
    Benchmarks statevector simulation performance using Qiskit Aer on GPU.
    
    Args:
        qubit_range: Range of qubit counts to test
        shots: Number of shots for each simulation
        print_topology: Whether to print circuit topology information
        circuit_type: Type of circuit to benchmark ('qft', 'ghz', 'vqe', 'qaoa')
        include_init_time: Whether to measure circuit initialization time separately
    """
    # print("--- Running Qiskit Aer Benchmark (GPU) ---")
    results = []

    simulator = AerSimulator(method='statevector', device='GPU')
    
    # Print topology information once before benchmarking
    if print_topology:
        print("\n🔍 Circuit Topology Analysis:")
        for nq in qubit_range:
            circuit_obj = QuantumCircuitFactory.create_circuit(circuit_type, nq)
            print(f"\n📊 {circuit_type.upper()} Circuit ({nq} qubits) - {circuit_obj.summary()}")
            circuit_obj.print_topology('qiskit')
        print("\n" + "="*50)
        print("🏃 Starting benchmark timing...")
        print("="*50)

    for nqubits in qubit_range:
        # print(f"Testing {nqubits} qubits...", end="", flush=True)
        
        def create_circuit():
            """Circuit creation and preparation phase."""
            circuit_obj = QuantumCircuitFactory.create_circuit(circuit_type, nqubits)
            qc = circuit_obj.get_qiskit_circuit()
            qc.measure_all()
            transpiled_circuit = transpile(qc, simulator)
            return transpiled_circuit
        
        def execute_circuit(transpiled_circuit):
            """Circuit execution phase."""
            result = simulator.run(transpiled_circuit, shots=shots).result()
            return result
        
        # Use high-precision timing with separate init/execution phases
        timing_result = time_quantum_benchmark(
            create_circuit, 
            execute_circuit, 
            include_init_time=include_init_time
        )

        # print(f" Total time: {timing_result.total_time:.4f}s (Init: {timing_result.init_time:.4f}s, Exec: {timing_result.execution_time:.4f}s)")
        results.append(timing_result)

    return results

if __name__ == '__main__':
    qubit_range_to_test = range(10, 21, 2)
    qiskit_results = benchmark_qiskit_statevector_gpu(qubit_range_to_test)

    print("\n--- Qiskit Statevector GPU Benchmark Results ---")
    for nq, timing_result in zip(qubit_range_to_test, qiskit_results):
        print(f"Qubits: {nq}, Total: {timing_result.total_time:.4f}s, "
              f"Init: {timing_result.init_time:.4f}s, Exec: {timing_result.execution_time:.4f}s")
