import cudaq
import time
from quantum_circuits import QuantumCircuitFactory
from timing_utils import PrecisionTimer, TimingResult, time_quantum_benchmark

# Benchmark metadata
BENCHMARK_NAME = 'cudaq_statevector_gpu'
BENCHMARK_DESCRIPTION = 'CUDA Quantum - Statevector (GPU)'

def benchmark_cudaq_statevector_gpu(qubit_range, shots=1024, print_topology=False, circuit_type='ghz',
                                   include_init_time=False):
    """
    Runs CUDA-Q statevector simulations on GPU and measures execution times.
    
    Args:
        qubit_range: Range of qubit counts to test
        shots: Number of shots for each simulation
        print_topology: Whether to print circuit topology information
        circuit_type: Type of circuit to benchmark ('qft', 'ghz', 'vqe', 'qaoa')
        include_init_time: Whether to measure circuit initialization time separately
    """
    if cudaq.has_target("nvidia"):
        cudaq.set_target("nvidia")
        # print("Using NVIDIA GPU for CUDA-Q simulations.")
    else:
        # print("NVIDIA GPU target not available. Using CPU.")
        cudaq.set_target("default")

    results = []
    
    # Print topology information once before benchmarking
    if print_topology:
        print("\n🔍 Circuit Topology Analysis:")
        for nq in qubit_range:
            circuit_obj = QuantumCircuitFactory.create_circuit(circuit_type, nq)
            print(f"\n📊 {circuit_type.upper()} Kernel ({nq} qubits) - {circuit_obj.summary()}")
            circuit_obj.print_topology('cudaq')
        print("\n" + "="*50)
        print("🏃 Starting benchmark timing...")
        print("="*50)

    for num_qubits in qubit_range:
        # print(f"\nSimulating {circuit_type.upper()} circuit with {num_qubits} qubits...")
        
        def create_circuit():
            """Circuit creation and preparation phase."""
            circuit_obj = QuantumCircuitFactory.create_circuit(circuit_type, num_qubits)
            kernel = circuit_obj.get_cudaq_kernel()
            return kernel
        
        def execute_circuit(kernel):
            """Circuit execution phase."""
            result = cudaq.sample(kernel, shots_count=shots)
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
    qubit_counts = [10, 20, 30, 40, 50] 
    cudaq_results = benchmark_cudaq_statevector_gpu(qubit_counts)
    
    print("\n--- CUDA-Q Statevector GPU Benchmark Results ---")
    for nq, timing_result in zip(qubit_counts, cudaq_results):
        print(f"Qubits: {nq}, Total: {timing_result.total_time:.4f}s, "
              f"Init: {timing_result.init_time:.4f}s, Exec: {timing_result.execution_time:.4f}s")
