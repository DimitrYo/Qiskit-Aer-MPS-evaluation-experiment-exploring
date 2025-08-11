import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from timing_utils import TimingResult, average_timing_results, validate_timing_result

def average_benchmark(benchmark_func, qubit_range, repeats=10, warmup=1, desc=None, 
                     return_detailed_timing=False):
    """
    Runs benchmark_func for each qubit count in qubit_range, repeats times, and returns average times.
    Shows progress bar with tqdm.
    
    Args:
        benchmark_func: Function to benchmark
        qubit_range: Range of qubit counts to test
        repeats: Number of timing iterations (default: 10)
        warmup: Number of warm-up iterations that are not counted (default: 1)
        desc: Description for progress bar
        return_detailed_timing: If True, returns TimingResult objects with detailed timing
    """
    avg_times = []
    detailed_results = []
    
    for nq in tqdm(qubit_range, desc=desc):
        # Warm-up iterations (not counted)
        for _ in range(warmup):
            benchmark_func([nq])
        
        # Actual timing iterations
        timing_results = []
        for _ in range(repeats):
            result = benchmark_func([nq])
            
            # Handle both old (float) and new (TimingResult) return formats
            if isinstance(result[0], TimingResult):
                timing_result = validate_timing_result(result[0])
                timing_results.append(timing_result)
            else:
                # Legacy format: assume it's execution time only
                execution_time = max(0.0, float(result[0]))  # Prevent negative times
                timing_result = TimingResult(
                    total_time=execution_time,
                    execution_time=execution_time,
                    init_time=0.0
                )
                timing_results.append(timing_result)
        
        # Calculate averages
        if timing_results:
            avg_timing = average_timing_results(timing_results)
            avg_times.append(avg_timing.total_time)
            detailed_results.append(avg_timing)
        else:
            avg_times.append(0.0)
            detailed_results.append(TimingResult(0.0, 0.0, 0.0))
    
    if return_detailed_timing:
        return detailed_results
    else:
        return avg_times

def print_results_table(qubit_range, labels, values_matrix):
    """
    Prints a formatted table of results for any number of columns.
    """
    col_width = 22
    header = f"{'Qubits':<15}" + "".join([f"{label:<{col_width}}" for label in labels])
    print("\n\n--- Benchmark Results (averaged over multiple runs) ---")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for i, nq in enumerate(qubit_range):
        row = f"{nq:<15}" + "".join([f"{values[i]:<{col_width}.4f}" for values in values_matrix])
        print(row)
    print("=" * len(header))


def print_detailed_timing_table(qubit_range, labels, detailed_results_matrix):
    """
    Prints a detailed timing table showing total, execution, and initialization times.
    
    Args:
        qubit_range: Range of qubit counts
        labels: List of benchmark labels
        detailed_results_matrix: Matrix of TimingResult objects
    """
    col_width = 18
    
    # Create headers for each benchmark (Total/Exec/Init columns)
    headers = []
    for label in labels:
        headers.extend([f"{label[:15]}-Total", f"{label[:15]}-Exec", f"{label[:15]}-Init"])
    
    header_line = f"{'Qubits':<10}" + "".join([f"{h:<{col_width}}" for h in headers])
    
    print("\n\n--- Detailed Timing Results (Total/Execution/Initialization) ---")
    print("=" * len(header_line))
    print(header_line)
    print("-" * len(header_line))
    
    for i, nq in enumerate(qubit_range):
        row_parts = [f"{nq:<10}"]
        
        for results in detailed_results_matrix:
            if i < len(results):
                timing = results[i]
                row_parts.extend([
                    f"{timing.total_time:<{col_width}.4f}",
                    f"{timing.execution_time:<{col_width}.4f}",
                    f"{timing.init_time:<{col_width}.4f}"
                ])
            else:
                row_parts.extend([f"{'N/A':<{col_width}}" for _ in range(3)])
        
        print("".join(row_parts))
    
    print("=" * len(header_line))


def save_timing_results_table(qubit_range, labels, values_matrix, detailed_results_matrix=None, filename_base="benchmark_results"):
    """
    Save all timing results as a single comprehensive numpy file (.npz format).
    
    Args:
        qubit_range: Range of qubit counts
        labels: List of benchmark labels
        values_matrix: Matrix of timing values (total times)
        detailed_results_matrix: Optional matrix of TimingResult objects for detailed timing
        filename_base: Base filename without extension
    """
    # Convert to numpy arrays
    qubit_array = np.array(list(qubit_range))
    timing_matrix = np.array(values_matrix).T  # Transpose so rows are qubits, columns are benchmarks
    
    # Prepare data dictionary for comprehensive save
    data_dict = {
        'qubits': qubit_array,
        'basic_timing': timing_matrix,
        'labels': np.array(labels),
        'metadata': {
            'description': 'Comprehensive quantum benchmark timing results',
            'units': 'seconds',
            'basic_columns': ['qubits'] + labels,
            'timestamp': np.datetime64('now'),
            'has_detailed_timing': False
        }
    }
    
    # Add detailed timing if available
    if detailed_results_matrix and any(results is not None for results in detailed_results_matrix):
        detailed_data = []
        detailed_columns = []
        
        for i, (label, results) in enumerate(zip(labels, detailed_results_matrix)):
            if results is not None:
                total_times = np.array([r.total_time for r in results])
                exec_times = np.array([r.execution_time for r in results])
                init_times = np.array([r.init_time for r in results])
                
                # Store as separate arrays in the data dictionary
                data_dict[f'{label}_total'] = total_times
                data_dict[f'{label}_execution'] = exec_times
                data_dict[f'{label}_initialization'] = init_times
                
                detailed_columns.extend([
                    f"{label}_total",
                    f"{label}_execution", 
                    f"{label}_initialization"
                ])
            else:
                # Fill with zeros if no detailed data available
                zeros = np.zeros(len(qubit_range))
                data_dict[f'{label}_total'] = zeros
                data_dict[f'{label}_execution'] = zeros
                data_dict[f'{label}_initialization'] = zeros
                
                detailed_columns.extend([
                    f"{label}_total",
                    f"{label}_execution",
                    f"{label}_initialization"
                ])
        
        # Update metadata to include detailed timing info
        data_dict['metadata']['has_detailed_timing'] = True
        data_dict['metadata']['detailed_columns'] = detailed_columns
    
    # Save everything in a single .npz file
    output_filename = f"{filename_base}_timing.npz"
    np.savez_compressed(output_filename, **data_dict)
    
    print(f"All timing results saved to '{output_filename}'")
    
    # Print summary of what was saved
    if data_dict['metadata']['has_detailed_timing']:
        print(f"  - Basic timing data for {len(labels)} benchmarks")
        print(f"  - Detailed timing breakdown (total/execution/initialization)")
        print(f"  - {len(qubit_array)} qubit values: {qubit_array[0]} to {qubit_array[-1]}")
    else:
        print(f"  - Basic timing data for {len(labels)} benchmarks")
        print(f"  - {len(qubit_array)} qubit values: {qubit_array[0]} to {qubit_array[-1]}")


def load_timing_results(filename):
    """
    Load timing results from a saved .npz file.
    
    Args:
        filename: Path to the .npz file
    
    Returns:
        dict: Dictionary containing all timing data and metadata
        
    Example:
        data = load_timing_results('benchmark_results_20250805_183000_timing.npz')
        print("Qubits:", data['qubits'])
        print("Labels:", data['labels'])
        print("Basic timing shape:", data['basic_timing'].shape)
        if data['metadata']['has_detailed_timing']:
            for label in data['labels']:
                print(f"{label} total times:", data[f'{label}_total'])
    """
    with np.load(filename, allow_pickle=True) as data:
        result = dict(data)
        # Convert metadata back to dictionary if it was saved as numpy array
        if 'metadata' in result:
            result['metadata'] = result['metadata'].item()
        return result


def print_timing_data_summary(filename):
    """
    Print a summary of timing data from a saved .npz file.
    
    Args:
        filename: Path to the .npz file
    """
    try:
        data = load_timing_results(filename)
        
        print(f"\n=== Timing Data Summary: {filename} ===")
        print(f"Timestamp: {data['metadata']['timestamp']}")
        print(f"Description: {data['metadata']['description']}")
        print(f"Units: {data['metadata']['units']}")
        print(f"Qubit range: {data['qubits'][0]} to {data['qubits'][-1]} ({len(data['qubits'])} points)")
        print(f"Benchmarks: {', '.join(data['labels'])}")
        
        if data['metadata']['has_detailed_timing']:
            print("Detailed timing: Available (total/execution/initialization)")
            print("\nDetailed timing columns:")
            for col in data['metadata']['detailed_columns']:
                print(f"  - {col}")
        else:
            print("Detailed timing: Not available")
        
        print(f"\nBasic timing data shape: {data['basic_timing'].shape}")
        print("Basic timing preview (first 3 rows):")
        for i, nq in enumerate(data['qubits'][:3]):
            row_data = [f"{nq:2d}"] + [f"{data['basic_timing'][i,j]:.4f}" for j in range(data['basic_timing'].shape[1])]
            print(f"  {' | '.join(row_data)}")
        
        if len(data['qubits']) > 3:
            print("  ...")
            
    except Exception as e:
        print(f"Error loading timing data: {e}")


def plot_results(qubit_range, labels, values_matrix, filename="benchmark_results.png", colors=None, markers=None):
    """
    Plots benchmark results for any number of columns.
    
    Args:
        qubit_range: Range of qubit counts
        labels: List of benchmark labels  
        values_matrix: Matrix of timing values
        filename: Output filename for the plot
        colors: Optional list of colors (defaults to matplotlib color cycle)
        markers: Optional list of markers (defaults to standard markers)
    """
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        # Fallback if seaborn style is not available
        plt.style.use('default')
        
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Default colors from matplotlib's color cycle (hex format for reliability)
    if colors is None:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Default markers
    if markers is None:
        markers = ['o', 's', 'd', '^', 'v', '<', '>', 'p', '*', 'h']
    
    # Ensure we have enough colors and markers
    num_benchmarks = len(labels)
    while len(colors) < num_benchmarks:
        colors.extend(colors)
    while len(markers) < num_benchmarks:
        markers.extend(markers)
    
    # Plot each benchmark
    for idx, (label, values) in enumerate(zip(labels, values_matrix)):
        ax.plot(qubit_range, values, 
                marker=markers[idx % len(markers)], 
                label=label,
                color=colors[idx % len(colors)], 
                linewidth=2, 
                markersize=8)
    
    ax.set_xlabel('Number of Qubits', fontsize=14)
    ax.set_ylabel('Execution Time (seconds)', fontsize=14)
    ax.set_title('Quantum Circuit Benchmark Comparison', fontsize=16)
    ax.legend(fontsize=12, loc='best')
    ax.set_xticks(list(qubit_range))
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nBenchmark plot saved as '{filename}'")
    
    # Only show plot if running interactively
    try:
        plt.show()
    except:
        plt.close()