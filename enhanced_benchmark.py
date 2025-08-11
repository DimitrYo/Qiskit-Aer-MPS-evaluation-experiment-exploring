#!/usr/bin/env python3
"""
Quantum Benchmark Suite with Statistical Analysis and Advanced Plotting
======================================================================

This benchmark provides:
- Statistical analysis with standard deviations and confidence intervals
- Advanced plotting with error bars and deviation visualization
- Improved runtime stability and reproducibility
- Comprehensive data saving with statistical metadata
- System stability monitoring and outlier detection
"""

import argparse
import os
import sys
import time
import gc
import psutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Import existing benchmark modules
from benchmark_qiskit__mps_cpu import (
    benchmark_qiskit_mps_cpu, BENCHMARK_NAME as qiskit_mps_name, BENCHMARK_DESCRIPTION as qiskit_mps_desc
)
from benchmark_qiskit_statevector_cpu import (
    benchmark_qiskit_statevector_cpu, BENCHMARK_NAME as qiskit_statevector_cpu_name, BENCHMARK_DESCRIPTION as qiskit_statevector_cpu_desc
)
from benchmark_qiskit_statevector_gpu import (
    benchmark_qiskit_statevector_gpu, BENCHMARK_NAME as qiskit_statevector_gpu_name, BENCHMARK_DESCRIPTION as qiskit_statevector_gpu_desc
)
from benchmark_qiskit_tensor_network_gpu import (
    benchmark_qiskit_tensornet_gpu, BENCHMARK_NAME as qiskit_tensornet_name, BENCHMARK_DESCRIPTION as qiskit_tensornet_desc
)
from benchmark_cudaq_gpu import (
    benchmark_cudaq_statevector_gpu, BENCHMARK_NAME as cudaq_statevector_name, BENCHMARK_DESCRIPTION as cudaq_statevector_desc
)
from timing_utils import TimingResult, validate_timing_result
from plot_utils import load_timing_results


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    if seed is not None:
        np.random.seed(seed)
        # Set other library seeds if available
        try:
            import random
            random.seed(seed)
        except ImportError:
            pass


def get_available_benchmarks():
    """Return dictionary of available benchmark functions."""
    return {
        qiskit_mps_name: (benchmark_qiskit_mps_cpu, qiskit_mps_desc),
        qiskit_statevector_cpu_name: (benchmark_qiskit_statevector_cpu, qiskit_statevector_cpu_desc),
        qiskit_statevector_gpu_name: (benchmark_qiskit_statevector_gpu, qiskit_statevector_gpu_desc),
        qiskit_tensornet_name: (benchmark_qiskit_tensornet_gpu, qiskit_tensornet_desc),
        cudaq_statevector_name: (benchmark_cudaq_statevector_gpu, cudaq_statevector_desc),
    }


def create_benchmark_wrapper(benchmark_func, shots=1024, print_topology=False, circuit_type=None, 
                           include_init_time=False, **kwargs):
    """Create a wrapper function that passes parameters to the benchmark."""
    def wrapper(qubit_range):
        # Always pass circuit_type to benchmark functions
        kwargs['circuit_type'] = circuit_type
        kwargs['include_init_time'] = include_init_time
        return benchmark_func(qubit_range, shots=shots, print_topology=print_topology, **kwargs)
    return wrapper


def get_benchmark_function(benchmark_name, circuit_type, shots=1024):
    """Get the appropriate benchmark function for the given benchmark name and circuit type."""
    available_benchmarks = get_available_benchmarks()
    
    if benchmark_name not in available_benchmarks:
        print(f"❌ Unknown benchmark: {benchmark_name}")
        print(f"Available benchmarks: {', '.join(available_benchmarks.keys())}")
        return None
    
    benchmark_func, description = available_benchmarks[benchmark_name]
    
    # Create wrapper with the specified parameters
    wrapper = create_benchmark_wrapper(
        benchmark_func, 
        shots=shots, 
        circuit_type=circuit_type,
        include_init_time=True  # Enable detailed timing for enhanced analysis
    )
    
    return wrapper

class SystemStabilizer:
    """Ensures system stability for reproducible benchmarks."""
    
    def __init__(self):
        self.initial_cpu_percent = None
        self.initial_memory_percent = None
        
    def check_system_stability(self, max_cpu_percent=80, max_memory_percent=85):
        """Check if system is stable enough for benchmarking."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        
        if cpu_percent > max_cpu_percent:
            print(f"⚠️  Warning: High CPU usage ({cpu_percent:.1f}%) detected")
            print("   Consider waiting for system to stabilize")
            
        if memory_percent > max_memory_percent:
            print(f"⚠️  Warning: High memory usage ({memory_percent:.1f}%) detected")
            print("   Consider closing unnecessary applications")
            
        return cpu_percent <= max_cpu_percent and memory_percent <= max_memory_percent
    
    def stabilize_system(self):
        """Perform system stabilization operations."""
        # Force garbage collection
        gc.collect()
        
        # Brief pause to let system settle
        time.sleep(2)
        
        # Record baseline system state
        self.initial_cpu_percent = psutil.cpu_percent(interval=0.5)
        self.initial_memory_percent = psutil.virtual_memory().percent


class StatisticalAnalyzer:
    """Performs statistical analysis on benchmark results."""
    
    @staticmethod
    def analyze_timings(timings):
        """Analyze timing results and return statistics."""
        if not timings:
            return None
            
        times = np.array([t.total_time for t in timings])
        
        # Remove outliers using IQR method
        q1, q3 = np.percentile(times, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Filter outliers
        filtered_times = times[(times >= lower_bound) & (times <= upper_bound)]
        outliers_removed = len(times) - len(filtered_times)
        
        if len(filtered_times) < 3:
            # If too many outliers, use original data
            filtered_times = times
            outliers_removed = 0
        
        return {
            'mean': np.mean(filtered_times),
            'std': np.std(filtered_times, ddof=1),
            'median': np.median(filtered_times),
            'min': np.min(filtered_times),
            'max': np.max(filtered_times),
            'count': len(filtered_times),
            'outliers_removed': outliers_removed,
            'confidence_interval_95': stats.t.interval(
                0.95, len(filtered_times)-1, 
                loc=np.mean(filtered_times), 
                scale=stats.sem(filtered_times)
            ) if len(filtered_times) > 1 else (np.mean(filtered_times), np.mean(filtered_times)),
            'coefficient_of_variation': np.std(filtered_times, ddof=1) / np.mean(filtered_times) * 100 if np.mean(filtered_times) > 0 else 0
        }


def statistical_benchmark_runner(benchmark_func, qubit_range, repeats=15, warmup=5, 
                            desc=None, stability_check=True):
    """
    Statistical benchmark runner with statistical analysis and stability monitoring.
    
    Args:
        benchmark_func: Function to benchmark
        qubit_range: Range of qubit counts to test
        repeats: Number of timing iterations (increased default for better statistics)
        warmup: Number of warm-up iterations (increased for stability)
        desc: Description for progress bar
        stability_check: Whether to perform system stability checks
    
    Returns:
        tuple: (mean_times, std_times, statistics_data, raw_timings)
    """
    stabilizer = SystemStabilizer()
    analyzer = StatisticalAnalyzer()
    
    if stability_check:
        print(f"🔍 Checking system stability for: {desc}")
        if not stabilizer.check_system_stability():
            print("⚠️  System may not be stable - results may vary")
        stabilizer.stabilize_system()
    
    mean_times = []
    std_times = []
    statistics_data = []
    raw_timings = []
    
    for nq in tqdm(qubit_range, desc=desc, unit="qubits"):
        # Extended warm-up for stability
        for _ in range(warmup):
            try:
                benchmark_func([nq])
                gc.collect()  # Clean up between warmup runs
            except Exception as e:
                print(f"Warning: Warmup failed for {nq} qubits: {e}")
        
        # Brief pause between warmup and actual timing
        time.sleep(0.5)
        
        # Actual timing iterations with error handling
        timing_results = []
        failed_runs = 0
        
        for run_idx in range(repeats):
            try:
                # Small pause between runs for stability
                if run_idx > 0:
                    time.sleep(0.1)
                
                result = benchmark_func([nq])
                
                # Handle both old and new return formats
                if isinstance(result[0], TimingResult):
                    timing_result = validate_timing_result(result[0])
                    timing_results.append(timing_result)
                else:
                    # Legacy format
                    execution_time = max(0.0, float(result[0]))
                    timing_result = TimingResult(
                        total_time=execution_time,
                        execution_time=execution_time,
                        init_time=0.0
                    )
                    timing_results.append(timing_result)
                    
            except Exception as e:
                failed_runs += 1
                print(f"Warning: Run {run_idx+1} failed for {nq} qubits: {e}")
                if failed_runs > repeats // 2:
                    print(f"Too many failures for {nq} qubits, skipping...")
                    break
        
        # Analyze results
        if len(timing_results) >= 3:  # Minimum for meaningful statistics
            stats = analyzer.analyze_timings(timing_results)
            
            if stats:
                mean_times.append(stats['mean'])
                std_times.append(stats['std'])
                statistics_data.append(stats)
                raw_timings.append(timing_results)
                
                # Print stability warning if high variation
                if stats['coefficient_of_variation'] > 20:  # >20% CV
                    print(f"⚠️  High variation detected for {nq} qubits: {stats['coefficient_of_variation']:.1f}% CV")
            else:
                mean_times.append(0.0)
                std_times.append(0.0)
                statistics_data.append(None)
                raw_timings.append([])
        else:
            print(f"❌ Insufficient data for {nq} qubits (only {len(timing_results)} successful runs)")
            mean_times.append(0.0)
            std_times.append(0.0)
            statistics_data.append(None)
            raw_timings.append([])
    
    return mean_times, std_times, statistics_data, raw_timings


def create_confidence_interval_plot(qubit_range, benchmark_data, output_path, title_prefix="Benchmark"):
    """
    Create a dedicated plot showing 95% confidence intervals.
    
    Args:
        qubit_range: Range of qubit counts
        benchmark_data: Dictionary with benchmark results
        output_path: Path for the confidence interval plot
        title_prefix: Prefix for plot titles
    """
    plt.style.use('default')
    sns.set_palette("husl")
    
    plt.figure(figsize=(12, 8))
    
    # Generate colors
    colors = plt.cm.Set1(np.linspace(0, 1, len(benchmark_data)))
    colors = [tuple(c) for c in colors]
    
    for i, (label, data) in enumerate(benchmark_data.items()):
        mean_times = data['mean_times']
        ci_lower = []
        ci_upper = []
        
        for stats in data['statistics_data']:
            if stats and stats['confidence_interval_95']:
                ci_lower.append(stats['confidence_interval_95'][0])
                ci_upper.append(stats['confidence_interval_95'][1])
            else:
                ci_lower.append(0)
                ci_upper.append(0)
        
        # Plot mean line
        plt.plot(qubit_range, mean_times, marker='o', label=f'{label} (mean)', 
                linewidth=2, markersize=6, color=colors[i])
        
        # Fill confidence interval
        plt.fill_between(qubit_range, ci_lower, ci_upper, 
                        alpha=0.3, color=colors[i])
    
    plt.xlabel('Number of Qubits', fontsize=14)
    plt.ylabel('Execution Time (seconds)', fontsize=14)
    plt.title(f'{title_prefix}', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the confidence interval plot
    ci_plot_path = output_path.replace('.png', '_confidence_intervals.png')
    plt.savefig(ci_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 Confidence interval plot saved as '{ci_plot_path}'")
    plt.close()
    
    return ci_plot_path


def create_statistical_plots(qubit_range, benchmark_data, output_path, title_prefix="Benchmark"):
    """
    Create statistical plots with error bars, confidence intervals, and statistical information.
    
    Args:
        qubit_range: Range of qubit counts
        benchmark_data: Dictionary with benchmark results
        output_path: Base path for output files
        title_prefix: Prefix for plot titles
    """
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{title_prefix} - Statistical Analysis', fontsize=16, fontweight='bold')
    
    # Generate colors in proper format
    colors = plt.cm.Set1(np.linspace(0, 1, len(benchmark_data)))
    # Ensure colors are RGBA tuples in 0-1 range
    colors = [tuple(c) for c in colors]
    
    # Plot 1: Main benchmark with error bars
    ax1.set_title('Execution Time with Standard Deviation', fontsize=14, fontweight='bold')
    
    for i, (label, data) in enumerate(benchmark_data.items()):
        mean_times, std_times = data['mean_times'], data['std_times']
        
        # Main line with error bars
        ax1.errorbar(qubit_range, mean_times, yerr=std_times, 
                    label=label, marker='o', capsize=5, capthick=2,
                    linewidth=2, markersize=6, color=colors[i])
        
        # Fill between for standard deviation
        ax1.fill_between(qubit_range, 
                        np.array(mean_times) - np.array(std_times),
                        np.array(mean_times) + np.array(std_times),
                        alpha=0.2, color=colors[i])
    
    ax1.set_xlabel('Number of Qubits', fontsize=12)
    ax1.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax1.set_yscale('log')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Coefficient of Variation (stability metric)
    ax2.set_title('Measurement Stability (Coefficient of Variation)', fontsize=14, fontweight='bold')
    
    for i, (label, data) in enumerate(benchmark_data.items()):
        cv_values = []
        for stats in data['statistics_data']:
            if stats:
                cv_values.append(stats['coefficient_of_variation'])
            else:
                cv_values.append(0)
        
        ax2.plot(qubit_range, cv_values, marker='s', label=label, 
                linewidth=2, markersize=6, color=colors[i])
    
    ax2.axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='Good Stability (10%)')
    ax2.axhline(y=20, color='red', linestyle='--', alpha=0.7, label='Poor Stability (20%)')
    ax2.set_xlabel('Number of Qubits', fontsize=12)
    ax2.set_ylabel('Coefficient of Variation (%)', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Confidence Intervals
    ax3.set_title('95% Confidence Intervals', fontsize=14, fontweight='bold')
    
    for i, (label, data) in enumerate(benchmark_data.items()):
        mean_times = data['mean_times']
        ci_lower = []
        ci_upper = []
        
        for stats in data['statistics_data']:
            if stats and stats['confidence_interval_95']:
                ci_lower.append(stats['confidence_interval_95'][0])
                ci_upper.append(stats['confidence_interval_95'][1])
            else:
                ci_lower.append(0)
                ci_upper.append(0)
        
        # Plot mean line
        ax3.plot(qubit_range, mean_times, marker='o', label=f'{label} (mean)', 
                linewidth=2, markersize=4, color=colors[i])
        
        # Fill confidence interval
        ax3.fill_between(qubit_range, ci_lower, ci_upper, 
                        alpha=0.3, color=colors[i], label=f'{label} (95% CI)')
    
    ax3.set_xlabel('Number of Qubits', fontsize=12)
    ax3.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax3.set_yscale('log')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Distribution comparison (box plot style for last few qubit counts)
    ax4.set_title('Timing Distribution (Recent Qubit Counts)', fontsize=14, fontweight='bold')
    
    # Show distributions for the last 3-4 qubit counts
    recent_qubits = list(qubit_range)[-min(4, len(qubit_range)):]
    box_data = []
    box_labels = []
    
    for label, data in benchmark_data.items():
        for i, nq in enumerate(recent_qubits):
            if i < len(data['raw_timings']) and data['raw_timings'][-(len(recent_qubits)-i)]:
                timings = [t.total_time for t in data['raw_timings'][-(len(recent_qubits)-i)]]
                box_data.append(timings)
                box_labels.append(f'{label}\n{nq}q')
    
    if box_data:
        bp = ax4.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
        # Ensure colors are in the right format (0-1 range)
        colors_normalized = []
        for color in colors:
            if isinstance(color, str) and color.startswith('#'):
                # Convert hex to RGB
                color = color.lstrip('#')
                rgb = tuple(int(color[i:i+2], 16)/255.0 for i in (0, 2, 4))
                colors_normalized.append(rgb)
            elif isinstance(color, (tuple, list)) and max(color) > 1:
                # Convert 0-255 range to 0-1 range
                colors_normalized.append(tuple(c/255.0 for c in color))
            else:
                colors_normalized.append(color)
        
        for patch, color in zip(bp['boxes'], colors_normalized * len(recent_qubits)):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax4.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save the statistical plot
    statistical_plot_path = output_path.replace('.png', '_statistical.png')
    plt.savefig(statistical_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 Statistical plot saved as '{statistical_plot_path}'")
    
    # Create a simple comparison plot (similar to original)
    plt.figure(figsize=(12, 8))
    for i, (label, data) in enumerate(benchmark_data.items()):
        plt.errorbar(qubit_range, data['mean_times'], yerr=data['std_times'],
                    label=label, marker='o', capsize=4, linewidth=2, markersize=6)
    
    plt.xlabel('Number of Qubits', fontsize=14)
    plt.ylabel('Execution Time (seconds)', fontsize=14)
    plt.title(f'{title_prefix} - Execution Time Comparison', fontsize=16)
    plt.legend(fontsize=12)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📈 Standard plot saved as '{output_path}'")
    plt.close('all')
    
    # Create separate confidence interval plot
    create_confidence_interval_plot(qubit_range, benchmark_data, output_path, title_prefix)


def save_statistical_results(qubit_range, benchmark_data, filename_base):
    """Save comprehensive benchmark results with statistical analysis."""
    
    # Prepare comprehensive data dictionary
    data_dict = {
        'qubits': np.array(list(qubit_range)),
        'metadata': {
            'description': 'Quantum benchmark results with statistical analysis',
            'units': 'seconds',
            'timestamp': np.datetime64('now'),
            'has_statistical_analysis': True,
            'benchmarks': list(benchmark_data.keys())
        }
    }
    
    # Add data for each benchmark
    for label, data in benchmark_data.items():
        data_dict[f'{label}_mean'] = np.array(data['mean_times'])
        data_dict[f'{label}_std'] = np.array(data['std_times'])
        
        # Add detailed statistics
        stats_arrays = {}
        for stat_name in ['median', 'min', 'max', 'coefficient_of_variation']:
            stats_arrays[stat_name] = []
            for stats in data['statistics_data']:
                if stats:
                    stats_arrays[stat_name].append(stats[stat_name])
                else:
                    stats_arrays[stat_name].append(0.0)
        
        for stat_name, values in stats_arrays.items():
            data_dict[f'{label}_{stat_name}'] = np.array(values)
        
        # Add confidence intervals
        ci_lower, ci_upper = [], []
        for stats in data['statistics_data']:
            if stats and stats['confidence_interval_95']:
                ci_lower.append(stats['confidence_interval_95'][0])
                ci_upper.append(stats['confidence_interval_95'][1])
            else:
                ci_lower.append(0.0)
                ci_upper.append(0.0)
        
        data_dict[f'{label}_ci_lower'] = np.array(ci_lower)
        data_dict[f'{label}_ci_upper'] = np.array(ci_upper)
    
    # Save comprehensive results
    output_filename = f"{filename_base}_statistical_results.npz"
    np.savez_compressed(output_filename, **data_dict)
    
    print(f"💾 Statistical results saved to '{output_filename}'")
    print(f"   - Statistical analysis for {len(benchmark_data)} benchmarks")
    print(f"   - {len(qubit_range)} qubit values: {min(qubit_range)} to {max(qubit_range)}")
    print(f"   - Includes: mean, std, median, min, max, CV, 95% CI")
    
    return output_filename


def print_statistical_summary(benchmark_data, qubit_range):
    """Print a comprehensive statistical summary."""
    print("\n" + "="*80)
    print("📊 STATISTICAL SUMMARY")
    print("="*80)
    
    for label, data in benchmark_data.items():
        print(f"\n🔬 {label.upper()}")
        print("-" * 50)
        
        # Overall statistics
        all_means = [m for m in data['mean_times'] if m > 0]
        if all_means:
            print(f"Overall mean execution time: {np.mean(all_means):.6f} ± {np.std(all_means):.6f} seconds")
        
        # Stability analysis
        cv_values = [stats['coefficient_of_variation'] for stats in data['statistics_data'] if stats]
        if cv_values:
            avg_cv = np.mean(cv_values)
            print(f"Average measurement stability: {avg_cv:.1f}% CV", end="")
            if avg_cv < 10:
                print(" (Excellent ✅)")
            elif avg_cv < 20:
                print(" (Good ⚠️)")
            else:
                print(" (Poor ❌)")
        
        # Show detailed stats for a few qubit counts
        print(f"\nDetailed statistics (sample qubit counts):")
        sample_indices = [0, len(qubit_range)//2, -1] if len(qubit_range) > 2 else range(len(qubit_range))
        
        for i in sample_indices:
            if i < len(data['statistics_data']) and data['statistics_data'][i]:
                stats = data['statistics_data'][i]
                nq = list(qubit_range)[i]
                print(f"  {nq:2d} qubits: {stats['mean']:.6f}s ± {stats['std']:.6f}s "
                      f"(CV: {stats['coefficient_of_variation']:.1f}%, n={stats['count']})")


def main():
    parser = argparse.ArgumentParser(
        description='Quantum Benchmark Suite with Statistical Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --benchmarks qiskit_mps_cpu cudaq_statevector_gpu --circuit-type vqe --repeats 20
  %(prog)s --benchmarks all --min-qubits 3 --max-qubits 15 --enhanced-stability
        """
    )
    
    # Benchmark selection
    parser.add_argument('--benchmarks', nargs='+', required=True,
                       help='Benchmark backends to run (e.g., qiskit_mps_cpu, cudaq_statevector_gpu)')
    parser.add_argument('--circuit-type', nargs='+', required=True,
                       help='Circuit types to benchmark (e.g., vqe, qaoa, qft, ghz)')
    
    # Qubit range
    parser.add_argument('--min-qubits', type=int, default=3,
                       help='Minimum number of qubits (default: 3)')
    parser.add_argument('--max-qubits', type=int, default=15,
                       help='Maximum number of qubits (default: 15)')
    
    # Timing parameters (enhanced defaults)
    parser.add_argument('--repeats', type=int, default=20,
                       help='Number of timing repetitions (default: 20, increased for better statistics)')
    parser.add_argument('--warmup', type=int, default=5,
                       help='Number of warmup iterations (default: 5, increased for stability)')
    
    # Statistical features
    parser.add_argument('--stability-check', action='store_true',
                       help='Enable stability checking and system monitoring')
    parser.add_argument('--outlier-removal', action='store_true', default=True,
                       help='Remove statistical outliers (default: enabled)')
    
    # Output options
    parser.add_argument('--results-dir', type=str, default='statistical_results',
                       help='Directory to save results (default: statistical_results)')
    parser.add_argument('--output-filename', type=str,
                       help='Output filename for plots (default: auto-generated)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip generating plots')
    parser.add_argument('--no-table', action='store_true',
                       help='Skip printing results table')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    set_random_seed(args.seed)
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Print configuration
    print("="*80)
    print("🚀 QUANTUM BENCHMARK SUITE")
    print("="*80)
    print(f"📊 Benchmarks: {', '.join(args.benchmarks)}")
    print(f"🔬 Circuit types: {', '.join(args.circuit_type)}")
    print(f"⚙️  Configuration:")
    print(f"   • Random seed: {args.seed}")
    print(f"   • Repetitions: {args.repeats} (with {args.warmup} warmup runs)")
    print(f"   • Qubit range: {args.min_qubits} to {args.max_qubits}")
    print(f"   • Stability check: {'Enabled' if args.stability_check else 'Disabled'}")
    print(f"   • Outlier removal: {'Enabled' if args.outlier_removal else 'Disabled'}")
    print("="*80)
    
    # Prepare qubit range and benchmark data storage
    qubit_range = range(args.min_qubits, args.max_qubits + 1)
    benchmark_data = {}
    
    # Run benchmarks for each circuit type
    for circuit_type in args.circuit_type:
        print(f"\n🧪 Running {circuit_type.upper()} circuit benchmarks...")
        
        for benchmark_name in args.benchmarks:
            try:
                # Get benchmark function
                benchmark_func = get_benchmark_function(benchmark_name, circuit_type)
                if benchmark_func is None:
                    print(f"❌ Benchmark {benchmark_name} not found for {circuit_type}")
                    continue
                
                # Create descriptive label
                label = f"{benchmark_name}_{circuit_type}"
                desc = f"{benchmark_name} ({circuit_type})"
                
                # Run statistical benchmark
                mean_times, std_times, statistics_data, raw_timings = statistical_benchmark_runner(
                    benchmark_func, qubit_range,
                    repeats=args.repeats,
                    warmup=args.warmup,
                    desc=desc,
                    stability_check=args.stability_check
                )
                
                # Store results
                benchmark_data[label] = {
                    'mean_times': mean_times,
                    'std_times': std_times,
                    'statistics_data': statistics_data,
                    'raw_timings': raw_timings
                }
                
                print(f"✅ Completed: {desc}")
                
            except Exception as e:
                print(f"❌ Error running {benchmark_name} with {circuit_type}: {e}")
                continue
    
    if not benchmark_data:
        print("❌ No successful benchmarks completed!")
        return 1
    
    # Print statistical summary
    if not args.no_table:
        print_statistical_summary(benchmark_data, qubit_range)
    
    # Generate enhanced plots
    if not args.no_plot:
        # Generate output filename
        if args.output_filename:
            output_filename = args.output_filename
        else:
            circuit_str = "_".join(args.circuit_type)
            benchmark_str = "_".join(args.benchmarks[:3])  # Limit length
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_filename = f"benchmark_{circuit_str}_{benchmark_str}_{timestamp}.png"
        
        output_path = os.path.join(args.results_dir, output_filename)
        
        # Create statistical plots
        create_statistical_plots(qubit_range, benchmark_data, output_path, 
                            title_prefix=f"Benchmark - {', '.join(args.circuit_type).upper()}")
    
    # Save statistical results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename_base = os.path.join(args.results_dir, f"benchmark_{timestamp}")
    save_statistical_results(qubit_range, benchmark_data, filename_base)
    
    print("\n" + "="*80)
    print("🎉 BENCHMARK SUITE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"📁 Results saved in: {args.results_dir}")
    print(f"🕒 Timestamp: {timestamp}")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
