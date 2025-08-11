#!/bin/bash

# Quantum Benchmark Experiment Suite
# ==================================
# Execute comprehensive quantum computing benchmarks with statistical analysis,
# error bars, deviation visualization, and improved runtime stability.
#
# Usage: ./enhanced_experiment.sh [--seed SEED]
#
# Features:
# - Statistical analysis with confidence intervals and outlier detection
# - Statistical plotting with error bars and distribution analysis
# - System stability monitoring for reproducible results
# - Comprehensive data saving with statistical metadata
# - Improved runtime stability through extended warmup and system checks

set -e  # Exit on any error

# Parse command line arguments
SEED="42"  # Default seed

while [[ $# -gt 0 ]]; do
    case $1 in
        --seed)
            SEED="$2"
            shift 2
            ;;
        -h|--help)
            echo "Quantum Benchmark Experiment Suite"
            echo ""
            echo "Usage: $0 [--seed SEED]"
            echo ""
            echo "Options:"
            echo "  --seed SEED    Random seed for reproducibility (default: 42)"
            echo "  -h, --help     Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --seed 12345"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "    🚀 Quantum Computing Benchmark Suite"
echo "============================================================"

# Get timestamp for unique result files and create organized folder structure
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_RESULTS_DIR="statistical_results"
RESULTS_DIR="$BASE_RESULTS_DIR/run_$TIMESTAMP"

echo "📁 Creating timestamped results directory: $RESULTS_DIR"
mkdir -p "$RESULTS_DIR"

# Create subdirectories for better organization
mkdir -p "$RESULTS_DIR/individual_circuits"
mkdir -p "$RESULTS_DIR/mps_extended" 
mkdir -p "$RESULTS_DIR/multi_circuit_comparison"
mkdir -p "$RESULTS_DIR/data_files"

# Configuration Parameters (optimized for stability and statistics)
readonly SEED                            # Random seed for reproducibility (set via command line)
readonly REPEATS="20"                    # Increased repetitions for better statistics
readonly WARMUP="5"                      # Extended warmup for stability
readonly MIN_QUBITS="3"                  # Minimum number of qubits to test
readonly MAX_QUBITS="25"                 # Reasonable maximum for statistical analysis

# Backend configurations
readonly ALL_BACKENDS="qiskit_mps_cpu qiskit_statevector_cpu qiskit_statevector_gpu cudaq_statevector_gpu"
readonly MPS_BACKEND="qiskit_mps_cpu"
readonly MPS_MAX_QUBITS="45"             # Extended range for MPS backend

# Circuit Types to benchmark
readonly CIRCUIT_TYPES="vqe qaoa qft ghz"

echo "⚙️  Configuration:"
echo "   • Random Seed: $SEED"
echo "   • Repetitions: $REPEATS (with $WARMUP warmup runs)"
echo "   • Standard Qubit Range: $MIN_QUBITS to $MAX_QUBITS"
echo "   • Extended MPS Range: $MIN_QUBITS to $MPS_MAX_QUBITS"
echo "   • Stability Check: Enabled"
echo "   • Statistical Analysis: Full (mean, std, CI, outlier detection)"
echo "============================================================"
echo ""

# Function to run statistical benchmark with comprehensive parameters
run_statistical_benchmark() {
    local benchmarks="$1"
    local circuit_types="$2"
    local max_qubits="$3"
    local description="$4"
    local extra_flags="$5"
    local output_subdir="${6:-individual_circuits}"  # Default to individual_circuits subdirectory
    
    # Create full output directory path
    local full_output_dir="$RESULTS_DIR/$output_subdir"
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔬 Running: $description"
    echo "   Backends: $benchmarks"
    echo "   Circuits: $circuit_types"
    echo "   Qubit Range: $MIN_QUBITS to $max_qubits"
    echo "   Output Directory: $output_subdir/"
    echo "   Statistical Features: Statistical Analysis + Stability Monitoring"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    python3 enhanced_benchmark.py \
        --benchmarks $benchmarks \
        --circuit-type $circuit_types \
        --min-qubits "$MIN_QUBITS" \
        --max-qubits "$max_qubits" \
        --repeats "$REPEATS" \
        --warmup "$WARMUP" \
        --seed "$SEED" \
        --results-dir "$full_output_dir" \
        --stability-check \
        --outlier-removal \
        $extra_flags
    
    echo "✅ Completed: $description"
    echo ""
    echo "🔍 System stabilization pause (10 seconds)..."
    sleep 10
    echo ""
}

# =============================================================================
# 🧬 VQE (Variational Quantum Eigensolver) Benchmark
# =============================================================================
echo "🧬 Running VQE Circuit Benchmark..."
echo "   📊 Features: Error bars, confidence intervals, stability analysis"

run_statistical_benchmark "$ALL_BACKENDS" "vqe" "$MAX_QUBITS" \
    "VQE Circuit Performance Analysis - All Backends" \
    "--output-filename vqe_all_backends_statistical_analysis.png" \
    "individual_circuits"

# =============================================================================
# 🔄 QAOA (Quantum Approximate Optimization Algorithm) Benchmark
# =============================================================================
echo "🔄 Running QAOA Circuit Benchmark..."
echo "   📊 Features: Statistical analysis, outlier detection, distribution plots"

run_statistical_benchmark "$ALL_BACKENDS" "qaoa" "$MAX_QUBITS" \
    "QAOA Circuit Performance Analysis - All Backends" \
    "--output-filename qaoa_all_backends_statistical_analysis.png" \
    "individual_circuits"

# =============================================================================
# 🌊 QFT (Quantum Fourier Transform) Benchmark
# =============================================================================
echo "🌊 Running QFT Circuit Benchmark..."
echo "   📊 Features: Coefficient of variation analysis, stability metrics"

run_statistical_benchmark "$ALL_BACKENDS" "qft" "$MAX_QUBITS" \
    "QFT Circuit Performance Analysis - All Backends" \
    "--output-filename qft_all_backends_statistical_analysis.png" \
    "individual_circuits"

# =============================================================================
# 🔗 GHZ (Greenberger-Horne-Zeilinger) State Benchmark
# =============================================================================
echo "🔗 Running GHZ State Benchmark..."
echo "   📊 Features: Multi-panel analysis, box plots, confidence intervals"

run_statistical_benchmark "$ALL_BACKENDS" "ghz" "$MAX_QUBITS" \
    "GHZ State Performance Analysis - All Backends" \
    "--output-filename ghz_all_backends_statistical_analysis.png" \
    "individual_circuits"


# =============================================================================
# 📊 Multi-Circui MPS Extended Range Benchmark
# =============================================================================
echo "📊 Running Multi-Circuit Comparison Analysis..."
echo "   🔬 Comparing all circuits on primary backends for comprehensive analysis"

run_statistical_benchmark "qiskit_mps_cpu" \
    "$CIRCUIT_TYPES" "$MAX_QUBITS" \
    "Multi-Circui MPS" \
    "--output-filename multi_circuit_comparison_primary_backends.png" \
    "multi_circuit_comparison"

# =============================================================================
# 🎉 Experiment Complete
# =============================================================================
echo "============================================================"
echo "🎉 BENCHMARK SUITE COMPLETED SUCCESSFULLY!"
echo "============================================================"
echo "📊 Analysis Summary:"
echo "   ✅ VQE Circuit - Statistical Analysis with Error Bars"
echo "   ✅ QAOA Circuit - Confidence Intervals & Outlier Detection" 
echo "   ✅ QFT Circuit - Stability Analysis & Distribution Plots"
echo "   ✅ GHZ State - Multi-panel Statistical Visualization"
echo "   ✅ MPS Multi-Circuit - Comprehensive Comparative Analysis"
echo ""
echo "🔬 Statistical Features Applied:"
echo "   • Mean ± Standard Deviation with Error Bars"
echo "   • 95% Confidence Intervals"
echo "   • Coefficient of Variation (Stability Metric)"
echo "   • Outlier Detection and Removal (IQR Method)"
echo "   • Distribution Analysis with Box Plots"
echo "   • System Stability Monitoring"
echo ""
echo "📁 All statistical results saved in: $RESULTS_DIR"
echo "🕒 Experiment timestamp: $TIMESTAMP"
echo ""
echo "📈 Plot Files Generated:"
echo "   • *_comparison.png (Standard Comparison)"
echo "   • *_statistical.png (4-Panel Statistical Analysis)"
echo "   • *_confidence_intervals.png (95% Confidence Intervals)"
echo ""
echo "💾 Data Files Generated:"
echo "   • benchmark_*_statistical_results.npz (Statistical Data)"
echo "   • Includes: means, std devs, confidence intervals, raw data"
echo "============================================================"
