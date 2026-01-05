#!/usr/bin/env python3
"""
Cholesky Decomposition MPI+OpenMP Benchmark Analysis
Comprehensive visualization and performance metrics
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib.ticker import ScalarFormatter

# Configuration
CSV_FILE = "benchmarks/benchmark_results.csv"
PLOTS_DIR = "plots"

# Styling
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.dpi': 100,
    'figure.figsize': (10, 6)
})

COLORS = {
    'mpi': '#2E86AB',
    'hybrid': '#A23B72',
    'ideal': '#666666',
    'threshold': '#F18F01'
}

def setup():
    """Initialize output directory"""
    Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {PLOTS_DIR}/\n")

def load_data():
    """Load and preprocess benchmark data"""
    df = pd.read_csv(CSV_FILE)
    
    # Normalize OpenMP flag
    if df['OpenMPEnabled'].dtype == object:
        df['OpenMPEnabled'] = df['OpenMPEnabled'].astype(str).str.lower().isin(['true', 'yes', '1'])
    else:
        df['OpenMPEnabled'] = df['OpenMPEnabled'].astype(bool)
    
    df['NumThreads'] = df['NumThreads'].fillna(1).astype(int)
    df['Implementation'] = df['OpenMPEnabled'].map({True: 'Hybrid', False: 'Pure MPI'})
    
    print(f"Loaded {len(df)} benchmark records")
    print(f"  Strong scaling: {len(df[df['TestCategory']=='strong'])} records")
    print(f"  Weak scaling: {len(df[df['TestCategory']=='weak'])} records\n")
    
    return df

def compute_scaling_metrics(data):
    """Calculate speedup and efficiency metrics"""
    data = data.sort_values('NumProcesses')
    base_time = data['AvgTime'].iloc[0]
    base_procs = data['NumProcesses'].iloc[0]
    
    data['Speedup'] = base_time / data['AvgTime']
    data['IdealSpeedup'] = data['NumProcesses'] / base_procs
    data['Efficiency'] = (data['Speedup'] / data['IdealSpeedup']) * 100
    
    return data

def plot_strong_scaling_comparison(df):
    """Generate strong scaling plots comparing MPI vs Hybrid"""
    print("Generating Strong Scaling Analysis...")
    
    strong = df[df['TestCategory'] == 'strong'].copy()
    matrix_sizes = sorted(strong['MatrixSize'].unique())
    
    for N in matrix_sizes:
        subset = strong[strong['MatrixSize'] == N]
        mpi = subset[~subset['OpenMPEnabled']].copy()
        hybrid = subset[subset['OpenMPEnabled']].copy()
        
        if mpi.empty and hybrid.empty:
            continue
        
        mpi = compute_scaling_metrics(mpi)
        hybrid = compute_scaling_metrics(hybrid)
        
        all_procs = sorted(subset['NumProcesses'].unique())
        threads = hybrid['NumThreads'].iloc[0] if not hybrid.empty else 1
        
        # 1. EXECUTION TIME COMPARISON
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if not mpi.empty:
            ax.plot(mpi['NumProcesses'], mpi['AvgTime'], 'o-', 
                   color=COLORS['mpi'], linewidth=2.5, markersize=8, 
                   label='Pure MPI', alpha=0.9)
        
        if not hybrid.empty:
            ax.plot(hybrid['NumProcesses'], hybrid['AvgTime'], 's--', 
                   color=COLORS['hybrid'], linewidth=2.5, markersize=8,
                   label=f'Hybrid (MPI+{threads} OpenMP)', alpha=0.9)
        
        ax.set_xlabel('Number of MPI Processes', fontweight='bold')
        ax.set_ylabel('Execution Time (seconds)', fontweight='bold')
        ax.set_title(f'Strong Scaling: Execution Time (N={N})', fontweight='bold', pad=15)
        ax.set_xticks(all_procs)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(framealpha=0.95)
        
        # Use scalar formatter to avoid scientific notation
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(style='plain', axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/strong_N{N}_time.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. SPEEDUP CURVES
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if not mpi.empty:
            ax.plot(mpi['NumProcesses'], mpi['Speedup'], 'o-',
                   color=COLORS['mpi'], linewidth=2.5, markersize=8,
                   label='Pure MPI', alpha=0.9)
            ax.plot(mpi['NumProcesses'], mpi['IdealSpeedup'], ':',
                   color=COLORS['ideal'], linewidth=2, alpha=0.6,
                   label='Ideal Linear Speedup')
        
        if not hybrid.empty:
            ax.plot(hybrid['NumProcesses'], hybrid['Speedup'], 's--',
                   color=COLORS['hybrid'], linewidth=2.5, markersize=8,
                   label=f'Hybrid ({threads} threads)', alpha=0.9)
        
        ax.set_xlabel('Number of MPI Processes', fontweight='bold')
        ax.set_ylabel('Speedup', fontweight='bold')
        ax.set_title(f'Strong Scaling: Speedup (N={N})', fontweight='bold', pad=15)
        ax.set_xticks(all_procs)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(framealpha=0.95)
        
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/strong_N{N}_speedup.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. PARALLEL EFFICIENCY (only for Pure MPI)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if not mpi.empty:
            ax.plot(mpi['NumProcesses'], mpi['Efficiency'], 'o-',
                   color=COLORS['mpi'], linewidth=2.5, markersize=8,
                   label='Pure MPI', alpha=0.9)
        
            # Dynamic y-axis limit based on data
            max_eff = mpi['Efficiency'].max()
            y_limit = max(115, max_eff + 10)
        
            ax.axhline(100, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
            ax.axhline(80, color=COLORS['threshold'], linestyle=':', linewidth=1.5, 
                      alpha=0.7, label='80% Efficiency Threshold')
        
            ax.set_xlabel('Number of MPI Processes', fontweight='bold')
            ax.set_ylabel('Parallel Efficiency (%)', fontweight='bold')
            ax.set_title(f'Strong Scaling: Pure MPI Efficiency (N={N})', fontweight='bold', pad=15)
            ax.set_xticks(all_procs)
            ax.set_ylim(0, y_limit)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(framealpha=0.95)
        
            plt.tight_layout()
            plt.savefig(f'{PLOTS_DIR}/strong_N{N}_efficiency.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. GFLOPS COMPARISON
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if not mpi.empty:
            ax.plot(mpi['NumProcesses'], mpi['GFLOPS'], 'o-',
                   color=COLORS['mpi'], linewidth=2.5, markersize=8,
                   label='Pure MPI', alpha=0.9)
        
        if not hybrid.empty:
            ax.plot(hybrid['NumProcesses'], hybrid['GFLOPS'], 's--',
                   color=COLORS['hybrid'], linewidth=2.5, markersize=8,
                   label=f'Hybrid ({threads} threads)', alpha=0.9)
        
        ax.set_xlabel('Number of MPI Processes', fontweight='bold')
        ax.set_ylabel('GFLOPS', fontweight='bold')
        ax.set_title(f'Strong Scaling: Computational Throughput (N={N})', fontweight='bold', pad=15)
        ax.set_xticks(all_procs)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(framealpha=0.95)
        
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/strong_N{N}_gflops.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  N={N}: Generated 4 plots")

def plot_weak_scaling_analysis(df):
    """Analyze weak scaling with work-per-process focus"""
    print("\nGenerating Weak Scaling Analysis...")
    
    weak = df[df['TestCategory'] == 'weak'].copy()
    
    if weak.empty:
        print("  No weak scaling data found")
        return
    
    mpi = weak[~weak['OpenMPEnabled']].sort_values('NumProcesses')
    hybrid = weak[weak['OpenMPEnabled']].sort_values('NumProcesses')
    
    all_procs = sorted(weak['NumProcesses'].unique())
    threads = hybrid['NumThreads'].iloc[0] if not hybrid.empty else 1
    
    # 1. TIME CONSISTENCY (should be flat for perfect weak scaling)
    fig, ax = plt.subplots(figsize=(11, 6))
    
    if not mpi.empty:
        ax.plot(mpi['NumProcesses'], mpi['AvgTime'], 'o-',
               color=COLORS['mpi'], linewidth=2.5, markersize=8,
               label='Pure MPI', alpha=0.9)
        baseline = mpi['AvgTime'].iloc[0]
        ax.axhline(baseline, color=COLORS['mpi'], linestyle=':', alpha=0.4)
    
    if not hybrid.empty:
        ax.plot(hybrid['NumProcesses'], hybrid['AvgTime'], 's--',
               color=COLORS['hybrid'], linewidth=2.5, markersize=8,
               label=f'Hybrid (MPI+{threads} OpenMP)', alpha=0.9)
        baseline_h = hybrid['AvgTime'].iloc[0]
        ax.axhline(baseline_h, color=COLORS['hybrid'], linestyle=':', alpha=0.4)
    
    ax.set_xlabel('Number of Processes', fontweight='bold')
    ax.set_ylabel('Execution Time (seconds)', fontweight='bold')
    ax.set_title('Weak Scaling: Time Stability (Ideal = Flat Line)', 
                fontweight='bold', pad=15)
    ax.set_xticks(all_procs)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(framealpha=0.95)
    
    # Use scalar formatter
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(style='plain', axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/weak_scaling_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. GFLOPS SCALING
    fig, ax = plt.subplots(figsize=(11, 6))
    
    if not mpi.empty:
        ax.plot(mpi['NumProcesses'], mpi['GFLOPS'], 'o-',
               color=COLORS['mpi'], linewidth=2.5, markersize=8,
               label='Pure MPI', alpha=0.9)
    
    if not hybrid.empty:
        ax.plot(hybrid['NumProcesses'], hybrid['GFLOPS'], 's--',
               color=COLORS['hybrid'], linewidth=2.5, markersize=8,
               label=f'Hybrid ({threads} threads)', alpha=0.9)
    
    ax.set_xlabel('Number of Processes', fontweight='bold')
    ax.set_ylabel('GFLOPS', fontweight='bold')
    ax.set_title('Weak Scaling: Aggregate Performance', fontweight='bold', pad=15)
    ax.set_xticks(all_procs)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/weak_scaling_gflops.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. WORK PER PROCESS ANALYSIS
    fig, ax = plt.subplots(figsize=(11, 6))
    
    if not mpi.empty:
        ax.plot(mpi['NumProcesses'], mpi['WorkPerProcess'] / 1e6, 'o-',
               color=COLORS['mpi'], linewidth=2.5, markersize=8,
               label='Pure MPI', alpha=0.9)
    
    if not hybrid.empty:
        ax.plot(hybrid['NumProcesses'], hybrid['WorkPerProcess'] / 1e6, 's--',
               color=COLORS['hybrid'], linewidth=2.5, markersize=8,
               label=f'Hybrid ({threads} threads)', alpha=0.9)
    
    ax.set_xlabel('Number of Processes', fontweight='bold')
    ax.set_ylabel('Work Per Process (Million elements)', fontweight='bold')
    ax.set_title('Weak Scaling: Workload Distribution', fontweight='bold', pad=15)
    ax.set_xticks(all_procs)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(framealpha=0.95)
    
    # Use scalar formatter
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(style='plain', axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/weak_scaling_workload.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. WEAK SCALING EFFICIENCY (MPI only)
    fig, ax = plt.subplots(figsize=(11, 6))
    
    if not mpi.empty:
        base_t = mpi['AvgTime'].iloc[0]
        eff = (base_t / mpi['AvgTime']) * 100
        max_eff = eff.max()
        y_limit = max(115, max_eff + 10)
        
        ax.plot(mpi['NumProcesses'], eff, 'o-',
               color=COLORS['mpi'], linewidth=2.5, markersize=8,
               label='Pure MPI', alpha=0.9)
        
        ax.axhline(100, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.axhline(90, color=COLORS['threshold'], linestyle=':', linewidth=1.5,
                  alpha=0.7, label='90% Threshold')
        
        ax.set_xlabel('Number of Processes', fontweight='bold')
        ax.set_ylabel('Weak Scaling Efficiency (%)', fontweight='bold')
        ax.set_title('Weak Scaling: Pure MPI Efficiency', fontweight='bold', pad=15)
        ax.set_xticks(all_procs)
        ax.set_ylim(0, y_limit)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(framealpha=0.95)
        
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/weak_scaling_efficiency.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"  Generated 4 weak scaling plots")

def main():
    setup()
    df = load_data()
    
    plot_strong_scaling_comparison(df)
    plot_weak_scaling_analysis(df)
    
    print(f"\nAnalysis complete. All plots saved to: {PLOTS_DIR}/")

if __name__ == "__main__":
    main()


































# from pathlib import Path
#
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
#
# # Configuration
# CSV_FILE = "benchmarks/benchmark_results.csv"
# PLOTS_DIR = "plots"
#
# # Professional styling
# plt.rcParams["font.size"] = 11
# plt.rcParams["axes.labelsize"] = 12
# plt.rcParams["axes.titlesize"] = 14
# plt.rcParams["legend.fontsize"] = 10
# plt.rcParams["figure.dpi"] = 100
#
# # Color palette: Blue for MPI, Orange for Hybrid
# COLORS = {
#     "mpi": "#1f77b4",      # Blue
#     "hybrid": "#ff7f0e",   # Orange
#     "ideal": "black",
#     "threshold": "gray"
# }
#
# def setup():
#     """Create output directory"""
#     Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)
#     print(f"Output directory: {PLOTS_DIR}/\n")
#
#
# def load_data():
#     """Load benchmark data from CSV and normalize columns"""
#     try:
#         df = pd.read_csv(CSV_FILE)
#
#         # Ensure OpenMPEnabled is boolean
#         # Handles cases like 0/1, 'True'/'False', 'yes'/'no'
#         if df["OpenMPEnabled"].dtype == object:
#             df["OpenMPEnabled"] = df["OpenMPEnabled"].astype(str).str.lower().map({
#                 'true': True, 'yes': True, '1': True,
#                 'false': False, 'no': False, '0': False
#             })
#         else:
#             df["OpenMPEnabled"] = df["OpenMPEnabled"].astype(bool)
#
#         # Fill NumThreads with 1 where it might be NaN (pure MPI case)
#         if "NumThreads" in df.columns:
#              df["NumThreads"] = df["NumThreads"].fillna(1).astype(int)
#
#         print(f"Loaded data from {CSV_FILE}, total records: {len(df)}")
#         return df
#     except Exception as e:
#         print(f"Error loading data: {e}")
#         exit(1)
#
#
# def calculate_metrics(subset):
#     """Calculate Speedup and Efficiency for a given subset"""
#     if len(subset) < 1:
#         return subset
#
#     subset = subset.sort_values("NumProcesses")
#     procs = subset["NumProcesses"].values
#     times = subset["AvgTime"].values
#
#     # Base is the experiment with the smallest number of processes in this subset
#     base_time = times[0]
#     base_procs = procs[0]
#
#     subset["Speedup"] = base_time / subset["AvgTime"]
#     subset["IdealSpeedup"] = subset["NumProcesses"] / base_procs
#     subset["Efficiency"] = (subset["Speedup"] / subset["IdealSpeedup"]) * 100
#
#     return subset
#
#
# def plot_strong_scaling(df):
#     """
#     Strong Scaling Comparison: Pure MPI vs Hybrid
#     """
#     print("=== Generating Strong Scaling Comparative Plots ===")
#
#     strong_df = df[df["TestCategory"] == "strong"].copy()
#     matrix_sizes = sorted(strong_df["MatrixSize"].unique())
#
#     for matrix_size in matrix_sizes:
#         print(f"  Processing Matrix Size N={matrix_size}...")
#
#         subset = strong_df[strong_df["MatrixSize"] == matrix_size]
#
#         # Split into MPI and Hybrid
#         mpi_data = subset[subset["OpenMPEnabled"] == False].copy()
#         hybrid_data = subset[subset["OpenMPEnabled"] == True].copy()
#
#         # Skip if absolutely no data
#         if mpi_data.empty and hybrid_data.empty:
#             continue
#
#         # Calculate metrics independently for proper scaling curves
#         mpi_data = calculate_metrics(mpi_data)
#         hybrid_data = calculate_metrics(hybrid_data)
#
#         # ---------------------------------------------------------
#         # Plot 1: Execution Time (Absolute Performance Comparison)
#         # ---------------------------------------------------------
#         fig, ax = plt.subplots(figsize=(10, 6))
#
#         all_procs = sorted(subset["NumProcesses"].unique())
#
#         # Plot Pure MPI
#         if not mpi_data.empty:
#             ax.plot(mpi_data["NumProcesses"], mpi_data["AvgTime"], "o-", 
#                     color=COLORS["mpi"], linewidth=2, markersize=8, label="Pure MPI")
#
#         # Plot Hybrid
#         if not hybrid_data.empty:
#             threads = hybrid_data["NumThreads"].iloc[0] if not hybrid_data.empty else "?"
#             ax.plot(hybrid_data["NumProcesses"], hybrid_data["AvgTime"], "s--", 
#                     color=COLORS["hybrid"], linewidth=2, markersize=8, label=f"Hybrid (MPI + {threads} OMP)")
#
#         ax.set_xlabel("Number of MPI Processes", fontweight="bold")
#         ax.set_ylabel("Execution Time (seconds)", fontweight="bold")
#         ax.set_title(f"Execution Time Comparison (N={matrix_size})", fontweight="bold")
#         ax.set_xticks(all_procs)
#         ax.grid(True, alpha=0.3)
#         ax.legend()
#
#         filename = f"{PLOTS_DIR}/strong_scaling_N{matrix_size}_time.png"
#         plt.savefig(filename, dpi=300, bbox_inches="tight")
#         plt.close()
#
#         # ---------------------------------------------------------
#         # Plot 2: Speedup (Scaling Behavior)
#         # ---------------------------------------------------------
#         fig, ax = plt.subplots(figsize=(10, 6))
#
#         if not mpi_data.empty:
#             ax.plot(mpi_data["NumProcesses"], mpi_data["Speedup"], "o-", 
#                     color=COLORS["mpi"], label="Pure MPI Speedup")
#             # Ideal line based on MPI procs
#             ax.plot(mpi_data["NumProcesses"], mpi_data["IdealSpeedup"], ":", 
#                     color="gray", alpha=0.6, label="Ideal Linear")
#
#         if not hybrid_data.empty:
#             threads = hybrid_data["NumThreads"].iloc[0]
#             ax.plot(hybrid_data["NumProcesses"], hybrid_data["Speedup"], "s--", 
#                     color=COLORS["hybrid"], label=f"Hybrid Speedup ({threads} thr)")
#
#         ax.set_xlabel("Number of MPI Processes", fontweight="bold")
#         ax.set_ylabel("Speedup (Relative to base)", fontweight="bold")
#         ax.set_title(f"Speedup Scaling (N={matrix_size})", fontweight="bold")
#         ax.set_xticks(all_procs)
#         ax.grid(True, alpha=0.3)
#         ax.legend()
#
#         filename = f"{PLOTS_DIR}/strong_scaling_N{matrix_size}_speedup.png"
#         plt.savefig(filename, dpi=300, bbox_inches="tight")
#         plt.close()
#
#         # ---------------------------------------------------------
#         # Plot 3: Parallel Efficiency
#         # ---------------------------------------------------------
#         fig, ax = plt.subplots(figsize=(10, 6))
#
#         if not mpi_data.empty:
#             ax.plot(mpi_data["NumProcesses"], mpi_data["Efficiency"], "o-", 
#                     color=COLORS["mpi"], label="Pure MPI")
#
#         if not hybrid_data.empty:
#             threads = hybrid_data["NumThreads"].iloc[0]
#             ax.plot(hybrid_data["NumProcesses"], hybrid_data["Efficiency"], "s--", 
#                     color=COLORS["hybrid"], label=f"Hybrid ({threads} thr)")
#
#         ax.axhline(y=100, color="black", linestyle="--", linewidth=1, alpha=0.5)
#         ax.axhline(y=80, color="orange", linestyle=":", linewidth=1, alpha=0.5, label="80% Threshold")
#
#         ax.set_xlabel("Number of MPI Processes", fontweight="bold")
#         ax.set_ylabel("Efficiency (%)", fontweight="bold")
#         ax.set_title(f"Parallel Efficiency (N={matrix_size})", fontweight="bold")
#         ax.set_xticks(all_procs)
#         # ax.set_ylim(0, 115)
#         ax.grid(True, alpha=0.3)
#         ax.legend()
#
#         filename = f"{PLOTS_DIR}/strong_scaling_N{matrix_size}_efficiency.png"
#         plt.savefig(filename, dpi=300, bbox_inches="tight")
#         plt.close()
#
#
# def plot_weak_scaling(df):
#     """
#     Weak Scaling Comparison
#     """
#     print("=== Generating Weak Scaling Comparative Plots ===")
#
#     weak_df = df[df["TestCategory"] == "weak"].copy()
#
#     if weak_df.empty:
#         print("No weak scaling data found.")
#         return
#
#     # Split Data
#     mpi_data = weak_df[weak_df["OpenMPEnabled"] == False].sort_values("NumProcesses")
#     hybrid_data = weak_df[weak_df["OpenMPEnabled"] == True].sort_values("NumProcesses")
#
#     all_procs = sorted(weak_df["NumProcesses"].unique())
#
#     # Calculate Efficiency (Base Time / Current Time)
#     if not mpi_data.empty:
#         base_t = mpi_data["AvgTime"].iloc[0]
#         mpi_data["WeakEfficiency"] = (base_t / mpi_data["AvgTime"]) * 100
#
#     if not hybrid_data.empty:
#         base_t = hybrid_data["AvgTime"].iloc[0]
#         hybrid_data["WeakEfficiency"] = (base_t / hybrid_data["AvgTime"]) * 100
#
#     # ---------------------------------------------------------
#     # Plot 1: Weak Scaling Time (Should be flat)
#     # ---------------------------------------------------------
#     fig, ax = plt.subplots(figsize=(10, 6))
#
#     if not mpi_data.empty:
#         ax.plot(mpi_data["NumProcesses"], mpi_data["AvgTime"], "o-", 
#                 color=COLORS["mpi"], linewidth=2, label="Pure MPI")
#
#     if not hybrid_data.empty:
#         threads = hybrid_data["NumThreads"].iloc[0]
#         ax.plot(hybrid_data["NumProcesses"], hybrid_data["AvgTime"], "s--", 
#                 color=COLORS["hybrid"], linewidth=2, label=f"Hybrid (MPI+{threads} OMP)")
#
#     ax.set_xlabel("Number of Processes", fontweight="bold")
#     ax.set_ylabel("Execution Time (s)", fontweight="bold")
#     ax.set_title("Weak Scaling: Time Stability (Lower is Better)", fontweight="bold")
#     ax.set_xticks(all_procs)
#     ax.grid(True, alpha=0.3)
#     ax.legend()
#
#     filename = f"{PLOTS_DIR}/weak_scaling_time.png"
#     plt.savefig(filename, dpi=300, bbox_inches="tight")
#     plt.close()
#
#     # ---------------------------------------------------------
#     # Plot 2: GFLOPS (Performance Growth)
#     # ---------------------------------------------------------
#     fig, ax = plt.subplots(figsize=(10, 6))
#
#     if not mpi_data.empty:
#         ax.plot(mpi_data["NumProcesses"], mpi_data["GFLOPS"], "o-", 
#                 color=COLORS["mpi"], label="Pure MPI GFLOPS")
#
#     if not hybrid_data.empty:
#         threads = hybrid_data["NumThreads"].iloc[0]
#         ax.plot(hybrid_data["NumProcesses"], hybrid_data["GFLOPS"], "s--", 
#                 color=COLORS["hybrid"], label=f"Hybrid GFLOPS ({threads} thr)")
#
#     ax.set_xlabel("Number of Processes", fontweight="bold")
#     ax.set_ylabel("GFLOPS", fontweight="bold")
#     ax.set_title("Weak Scaling: Aggregate Performance", fontweight="bold")
#     ax.set_xticks(all_procs)
#     ax.grid(True, alpha=0.3)
#     ax.legend()
#
#     filename = f"{PLOTS_DIR}/weak_scaling_gflops.png"
#     plt.savefig(filename, dpi=300, bbox_inches="tight")
#     plt.close()
#
#     # ---------------------------------------------------------
#     # Plot 3: Weak Efficiency
#     # ---------------------------------------------------------
#     fig, ax = plt.subplots(figsize=(10, 6))
#
#     if not mpi_data.empty:
#         ax.plot(mpi_data["NumProcesses"], mpi_data["WeakEfficiency"], "o-", 
#                 color=COLORS["mpi"], label="Pure MPI")
#
#     if not hybrid_data.empty:
#         threads = hybrid_data["NumThreads"].iloc[0]
#         ax.plot(hybrid_data["NumProcesses"], hybrid_data["WeakEfficiency"], "s--", 
#                 color=COLORS["hybrid"], label=f"Hybrid ({threads} thr)")
#
#     ax.axhline(y=100, color="black", linestyle="--", alpha=0.5)
#     ax.set_xlabel("Number of Processes", fontweight="bold")
#     ax.set_ylabel("Weak Efficiency (%)", fontweight="bold")
#     ax.set_title("Weak Scaling Efficiency", fontweight="bold")
#     ax.set_xticks(all_procs)
#     ax.set_ylim(0, 115)
#     ax.grid(True, alpha=0.3)
#     ax.legend()
#
#     filename = f"{PLOTS_DIR}/weak_scaling_efficiency.png"
#     plt.savefig(filename, dpi=300, bbox_inches="tight")
#     plt.close()
#
#
# def main():
#     setup()
#     df = load_data()
#
#     plot_strong_scaling(df)
#     plot_weak_scaling(df)
#
#     print("\n=== Analysis Complete ===")
#     print(f"Plots saved in: {PLOTS_DIR}/")
#
#
# if __name__ == "__main__":
#     main()
