#!/usr/bin/env python3
"""
Complete Cholesky MPI Benchmark Analysis
Generates publication-ready plots for each test category
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configuration
CSV_FILE = "benchmarks/benchmark_results.csv"
PLOTS_DIR = "plots"

# Professional styling
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.dpi"] = 100

COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def setup():
    """Create output directory"""
    Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {PLOTS_DIR}/\n")


def load_data():
    """Load benchmark data from CSV"""
    df = pd.read_csv(CSV_FILE)
    print(f"Loaded data from {CSV_FILE}, total records: {len(df)}")
    return df


def plot_strong_scaling(df):
    """
    Strong Scaling: Fixed problem size, varying processes
    Expected: Speedup should approach linear (ideal line)
    """
    print("=== Generating Strong Scaling Plots ===")

    strong = df[df["TestCategory"] == "strong"].copy()
    matrix_sizes = sorted(strong["MatrixSize"].unique())

    # Create 2x2 subplot for each matrix size
    for matrix_size in matrix_sizes:
        subset = strong[strong["MatrixSize"] == matrix_size].sort_values("NumProcesses")

        # Skip if insufficient data
        if len(subset) < 2:
            continue

        procs = subset["NumProcesses"].values
        times = subset["AvgTime"].values
        gflops = subset["GFLOPS"].values

        # Calculate speedup
        base_time = times[0]
        base_procs = procs[0]
        speedup = base_time / times
        ideal_speedup = procs / base_procs
        efficiency = (speedup / ideal_speedup) * 100

        # Plot execution time
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            procs,
            times,
            "o-",
            color=COLORS[0],
            linewidth=2,
            markersize=8,
            label="Measured",
        )
        ax.set_xlabel("Number of Processes", fontweight="bold")
        ax.set_ylabel("Execution Time (seconds)", fontweight="bold")
        ax.set_title(f"Execution Time vs Processes (N={matrix_size})", fontweight="bold")
        ax.set_xscale("linear")
        ax.set_xticks(procs)
        ax.set_xticklabels(procs)
        ax.set_yscale("linear")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        filename = f"{PLOTS_DIR}/strong_scaling_N{matrix_size}_time.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"  Saved: {filename}")
        plt.close()

        # Plot Speedup vs Ideal
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            procs,
            speedup,
            "o-",
            color=COLORS[1],
            linewidth=2,
            markersize=8,
            label="Actual Speedup",
        )
        ax.plot(
            procs,
            ideal_speedup,
            "--",
            color="black",
            linewidth=2,
            alpha=0.7,
            label="Ideal Speedup",
        )
        ax.set_xlabel("Number of Processes", fontweight="bold")
        ax.set_ylabel("Speedup", fontweight="bold")
        ax.set_title(f"Speedup vs Ideal (N={matrix_size})", fontweight="bold")
        ax.set_xscale("linear")
        ax.set_xticks(procs)
        ax.set_xticklabels(procs)
        ax.set_yscale("linear")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        filename = f"{PLOTS_DIR}/strong_scaling_N{matrix_size}_speedup.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"  Saved: {filename}")
        plt.close()

        # Plot Efficiency
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            procs,
            efficiency,
            "s-",
            color=COLORS[2],
            linewidth=2,
            markersize=8,
            label="Efficiency",
        )
        ax.axhline(
            y=100,
            color="black",
            linestyle="--",
            linewidth=1.5,
            alpha=0.5,
            label="Ideal (100%)",
        )
        ax.axhline(
            y=80,
            color="orange",
            linestyle=":",
            linewidth=1.5,
            alpha=0.5,
            label="80% threshold",
        )
        ax.set_xlabel("Number of Processes", fontweight="bold")
        ax.set_ylabel("Parallel Efficiency (%)", fontweight="bold")
        ax.set_title(f"Parallel Efficiency (N={matrix_size})", fontweight="bold")
        ax.set_xscale("linear")
        ax.set_xticks(procs)
        ax.set_xticklabels(procs)
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add efficiency values
        for i, (p, e) in enumerate(zip(procs, efficiency)):
            ax.annotate(
                f"{e:.1f}%",
                (p, e),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=8,
            )

        plt.tight_layout()
        filename = f"{PLOTS_DIR}/strong_scaling_N{matrix_size}_efficiency.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"  Saved: {filename}")
        plt.close()


def plot_weak_scaling(df):
    """
    Weak Scaling: Problem size grows with number of processes
    Expected: Execution time should remain approximately constant
    """
    print("=== Generating Weak Scaling Plots ===")

    weak = df[df["TestCategory"] == "weak"].copy()

    # Sort by number of processes
    weak = weak.sort_values("NumProcesses")

    # Skip if insufficient data
    if len(weak) < 2:
        print("  Not enough weak scaling data, skipping.")
        return

    procs = weak["NumProcesses"].values
    matrix_sizes = weak["MatrixSize"].values
    times = weak["AvgTime"].values
    gflops = weak["GFLOPS"].values

    # Baseline for weak scaling efficiency (smallest P)
    base_time = times[0]
    base_procs = procs[0]
    efficiency = (base_time / times) * 100

    # Plot Execution Time vs Processes
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        procs,
        times,
        "o-",
        color=COLORS[0],
        linewidth=2,
        markersize=8,
        label="Measured",
    )
    ax.set_xlabel("Number of Processes", fontweight="bold")
    ax.set_ylabel("Execution Time (seconds)", fontweight="bold")
    ax.set_title("Weak Scaling: Execution Time vs Processes", fontweight="bold")
    ax.set_xscale("linear")
    ax.set_xticks(procs)
    ax.set_xticklabels(procs)
    ax.set_yscale("linear")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Annotate matrix sizes
    for p, n, t in zip(procs, matrix_sizes, times):
        ax.annotate(
            f"N={n}",
            (p, t),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )

    plt.tight_layout()
    filename = f"{PLOTS_DIR}/weak_scaling_time.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"  Saved: {filename}")
    plt.close()

    # Plot GFLOPS vs Processes
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        procs,
        gflops,
        "s-",
        color=COLORS[1],
        linewidth=2,
        markersize=8,
        label="Achieved GFLOPS",
    )
    ax.set_xlabel("Number of Processes", fontweight="bold")
    ax.set_ylabel("GFLOPS", fontweight="bold")
    ax.set_title("Weak Scaling: Performance Growth", fontweight="bold")
    ax.set_xscale("linear")
    ax.set_xticks(procs)
    ax.set_xticklabels(procs)
    ax.set_yscale("linear")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    filename = f"{PLOTS_DIR}/weak_scaling_gflops.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"  Saved: {filename}")
    plt.close()

    # Plot Weak Scaling Efficiency
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        procs,
        efficiency,
        "o-",
        color=COLORS[2],
        linewidth=2,
        markersize=8,
        label="Weak Scaling Efficiency",
    )
    ax.axhline(
        y=100,
        color="black",
        linestyle="--",
        linewidth=1.5,
        alpha=0.5,
        label="Ideal (100%)",
    )
    ax.axhline(
        y=80,
        color="orange",
        linestyle=":",
        linewidth=1.5,
        alpha=0.5,
        label="80% threshold",
    )
    ax.set_xlabel("Number of Processes", fontweight="bold")
    ax.set_ylabel("Efficiency (%)", fontweight="bold")
    ax.set_title("Weak Scaling Efficiency", fontweight="bold")
    ax.set_xscale("linear")
    ax.set_xticks(procs)
    ax.set_xticklabels(procs)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add efficiency values
    for p, e in zip(procs, efficiency):
        ax.annotate(
            f"{e:.1f}%",
            (p, e),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=8,
        )

    plt.tight_layout()
    filename = f"{PLOTS_DIR}/weak_scaling_efficiency.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"  Saved: {filename}")
    plt.close()


def main():
    setup()
    df = load_data()

    plot_strong_scaling(df)
    plot_weak_scaling(df)
    # plot_size_sweep(df)
    # plot_placement_comparison(df)
    # plot_overhead_analysis(df)
    # plot_sweetspot_analysis(df)
    # plot_granularity_analysis(df)

    print("\n=== Analysis Complete ===")
    print(f"Plots saved in: {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
