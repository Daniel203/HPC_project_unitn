#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "config.h"

// Compute timing statistics from multiple runs
TimingResults compute_timing_stats(const double* times, int num_runs,
                                   int matrix_size);

// Log benchmark results to CSV file
void log_to_csv(const ProcGrid* grid, const Config* cfg,
                const TimingResults* results);

// Print performance summary
void print_performance_summary(const ProcGrid* grid, const Config* cfg,
                               const TimingResults* results);

// Execute complete benchmark (multiple runs)
TimingResults execute_benchmark(double* global_A, double* global_A_backup,
                                double* global_L, ProcGrid* grid, Config* cfg);

// Verify factorization and print result
void verify_and_report(double* global_A_backup, double* global_L,
                       const Config* cfg);

#endif // BENCHMARK_H
