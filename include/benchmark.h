#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "config.h"

// Compute timing statistics from multiple runs
TimingResults compute_timing_stats(const double* times, int num_runs,
                                   int matrix_size);

// Log benchmark results to CSV file
void log_to_csv(const CholeskyContext* ctx, const TimingResults* results);

// Print performance summary
void print_performance_summary(const CholeskyContext* ctx,
                               const TimingResults* results);

// Execute complete benchmark (multiple runs)
TimingResults execute_benchmark(CholeskyContext* ctx);

// Verify factorization and print result
void verify_and_report(const CholeskyContext* ctx);

#endif // BENCHMARK_H
