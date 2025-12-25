#include "../include/benchmark.h"
#include "../include/cholesky.h"
#include "../include/matrix_ops.h"
#include "../include/mpi_utils.h"
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

TimingResults compute_timing_stats(const double* times, int num_runs,
                                   int matrix_size) {
    TimingResults results;

    results.avg_time = 0.0;
    for (int i = 0; i < num_runs; i++) {
        results.avg_time += times[i];
    }
    results.avg_time /= num_runs;

    results.min_time = times[0];
    results.max_time = times[0];
    for (int i = 1; i < num_runs; i++) {
        if (times[i] < results.min_time)
            results.min_time = times[i];
        if (times[i] > results.max_time)
            results.max_time = times[i];
    }

    double variance = 0.0;
    for (int i = 0; i < num_runs; i++) {
        double diff = times[i] - results.avg_time;
        variance += diff * diff;
    }
    results.std_dev = sqrt(variance / num_runs);

    double flops = (1.0 / 3.0) * matrix_size * matrix_size * matrix_size;
    results.gflops = (flops / results.avg_time) / 1e9;

    return results;
}

void log_to_csv(const CholeskyContext* ctx, const TimingResults* results) {
    const char* csv_filename = "benchmarks/benchmark_results.csv";
    FILE* csv_file = fopen(csv_filename, "a");

    if (csv_file == NULL) {
        printf("\n✗ Warning: Could not open %s for writing\n", csv_filename);
        return;
    }

    fseek(csv_file, 0, SEEK_END);
    if (ftell(csv_file) == 0) {
        fprintf(csv_file, "MatrixSize,NumProcesses,GridRows,GridCols,");
        fprintf(csv_file, "BlockSize,NumBlocks,");
        fprintf(csv_file, "TestCategory,Placement,");
        fprintf(csv_file, "AvgTime,MinTime,MaxTime,StdDev,GFLOPS,");
        fprintf(csv_file, "WorkPerProcess,");
        fprintf(csv_file, "Timestamp\n");
    }

    time_t now = time(NULL);
    struct tm* t = localtime(&now);
    char timestamp[64];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", t);

    int num_blocks = (ctx->config.matrix_size + ctx->config.block_size - 1) /
                     ctx->config.block_size;

    // Calculate work per process (elements)
    long long total_elements =
        (long long)ctx->config.matrix_size * ctx->config.matrix_size;
    long long work_per_process = total_elements / ctx->grid.np;

    fprintf(csv_file, "%d,%d,%d,%d,%d,%d,", ctx->config.matrix_size,
            ctx->grid.np, ctx->grid.grid_rows, ctx->grid.grid_cols,
            ctx->config.block_size, num_blocks);
    fprintf(csv_file, "%s,%s,", ctx->config.test_category,
            ctx->config.placement);
    fprintf(csv_file, "%.6f,%.6f,%.6f,%.6f,%.4f,", results->avg_time,
            results->min_time, results->max_time, results->std_dev,
            results->gflops);
    fprintf(csv_file, "%lld,", work_per_process);
    fprintf(csv_file, "%s\n", timestamp);

    fclose(csv_file);
    printf("\n✓ Results appended to %s\n", csv_filename);
}

void print_performance_summary(const CholeskyContext* ctx,
                               const TimingResults* results) {
    printf("\n=== Performance Summary ===\n");
    printf("Average time: %.4f seconds\n", results->avg_time);
    printf("Min time:     %.4f seconds\n", results->min_time);
    printf("Max time:     %.4f seconds\n", results->max_time);
    printf("Std dev:      %.4f seconds (%.2f%%)\n", results->std_dev,
           (results->std_dev / results->avg_time) * 100);

    printf("\nComputational metrics:\n");
    double flops = (1.0 / 3.0) * ctx->config.matrix_size *
                   ctx->config.matrix_size * ctx->config.matrix_size;
    printf("Total FLOPs:  %.2e\n", flops);
    printf("Performance:  %.2f GFLOPS\n", results->gflops);
    printf("Time per element: %.2f ns\n",
           (results->avg_time * 1e9) /
               (ctx->config.matrix_size * ctx->config.matrix_size));

    printf("\nParallel metrics:\n");
    printf("Processes:    %d (grid: %dx%d)\n", ctx->grid.np,
           ctx->grid.grid_rows, ctx->grid.grid_cols);
    printf("Work per process: %.2f million elements\n",
           (ctx->config.matrix_size * ctx->config.matrix_size /
            (double)ctx->grid.np) /
               1e6);
}

static double run_iteration(CholeskyContext* ctx, double** local_A,
                            int* local_rows, int* local_cols, int is_last_run) {

    if (*local_A == NULL) {
        compute_local_dimensions(&ctx->grid, ctx->config.matrix_size,
                                 ctx->config.block_size, local_rows,
                                 local_cols);
        *local_A =
            (double*)malloc((*local_rows) * (*local_cols) * sizeof(double));
    }

    // Each process generates its own blocks
    generate_spd_matrix_distributed(
        *local_A, &ctx->grid, ctx->config.matrix_size, ctx->config.block_size,
        *local_rows, *local_cols);

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    parallel_cholesky(*local_A, &ctx->grid, &ctx->config, *local_rows,
                      *local_cols);

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();

    // Gather only if verification is needed
    if (is_last_run && ctx->config.enable_verify) {
        gather_matrix(*local_A, ctx->global_L, &ctx->grid,
                      ctx->config.matrix_size, ctx->config.block_size,
                      *local_rows, *local_cols);
    }

    return end - start;
}

TimingResults execute_benchmark(CholeskyContext* ctx) {
    double* local_A = NULL;
    int local_rows, local_cols;
    double* times = (double*)malloc(ctx->config.num_runs * sizeof(double));

    for (int run = 0; run < ctx->config.num_runs; run++) {
        if (ctx->grid.rank == 0 && run > 0) {
            memcpy(ctx->global_A, ctx->global_A_backup,
                   ctx->config.matrix_size * ctx->config.matrix_size *
                       sizeof(double));
        }

        int is_last_run = (run == ctx->config.num_runs - 1);
        times[run] =
            run_iteration(ctx, &local_A, &local_rows, &local_cols, is_last_run);

        if (ctx->grid.rank == 0) {
            printf("Run %d: %.4f seconds\n", run + 1, times[run]);
        }

        if (run < ctx->config.num_runs - 1) {
            free(local_A);
        }
    }

    TimingResults results = compute_timing_stats(times, ctx->config.num_runs,
                                                 ctx->config.matrix_size);
    free(times);
    free(local_A);

    return results;
}

void verify_and_report(const CholeskyContext* ctx) {
    if (ctx->config.matrix_size <= 8) {
        print_matrix("\nLower triangular L", ctx->global_L,
                     ctx->config.matrix_size, 8);
    }

    printf("\n=== Verification ===\n");
    int success =
        verify_factorization(ctx->global_A_backup, ctx->global_L,
                             ctx->config.matrix_size, ctx->config.tolerance);

    if (success) {
        printf("✓ Verification PASSED: L * L^T = A (within tolerance)\n");
    } else {
        printf("✗ Verification FAILED: L * L^T ≠ A\n");
    }
}
