#ifndef CONFIG_H
#define CONFIG_H

#include <mpi.h>
#include <stdlib.h>

// Runtime configuration flags
#define TOLERANCE 1e-4
#define NUM_RUNS 3
#define ENABLE_VERIFICATION 1
#define ENABLE_CSV_LOGGING 1

// Configuration structure
typedef struct {
    int matrix_size;
    int block_size;
    int num_runs;
    double tolerance;
    int enable_verify;
    int enable_csv;
} Config;

// Process grid for 2D distribution
typedef struct {
    int rank;
    int np;
    int grid_rows;
    int grid_cols;
    int my_row;
    int my_col;
    MPI_Comm row_comm;
    MPI_Comm col_comm;
} ProcGrid;

// Timing results
typedef struct {
    double avg_time;
    double min_time;
    double max_time;
    double std_dev;
    double gflops;
} TimingResults;

// Global context - bundles all important state
typedef struct {
    Config config;
    ProcGrid grid;
    double* global_A;
    double* global_L;
    double* global_A_backup;
} CholeskyContext;

// Initialize default configuration
static inline Config config_default(int matrix_size) {
    Config cfg;
    cfg.matrix_size = matrix_size;
    cfg.block_size = 0;
    cfg.num_runs = NUM_RUNS;
    cfg.tolerance = TOLERANCE;
    cfg.enable_verify = ENABLE_VERIFICATION;
    cfg.enable_csv = ENABLE_CSV_LOGGING;
    return cfg;
}

// Initialize context
static inline CholeskyContext* context_create(int matrix_size) {
    CholeskyContext* ctx = (CholeskyContext*)malloc(sizeof(CholeskyContext));
    ctx->config = config_default(matrix_size);
    ctx->global_A = NULL;
    ctx->global_L = NULL;
    ctx->global_A_backup = NULL;
    return ctx;
}

// Free context
static inline void context_destroy(CholeskyContext* ctx) {
    if (ctx->global_A) free(ctx->global_A);
    if (ctx->global_L) free(ctx->global_L);
    if (ctx->global_A_backup) free(ctx->global_A_backup);
    free(ctx);
}

#endif // CONFIG_H
