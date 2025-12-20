#include "../include/benchmark.h"
#include "../include/config.h"
#include "../include/matrix_ops.h"
#include "../include/mpi_utils.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Parse matrix size from command line
static int parse_matrix_size(int argc, char* argv[]) {
    if (argc >= 2) {
        int n = atoi(argv[1]);
        if (n > 0) return n;
        printf("Invalid matrix size '%s', using default 1024\n", argv[1]);
    }
    return 1024;
}

// Allocate global matrices (root only)
static void allocate_global_matrices(CholeskyContext* ctx) {
    int n = ctx->config.matrix_size;
    ctx->global_A = (double*)malloc(n * n * sizeof(double));
    ctx->global_L = (double*)malloc(n * n * sizeof(double));
    ctx->global_A_backup = (double*)malloc(n * n * sizeof(double));

    generate_spd_matrix(ctx->global_A, n);
    memcpy(ctx->global_A_backup, ctx->global_A, n * n * sizeof(double));

    if (n <= 8) {
        print_matrix("Original matrix A", ctx->global_A, n, 8);
    }
}

// Print initial configuration info
static void print_config_info(const CholeskyContext* ctx) {
    printf("Matrix size: %d x %d\n", ctx->config.matrix_size, 
           ctx->config.matrix_size);
    printf("Number of blocks: %d\n",
           (ctx->config.matrix_size + ctx->config.block_size - 1) / 
           ctx->config.block_size);
    printf("Total matrix elements: %.2f million\n",
           (ctx->config.matrix_size * ctx->config.matrix_size) / 1e6);
    printf("Memory per matrix: %.2f MB\n",
           (ctx->config.matrix_size * ctx->config.matrix_size * sizeof(double)) / 
           (1024.0 * 1024.0));
    printf("Running %d iterations for timing...\n\n", ctx->config.num_runs);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    // Parse and broadcast matrix size
    int matrix_size = parse_matrix_size(argc, argv);
    MPI_Bcast(&matrix_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Create context
    CholeskyContext* ctx = context_create(matrix_size);

    // Initialize process grid
    init_process_grid(&ctx->grid, &ctx->config);

    // Root allocates and initializes matrices
    if (ctx->grid.rank == 0) {
        print_config_info(ctx);
        allocate_global_matrices(ctx);
    }

    // Execute benchmark
    TimingResults results = execute_benchmark(ctx);

    // Root prints results
    if (ctx->grid.rank == 0) {
        print_performance_summary(ctx, &results);

        if (ctx->config.enable_csv) {
            log_to_csv(ctx, &results);
        }

        if (ctx->config.enable_verify) {
            verify_and_report(ctx);
        } else {
            printf("\n(Verification disabled for performance testing)\n");
        }
    }

    // Cleanup
    cleanup_process_grid(&ctx->grid);
    context_destroy(ctx);
    MPI_Finalize();

    return 0;
}
