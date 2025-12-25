#include "../include/benchmark.h"
#include "../include/config.h"
#include "../include/matrix_ops.h"
#include "../include/mpi_utils.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/benchmark.h"
#include "../include/config.h"
#include "../include/matrix_ops.h"
#include "../include/mpi_utils.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Parse command line options
static void parse_command_line_args(int argc, char* argv[], int* matrix_size,
                                    char* test_category, char* placement) {
    // Defaults
    *matrix_size = 1024;
    strncpy(test_category, "unknown", 64);
    strncpy(placement, "unknown", 16);

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--matrix-size") == 0 && i + 1 < argc) {
            *matrix_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--test-category") == 0 && i + 1 < argc) {
            strncpy(test_category, argv[++i], 64 - 1);
        } else if (strcmp(argv[i], "--placement") == 0 && i + 1 < argc) {
            strncpy(placement, argv[++i], 16 - 1);
        }
    }
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

// Allocate only for verification/output (optional, root only)
static void allocate_verification_matrices(CholeskyContext* ctx) {
    if (ctx->config.enable_verify) {
        int n = ctx->config.matrix_size;
        ctx->global_L = (double*)malloc(n * n * sizeof(double));
        ctx->global_A_backup = (double*)malloc(n * n * sizeof(double));
        
        // Generate reference matrix for verification only
        ctx->global_A_backup = (double*)malloc(n * n * sizeof(double));
        generate_spd_matrix(ctx->global_A_backup, n);
        
        if (n <= 8) {
            print_matrix("Original matrix A", ctx->global_A_backup, n, 8);
        }
    } else {
        // No global matrices needed at all!
        ctx->global_A = NULL;
        ctx->global_L = NULL;
        ctx->global_A_backup = NULL;
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
    printf(
        "Memory per matrix: %.2f MB\n",
        (ctx->config.matrix_size * ctx->config.matrix_size * sizeof(double)) /
            (1024.0 * 1024.0));
    printf("Test category: %s\n", ctx->config.test_category);
    printf("Placement: %s\n", ctx->config.placement);
    printf("Running %d iterations for timing...\n\n", ctx->config.num_runs);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    // Parse and broadcast cmd argumenst
    int matrix_size;
    char test_category[64];
    char placement[16];

    parse_command_line_args(argc, argv, &matrix_size, test_category, placement);

    MPI_Bcast(&matrix_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(test_category, 64, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(placement, 16, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Create context
    CholeskyContext* ctx = context_create(matrix_size);
    strncpy(ctx->config.test_category, test_category, 64);
    strncpy(ctx->config.placement, placement, 16);


    // Initialize process grid
    init_process_grid(&ctx->grid, &ctx->config);

    // Root allocates and initializes matrices
    if (ctx->grid.rank == 0) {
        print_config_info(ctx);
        // allocate_global_matrices(ctx);
        // Only allocate global matrices if verification is enabled
        allocate_verification_matrices(ctx);
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
