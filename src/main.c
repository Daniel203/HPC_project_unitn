#include "../include/benchmark.h"
#include "../include/config.h"
#include "../include/matrix_ops.h"
#include "../include/mpi_utils.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// Parse command line options
static void parse_command_line_args(int argc, char* argv[], int* matrix_size,
                                    char* test_category, char* placement,
                                    int* enable_openmp, int* num_threads) {
    // Defaults
    *matrix_size = 1024;
    strncpy(test_category, "unknown", 64);
    strncpy(placement, "unknown", 16);
    *enable_openmp = 0;  // Default: OpenMP disabled
    *num_threads = 0;    // 0 = auto-detect

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--matrix-size") == 0 && i + 1 < argc) {
            *matrix_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--test-category") == 0 && i + 1 < argc) {
            strncpy(test_category, argv[++i], 64 - 1);
        } else if (strcmp(argv[i], "--placement") == 0 && i + 1 < argc) {
            strncpy(placement, argv[++i], 16 - 1);
        } else if (strcmp(argv[i], "--enable-openmp") == 0) {
            *enable_openmp = 1;
        } else if (strcmp(argv[i], "--num-threads") == 0 && i + 1 < argc) {
            *num_threads = atoi(argv[++i]);
        }
    }
}

// Allocate only for verification/output (optional, root only)
static void allocate_verification_matrices(CholeskyContext* ctx) {
    if (ctx->config.enable_verify) {
        int n = ctx->config.matrix_size;
        ctx->global_L = (double*)malloc(n * n * sizeof(double));
        ctx->global_A_backup = (double*)malloc(n * n * sizeof(double));
        
        // Generate reference matrix for verification only
        generate_spd_matrix(ctx->global_A_backup, n);
        
        if (n <= 8) {
            print_matrix("Original matrix A", ctx->global_A_backup, n, 8);
        }
    } else {
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
    printf("Memory per matrix: %.2f MB\n",
           (ctx->config.matrix_size * ctx->config.matrix_size * sizeof(double)) /
               (1024.0 * 1024.0));
    printf("Test category: %s\n", ctx->config.test_category);
    printf("Placement: %s\n", ctx->config.placement);
    
    #ifdef _OPENMP
    if (ctx->config.enable_openmp) {
        printf("OpenMP: ENABLED (%d threads per MPI process)\n", ctx->config.num_threads);
    } else {
        printf("OpenMP: DISABLED (MPI only)\n");
    }
    #else
    printf("OpenMP: NOT COMPILED\n");
    #endif
    
    printf("Running %d iterations for timing...\n\n", ctx->config.num_runs);
}

int main(int argc, char* argv[]) {
    // Initialize MPI with thread support
    int provided;
    #ifdef _OPENMP
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
            printf("Warning: MPI thread support level insufficient\n");
        }
    }
    #else
    MPI_Init(&argc, &argv);
    #endif

    // Parse command line arguments
    int matrix_size;
    char test_category[64];
    char placement[16];
    int enable_openmp;
    int num_threads;

    parse_command_line_args(argc, argv, &matrix_size, test_category, placement,
                           &enable_openmp, &num_threads);

    // Broadcast parameters to all processes
    MPI_Bcast(&matrix_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(test_category, 64, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(placement, 16, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&enable_openmp, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_threads, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Create context
    CholeskyContext* ctx = context_create(matrix_size);
    strncpy(ctx->config.test_category, test_category, 64);
    strncpy(ctx->config.placement, placement, 16);
    ctx->config.enable_openmp = enable_openmp;
    
    #ifdef _OPENMP
    if (enable_openmp) {
        if (num_threads > 0) {
            omp_set_num_threads(num_threads);
            ctx->config.num_threads = num_threads;
        } else {
            // Auto-detect
            #pragma omp parallel
            {
                #pragma omp single
                ctx->config.num_threads = omp_get_num_threads();
            }
        }
    } else {
        ctx->config.num_threads = 1;
    }
    #else
    ctx->config.num_threads = 1;
    if (enable_openmp) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
            printf("Warning: OpenMP requested but not compiled in. Ignoring.\n");
        }
        ctx->config.enable_openmp = 0;
    }
    #endif

    // Initialize process grid
    init_process_grid(&ctx->grid, &ctx->config);

    // Root prints configuration
    if (ctx->grid.rank == 0) {
        int total_cores = ctx->grid.np * ctx->config.num_threads;
        printf("=== Hybrid Parallelization ===\n");
        printf("MPI processes: %d\n", ctx->grid.np);
        printf("OpenMP threads per process: %d\n", ctx->config.num_threads);
        printf("Total cores: %d\n", total_cores);
        printf("===============================\n\n");

        print_config_info(ctx);
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
