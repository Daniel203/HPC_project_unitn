#define _POSIX_C_SOURCE 199309L

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Configuration constants
#define N 3072              // Matrix dimension
#define TOLERANCE 1e-4      // Tolerance for verification

// Block size will be computed automatically based on N and number of processes
int BLOCK_SIZE;

// Process grid structure
typedef struct {
    int rank;           // Process rank
    int np;             // Total number of processes
    int grid_rows;      // Number of process rows
    int grid_cols;      // Number of process columns
    int my_row;         // My row in process grid
    int my_col;         // My column in process grid
    MPI_Comm row_comm;  // Row communicator
    MPI_Comm col_comm;  // Column communicator
} ProcGrid;

// Initialize process grid and compute optimal block size
void init_proc_grid(ProcGrid *grid) {
    MPI_Comm_rank(MPI_COMM_WORLD, &grid->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &grid->np);
    
    // Find best factorization of np into grid_rows x grid_cols
    grid->grid_rows = (int)sqrt(grid->np);
    while (grid->np % grid->grid_rows != 0) {
        grid->grid_rows--;
    }
    grid->grid_cols = grid->np / grid->grid_rows;
    
    grid->my_row = grid->rank / grid->grid_cols;
    grid->my_col = grid->rank % grid->grid_cols;
    
    MPI_Comm_split(MPI_COMM_WORLD, grid->my_row, grid->my_col, &grid->row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, grid->my_col, grid->my_row, &grid->col_comm);
    
    // Compute optimal block size
    // Strategy: We want num_blocks to be close to grid dimension for good load balance
    // But blocks shouldn't be too small (minimum 64) or too large (maximum N/2)
    int min_block_size = 64;
    int max_block_size = N / 2;
    
    // Target: num_blocks ≈ max(grid_rows, grid_cols) for even distribution
    int target_blocks = (grid->grid_rows > grid->grid_cols) ? grid->grid_rows : grid->grid_cols;
    
    // Calculate ideal block size
    BLOCK_SIZE = N / target_blocks;
    
    // Ensure block size is within bounds
    if (BLOCK_SIZE < min_block_size) BLOCK_SIZE = min_block_size;
    if (BLOCK_SIZE > max_block_size) BLOCK_SIZE = max_block_size;
    
    // Round to a nice number (power of 2 or multiple of 32 for cache efficiency)
    if (BLOCK_SIZE >= 128) {
        BLOCK_SIZE = ((BLOCK_SIZE + 127) / 128) * 128;
    } else if (BLOCK_SIZE >= 64) {
        BLOCK_SIZE = ((BLOCK_SIZE + 63) / 64) * 64;
    } else {
        BLOCK_SIZE = ((BLOCK_SIZE + 31) / 32) * 32;
    }
    
    if (grid->rank == 0) {
        int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        printf("Process grid: %d x %d = %d processes\n", 
               grid->grid_rows, grid->grid_cols, grid->np);
        printf("Auto-computed block size: %d (results in %d blocks)\n", 
               BLOCK_SIZE, num_blocks);
        if (num_blocks > grid->grid_rows * 2 || num_blocks > grid->grid_cols * 2) {
            printf("WARNING: Number of blocks (%d) is much larger than grid dimensions (%dx%d)\n",
                   num_blocks, grid->grid_rows, grid->grid_cols);
            printf("         This may cause load imbalance. Consider using more processes.\n");
        }
    }
}

// Compute local matrix dimensions
void compute_local_dims(ProcGrid *grid, int *local_rows, int *local_cols, 
                       int *row_start, int *col_start) {
    int rows_per_proc = N / grid->grid_rows;
    int cols_per_proc = N / grid->grid_cols;
    int extra_rows = N % grid->grid_rows;
    int extra_cols = N % grid->grid_cols;
    
    *row_start = grid->my_row * rows_per_proc + (grid->my_row < extra_rows ? grid->my_row : extra_rows);
    *col_start = grid->my_col * cols_per_proc + (grid->my_col < extra_cols ? grid->my_col : extra_cols);
    
    *local_rows = rows_per_proc + (grid->my_row < extra_rows ? 1 : 0);
    *local_cols = cols_per_proc + (grid->my_col < extra_cols ? 1 : 0);
}

// Generate symmetric positive definite matrix
void generate_spd_matrix(double *A, int n) {
    // Create a more strongly diagonally dominant matrix for numerical stability
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i * n + j] = (i == j) ? (n + 1.0) : 1.0 / (1.0 + abs(i - j));
        }
    }
}

// Distribute global matrix to local blocks
void distribute_matrix(double *global_A, double **local_A, ProcGrid *grid,
                      int *local_rows, int *local_cols) {
    int row_start, col_start;
    compute_local_dims(grid, local_rows, local_cols, &row_start, &col_start);
    
    *local_A = (double *)calloc((*local_rows) * (*local_cols), sizeof(double));
    
    if (grid->rank == 0) {
        for (int p = 0; p < grid->np; p++) {
            int dest_row = p / grid->grid_cols;
            int dest_col = p % grid->grid_cols;
            
            ProcGrid temp_grid = *grid;
            temp_grid.my_row = dest_row;
            temp_grid.my_col = dest_col;
            
            int dest_local_rows, dest_local_cols, dest_row_start, dest_col_start;
            compute_local_dims(&temp_grid, &dest_local_rows, &dest_local_cols, 
                             &dest_row_start, &dest_col_start);
            
            double *send_buf = (double *)malloc(dest_local_rows * dest_local_cols * sizeof(double));
            
            for (int i = 0; i < dest_local_rows; i++) {
                for (int j = 0; j < dest_local_cols; j++) {
                    send_buf[i * dest_local_cols + j] = 
                        global_A[(dest_row_start + i) * N + (dest_col_start + j)];
                }
            }
            
            if (p == 0) {
                memcpy(*local_A, send_buf, dest_local_rows * dest_local_cols * sizeof(double));
            } else {
                MPI_Send(send_buf, dest_local_rows * dest_local_cols, MPI_DOUBLE, 
                        p, 0, MPI_COMM_WORLD);
            }
            free(send_buf);
        }
    } else {
        MPI_Recv(*local_A, (*local_rows) * (*local_cols), MPI_DOUBLE, 
                0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

// Sequential Cholesky for a block
void cholesky_block(double *A, int n, int ld) {
    for (int k = 0; k < n; k++) {
        A[k * ld + k] = sqrt(A[k * ld + k]);
        for (int i = k + 1; i < n; i++) {
            A[i * ld + k] /= A[k * ld + k];
        }
        for (int j = k + 1; j < n; j++) {
            for (int i = j; i < n; i++) {
                A[i * ld + j] -= A[i * ld + k] * A[j * ld + k];
            }
        }
    }
}

// TRSM: solve L * X = B where L is lower triangular
void trsm_block(double *L, double *B, int n, int m, int ld_L, int ld_B) {
    for (int k = 0; k < n; k++) {
        for (int j = 0; j < m; j++) {
            B[j * ld_B + k] /= L[k * ld_L + k];
        }
        for (int i = k + 1; i < n; i++) {
            for (int j = 0; j < m; j++) {
                B[j * ld_B + i] -= L[i * ld_L + k] * B[j * ld_B + k];
            }
        }
    }
}

// SYRK: C = C - A * A^T (lower triangular C)
void syrk_block(double *A, double *C, int m, int n, int k, int ld_A, int ld_C) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j <= i && j < n; j++) {
            double sum = 0.0;
            for (int p = 0; p < k; p++) {
                sum += A[i * ld_A + p] * A[j * ld_A + p];
            }
            C[i * ld_C + j] -= sum;
        }
    }
}

// GEMM: C = C - A * B^T
void gemm_block(double *A, double *B, double *C, int m, int n, int k, 
                int ld_A, int ld_B, int ld_C) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int p = 0; p < k; p++) {
                sum += A[i * ld_A + p] * B[j * ld_B + p];
            }
            C[i * ld_C + j] -= sum;
        }
    }
}

// Parallel Cholesky factorization
void cholesky_factorization(double *local_A, ProcGrid *grid, 
                           int local_rows, int local_cols) {
    int row_start, col_start;
    int dummy_rows, dummy_cols;
    compute_local_dims(grid, &dummy_rows, &dummy_cols, &row_start, &col_start);
    int row_end = row_start + local_rows;
    int col_end = col_start + local_cols;
    
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (int k = 0; k < num_blocks; k++) {
        int block_start = k * BLOCK_SIZE;
        int block_size = (block_start + BLOCK_SIZE <= N) ? BLOCK_SIZE : N - block_start;
        int block_end = block_start + block_size;
        
        // Find which process owns this diagonal block
        int owner_row = -1, owner_col = -1;
        for (int pr = 0; pr < grid->grid_rows; pr++) {
            ProcGrid temp = *grid;
            temp.my_row = pr;
            temp.my_col = 0;
            int lr, lc, rs, cs;
            compute_local_dims(&temp, &lr, &lc, &rs, &cs);
            if (rs <= block_start && block_start < rs + lr) {
                owner_row = pr;
                break;
            }
        }
        for (int pc = 0; pc < grid->grid_cols; pc++) {
            ProcGrid temp = *grid;
            temp.my_row = 0;
            temp.my_col = pc;
            int lr, lc, rs, cs;
            compute_local_dims(&temp, &lr, &lc, &rs, &cs);
            if (cs <= block_start && block_start < cs + lc) {
                owner_col = pc;
                break;
            }
        }
        
        if (owner_row == -1 || owner_col == -1) {
            if (grid->rank == 0) {
                printf("ERROR: Could not find owner for block %d\n", k);
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        double *diag_block = (double *)calloc(BLOCK_SIZE * BLOCK_SIZE, sizeof(double));
        
        // Step 1: Factorize diagonal block
        if (grid->my_row == owner_row && grid->my_col == owner_col) {
            int local_offset_row = block_start - row_start;
            int local_offset_col = block_start - col_start;
            
            // Copy to diag_block
            for (int i = 0; i < block_size; i++) {
                for (int j = 0; j < block_size; j++) {
                    diag_block[i * BLOCK_SIZE + j] = 
                        local_A[(local_offset_row + i) * local_cols + (local_offset_col + j)];
                }
            }
            
            // Factorize
            cholesky_block(diag_block, block_size, BLOCK_SIZE);
            
            // Copy back (only lower triangular)
            for (int i = 0; i < block_size; i++) {
                for (int j = 0; j <= i && j < block_size; j++) {
                    local_A[(local_offset_row + i) * local_cols + (local_offset_col + j)] = 
                        diag_block[i * BLOCK_SIZE + j];
                }
            }
        }
        
        // Broadcast diagonal block to all processes
        MPI_Bcast(diag_block, BLOCK_SIZE * BLOCK_SIZE, MPI_DOUBLE, owner_col, grid->row_comm);
        MPI_Bcast(diag_block, BLOCK_SIZE * BLOCK_SIZE, MPI_DOUBLE, owner_row, grid->col_comm);
        
        // Step 2: Update column blocks below diagonal (TRSM)
        if (grid->my_col == owner_col && col_start <= block_start && block_start < col_end && row_start < N) {
            int local_offset_col = block_start - col_start;
            int start_local_row = (row_start < block_end) ? (block_end - row_start) : 0;
            
            if (start_local_row < local_rows) {
                for (int i = start_local_row; i < local_rows; i++) {
                    for (int j = 0; j < block_size; j++) {
                        double sum = local_A[i * local_cols + local_offset_col + j];
                        for (int p = 0; p < j; p++) {
                            sum -= local_A[i * local_cols + local_offset_col + p] * 
                                   diag_block[j * BLOCK_SIZE + p];
                        }
                        local_A[i * local_cols + local_offset_col + j] = 
                            sum / diag_block[j * BLOCK_SIZE + j];
                    }
                }
            }
        }
        
        // Gather updated column panel
        double *col_panel = (double *)calloc(N * BLOCK_SIZE, sizeof(double));
        
        if (grid->my_col == owner_col && col_start <= block_start && block_start < col_end) {
            int local_offset_col = block_start - col_start;
            
            for (int i = 0; i < local_rows; i++) {
                int global_i = row_start + i;
                if (global_i >= block_end) {
                    for (int j = 0; j < block_size; j++) {
                        col_panel[global_i * BLOCK_SIZE + j] = 
                            local_A[i * local_cols + local_offset_col + j];
                    }
                }
            }
        }
        
        MPI_Allreduce(MPI_IN_PLACE, col_panel, N * BLOCK_SIZE, MPI_DOUBLE, MPI_SUM, grid->col_comm);
        MPI_Bcast(col_panel, N * BLOCK_SIZE, MPI_DOUBLE, owner_col, grid->row_comm);
        
        // Step 3: Update trailing submatrix
        if (local_rows > 0 && local_cols > 0) {
            for (int i = 0; i < local_rows; i++) {
                int global_i = row_start + i;
                if (global_i >= block_end) {
                    for (int j = 0; j < local_cols; j++) {
                        int global_j = col_start + j;
                        if (global_j >= block_end && global_j <= global_i) {
                            double sum = 0.0;
                            for (int p = 0; p < block_size; p++) {
                                sum += col_panel[global_i * BLOCK_SIZE + p] * 
                                       col_panel[global_j * BLOCK_SIZE + p];
                            }
                            local_A[i * local_cols + j] -= sum;
                        }
                    }
                }
            }
        }
        
        free(col_panel);
        free(diag_block);
    }
}

// Gather local matrices to global matrix
void gather_matrix(double *local_A, double *global_L, ProcGrid *grid,
                  int local_rows, int local_cols) {
    if (grid->rank == 0) {
        memset(global_L, 0, N * N * sizeof(double));
        
        for (int p = 0; p < grid->np; p++) {
            int src_row = p / grid->grid_cols;
            int src_col = p % grid->grid_cols;
            
            ProcGrid temp_grid = *grid;
            temp_grid.my_row = src_row;
            temp_grid.my_col = src_col;
            
            int src_local_rows, src_local_cols, src_row_start, src_col_start;
            compute_local_dims(&temp_grid, &src_local_rows, &src_local_cols,
                             &src_row_start, &src_col_start);
            
            double *recv_buf;
            if (p == 0) {
                recv_buf = local_A;
            } else {
                recv_buf = (double *)malloc(src_local_rows * src_local_cols * sizeof(double));
                MPI_Recv(recv_buf, src_local_rows * src_local_cols, MPI_DOUBLE, 
                        p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            
            for (int i = 0; i < src_local_rows; i++) {
                for (int j = 0; j < src_local_cols; j++) {
                    global_L[(src_row_start + i) * N + (src_col_start + j)] = 
                        recv_buf[i * src_local_cols + j];
                }
            }
            
            if (p != 0) free(recv_buf);
        }
    } else {
        MPI_Send(local_A, local_rows * local_cols, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
}

// Verify that L * L^T = A
int verify_factorization(double *A, double *L, int n) {
    double max_error = 0.0;
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0.0;
            for (int k = 0; k <= j; k++) {
                sum += L[i * n + k] * L[j * n + k];
            }
            double error = fabs(sum - A[i * n + j]);
            if (error > max_error) max_error = error;
        }
    }
    
    printf("Maximum error: %e\n", max_error);
    return max_error < TOLERANCE;
}

void print_matrix(const char *name, double *A, int n, int max_print) {
    printf("%s:\n", name);
    int print_n = (n < max_print) ? n : max_print;
    for (int i = 0; i < print_n; i++) {
        for (int j = 0; j < print_n; j++) {
            printf("%8.4f ", A[i * n + j]);
        }
        printf("\n");
    }
    if (n > max_print) printf("... (showing %dx%d of %dx%d)\n", max_print, max_print, n, n);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    
    ProcGrid grid;
    init_proc_grid(&grid);
    
    double *global_A = NULL;
    double *global_L = NULL;
    
    if (grid.rank == 0) {
        printf("Matrix size: %d x %d\n", N, N);
        printf("Number of blocks: %d\n", (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
        
        global_A = (double *)malloc(N * N * sizeof(double));
        global_L = (double *)malloc(N * N * sizeof(double));
        generate_spd_matrix(global_A, N);
        
        if (N <= 8) {
            print_matrix("Original matrix A", global_A, N, N);
        }
    }
    
    double *local_A;
    int local_rows, local_cols;
    distribute_matrix(global_A, &local_A, &grid, &local_rows, &local_cols);
    
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    
    cholesky_factorization(local_A, &grid, local_rows, local_cols);
    
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    
    if (grid.rank == 0) {
        printf("\nCholesky factorization completed in %.4f seconds\n", end_time - start_time);
    }
    
    gather_matrix(local_A, global_L, &grid, local_rows, local_cols);
    
    if (grid.rank == 0) {
        if (N <= 8) {
            print_matrix("\nLower triangular L", global_L, N, N);
        }
        
        printf("\nVerifying factorization...\n");
        int success = verify_factorization(global_A, global_L, N);
        
        if (success) {
            printf("✓ Verification PASSED: L * L^T = A (within tolerance)\n");
        } else {
            printf("✗ Verification FAILED: L * L^T ≠ A\n");
        }
        
        free(global_A);
        free(global_L);
    }
    
    free(local_A);
    MPI_Comm_free(&grid.row_comm);
    MPI_Comm_free(&grid.col_comm);
    MPI_Finalize();
    
    return 0;
}
