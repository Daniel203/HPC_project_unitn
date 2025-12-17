#include "../include/cholesky.h"
#include "../include/mpi_utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Sequential Cholesky factorization for a single block
static void cholesky_block(double* A, int n, int ld) {
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

// Find which process owns a specific block
static void find_block_owner(const ProcGrid* grid, int block_start,
                             int matrix_size, int* owner_row, int* owner_col) {
    *owner_row = -1;
    *owner_col = -1;

    for (int pr = 0; pr < grid->grid_rows; pr++) {
        ProcGrid temp = *grid;
        temp.my_row = pr;
        temp.my_col = 0;
        int lr, lc, rs, cs;
        compute_local_dimensions(&temp, matrix_size, &lr, &lc, &rs, &cs);
        if (rs <= block_start && block_start < rs + lr) {
            *owner_row = pr;
            break;
        }
    }

    for (int pc = 0; pc < grid->grid_cols; pc++) {
        ProcGrid temp = *grid;
        temp.my_row = 0;
        temp.my_col = pc;
        int lr, lc, rs, cs;
        compute_local_dimensions(&temp, matrix_size, &lr, &lc, &rs, &cs);
        if (cs <= block_start && block_start < cs + lc) {
            *owner_col = pc;
            break;
        }
    }
}

// Update column below diagonal block (TRSM operation)
static void update_column_panel(double* local_A, const double* diag_block,
                                const ProcGrid* grid, const Config* cfg,
                                int local_rows, int local_cols, int row_start,
                                int col_start, int block_start, int block_size,
                                int block_end, int owner_col) {
    if (grid->my_col != owner_col || col_start > block_start ||
        block_start >= col_start + local_cols) {
        return;
    }

    int local_offset_col = block_start - col_start;
    int start_local_row = (row_start < block_end) ? (block_end - row_start) : 0;

    if (start_local_row >= local_rows)
        return;

    // Solve L * X = B where L is the diagonal block
    for (int i = start_local_row; i < local_rows; i++) {
        for (int j = 0; j < block_size; j++) {
            double sum = local_A[i * local_cols + local_offset_col + j];
            for (int p = 0; p < j; p++) {
                sum -= local_A[i * local_cols + local_offset_col + p] *
                       diag_block[j * cfg->block_size + p];
            }
            local_A[i * local_cols + local_offset_col + j] =
                sum / diag_block[j * cfg->block_size + j];
        }
    }
}

// Gather column panel from all processes in the same column
static void gather_column_panel(double* col_panel, const double* local_A,
                                const ProcGrid* grid, const Config* cfg,
                                int local_rows, int local_cols, int row_start,
                                int col_start, int block_start, int block_size,
                                int block_end, int owner_col) {
    memset(col_panel, 0, cfg->matrix_size * cfg->block_size * sizeof(double));

    if (grid->my_col == owner_col && col_start <= block_start &&
        block_start < col_start + local_cols) {
        int local_offset_col = block_start - col_start;

        for (int i = 0; i < local_rows; i++) {
            int global_i = row_start + i;
            if (global_i >= block_end) {
                for (int j = 0; j < block_size; j++) {
                    col_panel[global_i * cfg->block_size + j] =
                        local_A[i * local_cols + local_offset_col + j];
                }
            }
        }
    }

    // Combine contributions from all processes in this column
    MPI_Allreduce(MPI_IN_PLACE, col_panel, cfg->matrix_size * cfg->block_size,
                  MPI_DOUBLE, MPI_SUM, grid->col_comm);

    // Broadcast to all processes in each row
    MPI_Bcast(col_panel, cfg->matrix_size * cfg->block_size, MPI_DOUBLE,
              owner_col, grid->row_comm);
}

// Update trailing submatrix (SYRK operation)
static void update_trailing_matrix(double* local_A, const double* col_panel,
                                   const Config* cfg, int local_rows,
                                   int local_cols, int row_start, int col_start,
                                   int block_size, int block_end) {
    if (local_rows == 0 || local_cols == 0)
        return;

    // Update lower triangular part only
    for (int i = 0; i < local_rows; i++) {
        int global_i = row_start + i;
        if (global_i < block_end)
            continue;

        for (int j = 0; j < local_cols; j++) {
            int global_j = col_start + j;
            if (global_j < block_end || global_j > global_i)
                continue;

            // Compute A[i,j] -= L[i,:] * L[j,:]^T
            double sum = 0.0;
            for (int p = 0; p < block_size; p++) {
                sum += col_panel[global_i * cfg->block_size + p] *
                       col_panel[global_j * cfg->block_size + p];
            }
            local_A[i * local_cols + j] -= sum;
        }
    }
}

void parallel_cholesky(double* local_A, const ProcGrid* grid, const Config* cfg,
                       int local_rows, int local_cols) {
    int row_start, col_start, dummy_rows, dummy_cols;
    compute_local_dimensions(grid, cfg->matrix_size, &dummy_rows, &dummy_cols,
                             &row_start, &col_start);

    int num_blocks = (cfg->matrix_size + cfg->block_size - 1) / cfg->block_size;

    for (int k = 0; k < num_blocks; k++) {
        int block_start = k * cfg->block_size;
        int block_size = (block_start + cfg->block_size <= cfg->matrix_size)
                             ? cfg->block_size
                             : cfg->matrix_size - block_start;
        int block_end = block_start + block_size;

        // Find owner of diagonal block
        int owner_row, owner_col;
        find_block_owner(grid, block_start, cfg->matrix_size, &owner_row,
                         &owner_col);

        if (owner_row == -1 || owner_col == -1) {
            if (grid->rank == 0) {
                printf("ERROR: Could not find owner for block %d\n", k);
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        double* diag_block =
            (double*)calloc(cfg->block_size * cfg->block_size, sizeof(double));

        // Step 1: Factorize diagonal block
        if (grid->my_row == owner_row && grid->my_col == owner_col) {
            int local_offset_row = block_start - row_start;
            int local_offset_col = block_start - col_start;

            // Copy diagonal block
            for (int i = 0; i < block_size; i++) {
                for (int j = 0; j < block_size; j++) {
                    diag_block[i * cfg->block_size + j] =
                        local_A[(local_offset_row + i) * local_cols +
                                (local_offset_col + j)];
                }
            }

            // Factorize
            cholesky_block(diag_block, block_size, cfg->block_size);

            // Copy back (lower triangular only)
            for (int i = 0; i < block_size; i++) {
                for (int j = 0; j <= i && j < block_size; j++) {
                    local_A[(local_offset_row + i) * local_cols +
                            (local_offset_col + j)] =
                        diag_block[i * cfg->block_size + j];
                }
            }
        }

        // Broadcast diagonal block to all processes
        MPI_Bcast(diag_block, cfg->block_size * cfg->block_size, MPI_DOUBLE,
                  owner_col, grid->row_comm);
        MPI_Bcast(diag_block, cfg->block_size * cfg->block_size, MPI_DOUBLE,
                  owner_row, grid->col_comm);

        // Step 2: Update column panel below diagonal
        update_column_panel(local_A, diag_block, grid, cfg, local_rows,
                            local_cols, row_start, col_start, block_start,
                            block_size, block_end, owner_col);

        // Step 3: Gather updated column panel
        double* col_panel =
            (double*)calloc(cfg->matrix_size * cfg->block_size, sizeof(double));
        gather_column_panel(col_panel, local_A, grid, cfg, local_rows,
                            local_cols, row_start, col_start, block_start,
                            block_size, block_end, owner_col);

        // Step 4: Update trailing submatrix
        update_trailing_matrix(local_A, col_panel, cfg, local_rows, local_cols,
                               row_start, col_start, block_size, block_end);

        free(col_panel);
        free(diag_block);
    }
}
