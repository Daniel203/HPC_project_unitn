#include "../include/mpi_utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Find optimal block size for matrix distribution
static int compute_optimal_block_size(int matrix_size, int grid_dim) {
    const int MIN_BLOCK_SIZE = 64;
    int target_blocks = grid_dim;
    int block_size = matrix_size / target_blocks;

    if (block_size < MIN_BLOCK_SIZE) {
        return MIN_BLOCK_SIZE;
    }

    // Try to find a divisor of matrix_size close to our target
    if (matrix_size % block_size != 0) {
        int best_block = block_size;
        int min_diff = abs(matrix_size % block_size);

        for (int bs = block_size - 32; bs <= block_size + 32; bs += 16) {
            if (bs >= MIN_BLOCK_SIZE && matrix_size % bs == 0) {
                int diff = abs(bs - block_size);
                if (diff < min_diff ||
                    (matrix_size % bs == 0 && matrix_size % best_block != 0)) {
                    best_block = bs;
                    min_diff = diff;
                    if (matrix_size % bs == 0)
                        break;
                }
            }
        }
        block_size = best_block;
    }

    return block_size;
}

void init_process_grid(ProcGrid* grid, Config* cfg) {
    MPI_Comm_rank(MPI_COMM_WORLD, &grid->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &grid->np);

    // Factorize number of processes into grid_rows x grid_cols
    grid->grid_rows = (int)sqrt(grid->np);
    while (grid->np % grid->grid_rows != 0) {
        grid->grid_rows--;
    }
    grid->grid_cols = grid->np / grid->grid_rows;

    // Compute this process's position
    grid->my_row = grid->rank / grid->grid_cols;
    grid->my_col = grid->rank % grid->grid_cols;

    // Create row and column communicators
    MPI_Comm_split(MPI_COMM_WORLD, grid->my_row, grid->my_col, &grid->row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, grid->my_col, grid->my_row, &grid->col_comm);

    // Compute optimal block size
    int grid_dim =
        (grid->grid_rows > grid->grid_cols) ? grid->grid_rows : grid->grid_cols;
    cfg->block_size = compute_optimal_block_size(cfg->matrix_size, grid_dim);

    if (grid->rank == 0) {
        int num_blocks =
            (cfg->matrix_size + cfg->block_size - 1) / cfg->block_size;
        printf("Process grid: %d x %d = %d processes\n", grid->grid_rows,
               grid->grid_cols, grid->np);
        printf("Block size: %d (results in %d blocks)\n", cfg->block_size,
               num_blocks);

        if (cfg->matrix_size % cfg->block_size == 0) {
            printf("✓ Block size evenly divides matrix\n");
        } else {
            int last_block =
                cfg->matrix_size - ((num_blocks - 1) * cfg->block_size);
            printf("⚠ Last block will be smaller: %d elements\n", last_block);
        }
    }
}

void compute_local_dimensions(const ProcGrid* grid, int matrix_size,
                              int* local_rows, int* local_cols, int* row_start,
                              int* col_start) {
    int rows_per_proc = matrix_size / grid->grid_rows;
    int cols_per_proc = matrix_size / grid->grid_cols;
    int extra_rows = matrix_size % grid->grid_rows;
    int extra_cols = matrix_size % grid->grid_cols;

    *row_start = grid->my_row * rows_per_proc +
                 (grid->my_row < extra_rows ? grid->my_row : extra_rows);
    *col_start = grid->my_col * cols_per_proc +
                 (grid->my_col < extra_cols ? grid->my_col : extra_cols);

    *local_rows = rows_per_proc + (grid->my_row < extra_rows ? 1 : 0);
    *local_cols = cols_per_proc + (grid->my_col < extra_cols ? 1 : 0);
}

void distribute_matrix(const double* global_A, double** local_A,
                       const ProcGrid* grid, int matrix_size, int* local_rows,
                       int* local_cols) {
    int row_start, col_start;
    compute_local_dimensions(grid, matrix_size, local_rows, local_cols,
                             &row_start, &col_start);

    *local_A = (double*)calloc((*local_rows) * (*local_cols), sizeof(double));

    if (grid->rank == 0) {
        // Root sends to all processes
        for (int p = 0; p < grid->np; p++) {
            int dest_row = p / grid->grid_cols;
            int dest_col = p % grid->grid_cols;

            ProcGrid temp_grid = *grid;
            temp_grid.my_row = dest_row;
            temp_grid.my_col = dest_col;

            int dest_rows, dest_cols, dest_row_start, dest_col_start;
            compute_local_dimensions(&temp_grid, matrix_size, &dest_rows,
                                     &dest_cols, &dest_row_start,
                                     &dest_col_start);

            double* send_buf =
                (double*)malloc(dest_rows * dest_cols * sizeof(double));

            // Pack submatrix for destination
            for (int i = 0; i < dest_rows; i++) {
                for (int j = 0; j < dest_cols; j++) {
                    send_buf[i * dest_cols + j] =
                        global_A[(dest_row_start + i) * matrix_size +
                                 (dest_col_start + j)];
                }
            }

            if (p == 0) {
                memcpy(*local_A, send_buf,
                       dest_rows * dest_cols * sizeof(double));
            } else {
                MPI_Send(send_buf, dest_rows * dest_cols, MPI_DOUBLE, p, 0,
                         MPI_COMM_WORLD);
            }
            free(send_buf);
        }
    } else {
        MPI_Recv(*local_A, (*local_rows) * (*local_cols), MPI_DOUBLE, 0, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

void gather_matrix(const double* local_A, double* global_L,
                   const ProcGrid* grid, int matrix_size, int local_rows,
                   int local_cols) {
    if (grid->rank == 0) {
        memset(global_L, 0, matrix_size * matrix_size * sizeof(double));

        // Receive from all processes
        for (int p = 0; p < grid->np; p++) {
            int src_row = p / grid->grid_cols;
            int src_col = p % grid->grid_cols;

            ProcGrid temp_grid = *grid;
            temp_grid.my_row = src_row;
            temp_grid.my_col = src_col;

            int src_rows, src_cols, src_row_start, src_col_start;
            compute_local_dimensions(&temp_grid, matrix_size, &src_rows,
                                     &src_cols, &src_row_start, &src_col_start);

            double* recv_buf;
            if (p == 0) {
                recv_buf = (double*)local_A;
            } else {
                recv_buf =
                    (double*)malloc(src_rows * src_cols * sizeof(double));
                MPI_Recv(recv_buf, src_rows * src_cols, MPI_DOUBLE, p, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            // Unpack into global matrix
            for (int i = 0; i < src_rows; i++) {
                for (int j = 0; j < src_cols; j++) {
                    global_L[(src_row_start + i) * matrix_size +
                             (src_col_start + j)] = recv_buf[i * src_cols + j];
                }
            }

            if (p != 0)
                free(recv_buf);
        }
    } else {
        MPI_Send(local_A, local_rows * local_cols, MPI_DOUBLE, 0, 0,
                 MPI_COMM_WORLD);
    }
}

void cleanup_process_grid(ProcGrid* grid) {
    MPI_Comm_free(&grid->row_comm);
    MPI_Comm_free(&grid->col_comm);
}
