#include "../include/mpi_utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Find optimal block size for matrix distribution
static int compute_optimal_block_size(int matrix_size, int grid_dim) {
    const int MIN_BLOCK_SIZE = 64;
    int target_blocks = grid_dim * BLOCK_CYCLIC_MULTIPLIER;
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
        printf("Distribution: Block-Cyclic (2D cyclic)\n");

        if (cfg->matrix_size % cfg->block_size == 0) {
            printf("✓ Block size evenly divides matrix\n");
        } else {
            int last_block =
                cfg->matrix_size - ((num_blocks - 1) * cfg->block_size);
            printf("⚠ Last block will be smaller: %d elements\n", last_block);
        }
    }
}

// Convert global block index to process coordinates (block-cyclic)
void global_block_to_proc(int block_row, int block_col, const ProcGrid* grid,
                         int* proc_row, int* proc_col) {
    *proc_row = block_row % grid->grid_rows;
    *proc_col = block_col % grid->grid_cols;
}

// Convert global block index to local block index for a process
void global_block_to_local(int block_row, int block_col, const ProcGrid* grid,
                          int* local_block_row, int* local_block_col) {
    *local_block_row = block_row / grid->grid_rows;
    *local_block_col = block_col / grid->grid_cols;
}

// Count how many blocks this process owns in each dimension
void count_local_blocks(const ProcGrid* grid, int matrix_size, int block_size,
                       int* num_local_block_rows, int* num_local_block_cols) {
    int total_block_rows = (matrix_size + block_size - 1) / block_size;
    int total_block_cols = (matrix_size + block_size - 1) / block_size;
    
    *num_local_block_rows = total_block_rows / grid->grid_rows;
    if (grid->my_row < (total_block_rows % grid->grid_rows)) {
        (*num_local_block_rows)++;
    }
    
    *num_local_block_cols = total_block_cols / grid->grid_cols;
    if (grid->my_col < (total_block_cols % grid->grid_cols)) {
        (*num_local_block_cols)++;
    }
}

void compute_local_dimensions(const ProcGrid* grid, int matrix_size, 
                              int block_size, int* local_rows, int* local_cols) {
    int num_local_block_rows, num_local_block_cols;
    count_local_blocks(grid, matrix_size, block_size, 
                      &num_local_block_rows, &num_local_block_cols);
    
    // Allocate enough space for all local blocks
    *local_rows = num_local_block_rows * block_size;
    *local_cols = num_local_block_cols * block_size;
}

// Helper: Get global position from local block indices
void local_block_to_global_pos(int local_block_row, int local_block_col,
                               const ProcGrid* grid, int block_size,
                               int* global_row, int* global_col) {
    int global_block_row = local_block_row * grid->grid_rows + grid->my_row;
    int global_block_col = local_block_col * grid->grid_cols + grid->my_col;
    
    *global_row = global_block_row * block_size;
    *global_col = global_block_col * block_size;
}

// Helper: Convert global position to local block storage index
int global_to_local_index(int global_row, int global_col, const ProcGrid* grid,
                         int block_size, int num_local_block_cols) {
    int block_row = global_row / block_size;
    int block_col = global_col / block_size;
    
    // Check if this process owns this block
    int owner_row = block_row % grid->grid_rows;
    int owner_col = block_col % grid->grid_cols;
    
    if (owner_row != grid->my_row || owner_col != grid->my_col) {
        return -1; // Not owned by this process
    }
    
    int local_block_row = block_row / grid->grid_rows;
    int local_block_col = block_col / grid->grid_cols;
    
    int in_block_row = global_row % block_size;
    int in_block_col = global_col % block_size;
    
    // Linear index in local storage
    return (local_block_row * block_size + in_block_row) * 
           (num_local_block_cols * block_size) +
           (local_block_col * block_size + in_block_col);
}

void distribute_matrix(const double* global_A, double** local_A,
                       const ProcGrid* grid, int matrix_size, int block_size,
                       int* local_rows, int* local_cols) {
    int num_local_block_rows, num_local_block_cols;
    count_local_blocks(grid, matrix_size, block_size,
                      &num_local_block_rows, &num_local_block_cols);
    
    *local_rows = num_local_block_rows * block_size;
    *local_cols = num_local_block_cols * block_size;
    
    *local_A = (double*)calloc((*local_rows) * (*local_cols), sizeof(double));
    
    if (grid->rank == 0) {
        // Root distributes blocks in cyclic fashion
        int total_block_rows = (matrix_size + block_size - 1) / block_size;
        int total_block_cols = (matrix_size + block_size - 1) / block_size;
        
        for (int p = 0; p < grid->np; p++) {
            int dest_row = p / grid->grid_cols;
            int dest_col = p % grid->grid_cols;
            
            // Count blocks for this process
            int dest_num_block_rows = total_block_rows / grid->grid_rows;
            if (dest_row < (total_block_rows % grid->grid_rows)) {
                dest_num_block_rows++;
            }
            int dest_num_block_cols = total_block_cols / grid->grid_cols;
            if (dest_col < (total_block_cols % grid->grid_cols)) {
                dest_num_block_cols++;
            }
            
            int dest_local_rows = dest_num_block_rows * block_size;
            int dest_local_cols = dest_num_block_cols * block_size;
            
            double* send_buf = (double*)calloc(dest_local_rows * dest_local_cols, 
                                              sizeof(double));
            
            // Pack blocks for this process
            for (int br = dest_row; br < total_block_rows; br += grid->grid_rows) {
                for (int bc = dest_col; bc < total_block_cols; bc += grid->grid_cols) {
                    int local_br = br / grid->grid_rows;
                    int local_bc = bc / grid->grid_cols;
                    
                    // Copy this block
                    int global_row_start = br * block_size;
                    int global_col_start = bc * block_size;
                    int block_rows = (global_row_start + block_size <= matrix_size) 
                                    ? block_size : matrix_size - global_row_start;
                    int block_cols = (global_col_start + block_size <= matrix_size)
                                    ? block_size : matrix_size - global_col_start;
                    
                    for (int i = 0; i < block_rows; i++) {
                        for (int j = 0; j < block_cols; j++) {
                            int local_row = local_br * block_size + i;
                            int local_col = local_bc * block_size + j;
                            int global_row = global_row_start + i;
                            int global_col = global_col_start + j;
                            
                            send_buf[local_row * dest_local_cols + local_col] =
                                global_A[global_row * matrix_size + global_col];
                        }
                    }
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
        MPI_Recv(*local_A, (*local_rows) * (*local_cols), MPI_DOUBLE, 0, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

void gather_matrix(const double* local_A, double* global_L,
                   const ProcGrid* grid, int matrix_size, int block_size,
                   int local_rows, int local_cols) {
    if (grid->rank == 0) {
        memset(global_L, 0, matrix_size * matrix_size * sizeof(double));
        
        int total_block_rows = (matrix_size + block_size - 1) / block_size;
        int total_block_cols = (matrix_size + block_size - 1) / block_size;
        
        for (int p = 0; p < grid->np; p++) {
            int src_row = p / grid->grid_cols;
            int src_col = p % grid->grid_cols;
            
            // Count blocks for this process
            int src_num_block_rows = total_block_rows / grid->grid_rows;
            if (src_row < (total_block_rows % grid->grid_rows)) {
                src_num_block_rows++;
            }
            int src_num_block_cols = total_block_cols / grid->grid_cols;
            if (src_col < (total_block_cols % grid->grid_cols)) {
                src_num_block_cols++;
            }
            
            int src_local_rows = src_num_block_rows * block_size;
            int src_local_cols = src_num_block_cols * block_size;
            
            double* recv_buf;
            if (p == 0) {
                recv_buf = (double*)local_A;
            } else {
                recv_buf = (double*)malloc(src_local_rows * src_local_cols * sizeof(double));
                MPI_Recv(recv_buf, src_local_rows * src_local_cols, MPI_DOUBLE,
                        p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            
            // Unpack blocks from this process
            for (int br = src_row; br < total_block_rows; br += grid->grid_rows) {
                for (int bc = src_col; bc < total_block_cols; bc += grid->grid_cols) {
                    int local_br = br / grid->grid_rows;
                    int local_bc = bc / grid->grid_cols;
                    
                    int global_row_start = br * block_size;
                    int global_col_start = bc * block_size;
                    int block_rows = (global_row_start + block_size <= matrix_size)
                                    ? block_size : matrix_size - global_row_start;
                    int block_cols = (global_col_start + block_size <= matrix_size)
                                    ? block_size : matrix_size - global_col_start;
                    
                    for (int i = 0; i < block_rows; i++) {
                        for (int j = 0; j < block_cols; j++) {
                            int local_row = local_br * block_size + i;
                            int local_col = local_bc * block_size + j;
                            int global_row = global_row_start + i;
                            int global_col = global_col_start + j;
                            
                            global_L[global_row * matrix_size + global_col] =
                                recv_buf[local_row * src_local_cols + local_col];
                        }
                    }
                }
            }
            
            if (p != 0) free(recv_buf);
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
