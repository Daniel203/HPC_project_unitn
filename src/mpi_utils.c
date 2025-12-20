#ifndef MPI_UTILS_H
#define MPI_UTILS_H

#include "../include/config.h"
#include <mpi.h>

// Initialize the 2D process grid
void init_process_grid(ProcGrid* grid, Config* cfg);

// Compute local matrix dimensions for this process (block-cyclic)
void compute_local_dimensions(const ProcGrid* grid, int matrix_size, 
                              int block_size, int* local_rows, int* local_cols);

// Distribute global matrix to local blocks (block-cyclic)
void distribute_matrix(const double* global_A, double** local_A,
                       const ProcGrid* grid, int matrix_size, int block_size,
                       int* local_rows, int* local_cols);

// Gather local blocks back to global matrix (block-cyclic)
void gather_matrix(const double* local_A, double* global_L,
                   const ProcGrid* grid, int matrix_size, int block_size,
                   int local_rows, int local_cols);

// Cleanup process grid communicators
void cleanup_process_grid(ProcGrid* grid);

// Block-cyclic distribution helper functions

// Convert global block indices to process coordinates
void global_block_to_proc(int block_row, int block_col, const ProcGrid* grid,
                         int* proc_row, int* proc_col);

// Convert global block indices to local block indices
void global_block_to_local(int block_row, int block_col, const ProcGrid* grid,
                          int* local_block_row, int* local_block_col);

// Count how many blocks this process owns
void count_local_blocks(const ProcGrid* grid, int matrix_size, int block_size,
                       int* num_local_block_rows, int* num_local_block_cols);

// Get global position from local block indices
void local_block_to_global_pos(int local_block_row, int local_block_col,
                               const ProcGrid* grid, int block_size,
                               int* global_row, int* global_col);

// Convert global position to local storage index
int global_to_local_index(int global_row, int global_col, const ProcGrid* grid,
                         int block_size, int num_local_block_cols);

#endif // MPI_UTILS_H
