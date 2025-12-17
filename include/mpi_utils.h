#ifndef MPI_UTILS_H
#define MPI_UTILS_H

#include "config.h"

// Initialize 2D process grid and compute optimal block size
void init_process_grid(ProcGrid* grid, Config* cfg);

// Compute local matrix dimensions for this process
void compute_local_dimensions(const ProcGrid* grid, int matrix_size,
                              int* local_rows, int* local_cols,
                              int* row_start, int* col_start);

// Distribute global matrix to all processes
void distribute_matrix(const double* global_A, double** local_A,
                       const ProcGrid* grid, int matrix_size,
                       int* local_rows, int* local_cols);

// Gather local matrices back to global matrix
void gather_matrix(const double* local_A, double* global_L,
                   const ProcGrid* grid, int matrix_size,
                   int local_rows, int local_cols);

// Cleanup process grid communicators
void cleanup_process_grid(ProcGrid* grid);

#endif // MPI_UTILS_H
