#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

#include "config.h"

// Generate symmetric positive definite matrix
void generate_spd_matrix(double* A, int n);

// Generate local blocks of SPD matrix directly (distributed generation)
void generate_spd_matrix_distributed(double* local_A, const ProcGrid* grid,
                                     int matrix_size, int block_size,
                                     int local_rows, int local_cols,
                                     const Config* cfg);

// Verify that L * L^T = A
int verify_factorization(const double* A, const double* L, int n,
                         double tolerance);

// Print matrix (for small matrices)
void print_matrix(const char* name, const double* A, int n, int max_print);

#endif // MATRIX_OPS_H
