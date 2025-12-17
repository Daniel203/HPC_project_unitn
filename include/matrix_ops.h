#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

// Generate symmetric positive definite matrix
void generate_spd_matrix(double* A, int n);

// Verify that L * L^T = A
int verify_factorization(const double* A, const double* L, int n,
                         double tolerance);

// Print matrix (for small matrices)
void print_matrix(const char* name, const double* A, int n, int max_print);

#endif // MATRIX_OPS_H
