#include "../include/matrix_ops.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void generate_spd_matrix(double* A, int n) {
    // Create diagonally dominant matrix for numerical stability
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i * n + j] = (i == j) ? (n + 1.0) : 1.0 / (1.0 + abs(i - j));
        }
    }
}

int verify_factorization(const double* A, const double* L, int n,
                         double tolerance) {
    double max_error = 0.0;

    // Compute L * L^T and compare with A
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0.0;
            for (int k = 0; k <= j; k++) {
                sum += L[i * n + k] * L[j * n + k];
            }
            double error = fabs(sum - A[i * n + j]);
            if (error > max_error) {
                max_error = error;
            }
        }
    }

    printf("Maximum error: %e\n", max_error);
    return max_error < tolerance;
}

void print_matrix(const char* name, const double* A, int n, int max_print) {
    printf("%s:\n", name);
    int print_n = (n < max_print) ? n : max_print;

    for (int i = 0; i < print_n; i++) {
        for (int j = 0; j < print_n; j++) {
            printf("%8.4f ", A[i * n + j]);
        }
        printf("\n");
    }

    if (n > max_print) {
        printf("... (showing %dx%d of %dx%d)\n", max_print, max_print, n, n);
    }
}
