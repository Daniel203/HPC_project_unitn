#define _POSIX_C_SOURCE 199309L

#include "utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Max value allowed for the generation of random values in the A matrix */
#define MAX_VALUE 10

/**
 * Populate the matrix A as a valid symmetric positive definite matrix
 * of size n x n.
 */
void generate_spd_matrix(int n, float A[n][n]) {
    // Seed random number generator
    srand(time(NULL));

    // Create a temporary matrix B
    float (*B)[n] = malloc(sizeof(float[n][n]));
    if (!B) {
        printf("Allocation failed\n");
        exit(1);
    }

    // Fill B with random values between 0 and `MAX_VALUE`
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            B[i][j] = (float)(rand() % (MAX_VALUE));
        }
    }

    // Compute A = B × B^T
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = 0;
            for (int k = 0; k < n; k++) {
                A[i][j] += B[i][k] * B[j][k];
            }
        }
    }

    free(B);
}

/**
 * Populate the matrix A as a valid symmetric positive definite matrix
 * of size n x n. Return in as a 1D dimensional array
 */
double* generate_spd_matrix_1d(int n) {
    double* A = (double*)malloc(n * n * sizeof(double));
    double* B_temp = (double*)malloc(n * n * sizeof(double));

    if (!A || !B_temp) {
        printf("Allocation failed\n");
        exit(1);
    }

    // Initialize B_temp with random values
    srand(time(NULL));
    for (int i = 0; i < n * n; i++) {
        B_temp[i] = ((double)rand() / RAND_MAX);
    }

    // A = B_temp * B_temp^T
    // Calculate only the half under the diagonal since it's symmetric
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0.0;

            for (int k = 0; k < n; k++) {
                // A[i][j] = SUM(k) B[i][k] * B[j][k]
                sum += B_temp[i * n + k] * B_temp[j * n + k];
            }

            A[i * n + j] = sum;
            if (i != j) {
                A[j * n + i] = sum;
            }
        }
    }

    free(B_temp);
    return A;
}

int verify_cholesky(int n, float A[n][n], float L[n][n]) {
    float max_error = 0.0f;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {

            // Compute entry (i,j) of L * L^T
            float sum = 0.0f;
            for (int k = 0; k <= (i < j ? i : j); k++) {
                sum += L[i][k] * L[j][k];
            }

            float error = fabsf(A[i][j] - sum);
            if (error > max_error)
                max_error = error;
        }
    }

    printf("Max reconstruction error: %g\n", max_error);

    // Valid only if the error is smaller than 1e-2
    return max_error < 0.01;
}

double now() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

