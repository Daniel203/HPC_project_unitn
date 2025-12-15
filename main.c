#include "utils.h"
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

/* The size of the matrix NxN */
#define N 1000
/* The size of the submatrix block BxB */
#define B 100
/* The number of blocks in a column in the matrix */
#define N_BLOCKS_COL N / B
/* The number of blocks in a row in the matrix */
#define N_BLOCKS_ROW N / B

/* Get the pointer to the submatrix at position i,j with size B */
#define BLOCK_PTR(A, i, j, B, N) (A + (i * (size_t)B * N) + (j * (size_t)B))

/* block_ptr: block to be updated.
 * block_size: dimension of the block (block_size X block_size).
 */
void potrf_block(double* block_ptr, int block_size) {
    // Iterate over columns (j)
    for (int j = 0; j < block_size; j++) {
        // ------------ diagonal ------------
        double sum_of_squares = 0.0;

        // Sum all the elements on the left (same row j, columns k < j)
        for (int k = 0; k < j; k++) {
            double val = block_ptr[j * block_size + k];
            sum_of_squares += val * val;
        }

        // Calculate the value on the diagonal element and write it in the matrix
        double A_jj = block_ptr[j * block_size + j];
        double L_jj = sqrt(A_jj - sum_of_squares);
        block_ptr[j * block_size + j] = L_jj;

        // ------------ non-diagonal ------------
        // Iterate over rows (i) below the diagonal
        for (int i = j + 1; i < block_size; i++) {
            double sum_of_products = 0.0;

            // Dot product between row i and row j (up to column j-1)
            for (int k = 0; k < j; k++) {
                double L_ik = block_ptr[i * block_size + k];
                double L_jk = block_ptr[j * block_size + k];
                sum_of_products += L_ik * L_jk;
            }

            double A_ij = block_ptr[i * block_size + j];
            block_ptr[i * block_size + j] = (A_ij - sum_of_products) / L_jj;
        }
    }
}

/* A_ik: block to be updated (rectangular block).
 * L_kk: diagonal block already calculated (triangular block).
 * block_size: dimension of the block (block_size X block_size).
 */
void trsm_block(double* A_ik, double* L_kk, int block_size) {
    // Iterate over every row of A_ik (i)
    for (int i = 0; i < block_size; i++) {
        // Iterate over every column of A_ik (j)
        for (int j = 0; j < block_size; j++) {
            double sum = 0.0;

            // Sum all the previous 'k' columns (k < j)
            // sum(x_k * L_jk) for every k=0..j-1
            for (int k = 0; k < j; k++) {
                double x_k = A_ik[i * block_size + k];  // Value already calculated in the same row
                double L_jk = L_kk[j * block_size + k]; // Value of L (corresponds to L^T in k,j)
                sum += x_k * L_jk;
            }

            double a_j = A_ik[i * block_size + j];  // Original value
            double L_jj = L_kk[j * block_size + j]; // Number on the diagonal

            // Override A_ik with the final result
            A_ik[i * block_size + j] = (a_j - sum) / L_jj;
        }
    }
}

/* A_ij: block to be updated
 * L_ik: block from row i, column k
 * L_jk: block from row j, column k
 * block_size: dimension of the block (block_size X block_size).
 *
 * Operation: A_ij = A_ij - L_ik * (L_jk)^T
 */
void gemm_block(double* A_ij, double* L_ik, double* L_jk, int block_size) {
    // Iterate over rows of A_ij
    for (int i = 0; i < block_size; i++) {
        // Iterate over columns of A_ij
        for (int j = 0; j < block_size; j++) {
            double sum = 0.0;

            // Standard Matrix Multiplication loop
            // We want to calculate: sum += L_ik[i][k] * L_jk^T[k][j]
            // Since L_jk^T[k][j] is equal to L_jk[j][k], we access L_jk at [j][k].
            for (int k = 0; k < block_size; k++) {
                double val_L_ik = L_ik[i * block_size + k];
                double val_L_jk = L_jk[j * block_size + k];
                sum += val_L_ik * val_L_jk;
            }

            A_ij[i * block_size + j] -= sum;
        }
    }
}

int main() {
    double* A = generate_spd_matrix_1d(N);

    double start = now();

    for (int k = 0; k < N_BLOCKS_ROW; k++) {
        // Factorization of current block
        potrf_block(BLOCK_PTR(A, k, k, B, N), B);

        // Resolve current column
        for (int i = k + 1; i < N_BLOCKS_COL; i++) {
            trsm_block(BLOCK_PTR(A, i, k, B, N), BLOCK_PTR(A, k, k, B, N), B);
        }

        // Update the remaining sub matrix (bottom-right)
        for (int j = k + 1; j < N_BLOCKS_COL; j++) {
            for (int i = j; i < N_BLOCKS_ROW; i++) {
                gemm_block(BLOCK_PTR(A, i, j, B, N), BLOCK_PTR(A, i, k, B, N),
                           BLOCK_PTR(A, j, k, B, N), B);
            }
        }
    }

    double end = now();

    printf("\n");
    printf("Time: %.6f seconds\n", end - start);

    free(A);
    return 0;
}
