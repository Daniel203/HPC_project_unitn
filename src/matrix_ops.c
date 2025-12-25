#include "../include/matrix_ops.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void generate_spd_matrix(double* A, int n) {
    // Create diagonally dominant matrix for numerical stability
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i * n + j] = (i == j) ? (n + 1.0) : 1.0 / (1.0 + abs(i - j));
        }
    }
}

void generate_spd_matrix_distributed(double* local_A, const ProcGrid* grid,
                                     int matrix_size, int block_size,
                                     int local_rows, int local_cols) {
    int num_local_block_cols = local_cols / block_size;
    int total_block_rows = (matrix_size + block_size - 1) / block_size;
    int total_block_cols = (matrix_size + block_size - 1) / block_size;
    
    // Initialize to zero
    memset(local_A, 0, local_rows * local_cols * sizeof(double));
    
    // Generate only the blocks owned by this process
    for (int br = grid->my_row; br < total_block_rows; br += grid->grid_rows) {
        for (int bc = grid->my_col; bc < total_block_cols; bc += grid->grid_cols) {
            // Only generate lower triangular part
            if (bc > br) continue;
            
            int local_br = br / grid->grid_rows;
            int local_bc = bc / grid->grid_cols;
            
            int global_row_start = br * block_size;
            int global_col_start = bc * block_size;
            
            int block_rows = (global_row_start + block_size <= matrix_size)
                            ? block_size : matrix_size - global_row_start;
            int block_cols = (global_col_start + block_size <= matrix_size)
                            ? block_size : matrix_size - global_col_start;
            
            // Generate elements for this block
            for (int i = 0; i < block_rows; i++) {
                for (int j = 0; j < block_cols; j++) {
                    int global_i = global_row_start + i;
                    int global_j = global_col_start + j;
                    
                    // Only lower triangular part
                    if (global_j > global_i) continue;
                    
                    int local_i = local_br * block_size + i;
                    int local_j = local_bc * block_size + j;
                    
                    // Same formula as before
                    double value = (global_i == global_j) 
                                  ? (matrix_size + 1.0) 
                                  : 1.0 / (1.0 + abs(global_i - global_j));
                    
                    local_A[local_i * local_cols + local_j] = value;
                }
            }
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
