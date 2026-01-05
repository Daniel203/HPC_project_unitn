#include "../include/cholesky.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// Sequential Cholesky factorization for a single block
static void cholesky_block(double* A, int n, int ld, const Config* cfg) {
    for (int k = 0; k < n; k++) {
        A[k * ld + k] = sqrt(A[k * ld + k]);
        
        #pragma omp parallel for if(cfg->enable_openmp && n > 64)
        for (int i = k + 1; i < n; i++) {
            A[i * ld + k] /= A[k * ld + k];
        }
        
        #pragma omp parallel for collapse(2) if(cfg->enable_openmp && n > 64)
        for (int j = k + 1; j < n; j++) {
            for (int i = j; i < n; i++) {
                A[i * ld + j] -= A[i * ld + k] * A[j * ld + k];
            }
        }
    }
}

// Find which process owns a specific block (block-cyclic)
static void find_block_owner(const ProcGrid* grid, int block_row, int block_col,
                             int* owner_row, int* owner_col) {
    *owner_row = block_row % grid->grid_rows;
    *owner_col = block_col % grid->grid_cols;
}

// Get pointer to local block storage
static double* get_local_block_ptr(double* local_A, int block_row, int block_col,
                                   const ProcGrid* grid, int block_size,
                                   int num_local_block_cols) {
    int owner_row, owner_col;
    find_block_owner(grid, block_row, block_col, &owner_row, &owner_col);
    
    if (owner_row != grid->my_row || owner_col != grid->my_col) {
        return NULL;
    }
    
    int local_br = block_row / grid->grid_rows;
    int local_bc = block_col / grid->grid_cols;
    
    int local_storage_cols = num_local_block_cols * block_size;
    return local_A + (local_br * block_size) * local_storage_cols + 
           (local_bc * block_size);
}

// Check if this process owns a block
static int owns_block(const ProcGrid* grid, int block_row, int block_col) {
    int owner_row = block_row % grid->grid_rows;
    int owner_col = block_col % grid->grid_cols;
    return (owner_row == grid->my_row && owner_col == grid->my_col);
}

// TRSM: Solve L_kk * L_ik^T = A_ik for column blocks below diagonal
static void trsm_column_blocks(double* local_A, const double* diag_block,
                               const ProcGrid* grid, const Config* cfg,
                               int k, int num_local_block_cols) {
    int total_block_rows = (cfg->matrix_size + cfg->block_size - 1) / cfg->block_size;
    int block_size = cfg->block_size;
    int owner_col = k % grid->grid_cols;
    
    if (grid->my_col != owner_col) {
        return;
    }
    
    // Count owned blocks for OpenMP decision
    int num_owned_blocks = 0;
    for (int i = k + 1; i < total_block_rows; i++) {
        if (owns_block(grid, i, k)) {
            num_owned_blocks++;
        }
    }
    
    #pragma omp parallel for schedule(dynamic) if(cfg->enable_openmp && num_owned_blocks > 1)
    for (int i = k + 1; i < total_block_rows; i++) {
        if (!owns_block(grid, i, k)) {
            continue;
        }
        
        double* block_ik = get_local_block_ptr(local_A, i, k, grid, 
                                               block_size, num_local_block_cols);
        if (block_ik == NULL) continue;
        
        int local_storage_cols = num_local_block_cols * block_size;
        int block_rows = ((i + 1) * block_size <= cfg->matrix_size) 
                        ? block_size : cfg->matrix_size - i * block_size;
        int block_cols = ((k + 1) * block_size <= cfg->matrix_size)
                        ? block_size : cfg->matrix_size - k * block_size;
        
        for (int row = 0; row < block_rows; row++) {
            for (int col = 0; col < block_cols; col++) {
                double sum = block_ik[row * local_storage_cols + col];
                for (int p = 0; p < col; p++) {
                    sum -= block_ik[row * local_storage_cols + p] * 
                           diag_block[col * block_size + p];
                }
                block_ik[row * local_storage_cols + col] = 
                    sum / diag_block[col * block_size + col];
            }
        }
    }
}

// Gather column panel into buffer for broadcast
static void gather_column_panel(double* col_panel, const double* local_A,
                                const ProcGrid* grid, const Config* cfg,
                                int k, int num_local_block_cols) {
    int total_block_rows = (cfg->matrix_size + cfg->block_size - 1) / cfg->block_size;
    int block_size = cfg->block_size;
    int owner_col = k % grid->grid_cols;
    
    memset(col_panel, 0, cfg->matrix_size * block_size * sizeof(double));
    
    if (grid->my_col == owner_col) {
        for (int i = k + 1; i < total_block_rows; i++) {
            if (!owns_block(grid, i, k)) {
                continue;
            }
            
            double* block_ik = get_local_block_ptr(local_A, i, k, grid,
                                                   block_size, num_local_block_cols);
            if (block_ik == NULL) continue;
            
            int local_storage_cols = num_local_block_cols * block_size;
            int global_row_start = i * block_size;
            int block_rows = ((i + 1) * block_size <= cfg->matrix_size)
                            ? block_size : cfg->matrix_size - global_row_start;
            int block_cols = ((k + 1) * block_size <= cfg->matrix_size)
                            ? block_size : cfg->matrix_size - k * block_size;
            
            for (int row = 0; row < block_rows; row++) {
                for (int col = 0; col < block_cols; col++) {
                    int global_row = global_row_start + row;
                    col_panel[global_row * block_size + col] =
                        block_ik[row * local_storage_cols + col];
                }
            }
        }
    }
    
    MPI_Allreduce(MPI_IN_PLACE, col_panel, cfg->matrix_size * block_size,
                  MPI_DOUBLE, MPI_SUM, grid->col_comm);
    
    MPI_Bcast(col_panel, cfg->matrix_size * block_size, MPI_DOUBLE,
              owner_col, grid->row_comm);
}

// SYRK: Update trailing submatrix
static void syrk_trailing_matrix(double* local_A, const double* col_panel,
                                 const ProcGrid* grid, const Config* cfg,
                                 int k, int num_local_block_cols) {
    int total_block_rows = (cfg->matrix_size + cfg->block_size - 1) / cfg->block_size;
    int block_size = cfg->block_size;
    
    // Count owned blocks for OpenMP decision
    int num_owned_blocks = 0;
    for (int i = k + 1; i < total_block_rows; i++) {
        for (int j = k + 1; j <= i; j++) {
            if (owns_block(grid, i, j)) {
                num_owned_blocks++;
            }
        }
    }
    
    #pragma omp parallel for collapse(2) schedule(guided) if(cfg->enable_openmp && num_owned_blocks > 4)
    for (int i = k + 1; i < total_block_rows; i++) {
        for (int j = k + 1; j <= i; j++) {
            if (!owns_block(grid, i, j)) {
                continue;
            }
            
            double* block_ij = get_local_block_ptr(local_A, i, j, grid,
                                                   block_size, num_local_block_cols);
            if (block_ij == NULL) continue;
            
            int local_storage_cols = num_local_block_cols * block_size;
            int global_row_start = i * block_size;
            int global_col_start = j * block_size;
            
            int block_rows = ((i + 1) * block_size <= cfg->matrix_size)
                            ? block_size : cfg->matrix_size - global_row_start;
            int block_cols = ((j + 1) * block_size <= cfg->matrix_size)
                            ? block_size : cfg->matrix_size - global_col_start;
            int panel_cols = ((k + 1) * block_size <= cfg->matrix_size)
                            ? block_size : cfg->matrix_size - k * block_size;
            
            for (int row = 0; row < block_rows; row++) {
                int global_row = global_row_start + row;
                
                for (int col = 0; col < block_cols; col++) {
                    int global_col = global_col_start + col;
                    
                    if (global_col > global_row) continue;
                    
                    double sum = 0.0;
                    for (int p = 0; p < panel_cols; p++) {
                        sum += col_panel[global_row * block_size + p] *
                               col_panel[global_col * block_size + p];
                    }
                    block_ij[row * local_storage_cols + col] -= sum;
                }
            }
        }
    }
}

void parallel_cholesky(double* local_A, const ProcGrid* grid, const Config* cfg,
                       int local_rows, int local_cols) {
    int num_blocks = (cfg->matrix_size + cfg->block_size - 1) / cfg->block_size;
    int block_size = cfg->block_size;
    
    int num_local_block_cols = local_cols / block_size;
    
    double* diag_block = (double*)calloc(block_size * block_size, sizeof(double));
    double* col_panel = (double*)calloc(cfg->matrix_size * block_size, sizeof(double));
    
    for (int k = 0; k < num_blocks; k++) {
        int owner_row, owner_col;
        find_block_owner(grid, k, k, &owner_row, &owner_col);
        
        // Step 1: Factorize diagonal block L_kk
        if (grid->my_row == owner_row && grid->my_col == owner_col) {
            double* block_kk = get_local_block_ptr(local_A, k, k, grid,
                                                   block_size, num_local_block_cols);
            
            int actual_block_size = ((k + 1) * block_size <= cfg->matrix_size)
                                   ? block_size : cfg->matrix_size - k * block_size;
            int local_storage_cols = num_local_block_cols * block_size;
            
            // Copy diagonal block to buffer
            for (int i = 0; i < actual_block_size; i++) {
                for (int j = 0; j <= i && j < actual_block_size; j++) {
                    diag_block[i * block_size + j] = 
                        block_kk[i * local_storage_cols + j];
                }
            }
            
            // Factorize (with optional OpenMP)
            cholesky_block(diag_block, actual_block_size, block_size, cfg);
            
            // Copy back to local storage
            for (int i = 0; i < actual_block_size; i++) {
                for (int j = 0; j <= i && j < actual_block_size; j++) {
                    block_kk[i * local_storage_cols + j] = 
                        diag_block[i * block_size + j];
                }
            }
        }
        
        // Broadcast diagonal block to all processes
        MPI_Bcast(diag_block, block_size * block_size, MPI_DOUBLE,
                  owner_col, grid->row_comm);
        MPI_Bcast(diag_block, block_size * block_size, MPI_DOUBLE,
                  owner_row, grid->col_comm);
        
        // Step 2: Update column panel below diagonal (TRSM)
        trsm_column_blocks(local_A, diag_block, grid, cfg, k, num_local_block_cols);
        
        // Step 3: Gather column panel for broadcast
        gather_column_panel(col_panel, local_A, grid, cfg, k, num_local_block_cols);
        
        // Step 4: Update trailing submatrix (SYRK)
        syrk_trailing_matrix(local_A, col_panel, grid, cfg, k, num_local_block_cols);
    }
    
    free(diag_block);
    free(col_panel);
}

