#pragma once

void generate_spd_matrix(int n, float A[n][n]);
double* generate_spd_matrix_1d(int n);
int verify_cholesky(int n, float A[n][n], float L[n][n]);
double now();
