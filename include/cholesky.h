#ifndef CHOLESKY_H
#define CHOLESKY_H

#include "config.h"

// Perform parallel Cholesky factorization
void parallel_cholesky(double* local_A, const ProcGrid* grid,
                      const Config* cfg, int local_rows, int local_cols);

#endif // CHOLESKY_H
