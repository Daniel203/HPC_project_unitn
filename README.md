# HPC Project: Hybrid Parallel Cholesky Decomposition

This repository contains a high-performance, distributed-memory implementation of the **Cholesky Decomposition** algorithm, developed for the High Performance Computing (HPC) course at the **University of Trento**.

The project focuses on a **hybrid MPI + OpenMP** approach designed to solve large-scale linear algebra problems by exploiting both inter-node and intra-node parallelism.

---

## Project Report

A comprehensive technical report is available in the repository. It contains the complete documentation of the parallel design, data dependency analysis, and performance benchmarking results.

**[Read the Full Report: report/main.pdf](report/main.pdf)**

The report covers:
* **Algorithm Design:** Details on the 2D block-cyclic distribution and the "Zero-Filling" assembly strategy.
* **Implementation Details:** A breakdown of the hybrid MPI+OpenMP architecture and memory management.
* **Performance Evaluation:** Benchmarks including strong and weak scaling analysis, speedup, and GFLOPS metrics.

---

## Key Features

- **Hybrid Parallelism:** Combines MPI for distributed-memory communication with OpenMP for shared-memory multi-threading within nodes.
- **Linear Memory Scalability:** Implements distributed matrix generation, ensuring that no single node ever holds the full matrix in RAM.
- **Efficient Communication:** Uses a 2D process grid topology and custom communicators (`row_comm`, `col_comm`) to minimize broadcast overhead.
- **Optimized Scheduling:** Employs specialized OpenMP scheduling (Static, Dynamic, and Guided) for different phases of the decomposition to maximize thread utilization.
- **No External Dependencies:** The core linear algebra routines are implemented from scratch without requiring BLAS or LAPACK.

---

## Repository Structure

- `src/`: Core implementation files (`main.c`, `cholesky.c`, `matrix_ops.c`, `mpi_utils.c`).
- `include/`: Header files defining the data structures and interfaces.
- `report/`: Documentation, including the PDF report and LaTeX source files.
- `bin/`: Directory where the compiled executable is placed.
- `benchmarks/`: Created automatically to store performance data.

---

## Getting Started

### Prerequisites
- **MPI Library:** (e.g., OpenMPI or MPICH).
- **Compiler:** `gcc` or any C99-compliant compiler with OpenMP support.

### Build
To compile the project using the provided `Makefile`:
```bash
make
```
This will create the executable `bin/cholesky_mpi`.

### Usage
Run the executable using `mpirun`. You can customize the execution using various flags:

```bash
mpirun -np <num_processes> ./bin/cholesky_mpi [options]
```

**Available Options:**
* **`--matrix-size <N>`**: Set the dimension of the square matrix (default: 1024).
* **`--enable-openmp`**: Enable intra-node multi-threading.
* **`--num-threads <T>`**: Specify the number of OpenMP threads per process (default: auto-detect).
* **`--test-category <name>`**: Label for the benchmark run (e.g., "strong_scaling").
* **`--placement <name>`**: Description of process placement (e.g., "scatter").

**Example:**
```bash
mpirun -np 4 ./bin/cholesky_mpi --matrix-size 4096 --enable-openmp --num-threads 4
```

---

## Implementation Highlights

### Process Grid Topology
* The application organizes the $P$ available MPI processes into a logical 2D grid of dimensions $P_{rows} \times P_{cols}$.
* Dimensions are calculated dynamically to produce the most compact rectangular or square grid possible ($P_{rows} \approx \sqrt{P}$) to minimize communication costs.
* Custom communicators (`row_comm` and `col_comm`) are created using `MPI_Comm_split` to restrict collective operations like `MPI_Bcast` to the relevant subsets of processes.

### "Zero-Filling" Assembly Strategy
* To perform the trailing matrix update, a **Full-Column Broadcast Strategy** is used to assemble and distribute column panels.
* Each processor allocates a single temporary buffer (`col_panel`) of size $N \times B$, which is **reused** for every iteration to keep the memory footprint constant.
* At each step, processors initialize this buffer to zero and copy only their local blocks into the correct global positions.
* An `MPI_Allreduce` with the `MPI_SUM` operator is performed on the `col_comm`; since the data distribution is disjoint, this arithmetic sum acts as a logical merge of the panel data.

### Parallel Design & Scheduling
The implementation is based on a systematic data dependency analysis to ensure correctness and efficiency:
* **Outer Block Iterations:** These are strictly serialized due to Read-after-Write (RAW) dependencies between consecutive steps.
* **Diagonal Factorization:** Parallelized using the **Static** schedule to minimize synchronization overhead for fixed-size, contiguous data.
* **Panel Update:** Utilizes `schedule(dynamic)` so threads can effectively request work units on demand, preventing load imbalance caused by the sparse ownership of blocks.
* **Trailing Update:** The most compute-intensive phase uses `collapse(2)` and `schedule(guided)` to handle the decreasing workload of the triangular loop structure efficiently.
