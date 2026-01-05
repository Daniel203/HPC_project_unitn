#!/bin/bash

# Cholesky MPI+OpenMP Benchmark Suite
# Comprehensive comparison of MPI-only vs MPI+OpenMP hybrid parallelization

mkdir -p benchmarks
mkdir -p benchmarks/execution_output
mkdir -p benchmarks/pbs_scripts

# Test configurations: "N:N_PROCESSES:N_NODES:N_CPUS_PER_NODE:PLACEMENT:OPENMP:THREADS:DESC"
CONFIGURATIONS=(
    # ==================================================================
    # STRONG SCALING: Matrix 2048x2048
    # ==================================================================
    "2048:1:1:1:scatter:0:1:strong_2k_mpi_p1"
    "2048:4:1:4:scatter:0:1:strong_2k_mpi_p4"
    "2048:9:1:9:scatter:0:1:strong_2k_mpi_p9"
    "2048:16:2:8:scatter:0:1:strong_2k_mpi_p16"

    "2048:1:1:1:scatter:1:4:strong_2k_hybrid_4t_p1"
    "2048:4:1:4:scatter:1:4:strong_2k_hybrid_4t_p4"
    "2048:9:1:9:scatter:1:4:strong_2k_hybrid_4t_p9"
    "2048:16:2:8:scatter:1:4:strong_2k_hybrid_4t_p16"

    # ==================================================================
    # STRONG SCALING: Matrix 4096x4096
    # ==================================================================
    "4096:1:1:1:scatter:0:1:strong_4k_mpi_p1"
    "4096:4:1:4:scatter:0:1:strong_4k_mpi_p4"
    "4096:9:1:9:scatter:0:1:strong_4k_mpi_p9"
    "4096:16:2:8:scatter:0:1:strong_4k_mpi_p16"

    "4096:1:1:1:scatter:1:4:strong_4k_hybrid_4t_p1"
    "4096:4:1:4:scatter:1:4:strong_4k_hybrid_4t_p4"
    "4096:9:1:9:scatter:1:4:strong_4k_hybrid_4t_p9"
    "4096:16:2:8:scatter:1:4:strong_4k_hybrid_4t_p16"

    # ==================================================================
    # STRONG SCALING: Matrix 8192x8192
    # ==================================================================
    "8192:1:1:1:scatter:0:1:strong_8k_mpi_p1"
    "8192:4:1:4:scatter:0:1:strong_8k_mpi_p4"
    "8192:9:1:9:scatter:0:1:strong_8k_mpi_p9"
    "8192:16:2:8:scatter:0:1:strong_8k_mpi_p16"
    "8192:25:3:9:scatter:0:1:strong_8k_mpi_p25"

    "8192:1:1:1:scatter:1:4:strong_8k_hybrid_4t_p1"
    "8192:4:1:4:scatter:1:4:strong_8k_hybrid_4t_p4"
    "8192:9:1:9:scatter:1:4:strong_8k_hybrid_4t_p9"
    "8192:16:2:8:scatter:1:4:strong_8k_hybrid_4t_p16"
    "8192:25:3:9:scatter:1:4:strong_8k_hybrid_4t_p25"

    # ==================================================================
    # STRONG SCALING: Matrix 16384x16384
    # ==================================================================
    "16384:4:1:4:scatter:0:1:strong_16k_mpi_p4"
    "16384:9:1:9:scatter:0:1:strong_16k_mpi_p9"
    "16384:16:2:8:scatter:0:1:strong_16k_mpi_p16"
    "16384:25:3:9:scatter:0:1:strong_16k_mpi_p25"
    "16384:36:4:9:scatter:0:1:strong_16k_mpi_p36"
    "16384:49:5:10:scatter:0:1:strong_16k_mpi_p49"

    "16384:4:1:4:scatter:1:4:strong_16k_hybrid_4t_p4"
    "16384:9:1:9:scatter:1:4:strong_16k_hybrid_4t_p9"
    "16384:16:2:8:scatter:1:4:strong_16k_hybrid_4t_p16"
    "16384:25:3:9:scatter:1:4:strong_16k_hybrid_4t_p25"
    "16384:36:4:9:scatter:1:4:strong_16k_hybrid_4t_p36"
    "16384:49:5:10:scatter:1:4:strong_16k_hybrid_4t_p49"

    # ==================================================================
    # STRONG SCALING: Matrix 32768x32768
    # ==================================================================
    "32768:9:1:9:scatter:0:1:strong_32k_mpi_p9"
    "32768:16:2:8:scatter:0:1:strong_32k_mpi_p16"
    "32768:25:3:9:scatter:0:1:strong_32k_mpi_p25"
    "32768:36:4:9:scatter:0:1:strong_32k_mpi_p36"
    "32768:49:5:10:scatter:0:1:strong_32k_mpi_p49"
    "32768:64:7:10:scatter:0:1:strong_32k_mpi_p64"

    "32768:9:1:9:scatter:1:4:strong_32k_hybrid_4t_p9"
    "32768:16:2:8:scatter:1:4:strong_32k_hybrid_4t_p16"
    "32768:25:3:9:scatter:1:4:strong_32k_hybrid_4t_p25"
    "32768:36:4:9:scatter:1:4:strong_32k_hybrid_4t_p36"
    "32768:49:5:10:scatter:1:4:strong_32k_hybrid_4t_p49"
    "32768:64:7:10:scatter:1:4:strong_32k_hybrid_4t_p64"

    # ==================================================================
    # STRONG SCALING: Matrix 65536x65536
    # ==================================================================
    "65536:25:3:9:scatter:0:1:strong_64k_mpi_p25"
    "65536:36:4:9:scatter:0:1:strong_64k_mpi_p36"
    "65536:49:5:10:scatter:0:1:strong_64k_mpi_p49"
    "65536:64:7:10:scatter:0:1:strong_64k_mpi_p64"

    "65536:25:3:9:scatter:1:4:strong_64k_hybrid_4t_p25"
    "65536:36:4:9:scatter:1:4:strong_64k_hybrid_4t_p36"
    "65536:49:5:10:scatter:1:4:strong_64k_hybrid_4t_p49"
    "65536:64:7:10:scatter:1:4:strong_64k_hybrid_4t_p64"

    # ==================================================================
    # WEAK SCALING
    # ==================================================================
    "4096:1:1:1:scatter:0:1:weak_mpi_p1"
    "8192:4:1:4:scatter:0:1:weak_mpi_p4"
    "12288:9:1:9:scatter:0:1:weak_mpi_p9"
    "16384:16:2:8:scatter:0:1:weak_mpi_p16"
    "20480:25:3:9:scatter:0:1:weak_mpi_p25"
    "24576:36:4:9:scatter:0:1:weak_mpi_p36"
    "28672:49:5:10:scatter:0:1:weak_mpi_p49"
    "32768:64:7:10:scatter:0:1:weak_mpi_p64"

    "4096:1:1:1:scatter:1:4:weak_hybrid_4t_p1"
    "8192:4:1:4:scatter:1:4:weak_hybrid_4t_p4"
    "12288:9:1:9:scatter:1:4:weak_hybrid_4t_p9"
    "16384:16:2:8:scatter:1:4:weak_hybrid_4t_p16"
    "20480:25:3:9:scatter:1:4:weak_hybrid_4t_p25"
    "24576:36:4:9:scatter:1:4:weak_hybrid_4t_p36"
    "28672:49:5:10:scatter:1:4:weak_hybrid_4t_p49"
    "32768:64:7:10:scatter:1:4:weak_hybrid_4t_p64"
)

JOB_IDS=()

for CONFIG in "${CONFIGURATIONS[@]}"; do
    IFS=':' read -r N_MATRIX N_PROCS N_NODES NCPUS PLACEMENT ENABLE_OPENMP NUM_THREADS DESCRIPTION <<< "$CONFIG"
    
    # Walltime estimation based on matrix size and parallelization
    if [ $N_MATRIX -le 4096 ]; then
        WALLTIME="0:05:00"
    elif [ $N_MATRIX -le 8192 ]; then
        WALLTIME="0:10:00"
    elif [ $N_MATRIX -le 16384 ]; then
        WALLTIME="0:20:00"
    elif [ $N_MATRIX -le 32768 ]; then
        WALLTIME="1:00:00"
    elif [ $N_MATRIX -le 65536 ]; then
        WALLTIME="2:00:00"
    else
        WALLTIME="4:00:00"
    fi

    # Single-process gets more time
    if [ $N_PROCS -eq 1 ]; then
        if [ $N_MATRIX -le 8192 ]; then
            WALLTIME="0:20:00"
        elif [ $N_MATRIX -le 16384 ]; then
            WALLTIME="1:00:00"
        elif [ $N_MATRIX -le 32768 ]; then
            WALLTIME="3:00:00"
        else
            WALLTIME="6:00:00"
        fi
    fi

    QUEUE="short_cpuQ"
    
    JOB_NAME="chol_${DESCRIPTION}"
    SAFE_JOB_NAME=${JOB_NAME//:/_}
    PBS_SCRIPT="benchmarks/pbs_scripts/${SAFE_JOB_NAME}.pbs"
    RANDOM_ID=$RANDOM

    TEST_CATEGORY=$(echo "${DESCRIPTION}" | cut -d'_' -f1)
    
    # Memory calculation: total matrix size / number of processes
    MEM_MB=$(( N_MATRIX * N_MATRIX * 8 / 1024 / 1024 / N_PROCS ))
    MEM_GB=$(( (MEM_MB + 1023) / 1024 + 1 ))
    
    # Build OpenMP configuration
    if [ $ENABLE_OPENMP -eq 1 ]; then
        OPENMP_FLAG="--enable-openmp --num-threads ${NUM_THREADS}"
        OPENMP_ENV="export OMP_NUM_THREADS=${NUM_THREADS}"
        TOTAL_CORES=$(( N_PROCS * NUM_THREADS ))
    else
        OPENMP_FLAG=""
        OPENMP_ENV="# OpenMP disabled (MPI only)"
        TOTAL_CORES=${N_PROCS}
    fi
    
    # Create PBS script
    cat > $PBS_SCRIPT << EOF
#!/bin/bash

#PBS -N ${SAFE_JOB_NAME}
#PBS -l select=${N_NODES}:ncpus=${NCPUS}:mem=${MEM_GB}gb -l place=${PLACEMENT}:excl
#PBS -l walltime=${WALLTIME}
#PBS -q ${QUEUE}
#PBS -o benchmarks/execution_output/${RANDOM_ID}_${SAFE_JOB_NAME}.out
#PBS -e benchmarks/execution_output/${RANDOM_ID}_${SAFE_JOB_NAME}.err

cd \$PBS_O_WORKDIR

echo "========================================================================"
echo "Job: ${DESCRIPTION}"
echo "Matrix size: ${N_MATRIX}x${N_MATRIX}"
echo "MPI processes: ${N_PROCS}"
echo "OpenMP threads per process: ${NUM_THREADS}"
echo "Total cores: ${TOTAL_CORES}"
echo "Nodes: ${N_NODES}, CPUs per node: ${NCPUS}"
echo "Placement: ${PLACEMENT}"
echo "Started: \$(date)"
echo "========================================================================"
echo ""

# Load required modules
module load mpich-3.2

# Set OpenMP configuration
${OPENMP_ENV}

echo "Configuration:"
echo "  Test category: ${TEST_CATEGORY}"
echo "  Placement: ${PLACEMENT}"
echo "  OpenMP enabled: ${ENABLE_OPENMP}"
if [ ${ENABLE_OPENMP} -eq 1 ]; then
    echo "  OMP_NUM_THREADS: \${OMP_NUM_THREADS}"
fi
echo ""

echo "Running Cholesky factorization..."
mpirun.actual -n ${N_PROCS} ./bin/cholesky_mpi \\
    --matrix-size ${N_MATRIX} \\
    --test-category ${TEST_CATEGORY} \\
    --placement ${PLACEMENT} \\
    ${OPENMP_FLAG}

EXIT_CODE=\$?

if [ \$EXIT_CODE -ne 0 ]; then
    echo ""
    echo "ERROR: Execution failed with exit code \$EXIT_CODE"
    exit 1
fi

echo ""
echo "========================================================================"
echo "Completed successfully: \$(date)"
echo "========================================================================"
EOF

    echo "Submitting: ${DESCRIPTION}"
    if [ $ENABLE_OPENMP -eq 1 ]; then
        echo "  N=${N_MATRIX}, P=${N_PROCS}, OpenMP=${NUM_THREADS}T, Total=${TOTAL_CORES} cores"
    else
        echo "  N=${N_MATRIX}, P=${N_PROCS}, MPI-only, Total=${TOTAL_CORES} cores"
    fi
    JOB_ID=$(qsub $PBS_SCRIPT)
    JOB_IDS+=($JOB_ID)
    echo "  Job ID: $JOB_ID"
    echo ""
    
    sleep 1
done

echo "========================================================================"
echo "All ${#CONFIGURATIONS[@]} jobs submitted!"
echo ""
echo "Results: benchmarks/benchmark_results.csv"
echo "Analysis: python3 analyze_benchmarks.py"
echo "========================================================================"

echo "${JOB_IDS[@]}" > benchmarks/submitted_job_ids.txt
