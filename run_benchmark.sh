#!/bin/bash

# Automation script for Cholesky MPI benchmark
# Automatically generates PBS jobs for different configurations

# Create directories for results if they do not exist
mkdir -p benchmarks
mkdir -p benchmarks/execution_output
mkdir -p benchmarks/pbs_scripts


# Clean the CSV file before starting (optional - comment out if you want to keep previous results)
# rm -f benchmark_results.csv

# Define the configurations to test
# Format: "N_MATRIX:N_PROCESSES:N_NODES:NCPUS:PLACEMENT"

CONFIGURATIONS=(
    # Scaling test with fixed matrix size (3072)
    "3072:1:1:1:scatter:excl"
    "3072:4:1:4:scatter:excl"
    "3072:9:1:9:scatter:excl"
    "3072:16:2:8:scatter:excl"
    "3072:25:3:10:scatter:excl"
    "3072:36:4:9:scatter:excl"
    "3072:49:5:10:scatter:excl"

    # # Scaling test with different matrix sizes (fixed processes = 16)
    "2048:16:2:8:scatter:excl"
    "4096:16:2:8:scatter:excl"
    "6144:16:2:8:scatter:excl"
    "8192:16:2:8:scatter:excl"

    # Scatter vs pack placement comparison
    "5120:25:3:10:scatter:excl"
    "5120:25:3:10:pack:excl"

    # Strong scaling test (same matrix, more processes)
    "10240:4:1:4:scatter:excl"
    "10240:16:2:8:scatter:excl"
    "10240:36:4:9:scatter:excl"
    "10240:64:7:10:scatter:excl"

    # Weak scaling test (matrix size proportional to processes)
    "3072:4:1:4:scatter:excl"      # ~2.4M elements per process
    "6144:16:2:8:scatter:excl"     # ~2.4M elements per process
    "9216:36:4:9:scatter:excl"     # ~2.4M elements per process
)

echo "=== Cholesky MPI Benchmark Automation ==="
echo "Generating ${#CONFIGURATIONS[@]} test configurations..."
echo ""

JOB_IDS=()

for CONFIG in "${CONFIGURATIONS[@]}"; do
    # Parse configuration
    IFS=':' read -r N_MATRIX N_PROCS N_NODES NCPUS PLACEMENT <<< "$CONFIG"
    
    # Compute walltime based on matrix size (estimate: larger = more time)
    if [ $N_MATRIX -le 4096 ]; then
        WALLTIME="0:10:00"
        QUEUE="short_cpuQ"
    elif [ $N_MATRIX -le 10240 ]; then
        WALLTIME="0:30:00"
        QUEUE="short_cpuQ"
    else
        WALLTIME="0:40:00"
        QUEUE="long_cpuQ"
    fi
    
    # Unique job name
    JOB_NAME="chol_N${N_MATRIX}_P${N_PROCS}_${PLACEMENT}"
    # Replace ':' with '_' in the job name for PBS and filenames
    SAFE_JOB_NAME=${JOB_NAME//:/_}
    PBS_SCRIPT="benchmarks/pbs_scripts/${SAFE_JOB_NAME}.pbs"

    RANDOM_ID=$RANDOM
    
    # Create PBS script
    cat > $PBS_SCRIPT << EOF
#!/bin/bash

#PBS -N ${SAFE_JOB_NAME}
#PBS -l select=${N_NODES}:ncpus=${NCPUS}:mem=4gb -l place=${PLACEMENT}
#PBS -l walltime=${WALLTIME}
#PBS -q ${QUEUE}
#PBS -o benchmarks/execution_output/${RANDOM_ID}_${SAFE_JOB_NAME}.o
#PBS -e benchmarks/execution_output/${RANDOM_ID}_${SAFE_JOB_NAME}.e

cd \$PBS_O_WORKDIR

echo "=========================================="
echo "Job: ${SAFE_JOB_NAME}"
echo "Matrix size: ${N_MATRIX}"
echo "Processes: ${N_PROCS}"
echo "Nodes: ${N_NODES}"
echo "CPUs per node: ${NCPUS}"
echo "Placement: ${PLACEMENT}"
echo "Started: \$(date)"
echo "=========================================="
echo ""

# Load MPI module
module load mpich-3.2

echo "Running MPI program..."
mpirun.actual -n ${N_PROCS} ./bin/cholesky_mpi ${N_MATRIX}

if [ $? -ne 0 ]; then
    echo "ERROR: MPI execution failed!"
    exit 1
fi


echo ""
echo "=========================================="
echo "Completed: \$(date)"
echo "=========================================="
EOF

    # Submit job
    echo "Submitting job: ${SAFE_JOB_NAME}"
    JOB_ID=$(qsub $PBS_SCRIPT)
    JOB_IDS+=($JOB_ID)
    echo "  Job ID: $JOB_ID"
    echo "  Script: $PBS_SCRIPT"
    echo ""
    
    # Small pause to avoid overloading the system
    sleep 1
done

echo "=========================================="
echo "All ${#CONFIGURATIONS[@]} jobs have been submitted!"
echo ""
echo ""
echo "To monitor jobs:"
echo "  qstat -u \$USER"
echo ""
echo "To see results in real time:"
echo "  tail -f benchmarks/execution_output/*.out"
echo ""
echo "Results will be saved in: benchmarks/benchmark_results.csv"
echo "=========================================="

