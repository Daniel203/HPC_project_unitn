#!/bin/bash

# Optimized Cholesky MPI Benchmark Suite
# Designed for comprehensive scaling analysis and performance characterization

# Create directories
mkdir -p benchmarks
mkdir -p benchmarks/execution_output
mkdir -p benchmarks/pbs_scripts

# Define test configurations
# Format: "N_MATRIX:N_PROCESSES:N_NODES:NCPUS:PLACEMENT:DESCRIPTION"
CONFIGURATIONS=(
    # ==================================================================
    # 1. STRONG SCALING 
    # ==================================================================
    "2048:1:1:1:scatter:strong_xxs_p1"
    "2048:4:1:4:scatter:strong_xxs_p4"
    "2048:9:1:9:scatter:strong_xxs_p9"
    "2048:16:2:8:scatter:strong_xxs_p16"

    "4096:1:1:1:scatter:strong_xs_p1"
    "4096:4:1:4:scatter:strong_xs_p4"
    "4096:9:1:9:scatter:strong_xs_p9"
    "4096:16:2:8:scatter:strong_xs_p16"

    "8192:1:1:1:scatter:strong_sm_p1"
    "8192:4:1:4:scatter:strong_sm_p4"
    "8192:9:1:9:scatter:strong_sm_p9"
    "8192:16:2:8:scatter:strong_sm_p16"
    "8192:25:3:9:scatter:strong_sm_p25"

    "16384:1:1:1:scatter:strong_md_p1"
    "16384:4:1:4:scatter:strong_md_p4"
    "16384:9:1:9:scatter:strong_md_p9"
    "16384:16:2:8:scatter:strong_md_p16"
    "16384:25:3:9:scatter:strong_md_p25"
    "16384:36:4:9:scatter:strong_md_p36"
    "16384:49:5:10:scatter:strong_md_p49"

    "32768:4:1:4:scatter:strong_lg_p4"
    "32768:9:1:9:scatter:strong_lg_p9"
    "32768:16:2:8:scatter:strong_lg_p16"
    "32768:25:3:9:scatter:strong_lg_p25"
    "32768:36:4:9:scatter:strong_lg_p36"
    "32768:49:5:10:scatter:strong_lg_p49"
    "32768:64:7:10:scatter:strong_lg_p64"

    "65536:25:3:9:scatter:strong_xl_p25"
    "65536:36:4:9:scatter:strong_xl_p36"
    "65536:49:5:10:scatter:strong_xl_p49"
    "65536:64:7:10:scatter:strong_xl_p64"


    # ==================================================================
    # 2. WEAK SCALING 
    # ==================================================================
    "4096:1:1:1:scatter:weak_xxs_p1"
    "6144:4:1:4:scatter:weak_xs_p4"
    "8192:9:1:9:scatter:weak_sm_p9"
    "10240:16:2:8:scatter:weak_md_p16"
    "12288:25:3:9:scatter:weak_lg_p25"
    "13312:36:4:9:scatter:weak_xl_p36"
    "15360:49:5:10:scatter:weak_xxl_p49"
    "16384:64:7:10:scatter:weak_xxxl_p64"

)

JOB_IDS=()

for CONFIG in "${CONFIGURATIONS[@]}"; do
    # Parse configuration
    IFS=':' read -r N_MATRIX N_PROCS N_NODES NCPUS PLACEMENT DESCRIPTION <<< "$CONFIG"
    
    # Walltime estimation (1 run, no serial verification)
    if [ $N_MATRIX -le 4096 ]; then
        WALLTIME="0:05:00"
    elif [ $N_MATRIX -le 8192 ]; then
        WALLTIME="0:10:00"
    elif [ $N_MATRIX -le 16384 ]; then
        WALLTIME="0:20:00"
    elif [ $N_MATRIX -le 32768 ]; then
        WALLTIME="1:30:00"
    elif [ $N_MATRIX -le 65536 ]; then
        WALLTIME="2:30:00"
    else
        WALLTIME="4:00:00"
    fi

    QUEUE="short_cpuQ"

    # Single-process penalty (true O(N^3))
    if [ $N_PROCS -eq 1 ]; then
        if [ $N_MATRIX -le 8192 ]; then
            WALLTIME="0:20:00"
        elif [ $N_MATRIX -le 16384 ]; then
            WALLTIME="1:00:00"
        elif [ $N_MATRIX -le 32768 ]; then
            WALLTIME="3:00:00"
        elif [ $N_MATRIX -le 65536 ]; then
            WALLTIME="5:00:00"
        else
            WALLTIME="6:00:00"
        fi
    fi


    # Job naming
    JOB_NAME="chol_${DESCRIPTION}"
    SAFE_JOB_NAME=${JOB_NAME//:/_}
    PBS_SCRIPT="benchmarks/pbs_scripts/${SAFE_JOB_NAME}.pbs"
    RANDOM_ID=$RANDOM

    # Other parameters
    TEST_CATEGORY=$(echo "${DESCRIPTION}" | cut -d'_' -f1)

    # Calculate needed RAM
    # Memory in MB (integer math)
    MEM_MB=$(( N_MATRIX * N_MATRIX * 8 / 1024 / 1024 / N_PROCS ))
    # Convert MB -> GB with ceiling
    MEM_GB=$(( (MEM_MB + 1023) / 1024 ))
    # Add safety margin
    MEM_GB=$(( MEM_GB + 1 ))
    
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
echo "Matrix size: ${N_MATRIX}"
echo "Processes: ${N_PROCS}"
echo "Nodes: ${N_NODES}"
echo "CPUs per node: ${NCPUS}"
echo "Placement: ${PLACEMENT}"
echo "Started: \$(date)"
echo "========================================================================"
echo ""

# Load MPI module
module load mpich-3.2

echo "Test category: \${TEST_CATEGORY}"
echo "Test placement: \${PLACEMENT}"
echo ""

echo "Running Cholesky factorization..."
mpirun.actual -n ${N_PROCS} ./bin/cholesky_mpi \
    --matrix-size ${N_MATRIX} \
    --test-category ${TEST_CATEGORY} \
    --placement ${PLACEMENT}

EXIT_CODE=\$?

if [ \$EXIT_CODE -ne 0 ]; then
    echo ""
    echo "ERROR: MPI execution failed with exit code \$EXIT_CODE"
    exit 1
fi

echo ""
echo "========================================================================"
echo "Completed successfully: \$(date)"
echo "========================================================================"
EOF

    # Submit job
    echo "Submitting: ${DESCRIPTION}"
    echo "  N=${N_MATRIX}, P=${N_PROCS}, Nodes=${N_NODES}, Placement=${PLACEMENT}"
    JOB_ID=$(qsub $PBS_SCRIPT)
    JOB_IDS+=($JOB_ID)
    echo "  Job ID: $JOB_ID"
    echo ""
    
    # Small delay to avoid overwhelming scheduler
    sleep 1
done

echo "========================================================================"
echo "All ${#CONFIGURATIONS[@]} jobs submitted successfully!"
echo ""
echo "Monitoring commands:"
echo "  qstat -u \$USER                    # Check job status"
echo "  qstat -u \$USER -n                 # Show node allocation"
echo "  tail -f benchmarks/execution_output/*.out  # Watch output"
echo ""
echo "Results:"
echo "  CSV: benchmarks/benchmark_results.csv"
echo "  Logs: benchmarks/execution_output/"
echo ""
echo "Analysis:"
echo "  python3 analyze_benchmarks.py     # Generate plots"
echo "========================================================================"

# Save job IDs for later reference
echo "${JOB_IDS[@]}" > benchmarks/submitted_job_ids.txt
echo "Job IDs saved to: benchmarks/submitted_job_ids.txt"
