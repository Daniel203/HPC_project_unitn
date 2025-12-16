#!/bin/bash
#PBS -l select=2:ncpus=4:mem=2gb -l place=scatter:excl
#PBS -l walltime=0:05:00
#PBS -q short_cpuQ

exec > execution_output/${PBS_JOBID}.out
exec 2> execution_output/${PBS_JOBID}.err

cd $PBS_O_WORKDIR

module load mpich-3.2
mpirun.actual -n 8 ./main_mpi 2048
