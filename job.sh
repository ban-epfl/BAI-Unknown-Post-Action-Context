#!/bin/bash
# Example SLURM job: sbatch job.sh grid   (adapt the resources and the module line to your cluster)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --time=02:00:00

module load gcc python
source .venv/bin/activate
JOBS=$SLURM_CPUS_PER_TASK ./run_all.sh "$@"
