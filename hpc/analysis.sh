#!/bin/bash
#SBATCH -J cobel # A single job name for the array
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # All cores on one Node
#SBATCH --mem 20000 # Memory request
#SBATCH -t 1-00:00 # Maximum execution time (D-HH:MM)
#SBATCH -o cobel_%A_%a.out # Standard output
#SBATCH -e cobel_%A_%a.err # Standard error
#module load tophat/2.0.13-fasrc02
#conda init
source activate cobel-spike
python run_analysis_all_sessions_arrayjob.py ${SLURM_ARRAY_TASK_ID}
