#!/bin/bash
#SBATCH -J cobel # A single job name for the array
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # All cores on one Node
#SBATCH --mem 20000 # Memory request
#SBATCH -t 1-00:00 # Maximum execution time (D-HH:MM)
#SBATCH -o cobel_%A_%a.out # Standard output
#SBATCH -e cobel_%A_%a.err # Standard error
module restore cobel-spike
#conda init
source /home/mohagmnr/projects/CoBeL-spike/packages/set_vars.sh

echo "running simulation for trial ${SLURM_ARRAY_TASK_ID}"
    
rm -r ./of
rm report_det.dat

python update_random_seed.py ${SLURM_ARRAY_TASK_ID} # Changing random seed for each trial
python create_dirs.py           		    # Create the data directory    
python update_music_file.py     		    # Make sure whether the updated parameters
                                   		    # are also updated in the music conf. file
python copy_par_files.py        		    # Copy parameter files to the simulation
                                    		    # output directory for later comparisons
python create_grid_positions.py 		    # Create place cells selectivity map
    
gymz-controller gym openfield.json &
gym_pid=$(echo $!)
mpirun -np 7 music nest_openfield.music
# sleep 1m
kill $gym_pid
python ../analysis/analysis_test.py
