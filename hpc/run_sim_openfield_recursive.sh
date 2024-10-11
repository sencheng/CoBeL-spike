#!/bin/sh

if [ $1 -le $2 ]; then

    echo "running simulation for trial $1 out of $2"
    
    rm -r ./of
    rm report_det.dat

    python update_random_seed.py $1 # Changing random seed for each trial
    python create_dirs.py           # Create the data directory    
    python update_music_file.py     # Make sure whether the updated parameters
                                    # are also updated in the music conf. file
    python copy_par_files.py        # Copy parameter files to the simulation
                                    # output directory for later comparisons
    python create_grid_positions.py # Create place cells selectivity map
    
    gymz-controller gym openfield.json &
    gym_pid=$(echo $!)
    mpirun -np 7 music parameter_sets/current_parameter/nest_openfield.music
    # sleep 1m
    kill $gym_pid
    python ../analysis/analysis_test.py
    
    
    ./run_sim_openfield_recursive.sh $(($1+1)) $2
fi