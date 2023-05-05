#!/bin/sh

rm -r ./of
rm report_det.dat
python update_music_file.py
python create_dir.py
gymz-controller gym openfield.json &
gym_pid=$(echo $!)
mpirun -np 7 music nest_openfield.music # TODO: change to equal num_core_per_sim
# sleep 1m
kill $gym_pid
python ../analysis/analysis_test.py

