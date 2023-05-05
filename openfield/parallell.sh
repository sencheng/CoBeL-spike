#!/bin/bash
# $1 $2 the same as in run_sim_openfield_recursive
# $3... location

# https://stackoverflow.com/questions/356100/how-to-wait-in-bash-for-several-subprocesses-to-finish-and-return-exit-code-0

for loc in "${@:3}"
  do
  mkdir -p $loc/logs
  touch $loc/logs/out.log
  touch $loc/logs/err.log
  cd $loc/openfield; ./run_sim_openfield_recursive.sh $1 $2  1>$loc/logs/out.log 2>$loc/logs/err.log &
  pids+=$!
done

for pid in ${pids[*]}; do
  wait $pid
done

echo "done"