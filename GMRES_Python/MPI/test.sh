#!/bin/bash

for size in 1000 2000 4000 8000 16000 20000 25000 30000
do
  python gen.py $size
  for threads in 1 2 4 6
  do
    echo "SIZE: $size, THREADS: $threads:"
    mpiexec -n $threads python3 GMRES.py $size
    echo ""
    echo "///////////////////////////////"
  done
done

