#!/bin/sh

for file in Trajectories/*; do
    bfile=$(basename "$file" .traj)

    echo "$bfile"

    cd HighOrder
    cp body.bkup body.cu
    cp build.bkup build.sh
    cp job.bkup job.lsf
    sed -i "s/replaceme/$bfile/g" body.cu
    sed -i "s/runme/$bfile/g" build.sh
    sed -i "s/runme/$bfile/g" job.lsf
    ./build.sh
    bsub < job.lsf
    cd ..
    
done
