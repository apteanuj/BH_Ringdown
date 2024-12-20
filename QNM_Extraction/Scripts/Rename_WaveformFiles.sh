#!/bin/bash

# switch to convention where for m > 0, file are labeled as hm{1,2,3,4}_... instead of hm0{1,2,3,4}
DATADIR=""
WAVEFORMS="$DATADIR"/Waveforms

for mlabel in {0,1,2,3,4}; do 
for thi in {thi126,thi137,thi150,thi180};do
for wavefile in "$WAVEFORMS"/"$thi"/hm0"$mlabel"*.dat; do
echo "$wavefile"
echo "${wavefile//hm0$mlabel/hm$mlabel}"
mv "$wavefile" "${wavefile//hm0$mlabel/hm$mlabel}"
done
done
done
