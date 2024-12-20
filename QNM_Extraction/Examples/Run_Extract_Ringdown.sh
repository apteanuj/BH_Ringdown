#!/bin/bash

RUNDIR="$(cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
CODEDIR="$RUNDIR"/..
KERRTABLE="$CODEDIR"/Tables/kerr
SWSHTABLE="$CODEDIR"/Tables/swsh
EXEC="$CODEDIR"/Scripts/Extract_Ringdown_MoreModes.py
TRAJECTORIES="$RUNDIR"
WAVEFORMS="$RUNDIR"
OUTDIR="$RUNDIR"

SPIN="0.7"
THINC=000
DATALIST=( 090.0_n )

for i in "${DATALIST[@]}";
do "$EXEC" \
--qnmtablefolder "$KERRTABLE" \
--swshtablefolder "$SWSHTABLE" \
--trajfilepath "$TRAJECTORIES"/a"$SPIN"_thi"$THINC"_thf"$i".traj \
--wavefilepaths "$WAVEFORMS"/hm{0..4}_a"$SPIN"_thi"$THINC"_thf"$i".dat "$WAVEFORMS"/hmm{1..4}_a"$SPIN"_thi"$THINC"_thf"$i".dat \
--modeindex {-4..4} \
--mmax 4 \
--thinc "$THINC" \
--thf "$i" \
--spinvalue "$SPIN" \
--outputdir "$OUTDIR" \
--verbose
done

"$CODEDIR"/Scripts/Postprocess.py --inputdir "$OUTDIR" --spinvalue "$SPIN" --outputdir "$OUTDIR" --thinc "$THINC" --modeindex {-4..4} --verbose --mmax 4

