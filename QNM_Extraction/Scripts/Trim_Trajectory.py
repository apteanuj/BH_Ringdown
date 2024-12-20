import numpy as np
import argparse
from pathlib import Path

def cuttrajectory(file,prebuffer,postbuffer,skip,retrograde):
  data = np.loadtxt(file)
  assert data.shape[1] == 10 
  omegacolumn = 5 # column corresponding to phidot

  # if prograde, search for phidot peak to cut data
  if retrograde == 0:
    peakindex = np.argmax(data[:,omegacolumn])

  # if prograde, search for peak negative phidot to cut data
  elif retrograde == 1:
    peakindex = np.argmin(data[:,omegacolumn])

  # calculate start and end array indices of trimmed data
  dt = data[peakindex,0] - data[peakindex-1,0]
  startindex = peakindex - int(prebuffer / dt)
  endindex = peakindex + int(postbuffer / dt)
  if endindex > len(data):
    endindex = -1

  return data[startindex:endindex:skip,:],data[startindex:endindex:skip,omegacolumn]

def main(args):

  # Convert command line inputs to Paths
  outputdir = Path(args.outputdir)
  trajfilepath = Path(args.trajfilepath)
  if args.savename != None:
    savename = args.savename
  else:
    savename = trajfilepath.name
  savefilepath = outputdir / savename
  print("Input Trajectory File: ", trajfilepath.as_posix())
  print("Output Trajectory File: ", savefilepath.as_posix())

  # check overwrite
  if args.overwrite == False:
    assert (savefilepath.exists() == False), "Savefilepath exists and overwrite flag set to 0"
 
  # Cut Trajectory
  data,omegaOrb = cuttrajectory(args.trajfilepath,args.prebuff,200,args.skip,args.retrograde)
  if args.retrograde == 0:
    print("(t_begin,t_end,tPeakOrb,npoints) = ", data[0,0],",",data[-1,0],",",data[np.argmax(omegaOrb),0],",",len(omegaOrb))
  elif args.retrograde == 1:
    print("(t_begin,t_end,tPeakOrb,npoints) = ", data[0,0],",",data[-1,0],",",data[np.argmin(omegaOrb),0],",",len(omegaOrb))

  # save
  np.savetxt(savefilepath,data)

if __name__== "__main__":
  parser = argparse.ArgumentParser(description="Trim Trajectory Files To Zoom in on Ringdown Portion, For Faster I/O in Extract_Ringdown.py")
  parser.add_argument("--trajfilepath",required=True,help="Filepath to trajectory data")
  parser.add_argument("--outputdir",required=True,help="Directory path of where to save trimmed trajectory file.")
  parser.add_argument("--savename",default=None,help="Optional filename of the trimmed trajectory file. Default is original file")
  parser.add_argument("--overwrite",action="store_true")
  parser.add_argument("--retrograde",action="store_true")
  parser.add_argument("--prebuff",default=50.0,type=float,help="Time before phidot peak to start trimmed trajectory. default 50M")
  parser.add_argument("--skip",default=3,type=int,help="Skip every SKIP lines, which is okay since lightring crossing algorithm will interpolate anyways")
  args = parser.parse_args()
  main(args)
