#!/usr/bin/env python3
import numpy as np
import argparse
import sys,os
import subprocess

sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'..'))
from QNM_Extraction.Library.ringdownlib import *

#========================================
# Main
#========================================
def mlabel(mval):
  if mval < 0:
    return "mm%1d" % abs(mval)
  else:
    return "m%1d" % mval

def main(args):
  m = args.modeindex
  mmax = args.mmax
  lmax = mmax + abs(m)
  lmin = max(2,abs(m))
  modelist = [[ll,m] for ll in range(lmin,lmax+1)]    
  nmodes = len(modelist)

  # Read in all data
  spinstr = "%1.2f" % args.spinvalue
  commandstr = 'ls | grep spheroidal_coefficients | grep ' +    \
    "_"+mlabel(m)+"_" + ' | grep thinc' + ("%.3d" % args.thinc) + " | grep a" + spinstr[-2:]
  proc = subprocess.run(commandstr, cwd=args.inputdir,stdout=subprocess.PIPE, shell=True)
  inputfiles = proc.stdout.decode('utf-8').split("\n")[:-1]
  if args.verbose:
    print("  Post processing ", mlabel(m), " fits")
    print("  Found %d Input Files" % (len(inputfiles)))
  assert len(inputfiles) > 0, "Need at least one inputfile to read in."

  # Define Containers to Store Results [(thfin, rel_amp), ...]
  relamp_positive = []
  relangle_positive = []
  relamp_negative = []
  relangle_negative = []

  # Loop over data to find excitation coefficients
  for j in range(len(inputfiles)):

    # Read in data, check normality
    data = np.transpose(np.loadtxt(args.inputdir + "/" + inputfiles[j]))
    assert len(data) == (1 + nmodes*4), \
      "Input data length mismatch: %d" % len(data)
    thfin = float(inputfiles[j].split("_")[-2][3:])
    #print "  Reading in data for theta_fin = ", thfin
    signthfin = inputfiles[j].split("_")[-1][0]
    time = data[0]

    # Get Amplitudes
    hamps = np.zeros((nmodes*2,len(time)))
    hangles = np.zeros((nmodes*2,len(time)))
    for i in range(nmodes*2):
      hamps[i] = np.sqrt(data[1+2*i]**2 + data[2+2*i]**2)
      hangles[i] = np.unwrap(np.angle(data[1+2*i] + 1j*data[2+2*i]))

    # Find when the amplitudes stabilize
    stdmin = 1e99
    # Default Time window
    timewindow = 50.00

    # Alternate Time window 
    # (a = 0.9, m = +-2, thinc = 20)
    # timewindow = 60.00

    # Alternate Time window
    # (a = 0.9, m = +-3, thinc = 20)
    # timewindow = 10.

    # # Alternate Time window 
    # # (a = 0.99)
    # timewindow = 30.0


    thigh = 0
    tstart = 0. # retrograde set this to 0, prograde set to 50 for optimize
    i = int(np.where(time == (time[0] + tstart))[0][0])
    tbestlow,besthigh = (0,0)
    while thigh < np.min((time[-1],time[0] + 250)):
      (tlowindex,tlow) = (i,time[i])
      thigh = tlow + timewindow
      index = np.where(np.abs(time - thigh) < 1e-10)[0]
      thighindex = int(index)
      hamps_mean = np.mean(hamps[:,tlowindex:thighindex],axis=1)
      hamps_std  = np.std(hamps[:,tlowindex:thighindex],axis=1)
      stdtotal = np.sum (hamps_std / hamps_mean)

      if stdtotal < stdmin:
        stdmin = stdtotal
        hamps_mean_best = hamps_mean
        hangles_mean_best = np.mean(hangles[:,tlowindex:thighindex],axis=1)
        hangles_mean_best = hangles_mean_best % (2*np.pi)
        (tbestlow,tbesthigh) = (tlow,thigh)
      i += 1
    if args.verbose == True:
      print("(cos(thf),sgnthfin,tlow, thigh) = ({:1.2f},{:s},{:1.2f}, {:1.2f})".format(np.cos(thfin*np.pi/180.),signthfin,tbestlow,tbesthigh))

    # Store in Results Container
    if signthfin == "n":
      relamp_negative.append(np.append([thfin],hamps_mean_best[:]))
      relangle_negative.append(np.append([thfin],hangles_mean_best[:]))

      if args.thinc == 0 or args.thinc == 180:
        relamp_positive.append(np.append([thfin],hamps_mean_best[:]))
        relangle_positive.append(np.append([thfin],hangles_mean_best[:]))

    elif signthfin == "p":
      relamp_positive.append(np.append([thfin],hamps_mean_best[:]))
      relangle_positive.append(np.append([thfin],hangles_mean_best[:]))

      if args.thinc == 0 or args.thinc == 180:
        relamp_negative.append(np.append([thfin],hamps_mean_best[:]))
        relangle_negative.append(np.append([thfin],hangles_mean_best[:]))

  # Save To File
  savename = "excitation_coefficients_a" + \
    ("%.2f" % args.spinvalue)[-2:] + "_thinc" + \
    ("%.3d" % args.thinc) + "_" + "h" + mlabel(m) + ".dat"
  outfile = open(args.outputdir + "/" + savename,'w')

  # Get Headings
  headings = "#[0] thetafin\n#[1] sign_thetafindot\n"
  for i in range(nmodes):
    if m >= 0:
      mposprime = ""
      mnegprime = "prime"
    elif m < 0:
      mposprime = "prime"
      mnegprime = ""
    headings += "#[%d] l%d%s%s_amp\n" % \
      (4*i+2,modelist[i][0],mlabel(abs(m)),mposprime)
    headings += "#[%d] l%d%s%s_phase\n" % \
      (4*i+3,modelist[i][0],mlabel(abs(m)),mposprime)
    headings += "#[%d] l%d%s%s_amp\n" % \
      (4*i+4,modelist[i][0],mlabel(-1*abs(m)),mnegprime)
    headings += "#[%d] l%d%s%s_phase\n" % \
      (4*i+5,modelist[i][0],mlabel(-1*abs(m)),mnegprime)
  outfile.write(headings)

  # Write Data 
  if len(relamp_negative) != 0:
    for i in range(len(relamp_negative)):
      outfile.write("%.2f\t%d" % (relamp_negative[i][0],-1))
      for j in range(1,len(relamp_negative[i])):
        outfile.write("\t%.4e" % (relamp_negative[i][j]))
        outfile.write("\t%.4e" % (relangle_negative[i][j]))
      outfile.write("\n")

  if len(relamp_positive) != 0:

    for i in range(len(relamp_positive)):
      outfile.write("%.2f\t%d" % (relamp_positive[i][0],1))
      for j in range(1,len(relamp_positive[i])):
        outfile.write("\t%.4e" % (relamp_positive[i][j]))
        outfile.write("\t%.4e" % (relangle_positive[i][j]))
      outfile.write("\n")

  outfile.close()

#========================================
# Parser and Decision Routines
#========================================
if __name__== "__main__":
  parser = argparse.ArgumentParser( \
    description="Calculate QNM excitation coefficients \
     from matrix inversion method.")
  parser.add_argument("--inputdir",required=True, \
    help="Directory where the fit results are loaded from")
  parser.add_argument("--spinvalue",type=float,required=True)
  parser.add_argument("--outputdir",default=None, \
    help="Directory where the excitation coefficients are loaded")
  parser.add_argument("--thinc",required=True,type=int)
  parser.add_argument("--mmax",required=True,type=int)
  parser.add_argument("--modeindex",required=True,type=int,nargs="+", \
    help="Angular Index of Modes to Extract  \
    (m = 2, 3 or 4) or list (m = -2,2,3,4)")
  parser.add_argument("--overtones",default=False,action='store_true')
  parser.add_argument("--verbose",default=False,action='store_true')
  args = parser.parse_args()
  marray = args.modeindex
  print("Post processing...")
  for mindex in marray:
    args.modeindex = mindex
    main(args)
