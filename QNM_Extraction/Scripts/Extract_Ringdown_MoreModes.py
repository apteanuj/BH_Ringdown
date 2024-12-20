#!/usr/bin/env python3
import numpy as np
import scipy.optimize
import argparse
import sys, os

sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'..'))
from QNM_Extraction.Library.ringdownlib import *

#========================================
# Support Functions
#========================================
def preparesystem(m,a,swsh_dir,qnm_dir,mmax ):
  '''                                                       \
  Return spherical-spheroidal inner products and            \
  qnm frequencies needed for                                \
  calculating elements of the LHS Linear System Matrix      \
  System := A[lp][l]                                        \
  larray  - l indices of spherical multipoles e.g. [2,3,4]   \
  lparray - l indices of spheroidal multipoles e.g. [2,3,4] \
  m - azimuthal index                                       \
  '''
  absm = abs(m)
  lmax = mmax + absm
  lmin = max(2,absm)

  # Setup the system matrix to solve
  larray  = [[ll,dd] for dd in [0,1] for ll in range(lmin,lmax+1)]
  lparray = [[ll,mm] for ll in range(lmin,lmax+1) for mm in [absm,-absm]]

  # Take into account m = 0 counterrotating modes degeneracy
  # (abs(l),0) and (-abs(l),0)
  if m == 0:
    lparray = [[lm[0]*(-1)**i,lm[1]] for i,lm in enumerate(lparray)]

  alphasystem = np.zeros((len(larray),len(larray),2),dtype=complex)

  for lindex in range(len(larray)):
    for lpindex in range(len(lparray)):
      # Be mindful about definition of overlap
      # Convention used here is S_l'm = \sum_{l} (mu_l'lm)^*  Y_lm 
      mp = lparray[lpindex][1]
      lp = lparray[lpindex][0]
      lv = larray[lindex][0]
      mure,muim = mulmlpnp(mp,lv,abs(lp),0,a,swsh_dir)
      if mp*m > 0: # corotating modes 
        mu = (1)**(lv) * (mure - 1j*muim)
      elif mp*m < 0: # counter rotating modes
        mu = (-1)**(lv)*(mure + 1j*muim)

      # Handle m = 0 case separately
      elif m == 0:
        if lp > 0:
          mu = (1)**(lv) * (mure - 1j*muim)
        elif lp < 0:
          mu = (-1)**(lv)*(mure + 1j*muim)

      # Get QNM frequencies, with Re(omega), Im(omega) < 0; 
      # For m > 0, expect omegare > 0 and omegaim > 0
      omegare,omegaim = \
        getqnmfrequencies([[lp,mp]],a,qnm_dir)[0]
      if mp*m > 0: # corotating modes
        omega = -1j * omegare - omegaim
      elif mp*m < 0: # counter rotating modes
        omega = 1j * omegare - omegaim

      # getqnmfrequencies returns negative Re(omega) 
      # automatically for counterrotating m = 0 mode
      elif m == 0:
        assert lp * omegare > 0, "Error loading m = 0 QNM frequency"
        omega = -1j * omegare - omegaim

      # If derivative, multiply mu by i*omega
      if larray[lindex][1] != 0:
        mu *= omega
      alphasystem[lindex][lpindex] = [mu,omega]

  return alphasystem

def getlinearsystem(alphasystem,t,tstart):
  musystem = alphasystem[:,:,0]
  omegasystem = alphasystem[:,:,1]
  return musystem * np.exp( omegasystem * (t-(tstart)))

def mlabel(mval):
  if mval < 0:
    return "mm%1d" % abs(mval)
  else:
    return "m%1d" % mval

#========================================
# Main
#========================================
def main(args, trajdata):

  # Command Line Options
  qnmdir  = args.qnmtablefolder
  swshdir = args.swshtablefolder
  mmax = args.mmax
  m = args.modeindex
  a = args.spinvalue
  thinc = args.thinc

  absm = abs(m)    
  lmax = mmax + absm
  lmin = max(2,absm)
  
  modelist = [[ll,m] for ll in range(lmin,lmax+1)]
  nmodes = len(modelist)

  #===============================
  # Adjust Waveform Data Type Here
  #===============================
  waveformdatas,[thetafin,signthetadot] = \
    loadwaveforms(args.wavefilepaths,modelist,mmax,verbose=args.verbose)

  # Assert Uniformity of Waveform Data 
  # waveformdatas[mode] = {time, hplus, hcross}
  for i in range(1,len(waveformdatas)):
    assert np.abs(np.sum(waveformdatas[0][0]-waveformdatas[i][0])) < 1e-10, \
      "Waveform Time Slice Mismatch"

  # Interpolate Data to Smooth Out Before Taking Derivatives
  oldtime = waveformdatas[0][0]
  newtime = np.arange(waveformdatas[0][0][0],waveformdatas[0][0][-1],.1)
  time_waveforms = newtime
  dt = time_waveforms[1] - time_waveforms[0]

  # Start Analysis 50M Before tPeakOrb
  i = 0
  while time_waveforms[i] < (tlrcross - 50.):
    i += 1
  timestartindex = i
  ntimes = (len(time_waveforms) - timestartindex) - 1

  # Define RHS of Linear System
  hpdataold = np.zeros((nmodes,len(oldtime)))
  hcdataold = np.zeros((nmodes,len(oldtime)))
  hpdatanew = np.zeros((nmodes,len(newtime)))
  hcdatanew = np.zeros((nmodes,len(newtime)))
  for i in range(nmodes):
    time, hpdataold[i], hcdataold[i] = waveformdatas[i]
    hpdatanew[i] = np.interp(newtime,oldtime,hpdataold[i])
    hcdatanew[i] = np.interp(newtime,oldtime,hcdataold[i])

  hmode_rd = np.zeros((nmodes,ntimes),dtype=complex)
  hmode_rd_deriv = np.zeros((nmodes,ntimes),dtype=complex)
  for i in range(nmodes):
    hmode_rd[i] = hpdatanew[i][timestartindex:-1] -  \
        1j * hcdatanew[i][timestartindex:-1]
    hmode_rd_deriv[i] = np.gradient(hmode_rd[i], dt)

  spherical_coefficients = np.transpose(np.vstack((hmode_rd,hmode_rd_deriv)))
  assert np.shape(spherical_coefficients)[1] == nmodes*2, \
    "RHS Data Length mismatch"
  assert np.shape(spherical_coefficients)[0] == ntimes, \
    "RHS Data Length mismatch"

  # Initialize LHS Solution Vector
  spheroidal_coefficients = \
    np.zeros((ntimes,len(modelist)*2),dtype=complex)

  # Prepare To Solve By Finding 
  # Inner products and QNMs relevant for inversion
  alphasystem = preparesystem(m,a,swshdir,qnmdir,mmax)

  # Solve the System at Each Time, 
  time_solution = time_waveforms[timestartindex:-1]

  #print("tpeak = ", waveformdatas[1][3])
  for t in range(ntimes):

    #=======================
    # Adjust t_0 here
    #=======================
    alplm = getlinearsystem(\
      alphasystem,time_solution[t],tlrcross) 
  
    spheroidal_coefficients[t] = \
        np.linalg.solve(alplm,spherical_coefficients[t])

  # Save Output To File  
  savename = os.path.join(args.outputdir, \
    "./spheroidal_coefficients_" + mlabel(m) + "_a" + \
    ("%.2f" % args.spinvalue)[-2:] + "_thinc" + \
    ("%.3d" % args.thinc) + "_thf" + args.thf + ".out")

  fout = open(savename,'w')

  # Get Headings
  headings = "#[0] time\n"
  for i in range(nmodes):
    headings += "#[%d] realamp_l%d%s\n" % (4*i+1,modelist[i][0],mlabel(m))
    headings += "#[%d] imagamp_l%d%s\n" % (4*i+2,modelist[i][0],mlabel(m))
    headings += "#[%d] realamp_l%d%s\n" % \
      (4*i+3,modelist[i][0],mlabel(-m))
    headings += "#[%d] imagamp_l%d%s\n" % \
      (4*i+4,modelist[i][0],mlabel(-m))
  fout.write(headings)
  
  # Write Data
  for t in range(len(spheroidal_coefficients)):
    fout.write("%.2f" % (time_solution[t]))
    for j in range(nmodes*2):
      fout.write("\t%.5e\t%.5e" % (
      float(np.real(spheroidal_coefficients[t][j])), \
      float(np.imag(spheroidal_coefficients[t][j]))))
    fout.write("\n")
  fout.close()
  if args.verbose:
    print("Wrote m = %d output to: " % (m), savename)

if __name__== "__main__":
  parser = argparse.ArgumentParser(description="Finds Spheroidal Mode Excitation Coefficients in Spherical Multipole Data.")

  # Tables
  parser.add_argument("--qnmtablefolder",type=str,required=True,help="Path to folder containing Kerr QNM frequencies")
  parser.add_argument("--swshtablefolder",type=str,required=True,help="Path to folder containing spherical-spheroidal overlaps")

  # Analysis Options
  parser.add_argument("--modeindex",required=True,type=int,nargs="+",help="Specify which angular degrees to extract, comma separated (e.g. --modeindex 0,-2,2,3,4)")
  parser.add_argument("--outputdir",default=None,help="Directory where the fit results and the fit plots will be dumped")

  # Specify Input Data
  parser.add_argument("--trajfilepath",required=True,help="Filepath to single trajectory data")
  parser.add_argument("--wavefilepaths",nargs="+",required=True,help="List of waveform datafiles (e.g. --wavefilepaths /dir/file1.dat /dir/file2.dat) for corresponding to trajectory in --trajfilepath")
  parser.add_argument("--spinvalue",required=True,type=float,help="Spin value ranging from 0.0 to 1.0")
  parser.add_argument("--thinc",required=True,type=int,help="Inclination of trajectory")
  parser.add_argument("--thf",required=True,type=str,help="Final polar angle of trajectory")  
  parser.add_argument("--mmax",required=True,type=int,help="Number of l modes to model beyond l = m. Supports 2 <= mmax <= 5")
  parser.add_argument("--verbose",action='store_true')
  args = parser.parse_args()

  lrradius = 2*(1+np.cos((2./3.)*np.arccos(-abs(args.spinvalue)))) # Prograde equatorial orbit of same BH spin
  assert (lrradius >= 1.0 and lrradius <= 4.0), "Invalid light-ring radius %1.1f" % (lrradius)
  if args.verbose:
    print("Equivalent Light Ring Radius = ",lrradius)

  # Load Trajectory
  trajdata = loadtrajectory(args.trajfilepath,lr_radius=lrradius,thinc=args.thinc,verbose=args.verbose)
  tOrb,omegaOrb, [tPeakOrb,tlrcross] = trajdata
  if args.verbose:
    print("  Trajectory Info: (tIniOrb,tFinOrb,tPeakOrb,tLRcross) = ", \
      tOrb[0],",",tOrb[-1],",",tPeakOrb, ",",tlrcross)
  
  print("Extracting QNM amplitudes...")
  marray = args.modeindex
  for mindex in marray:
    args.modeindex = mindex
    main(args,trajdata)
