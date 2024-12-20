import pandas
import numpy as np
import os

def loadtrajectory(file,verbose=0,lr_radius=None,thinc=0):
  print("Loading Trajectory File: ", file)
  data = np.loadtxt(file)
  if verbose > 0:
    print("Trajectory data has dimensions ", data.shape)
  if data.shape[1] == 10: # new inc data
    omegacolumn = 5
  elif data.shape[1] == 7: # old eq data
    omegacolumn = 4

  # Preprocess and interpolate
  deltat = 0. # in mathematica code this is set to 50... not sure why we need this
  dt = data[1,0] - data[0,0]
  data[:,0] = data[:,0] + deltat
  newtime = np.arange(data[:,0][0],data[:,0][-1],dt)  

  # Keep data that is after tpeak - deltat
  omegaorb = np.interp(newtime,data[:,0],data[:,omegacolumn])
  if thinc < 90:
    tpeak = newtime[np.argmax(omegaorb)]
  elif thinc > 90:
    tpeak = newtime[np.argmax(-omegaorb)]

  #===========================================
  # Used for Taracchini Comparison with Old Data
  #===========================================
  tpeak = newtime[np.argmax(omegaorb)]
  # tpeak = newtime[np.argmin(np.abs(omegaorb))]
  # tpeak = 1054.080000000437


  indexstart = np.argmin(np.abs(newtime-tpeak)) - int(deltat / dt)

  # Light Ring Crossing
  if lr_radius == None:
     return newtime[indexstart:],omegaorb[indexstart:],tpeak

  elif lr_radius != None:
    if verbose:
      print("LR radius: ", lr_radius)
    tindexlr = np.argmin(np.abs(data[:,1] - lr_radius))
    indexstart = np.amin([indexstart,tindexlr])
    return newtime[indexstart:],omegaorb[indexstart:],[tpeak,data[tindexlr][0]]

def loadwaveforms(wavefiles, modelist, mmax, verbose=0):
  '''Assumes wavefiles have format hm1_* for m >= 0 or hmm1_* for m < 0'''

  assert (mmax >=2) and (mmax <= 8), "Invalid mmax: {}".format(mmax)
  # Check each mode is consistent with mmax
  for mode in modelist:
    absm_test = abs(mode[1])
    lmax_test = mmax + absm_test
    lmin_test = max(2,absm_test)    
    assert (mode[0] >= lmin_test) and (mode[0] <= lmax_test), \
      "Mode ({},{}) out or range for mmax {}".format(mode[0],mode[1],mmax)

  wavefilenames = [os.path.basename(wfile) for wfile in wavefiles]

  ########################
  # Get filepaths for each mode
  ########################
  filemodelist = [] # which modes are in each respective file
  for ifile in range(len(wavefilenames)):

    # sign(m): check by scanning fourth char in filename
    fourth_char = wavefilenames[ifile][3]

    # m > 0 (hm1_a0.5....dat)     
    if fourth_char == "_":
      mf = int(wavefilenames[ifile][2])
      filemodelist.append([[mf+dm,mf] for dm in range(mmax+1)])

    # m < 0 (hmm1_a0.5....dat) 
    elif fourth_char != "_":
      mf = -int(wavefilenames[ifile][3])
      filemodelist.append([[abs(mf)+dm,mf] for dm in range(mmax+1)])

  ########################
  # Get data for each mode
  ########################
  results = []
  targetfile = None

  # for each desired mode
  for mode in modelist:

    # scan each file for desired mode
    targetfile = None
    for ifile in range(len(wavefiles)):
      if mode in filemodelist[ifile]:
        targetfile = ifile
    assert targetfile != None, \
      "Could not locate waveform file for mode %s" % (str(mode))
    if verbose != 0:
      print("Loading Waveform File for mode ", mode, ":", wavefiles[targetfile])

    # Load Waveform file
    data = pandas.read_csv(wavefiles[targetfile],delim_whitespace=True).values
    # Ensure data is uniform timestep
    dt = data[1,0] - data[0,0]
    (tini,tfin) = (data[0,0],data[-1,0])
    newtime = np.arange(tini,tfin,dt)

    # Find which columns have desired mode
    (iplus,icross) = (1+2*(mode[0]-np.abs(mode[1])),2+2*(mode[0]-np.abs(mode[1])))
    # return [time, hplus, hcross] 
    hplus1 = np.interp(newtime,data[:,0],data[:,iplus]) 
    hcross1 = np.interp(newtime,data[:,0],data[:,icross]) 
    results.append([newtime,hplus1,hcross1])

    # Extract thetafin, thetadotsign from filename
    thetafin = float(wavefilenames[0].split("_")[3][3:])
    thetadotsign = wavefilenames[0].split("_")[-1][0]
    if thetadotsign == 'p':
      thetadot = 1
    elif thetadotsign == 'n':
      thetadot = -1

  return results, [float(thetafin),int(thetadot)]

def mulmlpnp(m,l,lp,npr,a,swsh_dir):
  '''
  Returns mu_{m,l,l',n'}(j), where \
  mu_{m,l,l',n'}(j) =              \
  Intergrate[ Y_{l,m} S^*_{l',m,n'}(j) , dOmega] \
  '''
  if m >= 0:
    filepath = swsh_dir + "/l%1dm%1dlp%1dnp%1d.dat" % (l,m,lp,npr)
  elif m < 0:
    filepath = swsh_dir + "/l%1dmm%1dlp%1dnp%1d.dat" % (l,abs(m),lp,npr)
  assert os.path.isfile(filepath), "Error: %s does not exist" % (filepath)
  data = np.loadtxt(filepath)
  if (len(data.shape) == 1) and (a == data[0]):
    return data[1],data[2]
  elif (len(data.shape) == 2):
    realmu = np.interp(a,data[:,0],data[:,1])
    imagmu = np.interp(a,data[:,0],data[:,2])
    return realmu,imagmu

def loadqnmtable(l,m,n,a,qnmdir):
  while(m<0):
    a = -a
    m = -m
  if (a >= 0):
    data = np.loadtxt(qnmdir+'/n'+repr(n+1)+'l'+repr(l)+'m'+repr(m)+'.dat',usecols=(0,1,2))
    omegaqnm = np.interp(a,data[:,0],data[:,1])
    inversetauqnm = np.interp(a,data[:,0],data[:,2])
    return omegaqnm + 1j*inversetauqnm
  elif (a < 0):
    data = np.loadtxt(qnmdir+'/n'+repr(n+1)+'l'+repr(l)+'mm'+repr(m)+'.dat',usecols=(0,1,2))
    omegaqnm = np.interp(-a,data[:,0],data[:,1])
    inversetauqnm = np.interp(-a,data[:,0],data[:,2])
    return omegaqnm + 1j*inversetauqnm

def getqnmfrequencies(modelist,a,qnmdir,overtone=0):
  '''Given list of modes to fit, returns the real frequencies and real damping times. Will return negative real frequency, if pair conjugate mode, assuming that first mode in input modelist is the multipole'''
  frequencylist = []
  (l,m,nover) = (modelist[0][0],modelist[0][1],overtone)
  if m == 0:
    existingmzeromodes = []
  for i in range(len(modelist)):
    (lfind,mfind) = (modelist[i][0],modelist[i][1])

    # If mode to find frequency for is listed with l > 0, then list conjugate modes in order
    # positive real frequency comes first, followed by negative real frequency
    if lfind > 0:

      if mfind == m and m != 0:
        frequencylist.append([float(np.real(loadqnmtable(lfind,mfind,nover,a,qnmdir))),np.absolute(np.imag(loadqnmtable(lfind,mfind,nover,a,qnmdir)))])
      elif mfind == -m and m != 0:
        frequencylist.append([-np.real(loadqnmtable(lfind,mfind,nover,a,qnmdir)),np.absolute(np.imag(loadqnmtable(lfind,mfind,nover,a,qnmdir)))])

      # for m = 0, the conjugate frequencies are listed after the positive real frequencies
      elif m==0 and (modelist[i] not in existingmzeromodes):
        frequencylist.append([float(np.real(loadqnmtable(lfind,mfind,nover,a,qnmdir))),np.absolute(np.imag(loadqnmtable(lfind,mfind,nover,a,qnmdir)))])
        existingmzeromodes.append(modelist[i])
      elif m==0 and (modelist[i] in existingmzeromodes):
        frequencylist.append([-np.real(loadqnmtable(lfind,mfind,nover,a,qnmdir)),np.absolute(np.imag(loadqnmtable(lfind,mfind,nover,a,qnmdir)))])

    # If l < 0, then conjugate frequency may be listed in frequency list before a positive real frequency
    elif lfind < 0:
      frequencylist.append([-np.real(loadqnmtable(abs(lfind),mfind,nover,a,qnmdir)),np.absolute(np.imag(loadqnmtable(abs(lfind),mfind,nover,a,qnmdir)))])

  return frequencylist
