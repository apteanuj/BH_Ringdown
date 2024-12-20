#include "Globals.h"
#include "TP.h"
#include "Tensors.h"

#define DEGTORAD (M_PI/180.)

int main(int argc, char **argv)
{
  void spline(Real x[], Real y[], int n, Real yp1, Real ypn, Real y2[]);
  void splint(Real xa[], Real ya[], Real y2a[], int n, Real x, Real *y);
  void dsplint(Real xa[], Real ya[], Real y2a[], int n, Real x, Real *y);
  //
  Real dtdl(const Real a, const Real r, const Real th,
	    const Real E, const Real L, const Real Q);
  Real dphidt(const Real a, const Real r, const Real th,
	      const Real E, const Real L, const Real Q);
  Real dchidl(const Real a, const Real chi, const Real thi,
	      const Real E, const Real L, const Real Q);

  Real lamb[60000], r[60000], thi[60000], E[60000], L[60000], Q[60000];
  Real rpp[60000], thipp[60000], Epp[60000], Lpp[60000], Qpp[60000];

  if (argc != 6) {
    cerr << "Args: 1. a  2. Mino time trajectory filename" << endl;
    cerr << "      3. Desired total duration of worldline" << endl;
    cerr << "      4. Initial value of angular phase chi (degrees)" << endl;
    cerr << "      5. Initial value of axial angle phi (degrees)" << endl;
    exit(0);
  }
  const Real a = (Real)atof(argv[1]);
  char trajname[80];
  sprintf(trajname, "%s", argv[2]);
  const Real duration = (Real)atof(argv[3]);
  const Real chi0 = (Real)atof(argv[4])*DEGTORAD;
  const Real phi0 = (Real)atof(argv[5])*DEGTORAD;

  const Real end_padding = 500.; // Padding to the end to make sure the waveform ends cleanly.

  FILE *trajfile;
  trajfile = fopen(trajname, "r");
  char trajline[250];

  int i = 0;

  while (fgets(trajline, 250, trajfile) != NULL) {
    sscanf(trajline, "%lf %lf %lf %lf %lf %lf", 
	   &lamb[i], &r[i], &thi[i], &E[i], &L[i], &Q[i]);
    i++;
  }
  const int thiname = (int)thi[0];
  //
  // CAUTION: thi is in degrees!  Also, cos(thm) = sin(thi).
  //
  const int Ntraj = i - 1;
  spline(lamb, r, Ntraj, 1.e30, 1.e30, rpp);
  spline(lamb, thi, Ntraj, 1.e30, 1.e30, thipp);
  spline(lamb, E, Ntraj, 1.e30, 1.e30, Epp);
  spline(lamb, L, Ntraj, 1.e30, 1.e30, Lpp);
  spline(lamb, Q, Ntraj, 1.e30, 1.e30, Qpp);
  //
  // Parameters at the current location along the worldline
  //
  Real rh, drdlh, thih, Eh, Lh, Qh;
  Real costhh, thh;
  //
  // Derivatives at the current location along the worldline
  //
  Real dtdlh, dchidlh, dphidth;
  Real dthdlh;
  //
  // Variables for RK method integration ... no need for "_1",
  // it's identical to the h values.
  //
  Real dl_1, dchi_1, dphi_1;
  Real dl_2, dchi_2, dphi_2;
  Real dl_3, dchi_3, dphi_3;
  Real dl_4, dchi_4, dphi_4;
  Real r_2, th_2, costh_2, thi_2, E_2, L_2, Q_2;
  Real r_3, th_3, costh_3, thi_3, E_3, L_3, Q_3;
  Real r_4, th_4, costh_4, thi_4, E_4, L_4, Q_4;
  Real dtdl_2, dchidl_2, dphidt_2;
  Real dtdl_3, dchidl_3, dphidt_3;
  Real dtdl_4, dchidl_4, dphidt_4;
  //
  // The variables we integrate
  //
  Real l, t, dt = 0.001, phi, chi;
  //
  // We scan through the worldline once to determine the final value of
  // theta and the sign of its time derivative.  We then use that data
  // to set the filename and write it out on the second pass.
  //
  int pass = 0;
  FILE *outfile;
  char outname[60];
  Real thf;
  int thdotpos;
  while (pass < 2) {
    if (pass == 1) {
      if (thiname < 10) {
	if (thf < 10) {
	  if (thdotpos)
	    sprintf(outname, "a%2.1lf_thi00%d_thf00%2.1lf_p.traj", a, thiname, thf);
	  else
	    sprintf(outname, "a%2.1lf_thi00%d_thf00%2.1lf_n.traj", a, thiname, thf);
	} else if (thf < 100) {
	  if (thdotpos)
	    sprintf(outname, "a%2.1lf_thi00%d_thf0%3.1lf_p.traj", a, thiname, thf);
	  else
	    sprintf(outname, "a%2.1lf_thi00%d_thf0%3.1lf_n.traj", a, thiname, thf);
	} else {
	  if (thdotpos)
	    sprintf(outname, "a%2.1lf_thi00%d_thf%4.1lf_p.traj", a, thiname, thf);
	  else
	    sprintf(outname, "a%2.1lf_thi00%d_thf%4.1lf_n.traj", a, thiname, thf);
	}
      } else if (thiname < 100) {
	if (thf < 10) {
	  if (thdotpos)
	    sprintf(outname, "a%2.1lf_thi0%d_thf00%2.1lf_p.traj", a, thiname, thf);
	  else
	    sprintf(outname, "a%2.1lf_thi0%d_thf00%2.1lf_n.traj", a, thiname, thf);
	} else if (thf < 100) {
	  if (thdotpos)
	    sprintf(outname, "a%2.1lf_thi0%d_thf0%3.1lf_p.traj", a, thiname, thf);
	  else
	    sprintf(outname, "a%2.1lf_thi0%d_thf0%3.1lf_n.traj", a, thiname, thf);
	} else {
	  if (thdotpos)
	    sprintf(outname, "a%2.1lf_thi0%d_thf%4.1lf_p.traj", a, thiname, thf);
	  else
	    sprintf(outname, "a%2.1lf_thi0%d_thf%4.1lf_n.traj", a, thiname, thf);
	}
      } else {
	if (thf < 10) {
	  if (thdotpos)
	    sprintf(outname, "a%2.1lf_thi%d_thf00%2.1lf_p.traj", a, thiname, thf);
	  else
	    sprintf(outname, "a%2.1lf_thi%d_thf00%2.1lf_n.traj", a, thiname, thf);
	} else if (thf < 100) {
	  if (thdotpos)
	    sprintf(outname, "a%2.1lf_thi%d_thf0%3.1lf_p.traj", a, thiname, thf);
	  else
	    sprintf(outname, "a%2.1lf_thi%d_thf0%3.1lf_n.traj", a, thiname, thf);
	} else {
	  if (thdotpos)
	    sprintf(outname, "a%2.1lf_thi%d_thf%4.1lf_p.traj", a, thiname, thf);
	  else
	    sprintf(outname, "a%2.1lf_thi%d_thf%4.1lf_n.traj", a, thiname, thf);
	}
      }
      outfile = fopen(outname, "w");
    }
    //
    // Set initial conditions ... begin at a radius very close to the horizon.
    //
    Real r0 = 1. + sqrt(1. - a*a) + 0.01;
    //
    // Bisect to find the value of l corresponding to this radius
    //
    Real lhi = lamb[Ntraj], llo = lamb[0];
    Real rtest;
    while (lhi - llo > 1.e-8) {
      Real lmid = 0.5*(lhi + llo);
      splint(lamb, r, rpp, Ntraj, lmid, &rtest);
      if (rtest > r0) {
	// too far out: replace llo
	llo = lmid;
      } else {
	// too far in: replace lhi
	lhi = lmid;
      }
    }
    Real l0 = 0.5*(lhi + llo);
    //
    // Integrate out for an interval of time equal to duration - end_padding.
    // Don't bother to get the theta motion right for this estimate.
    //
    Real teul = 0., dt = 0.01;
    rh = r0;
    l = l0;
    while (teul < duration - end_padding) {
      splint(lamb, r, rpp, Ntraj, l, &rh);
      splint(lamb, E, Epp, Ntraj, l, &Eh);
      splint(lamb, L, Lpp, Ntraj, l, &Lh);
      splint(lamb, Q, Qpp, Ntraj, l, &Qh);
      dtdlh = dtdl(a, rh, 0.5*M_PI, Eh, Lh, Qh);
      teul += dt;
      l -= dt/dtdlh;
    }
    const Real lstart = l;
    ////////
    // END EXPERIMENTAL CODE
    ////////
    l = lstart;
    phi = phi0;
    chi = chi0;
    for (t = 0; t <= duration; t += dt) {
      //
      // Get all the data at the present value of l ("h"ere)
      //
      splint(lamb, r, rpp, Ntraj, l, &rh);
      dsplint(lamb, r, rpp, Ntraj, l, &drdlh);
      splint(lamb, thi, thipp, Ntraj, l, &thih);
      costhh = sin(thih*DEGTORAD)*cos(chi);
      thh = acos(costhh);
      splint(lamb, E, Epp, Ntraj, l, &Eh);
      splint(lamb, L, Lpp, Ntraj, l, &Lh);
      splint(lamb, Q, Qpp, Ntraj, l, &Qh);
      dtdlh = dtdl(a, rh, thh, Eh, Lh, Qh);
      dchidlh = dchidl(a, chi, thih, Eh, Lh, Qh);
      dphidth = dphidt(a, rh, thh, Eh, Lh, Qh);
      dthdlh = sin(thih*DEGTORAD)*sin(chi)*dchidlh/sqrt(1. - costhh*costhh);
      if (pass == 1)
	fprintf(outfile,
		"%.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e\n",
		t,
		rh, thh,
		drdlh/dtdlh, dthdlh/dtdlh,
		dphidth, phi,
		Eh, Lh, Qh);
      //
      // Step 1 of RK step.
      //
      dl_1 = dt/dtdlh;
      dchi_1 = dt*dchidlh/dtdlh;
      dphi_1 = dt*dphidth;
      //
      // Step 2: Data and derivatives at the first estimated mid point.
      //
      splint(lamb, r, rpp, Ntraj, l + 0.5*dl_1, &r_2);
      splint(lamb, thi, thipp, Ntraj, l + 0.5*dl_1, &thi_2);
      splint(lamb, E, Epp, Ntraj, l + 0.5*dl_1, &E_2);
      splint(lamb, L, Lpp, Ntraj, l + 0.5*dl_1, &L_2);
      splint(lamb, Q, Qpp, Ntraj, l + 0.5*dl_1, &Q_2);
      costh_2 = sin(thi_2*DEGTORAD)*cos(chi + 0.5*dchi_1);
      th_2 = acos(costh_2);
      dtdl_2 = dtdl(a, r_2, th_2, E_2, L_2, Q_2);
      dchidl_2 = dchidl(a, chi + 0.5*dchi_1, thi_2, E_2, L_2, Q_2);
      dphidt_2 = dphidt(a, r_2, th_2, E_2, L_2, Q_2);
      //
      dl_2 = dt/dtdl_2;
      dchi_2 = dt*dchidl_2/dtdl_2;
      dphi_2 = dt*dphidt_2;
      //
      // Step 3: Data and derivatives at the second estimated mid point.
      //
      splint(lamb, r, rpp, Ntraj, l + 0.5*dl_2, &r_3);
      splint(lamb, thi, thipp, Ntraj, l + 0.5*dl_2, &thi_3);
      splint(lamb, E, Epp, Ntraj, l + 0.5*dl_2, &E_3);
      splint(lamb, L, Lpp, Ntraj, l + 0.5*dl_2, &L_3);
      splint(lamb, Q, Qpp, Ntraj, l + 0.5*dl_2, &Q_3);
      costh_3 = sin(thi_3*DEGTORAD)*cos(chi + 0.5*dchi_2);
      th_3 = acos(costh_3);
      dtdl_3 = dtdl(a, r_3, th_3, E_3, L_3, Q_3);
      dchidl_3 = dchidl(a, chi + 0.5*dchi_2, thi_3, E_3, L_3, Q_3);
      dphidt_3 = dphidt(a, r_3, th_3, E_3, L_3, Q_3);
      //
      dl_3 = dt/dtdl_3;
      dchi_3 = dt*dchidl_3/dtdl_3;
      dphi_3 = dt*dphidt_3;
      //
      // Step 4: Data and derivatives at third estimated end point.
      //
      splint(lamb, r, rpp, Ntraj, l + dl_3, &r_4);
      splint(lamb, thi, thipp, Ntraj, l + dl_3, &thi_4);
      splint(lamb, E, Epp, Ntraj, l + dl_3, &E_4);
      splint(lamb, L, Lpp, Ntraj, l + dl_3, &L_4);
      splint(lamb, Q, Qpp, Ntraj, l + dl_3, &Q_4);
      costh_4 = sin(thi_4*DEGTORAD)*cos(chi + dchi_3);
      th_4 = acos(costh_4);
      dtdl_4 = dtdl(a, r_4, th_4, E_4, L_4, Q_4);
      dchidl_4 = dchidl(a, chi + dchi_3, thi_4, E_4, L_4, Q_4);
      dphidt_4 = dphidt(a, r_4, th_4, E_4, L_4, Q_4);
      //
      dl_4 = dt/dtdl_4;
      dchi_4 = dt*dchidl_4/dtdl_4;
      dphi_4 = dt*dphidt_4;
      //
      // Add the RK steps together
      //
      l += dl_1/6. + dl_2/3. + dl_3/3. + dl_4/6.;
      chi += dchi_1/6. + dchi_2/3. + dchi_3/3. + dchi_4/6.;
      phi += dphi_1/6. + dphi_2/3. + dphi_3/3. + dphi_4/6.;
    }
    if (pass == 0) {
      thf = thh/DEGTORAD;
      if (dthdlh > 0.) thdotpos = 1;
      else thdotpos = 0;
    }
    pass++;
  }
//   if (dthdlh > 0.)
//     fprintf(outfile, "# thf = %.1lf pos\n", thh/DEGTORAD);
//   else
//     fprintf(outfile, "# thf = %.1lf neg\n", thh/DEGTORAD);
}

void spline(Real x[], Real y[], int n, Real yp1, Real ypn, Real y2[])
{
  int i, k;
  Real p, qn, sig, un, *u;

  u = Tensor<Real>::vector(1, n - 1);
  if (yp1 > 0.99e30)
    y2[1] = u[1] = 0.0;
  else {
    y2[1] = -0.5;
    u[1] = (3.0/(x[2] - x[1]))*((y[2] - y[1])/(x[2] - x[1]) - yp1);
  }
  for (i = 2; i <= n - 1; i++) {
    sig = (x[i] - x[i - 1])/(x[i + 1] - x[i - 1]);
    p = sig*y2[i - 1] + 2.0;
    y2[i] = (sig - 1.0)/p;
    u[i] = (y[i + 1] - y[i])/(x[i + 1] - x[i]) -
      (y[i] - y[i - 1])/(x[i] - x[i - 1]);
    u[i] = (6.0*u[i]/(x[i + 1] - x[i - 1]) - sig*u[i - 1])/p;
  }
  if (ypn > 0.99e30)
    qn = un = 0.0;
  else {
    qn = 0.5;
    un = (3.0/(x[n] - x[n - 1]))*(ypn - (y[n] - y[n - 1])/(x[n] - x[n - 1]));
  }
  y2[n]=(un - qn*u[n - 1])/(qn*y2[n - 1] + 1.0);
  for (k = n - 1; k >= 1; k--)
    y2[k] = y2[k]*y2[k + 1] + u[k];
  Tensor<Real>::free_vector(u, 1, n - 1);
}

void splint(Real xa[], Real ya[], Real y2a[], int n, Real x, Real *y)
{
  int klo, khi, k;
  Real h, b, a;
  
  klo = 1; khi = n;
  while (khi - klo > 1) {
    k = (khi + klo) >> 1;
    if (xa[k] > x) khi = k;
    else klo = k;
  }
  h = xa[khi] - xa[klo];
  if (h == 0.0) {
    cerr << "Bad xa input to routine splint" << endl;
    exit(0);
  }
  a = (xa[khi] - x)/h;
  b = (x - xa[klo])/h;
  *y = a*ya[klo] + b*ya[khi] +
    ((a*a*a - a)*y2a[klo] + (b*b*b - b)*y2a[khi])*(h*h)/6.0;
}

//
// Spline interpolation for the derivative.
//
void dsplint(Real xa[], Real ya[], Real y2a[], int n, Real x, Real *y)
{
  int klo, khi, k;
  Real h, b, a, db, da;
  
  klo = 1; khi = n;
  while (khi - klo > 1) {
    k = (khi + klo) >> 1;
    if (xa[k] > x) khi = k;
    else klo = k;
  }
  h = xa[khi] - xa[klo];
  if (h == 0.0) {
    cerr << "Bad xa input to routine splint" << endl;
    exit(0);
  }
  a = (xa[khi] - x)/h; da = -1./h;
  b = (x - xa[klo])/h; db = 1./h;
  *y = da*ya[klo] + db*ya[khi] +
    ((3.*da*a*a - da)*y2a[klo] + (3.*db*b*b - db)*y2a[khi])*(h*h)/6.0;
}

Real dtdl(const Real a, const Real r, const Real th,
	  const Real E, const Real L, const Real Q)
{
  Real r2pa2 = r*r + a*a;
  Real sth = sin(th);
  Real delt = Kerr::Delta(r, a);

  Real term1 = E*(r2pa2*r2pa2/delt - a*a*sth*sth);
  Real term2 = -2.*a*r*L/delt;

  return(term1 + term2);
}

Real dphidt(const Real a, const Real r, const Real th,
	    const Real E, const Real L, const Real Q)
{
  Real st = sin(th);
  Real delt = Kerr::Delta(r, a);

  Real numer = a*(2.*E*r - a*L) + L*delt/(st*st);
  Real denom = a*a*a*a*E - 2.*a*L*r + 2.*a*a*E*r*r + E*r*r*r*r
    - a*a*E*delt*st*st;

  return(numer/denom);
}

Real dchidl(const Real a, const Real chi, const Real thi,
	    const Real E, const Real L, const Real Q)
{
  const Real A = a*a*(1. - E*E);
  const Real beta = A;
  const Real B = -L*L - Q - A;
  const Real zmin = sin(thi*DEGTORAD);
  const Real zsq = zmin*zmin*cos(chi)*cos(chi);

  const Real betazplussq = 0.5*(-B + sqrt(B*B - 4.*A*Q));

  return(sqrt(betazplussq - beta*zsq));
}
