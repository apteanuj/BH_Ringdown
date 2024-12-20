#include "Globals.h"
#include "TP.h"
#include "Tensors.h"

Insp::Insp(char *inspname, const Real a_in, const Real eta_in) : a(a_in), eta(eta_in)
{
  FILE *inspfile;
  char inspline[250];

  Real j1, j2;
  Real eflux, lflux, qflux;
  Real lhere;

  void spline(Real x[], Real y[], int n, Real yp1, Real ypn, Real y2[]);

  inspfile = fopen(inspname, "r");
  
  int i = 0;
  //
  // Read in and store data from the file.  We skip the time field
  // since we will use Mino time here, and we skip eccentricity since
  // all of our eccentricities are zero.
  //
  while (fgets(inspline, 250, inspfile) != NULL) {
    sscanf(inspline, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
	   &j1, &lhere, &r[i], &j2, &thi[i], &E[i], &L[i], &Q[i],
	   &eflux, &lflux, &qflux);
    dEdl[i] = -eflux; dLdl[i] = -lflux; dQdl[i] = -qflux;
    //
    // The adiabatic data has the mass ratio scaled out.  We would
    // like to put it back in.  Treat point zero as the origin (it
    // will typically have the value zero) and scale from there.
    //
    if (i == 0) l[i] = lhere;
    else l[i] = (lhere - l[0])/eta + l[0];
    i++;
  }
  Ninsp = i - 1;
  //
  // Store values at the LSO
  //
  r_lso = r[Ninsp];
  thi_lso = thi[Ninsp];
  E_lso = E[Ninsp];
  L_lso = L[Ninsp];
  Q_lso = Q[Ninsp];
  dEdl_lso = dEdl[Ninsp];
  dLdl_lso = dLdl[Ninsp];
  dQdl_lso = dQdl[Ninsp];
   //
  // Construct the spline data
  //
  spline(l, r, Ninsp, 1.e30, 1.e30, r_pp);
  spline(l, thi, Ninsp, 1.e30, 1.e30, thi_pp);
  spline(l, E, Ninsp, 1.e30, 1.e30, E_pp);
  spline(l, L, Ninsp, 1.e30, 1.e30, L_pp);
  spline(l, Q, Ninsp, 1.e30, 1.e30, Q_pp);
  spline(l, dEdl, Ninsp, 1.e30, 1.e30, dEdl_pp);
  spline(l, dLdl, Ninsp, 1.e30, 1.e30, dLdl_pp);
  spline(l, dQdl, Ninsp, 1.e30, 1.e30, dQdl_pp);
}

Real Insp::r_insp(const Real lamb)
{
  if (lamb < l[0] || lamb > l[Ninsp]) {
    cerr << "lambda out of range in Insp::r_insp" << endl;
    cerr << l[0] << " " << l[Ninsp] << " " << lamb << endl;
    exit(2);
  }
  void splint(Real xa[], Real ya[], Real y2a[], int n, Real x, Real *y);

  Real ans;
  splint(l, r, r_pp, Ninsp, lamb, &ans);

  return(ans);
}

Real Insp::thi_insp(const Real lamb)
{
  if (lamb < l[0] || lamb > l[Ninsp]) {
    cerr << "lambda out of range in Insp::thi_insp" << endl;
    exit(2);
  }
  void splint(Real xa[], Real ya[], Real y2a[], int n, Real x, Real *y);

  Real ans;
  splint(l, thi, thi_pp, Ninsp, lamb, &ans);

  return(ans);
}

Real Insp::E_insp(const Real lamb)
{
  if (lamb < l[0] || lamb > l[Ninsp]) {
    cerr << "lambda out of range in Insp::E_insp" << endl;
    exit(2);
  }
  void splint(Real xa[], Real ya[], Real y2a[], int n, Real x, Real *y);

  Real ans;
  splint(l, E, E_pp, Ninsp, lamb, &ans);

  return(ans);
}

Real Insp::L_insp(const Real lamb)
{
  if (lamb < l[0] || lamb > l[Ninsp]) {
    cerr << "lambda out of range in Insp::L_insp" << endl;
    exit(2);
  }
  void splint(Real xa[], Real ya[], Real y2a[], int n, Real x, Real *y);

  Real ans;
  splint(l, L, L_pp, Ninsp, lamb, &ans);

  return(ans);
}

Real Insp::Q_insp(const Real lamb)
{
  if (lamb < l[0] || lamb > l[Ninsp]) {
    cerr << "lambda out of range in Insp::Q_insp" << endl;
    exit(2);
  }
  void splint(Real xa[], Real ya[], Real y2a[], int n, Real x, Real *y);

  Real ans;
  splint(l, Q, Q_pp, Ninsp, lamb, &ans);

  return(ans);
}

Real Insp::dEdl_insp(const Real lamb)
{
  if (lamb < l[0] || lamb > l[Ninsp]) {
    cerr << "lambda out of range in Insp::dEdl_insp" << endl;
    exit(2);
  }
  void splint(Real xa[], Real ya[], Real y2a[], int n, Real x, Real *y);

  Real ans;
  splint(l, dEdl, dEdl_pp, Ninsp, lamb, &ans);

  return(ans);
}

Real Insp::dLdl_insp(const Real lamb)
{
  if (lamb < l[0] || lamb > l[Ninsp]) {
    cerr << "lambda out of range in Insp::dLdl_insp" << endl;
    exit(2);
  }
  void splint(Real xa[], Real ya[], Real y2a[], int n, Real x, Real *y);

  Real ans;
  splint(l, dLdl, dLdl_pp, Ninsp, lamb, &ans);

  return(ans);
}

Real Insp::dQdl_insp(const Real lamb)
{
  if (lamb < l[0] || lamb > l[Ninsp]) {
    cerr << "lambda out of range in Insp::dQdl_insp" << endl;
    exit(2);
  }
  void splint(Real xa[], Real ya[], Real y2a[], int n, Real x, Real *y);

  Real ans;
  splint(l, dQdl, dQdl_pp, Ninsp, lamb, &ans);

  return(ans);
}

////////////////////
////////////////////
////////////////////

Trans::Trans(char *OTname, const Real a_in, const Real eta_in,
	     const Real r_lso_in, const Real E_lso_in,
	     const Real L_lso_in, const Real Q_lso_in,
	     const Real dEdl_lso_in, const Real dLdl_lso_in,
	     const Real dQdl_lso_in) : a(a_in), eta(eta_in),
				       r_lso(r_lso_in), E_lso(E_lso_in), 
				       L_lso(L_lso_in), Q_lso(Q_lso_in), 
				       dEdl_lso(dEdl_lso_in), dLdl_lso(dLdl_lso_in),
				       dQdl_lso(dQdl_lso_in)
										 
{
  FILE *OTfile;
  char OTline[80];

  void spline(Real x[], Real y[], int n, Real yp1, Real ypn, Real y2[]);

  OTfile = fopen(OTname, "r");
  
  int i = 0;
  //
  // Read in and store data from the file.
  //
  while (fgets(OTline, 80, OTfile) != NULL) {
    sscanf(OTline, "%lf %lf", &L[i], &X[i]);
    i++;
  }
  Ntrans = i - 1;
  spline(L, X, Ntrans, 1.e30, 1.e30, X_pp);
  //
  // generalized OT parameters
  //
  const Real d3Rdr3 = 12. + 24.*(E_lso*E_lso - 1.)*r_lso;
  const Real d2RdEdr = -4.*a*L_lso + 8.*E_lso*r_lso*r_lso*r_lso + 4.*a*a*E_lso*(1. + r_lso);
  const Real d2RdLdr = -4.*(a*E_lso + L_lso*(r_lso - 1.));
  const Real d2RdQdr = 2.*(1 - r_lso);
  //
  A = -0.25*d3Rdr3;
  B = -0.5*(d2RdEdr*dEdl_lso + d2RdLdr*dLdl_lso + d2RdQdr*dQdl_lso);
  //
  ltrans_min = L[0]*pow(eta*A*B, -0.2);
  ltrans_max = L[Ntrans]*pow(eta*A*B, -0.2);
}

//
// Model 1: constant plus quadratic to make "constants" and fluxes
// continuous
//
void Trans::M1_Corr(const Real lTi, const Real EIf, const Real dEIdlf,
		    const Real LIf, const Real dLIdlf, const Real QIf, const Real dQIdlf)
{
  d2Edl2_M1 = eta*(dEIdlf - dEdl_lso)/lTi;
  CE = 0.5*(2.*(EIf - E_lso) - lTi*eta*(dEdl_lso + dEIdlf));
  d2Ldl2_M1 = eta*(dLIdlf - dLdl_lso)/lTi;
  CL = 0.5*(2.*(LIf - L_lso) - lTi*eta*(dLdl_lso + dLIdlf));
  d2Qdl2_M1 = eta*(dQIdlf - dQdl_lso)/lTi;
  CQ = 0.5*(2.*(QIf - Q_lso) - lTi*eta*(dQdl_lso + dQIdlf));
}

//
// Model 2: quadratic plus cubic to make "constants" and fluxes continuous
//
void Trans::M2_Corr(const Real lTi, const Real EIf, const Real dEIdlf,
		    const Real LIf, const Real dLIdlf, const Real QIf, const Real dQIdlf)
{
  d2Edl2_M2 = -2.*(3.*E_lso - 3.*EIf + (2.*dEdl_lso + dEIdlf)*eta*lTi)/(lTi*lTi);
  d3Edl3 = 6.*(2.*E_lso - 2.*EIf + (dEdl_lso + dEIdlf)*eta*lTi)/(lTi*lTi*lTi);
  d2Ldl2_M2 = -2.*(3.*L_lso - 3.*LIf + (2.*dLdl_lso + dLIdlf)*eta*lTi)/(lTi*lTi);
  d3Ldl3 = 6.*(2.*L_lso - 2.*LIf + (dLdl_lso + dLIdlf)*eta*lTi)/(lTi*lTi*lTi);
  d2Qdl2_M2 = -2.*(3.*Q_lso - 3.*QIf + (2.*dQdl_lso + dQIdlf)*eta*lTi)/(lTi*lTi);
  d3Qdl3 = 6.*(2.*Q_lso - 2.*QIf + (dQdl_lso + dQIdlf)*eta*lTi)/(lTi*lTi*lTi);
}

Real Trans::r_trans(const Real l)
{
  const Real Lh = l*pow(eta*A*B, 0.2);

  void splint(Real xa[], Real ya[], Real y2a[], int n, Real x, Real *y);

  Real Xh;
  splint(L, X, X_pp, Ntrans, Lh, &Xh);

  return(r_lso + Xh*pow(eta*B, 0.4)*pow(A, -0.6));
}

Real Trans::drdl_trans(const Real l)
{
  const Real Lh = l*pow(eta*A*B, 0.2);

  void dsplint(Real xa[], Real ya[], Real y2a[], int n, Real x, Real *y);

  Real dXdL;
  dsplint(L, X, X_pp, Ntrans, Lh, &dXdL);

  return(dXdL*pow(eta*B, 0.6)*pow(A, -0.4));
}

Real Trans::E_trans_M2(const Real l)
{
  return(E_lso + l*(eta*dEdl_lso + l*(0.5*d2Edl2_M2 + (l/6.)*d3Edl3)));
}

Real Trans::L_trans_M2(const Real l)
{
  return(L_lso + l*(eta*dLdl_lso + l*(0.5*d2Ldl2_M2 + (l/6.)*d3Ldl3)));
}

Real Trans::Q_trans_M2(const Real l)
{
  return(Q_lso + l*(eta*dQdl_lso + l*(0.5*d2Qdl2_M2 + (l/6.)*d3Qdl3)));
}

Real Trans::E_trans_M1(const Real l)
{
  return(CE + E_lso + l*(eta*dEdl_lso + l*0.5*d2Edl2_M1));
}

Real Trans::L_trans_M1(const Real l)
{
  return(CL + L_lso + l*(eta*dLdl_lso + l*0.5*d2Ldl2_M1));
}

Real Trans::Q_trans_M1(const Real l)
{
  return(CQ + Q_lso + l*(eta*dQdl_lso + l*0.5*d2Qdl2_M1));
}

////////////////////
////////////////////
////////////////////

Plunge::Plunge(const Real ain, const Real Ein, const Real Lin, const Real Qin) :
  a(ain), E(Ein), L(Lin), Q(Qin)
{
  //
  // No real body here .. just need to store the things that are
  // constants in the plunge!
  //
}

Real Plunge::dvdl(const Real r)
{
  const Real lzme = L - a*E;
  const Real r2pa2 = r*r + a*a;

  const Real term1 = -2.*r*(r2pa2 - 2.*r);
  const Real term2 = -2.*(r - 1.)*(r*r + Q + lzme*lzme);
  const Real term3 = 4.*E*r*(E*r2pa2 - a*L);

  return(0.5*(term1 + term2 + term3));
}

void Plunge::TakeAStep(const Real r, const Real v, const Real dl,
		       Real & dr, Real &dv)
{
  //
  // 4th order RK
  //
  Real dr1, dr2, dr3, dr4;
  Real dv1, dv2, dv3, dv4;
  //
  dr1 = v*dl;
  dv1 = dvdl(r)*dl;
  //
  dr2 = (v + 0.5*dv1)*dl;
  dv2 = dvdl(r + 0.5*dr1)*dl;
  //
  dr3 = (v + 0.5*dv2)*dl;
  dv3 = dvdl(r + 0.5*dr2)*dl;
  //
  dr4 = (v + dv3)*dl;
  dv4 = dvdl(r + dr3)*dl;
  //
  dr = dr1/6. + dr2/3. + dr3/3. + dr4/6.;
  dv = dv1/6. + dv2/3. + dv3/3. + dv4/6.;
}

////////////////////
////////////////////
////////////////////

int main(int argc, char **argv)
{
  ios::sync_with_stdio();
  Real thinc(const Real a, const Real eh, const Real lh, const Real qh);
  if (argc != 8) {
    cerr << "Arguments: 1. a  2. eta  3. adiabatic inspiral data filename" << endl;
    cerr << "           4. LTi, the Ori-Thorne time at which transition begins" << endl;
    cerr << "           5. LTf, the Ori-Thorne time at which transition ends" << endl;
    cerr << "           6. Value of r to start the inspiral" << endl;
    cerr << "           7. Model for evolution of integrals in transition:" << endl;
    cerr << "                1 == constant plus quadratic to enforce continuity" << endl;
    cerr << "                2 == quadratic plus cubic to enforce continuity" << endl;
    exit(1);
  }
  const Real a = (Real)atof(argv[1]);
  const Real eta = (Real)atof(argv[2]);
  const Real LTi = (Real)atof(argv[4]);
  if (LTi > -2.) {
    cerr << "Warning: Best choice of initial L appears to be near -2.7." << endl;
    cerr << "Continuing with LTi = " << LTi << endl;
  }
  const Real LTf = (Real)atof(argv[5]);
  const Real rstart = (Real)atof(argv[6]);
  const Real Model = atoi(argv[7]);
  if (!(Model == 1 || Model == 2)) {
    cerr << "Model type must be 1 or 2." << endl;
    exit(2);
  }
  //
  char inspname[80];
  sprintf(inspname, "%s", argv[3]);
  Insp insp(inspname, a, eta);
  //
  char OTname[80];
  sprintf(OTname, "OT_X_vs_L.dat");
  Trans trans(OTname, a, eta,
	      insp.r_lso, insp.E_lso, insp.L_lso, insp.Q_lso,
	      insp.dEdl_lso, insp.dLdl_lso, insp.dQdl_lso);
  //
  // Match the transition and the inspiral
  //
  // First, compute the radius at which the transition begins
  // according to our choice for Linitial.
  //
  const Real lTi = LTi*pow(eta*trans.A*trans.B, -0.2);
  const Real rTi = trans.r_trans(lTi);
  //
  // Find by bisection the value of l in the inspiral data at which
  // the inspiral ends.
  //
  Real lIlo = insp.l[0], lIhi = insp.l[insp.Ninsp];
  while (fabs(2.*(lIhi - lIlo)/(lIhi + lIlo)) > 1.e-14) {
    Real lImid = 0.5*(lIhi + lIlo);
    Real rImid = insp.r_insp(lImid);
    if (rImid < rTi)
      lIhi = lImid;
    else
      lIlo = lImid;
  }
  const Real lIf = 0.5*(lIlo + lIhi);
  //
  // Note that the span of Mino time in the inspiral data has a
  // different origin than the span in the transition (lI = 0 at start
  // of inspiral; lT = 0 in roughly the middle of the transition).
  // lInsp_offset will correct, setting everything to the same origin
  // as is used in the transition regime.
  //
  const Real lInsp_offset = lIf - lTi;
  //
  // Make sure that the end of transition is covered by our data.
  //
  if (LTf > trans.L[trans.Ntrans]) {
    cerr << "Requested lTf is beyond the range of a good OT solution." << endl;
    cerr << "Values near 2.5 - 3.0 work well." << endl;
    exit(3);
  }
  const Real lTf = LTf*pow(eta*trans.A*trans.B, -0.2);
  const Real rTf = trans.r_trans(lTf);
  if (Model == 1)
    //
    // Compute constant and quadratic corrections to the evolution of
    // the "constants" through the transition
    //
    trans.M1_Corr(lTi,
		  insp.E_insp(lTi + lInsp_offset), insp.dEdl_insp(lTi + lInsp_offset),
		  insp.L_insp(lTi + lInsp_offset), insp.dLdl_insp(lTi + lInsp_offset),
		  insp.Q_insp(lTi + lInsp_offset), insp.dQdl_insp(lTi + lInsp_offset));
  else
    //
    // Compute parameters for the quad plus cubic model.
    //
    trans.M2_Corr(lTi,
		  insp.E_insp(lTi + lInsp_offset), insp.dEdl_insp(lTi + lInsp_offset),
		  insp.L_insp(lTi + lInsp_offset), insp.dLdl_insp(lTi + lInsp_offset),
		  insp.Q_insp(lTi + lInsp_offset), insp.dQdl_insp(lTi + lInsp_offset));
  //
  // The plunge
  //
  Real E_Tf, L_Tf, Q_Tf;
  if (Model == 1) {
    E_Tf = trans.E_trans_M1(lTf);
    L_Tf = trans.L_trans_M1(lTf);
    Q_Tf = trans.Q_trans_M1(lTf);
  } else {
    E_Tf = trans.E_trans_M2(lTf);
    L_Tf = trans.L_trans_M2(lTf);
    Q_Tf = trans.Q_trans_M2(lTf);
  }
  Plunge plunge(a, E_Tf, L_Tf, Q_Tf);
  //
  // Now output.
  //
  Real l, rh, vh, thih, eh, lh, qh;
  Real dr, dv;
  //
  // Find the value of starting Mino time corresponding to our requested
  // inspiral start.
  //
  if (insp.r[0] < rstart) {
    cerr << "Requested start to inspiral is outside the range of adiabatic data."
	 << endl;
    exit(0);
  }
  //
  // Bisect
  lIlo = insp.l[0], lIhi = insp.l[insp.Ninsp];
  while (fabs(2.*(lIhi - lIlo)/(lIhi + lIlo)) > 1.e-14) {
    Real lImid = 0.5*(lIhi + lIlo);
    Real rImid = insp.r_insp(lImid);
    if (rImid < rstart)
      lIhi = lImid;
    else
      lIlo = lImid;
  }
  const Real lstart = 0.5*(lIlo + lIhi) - lInsp_offset;
  const Real lend = lTf + 100., dl = 0.1;
  for (l = lstart; l <= lend; l += dl) {
    if (l < lTi) {
      //
      // In inspiral
      rh = insp.r_insp(l + lInsp_offset);
      thih = insp.thi_insp(l + lInsp_offset);
      eh = insp.E_insp(l + lInsp_offset);
      lh = insp.L_insp(l + lInsp_offset);
      qh = insp.Q_insp(l + lInsp_offset);
    } else if (l > lTi && l < lTf){
      //
      // In transition
      rh = trans.r_trans(l);
      vh = trans.drdl_trans(l);
      if (Model == 1) {
	eh = trans.E_trans_M1(l);
	lh = trans.L_trans_M1(l);
	qh = trans.Q_trans_M1(l);
      } else {
	eh = trans.E_trans_M2(l);
	lh = trans.L_trans_M2(l);
	qh = trans.Q_trans_M2(l);
      }
      thih = thinc(a, eh, lh, qh);
    } else {
      plunge.TakeAStep(rh, vh, dl, dr, dv);
      rh += dr;
      vh += dv;
    }
    fprintf(stdout, "%.12lf %.12lf %.12lf %.12lf %.12lf %.12lf\n",
	    l, rh, thih, eh, lh, qh);
    if (rh < Kerr::rplus(a) - 0.5)
      break;
  }
}

void spline(Real x[], Real y[], int n, Real yp0, Real ypn, Real y2[])
{
  int i, k;
  Real p, qn, sig, un, *u;

  u = Tensor<Real>::vector(0, n);
  if (yp0 > 0.99e30)
    y2[0] = u[0] = 0.0;
  else {
    y2[0] = -0.5;
    u[1] = (3.0/(x[1] - x[0]))*((y[1] - y[0])/(x[1] - x[0]) - yp0);
  }
  for (i = 1; i <= n - 1; i++) {
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
  for (k = n - 1; k >= 0; k--)
    y2[k] = y2[k]*y2[k + 1] + u[k];
  Tensor<Real>::free_vector(u, 0, n);
}

void splint(Real xa[], Real ya[], Real y2a[], int n, Real x, Real *y)
{
  int klo, khi, k;
  Real h, b, a;
  
  klo = 0; khi = n;
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

Real thinc(const Real a, const Real eh, const Real lh, const Real qh)
{
  const Real A = a*a*(1. - eh*eh);
  const Real B = -lh*lh - qh - A;
  const Real qq = -0.5*(B - sqrt(B*B - 4.*A*qh));
  const Real zmin = sqrt(qh/qq);

  const Real thm = acos(zmin)*180./M_PI;
  if (lh > 0.)
    return(90. - thm);
  else
    return(90. + thm);
}
