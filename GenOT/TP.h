#include "Globals.h"

//
// A class that holds everything that comes from the FD inspiral data.
// Need to create an adiabatic inspiral using Spectre.  This class
// will then read it in, store everything important, and scale the
// data appropriately given the specified mass ratio.
//
class Insp {
public:
  Insp(char *inspname, const Real a_in, const Real eta_in);

  Real r_insp(const Real lamb), thi_insp(const Real lamb);
  Real E_insp(const Real lamb), L_insp(const Real lamb), Q_insp(const Real lamb);
  Real dEdl_insp(const Real lamb), dLdl_insp(const Real lamb), dQdl_insp(const Real lamb);

  Real a, eta;
  Real l[52000], r[52000], thi[52000];
  Real E[52000], L[52000], Q[52000];
  Real dEdl[52000], dLdl[52000], dQdl[52000];
  Real r_pp[52000], thi_pp[52000];
  Real E_pp[52000], L_pp[52000], Q_pp[52000];
  Real dEdl_pp[52000], dLdl_pp[52000], dQdl_pp[52000];
  int Ninsp;
  Real r_lso, thi_lso, E_lso, L_lso, Q_lso, dEdl_lso, dLdl_lso, dQdl_lso;
};

//
// A class that holds and computes everything related to the
// generalized OT transition.
//
class Trans {
public:
  Trans(char *OTname, const Real a_in, const Real eta_in,
	const Real r_lso_in, const Real E_lso_in,
	const Real L_lso_in, const Real Q_lso_in,
	const Real dEdl_lso_in,
	const Real dLdl_lso_in, const Real dQdl_lso_in);
  //
  Real r_trans(const Real l);
  Real drdl_trans(const Real l);
  Real E_trans_OT(const Real l), E_trans_M2(const Real l), E_trans_M1(const Real l);
  Real L_trans_OT(const Real l), L_trans_M2(const Real l), L_trans_M1(const Real l);
  Real Q_trans_OT(const Real l), Q_trans_M2(const Real l), Q_trans_M1(const Real l);
  //
  void M1_Corr(const Real lTi,
	       const Real EIf, const Real dEIdlf,
	       const Real LIf, const Real dLIdlf,
	       const Real QIf, const Real dQIdlf);
  void M2_Corr(const Real lTi,
	       const Real EIf, const Real dEIdlf,
	       const Real LIf, const Real dLIdlf,
	       const Real QIf, const Real dQIdlf);
  //
  Real a, eta;
  Real r_lso, E_lso, L_lso, Q_lso;
  Real dEdl_lso, dLdl_lso, dQdl_lso;
  //
  Real CE, d2Edl2_M2, d2Edl2_M1, d3Edl3;
  Real CL, d2Ldl2_M2, d2Ldl2_M1, d3Ldl3;
  Real CQ, d2Qdl2_M2, d2Qdl2_M1, d3Qdl3;
  //
  Real L[44000], X[44000];
  Real X_pp[44000];
  int Ntrans;
  //
  Real A, B;
  Real ltrans_min, ltrans_max;
};

//
// A class that makes the final plunge at the end of the transition.
//
class Plunge {
public:
  Plunge(const Real ain, const Real Ein, const Real Lin, const Real Qin);

  Real a, r, drdl, E, L, Q;

  Real dvdl(const Real r);
  void TakeAStep(const Real r, const Real v, const Real dl,
		 Real & dr, Real & dv);
};
