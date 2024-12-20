#ifndef _Globals_H
#define _Globals_H

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <complex>

using std::cout;
using std::cerr;
using std::endl;
using std::ios;

//
// Edit these to use a different level of precision.
//
typedef double Real;
#define REAL_MANT_DIG DBL_MANT_DIG

typedef std::complex<Real> Complex;
const Complex II = Complex(0.,1.);

// useful utilities //
extern "C" int isnan(double);
#define Isnan(x) isnan((double)x)
#define CIsnan(x) (Isnan(real(x)) || Isnan(imag(x)))
#define Cmaxpart(x) (Fmax(Fabs(real(x)),Fabs(imag(x))))
#define CNAN Complex(NAN,NAN)

//
// Container for some useful global functions.
//
class Kerr {
 public:
  //
  // r_+ = M + \sqrt{M^2 + a^2}; M \equiv 1
  //
  static Real rplus(const Real a)
    {
      return(1. + sqrt((1. - a)*(1. + a)));
    };
  //
  // r_- = M - \sqrt{M^2 - a^2}
  //
  static Real rminus(const Real a)
    {
      return(1. - sqrt((1. - a)*(1. + a)));
    };
  //
  // r^* = r + 2 r_+/(r_+ - r_-) \ln{(r - r_+)/2M}
  //         - 2 r_-/(r_+ - r_-) \ln{(r - r_-)/2M}
  //
  static Real rstar(const Real r, const Real a)
    {
      const Real rm = rminus(a);
      const Real rp = rplus(a);
      
      return(r + ((2.*rp)/(rp - rm))*log((r - rp)/2.) -
	     ((2.*rm)/(rp - rm))*log((r - rm)/2.));
    };
  //
  // \Delta  = r^2 - 2 M r + a^2
  //
  static inline Real Delta(const Real r, const Real a)
    {
      return(r*r - 2.*r + a*a);
    }
  //
  // d\Delta/dr  = 2 r - 2 M
  //
  static Real dr_Delta(const Real r)
    {
      return(2.*(r - 1.));
    }
  //
  // All right, so this one's kind of silly ...
  //
  // d^2\Delta/dr^2  = 2
  //
  static Real ddr_Delta()
    {
      return(2.);
    }
  //
  // \Sigma = r^2 + a^2 \cos^2\theta
  //
  static Real Sigma(const Real r, const Real a, const Real z)
    {
      return(r*r + a*a*z);
    };
  //
  // d\Sigma/dr = 2 r
  //
  static Real dr_Sigma(const Real r)
    {
      return(2.*r);
    };
  //
  // All right, so this one's kind of silly too ...
  //
  // d^2\Sigma/dr^2 = 2
  //
  static Real ddr_Sigma()
    {
      return(2.);
    };

  static Real Eeqpro(const Real r, const Real a)
    {
      const Real v = 1./sqrt(r);

      const Real numer = 1. - v*v*(2. - a*v);
      const Real denom = 1. - v*v*(3. - 2.*a*v);

      return(numer/sqrt(denom));
    };

  static Real Eeqret(const Real r, const Real a)
    {
      const Real v = 1./sqrt(r);
      
      const Real numer = 1. - v*v*(2. + a*v);
      const Real denom = 1. - v*v*(3. + 2.*a*v);
      
      return(numer/sqrt(denom));
    };
  
  static Real Lzeqpro(const Real r, const Real a)
    {
      const Real v = 1./sqrt(r);
      
      const Real numer = 1. - a*v*v*v*(2. - a*v);
      const Real denom = 1. - v*v*(3. - 2.*a*v);
      
      return(r*v*numer/sqrt(denom));
    };

  static Real Lzeqret(const Real r, const Real a)
    {
      const Real v = 1./sqrt(r);
      
      const Real numer = 1. + a*v*v*v*(2. + a*v);
      const Real denom = 1. - v*v*(3. + 2.*a*v);
      
      return(-r*v*numer/sqrt(denom));
    };

  static Real Omega_phi_eqpro(const Real r, const Real a)
    {
      return(1./(sqrt(r*r*r) + a));
    };

  static Real Omega_phi_eqret(const Real r, const Real a)
    {
      return(-1./(sqrt(r*r*r) - a));
    };

  static Real isco_pro(const Real a)
    {
      const Real Z1 = 1. + (pow(1. + a, 1./3.) + pow(1. - a, 1./3.))*
	pow((1. + a)*(1. - a), 1./3.);
      const Real Z2 = sqrt(3.*a*a + Z1*Z1);

      return(3. + Z2 - sqrt((3. - Z1)*(3. + Z1 + 2.*Z2)));
    };

  static Real isco_ret(const Real a)
    {
      const Real Z1 = 1. + (pow(1. + a, 1./3.) + pow(1. - a, 1./3.))*
	pow((1. + a)*(1. - a), 1./3.);
      const Real Z2 = sqrt(3.*a*a + Z1*Z1);

      return(3. + Z2 + sqrt((3. - Z1)*(3. + Z1 + 2.*Z2)));
    };
};

#endif
