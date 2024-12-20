#include <stdio.h>
#include <math.h>
#include "parm.h"

double swsh (double , double, double);
double fact (double);

double fact (double n){
double c;
double fac = 1.0; 
for (c = 1.0; c <= n; c++)
  {  fac = fac * c; }
return fac;
}

double swsh (double ell, double ml, double the){
double tmp, c1, c2, t, norm;

tmp = 0.0;

c1 = 0.0;
if ( (ml-2.0) > 0){ c1 = ml - 2.0; }

c2 = ell - 2.0;
if ( (ell-2.0) > (ell+ml) ){ c2 = ml + ell; }

for (t = c1; t <= c2; t++){
norm = pow(-1.0,t)*sqrt( fact(ell+ml)*fact(ell-ml)*fact(ell+2.0)*fact(ell-2.0) );
norm = norm / ( fact(ell+ml-t)*fact(ell-2.0-t)*fact(t)*fact(t+2.0-ml) );
tmp = tmp + norm*pow(cos(the/2.0),2.0*ell+ml-2.0-2.0*t)*pow(sin(the/2.0),2.0*t+2.0-ml);
}

tmp = sqrt( (2.0*ell+1.0)/4.0/Pi )*tmp;
return tmp;
}

double
project (double *qq, double l, double m)
{

  int i, j;
  double stheta, ctheta, dtheta, hlm, th;
  dtheta = Pi / M;
  hlm = 0.0;
  qq[M] = qq[M-1];

  for (i = 0; i < M / 8 ; i++)
    {

      th = 8.0 * dtheta * (double) i;
      stheta = sin (th);
      hlm += 989.0 * 4.0 * Pi * qq[8*i] * stheta * dtheta * swsh(l, m, th);

      th += dtheta;
      stheta = sin (th);
      hlm += 5888.0 * 4.0 * Pi * qq[8*i+1] * stheta * dtheta * swsh(l, m, th);

      th += dtheta;
      stheta = sin (th);
      hlm += (-928.0) * 4.0 * Pi * qq[8*i+2] * stheta * dtheta * swsh(l, m, th);

      th += dtheta;
      stheta = sin (th);
      hlm += 10496.0 * 4.0 * Pi * qq[8*i+3] * stheta * dtheta * swsh(l, m, th);

      th += dtheta;
      stheta = sin (th);
      hlm += (-4540.0) * 4.0 * Pi * qq[8*i+4] * stheta * dtheta * swsh(l, m, th);

      th += dtheta;
      stheta = sin (th);
      hlm +=  10496.0 * 4.0 * Pi * qq[8*i+5] * stheta * dtheta * swsh(l, m, th);

      th += dtheta;
      stheta = sin (th);
      hlm += (-928.0) * 4.0 * Pi * qq[8*i+6] * stheta * dtheta * swsh(l, m, th);

      th += dtheta;
      stheta = sin (th);
      hlm += 5888.0 * 4.0 * Pi * qq[8*i+7] * stheta * dtheta * swsh(l, m, th);

      th += dtheta;
      stheta = sin (th);
      hlm += 989.0 * 4.0 * Pi * qq[8*i+8] * stheta * dtheta * swsh(l, m, th);

    }

  hlm = 4.0 * hlm / 14175.0;
  return hlm;

}