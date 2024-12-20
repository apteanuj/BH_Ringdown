#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "parm.h"


/* Physical Parameters */

double mass, aa, ss, mm;

/* grid */

double x_h[N], x_c[N], r_h[N], r_c[N], theta[M];

/* fields */

double* qre;
double* qim;
double* pre;
double* pim;

/* miscellaneous */

double dx, dtheta, timer, dt, delta;

void grid (void);		/* set up the grid */
void body (double *, double *, double *, double *, double *, double *,
	   double *, double, double, double, double, double, double, double,
           int, int);

int
main (int argc, char **argv)
{				/* Main Code */

  int l, m, p, my_rank;

  p = 1;
  my_rank = 0;

  for (mm = -5; mm <= 5; mm++){

  if (my_rank==0) {
  printf ("\n");
  printf ("WELCOME TO TEUKOLSKY CODE!\n");
  printf ("\n");
  }

  mass = 1.0;
  aa = 1e-8;
  ss = -2.0;

  qre = (double*) malloc(N*M*sizeof(double));
  qim = (double*) malloc(N*M*sizeof(double));
  pre = (double*) malloc(N*M*sizeof(double));
  pim = (double*) malloc(N*M*sizeof(double));

  dx = (Xmax - Xmin) / N;	/* Radial grid increment */
  dtheta = Pi / M;		/* Angular grid increment */
//  dt = 0.32 * dx;
  dt = 0.002;

  grid ();			/* Setting up the grid */

  for (l = 0; l < N; l++)
    for (m = 0; m < M; m++)
      {
	qre[idx (m, l)] = 0.0;
	qim[idx (m, l)] = 0.0;
	pre[idx (m, l)] = 0.0;
	pim[idx (m, l)] = 0.0;
      }

  body (qre, qim, pre, pim, r_h, r_c, theta, dtheta, mass, aa, ss, mm, dx,
	dt, p, my_rank);

  if (my_rank==0) {
  printf ("All done.\n");
  printf ("Bye bye!\n");
  }


    free (qre);
    free (qim);
    free (pre);
    free (pim);

}

}

void
grid (void)
{

  int l, m;
  double tol, diff, drdx, rp, rm, r_old, r_new, x_o;
  double rpt, rin, Rrpt;
  double tx_c[N], tx_h[N];
  double Omega[N], Omega_h[N];

  for (l = 0; l < N; l++)
    {				/* Setting up the grid */

      x_c[l] = Xmin + (double) l * dx;
      x_h[l] = Xmin + ((double) l - 0.5) * dx;

    }


  for (m = 0; m < M; m++)
    {

      theta[m] = (double) m *dtheta;

    }

  rin = Xmin + (Xmax - Xmin) * 0.64;
  Rrpt = Xmax + 1e-6;

  for (l = 0; l < N; l++)
    {
  
     rpt = x_c[l];
     if (rpt < rin) {
     Omega[l] = 1.0;
     } else {
     Omega[l] = 1.0 - pow((rpt-rin)/(Rrpt-rin),4);     
     }

    }

  for (l = 0; l < N; l++)
    {
  
     rpt = x_h[l];
     if (rpt < rin) {
     Omega_h[l] = 1.0;
     } else {
     Omega_h[l] = 1.0 - pow((rpt-rin)/(Rrpt-rin),4);     
     }

    }


  for (l = 0; l < N; l++)
    {
         tx_c[l]  = x_c[l]/Omega[l];
         tx_h[l]  = x_h[l]/Omega_h[l];
    }   


  rp = mass + sqrt (mass * mass - aa * aa);
  rm = mass - sqrt (mass * mass - aa * aa);

  if (mass == 0.0)
    {

      for (l = 0; l < N; l++)
	{

	  r_h[l] = tx_h[l];
	  r_c[l] = tx_c[l];

	}
    }
  else
    {


      tol = 1e-12;

      r_old = tx_c[N - 1];
      for (l = N - 1; l >= 0; l--)
	{
	  diff = 2.0 * tol;

	  do
	    {

	      drdx =
		(r_old * r_old + aa * aa -
		 2.0 * mass * r_old) / (r_old * r_old + aa * aa);

	      if (mass == aa)
		{
		  x_o =
		    r_old + 2.0 * aa * log (r_old / aa - 1.0) -
		    2.0 * aa * aa / (r_old - aa);
		}
	      else
		{
		  if (aa == 0.0)
		    {
		      x_o =
			r_old + 2.0 * mass * log (r_old / (2.0 * mass) - 1.0);
		    }
		  else
		    {
		      x_o =
			r_old + (rp * rp + aa * aa) / (rp -
						       rm) * log (r_old / rp -
								  1.0) -
			(rm * rm + aa * aa) / (rp - rm) * log (r_old / rm -
							       1.0);
		    }
		}

	      r_new = r_old + (tx_c[l] - x_o) * drdx;
	      diff = fabs (r_new - r_old) / r_old;
	      r_old = r_new;

	    }
	  while (diff > tol);

	  r_c[l] = r_new;
	  r_old = r_new;

          //printf(" %f %f \n", x_c[l], r_old);

	}

      tol = 1e-12;

      r_old = tx_h[N - 1];
      for (l = N - 1; l >= 0; l--)
	{
	  diff = 2.0 * tol;

	  do
	    {

	      drdx =
		(r_old * r_old + aa * aa -
		 2.0 * mass * r_old) / (r_old * r_old + aa * aa);

	      if (mass == aa)
		{
		  x_o =
		    r_old + 2.0 * aa * log (r_old / aa - 1.0) -
		    2.0 * aa * aa / (r_old - aa);
		}
	      else
		{
		  if (aa == 0.0)
		    {
		      x_o =
			r_old + 2.0 * mass * log (r_old / (2.0 * mass) - 1.0);
		    }
		  else
		    {
		      x_o =
			r_old + (rp * rp + aa * aa) / (rp -
						       rm) * log (r_old / rp -
								  1.0) -
			(rm * rm + aa * aa) / (rp - rm) * log (r_old / rm -
							       1.0);
		    }
		}

	      r_new = r_old + (tx_h[l] - x_o) * drdx;
	      diff = fabs (r_new - r_old) / r_old;
	      r_old = r_new;

	    }
	  while (diff > tol);

	  r_h[l] = r_new;
	  r_old = r_new;

          //printf(" %f %f \n", x_h[l], r_old);

	}
    }
}