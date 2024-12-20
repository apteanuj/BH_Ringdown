#include <stdio.h>
#include <string.h>
#include "parm.h"

__global__ void kernel_init (double, double, double, double, double, double,
			     double, double *, double *, double *);
__global__ void kernel_average1 (double *, double *, double *, double *);
__global__ void kernel_rhs1 (double *, double *, double *, double *, double *,
			     double *);
__global__ void kernel_update1 ();
__global__ void kernel_boundary1 ();
__global__ void kernel_average2 ();
__global__ void kernel_rhs2 (double *, double *, double *, double *, double *,
			     double *);
__global__ void kernel_update2 (double *, double *, double *, double *);
__global__ void kernel_boundary2a (double *, double *, double *, double *);
__global__ void kernel_boundary2b (double *, double *, double *, double *, double *, double *);
__global__ void sourced (double *, double, double, double, double, double, 
			double, double, double, double, double, double, 
			double, double, double, double, double, double *);
__global__ void reset_sourced ();

double project (double *, double, double);

/* -------------------------------------------------- */

void
body (double *qre, double *qim, double *pre, double *pim, double *r_h,
      double *r_c, double *theta, double dtheta, double mass, double aa,
      double ss, double mm, double dx, double dt, int p, int my_rank)
{

  cudaSetDevice( my_rank%2 );

  double *d_r_h, *d_r_c, *d_theta, *d_qre_buff, *d_qim_buff;
  (cudaMalloc ((void **) &d_r_h, N * sizeof (double)));
  (cudaMalloc ((void **) &d_r_c, N * sizeof (double)));
  (cudaMalloc ((void **) &d_theta, M * sizeof (double)));
  (cudaMalloc ((void **) &d_qre_buff, M * sizeof (double)));
  (cudaMalloc ((void **) &d_qim_buff, M * sizeof (double)));

  cudaMemcpy (d_r_h, r_h, N * sizeof (double), cudaMemcpyHostToDevice);
  cudaMemcpy (d_r_c, r_c, N * sizeof (double), cudaMemcpyHostToDevice);
  cudaMemcpy (d_theta, theta, M * sizeof (double), cudaMemcpyHostToDevice);

  double *d_qre, *d_qim, *d_pre, *d_pim;
  (cudaMalloc ((void **) &d_qre, M * N * sizeof (double)));
  (cudaMalloc ((void **) &d_qim, M * N * sizeof (double)));
  (cudaMalloc ((void **) &d_pre, M * N * sizeof (double)));
  (cudaMalloc ((void **) &d_pim, M * N * sizeof (double)));

  cudaMemcpy (d_qre, qre, N * M * sizeof (double), cudaMemcpyHostToDevice);
  cudaMemcpy (d_qim, qim, N * M * sizeof (double), cudaMemcpyHostToDevice);
  cudaMemcpy (d_pre, pre, N * M * sizeof (double), cudaMemcpyHostToDevice);
  cudaMemcpy (d_pim, pim, N * M * sizeof (double), cudaMemcpyHostToDevice);

  dim3 threads (16, 8);
  dim3 grid (N / threads.x, M / threads.y);

  dim3 threads_source (8, 8);
  dim3 grid_source (PTSX / threads_source.x, PTSY / threads_source.y);

  dim3 threads_boundary (16);
  dim3 grid_boundary (N / threads_boundary.x);

  dim3 threads_boundary_m (8);
  dim3 grid_boundary_m (M / threads_boundary_m.x);

  kernel_init <<< grid, threads >>>
    (dtheta, mass, aa, ss, mm, dx, dt, d_theta, d_r_h, d_r_c);

  double timer = (T - Xmax - 200.0) / p * my_rank;
  double start = timer;
  double endtime = (T - Xmax - 200.0) / p * (my_rank + 1) + Xmax + 200.0;
  int nt = (int) ( (endtime - start) / dt);
  int k;
  char string[31], snum[1];

  sprintf (string, "%s", "hm0");
  if (mm < 0) 
  sprintf (string, "%s", "hmm");

  sprintf (snum, "%d", (int)abs(mm));
  strcat (string, snum);
  strcat (string, "_a0.9_thi060_thf150.3_n");
  strcat (string, ".dat");

  printf (" procs %d my_rank %d \n", p, my_rank);
  printf (" start %lf endtime %lf \n", timer, endtime);

  FILE *fpout;
  FILE *fpin;
  fpin = fopen("../Trajectories/a0.9_thi060_thf150.3_n.traj","r");                     
  fpout = fopen(string,"w");

  double ptime, rp, tp, tmp1, tmp2, tmp3, phip, E, lz, Q;
  double tmp11, tmp22, tmp33, d2rdt2, d2thdt2, d2phidt2;
  double psi2, dedt, qre_buff[M], qim_buff[M];
  double int_qre[M], int_qim[M], dint_qre[M], dint_qim[M];
  double rho2, rr, dvol, hplm[5], hclm[5];

  for (int q = 0; q < M; q++){    
   int_qre[q] = 0.0;
   int_qim[q] = 0.0;
   dint_qre[q] = 0.0;
   dint_qim[q] = 0.0;
  }

  tmp11 = 0.0;
  tmp22 = 0.0;
  tmp33 = 0.0;

  fscanf(fpin,"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", &ptime, &rp, &tp, &tmp1, &tmp2, &tmp3, &phip, &E, &lz, &Q); 

  do
    {
  fscanf(fpin,"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", &ptime, &rp, &tp, &tmp1, &tmp2, &tmp3, &phip, &E, &lz, &Q); 
    }
  while (ptime < timer);

  for (k = 0; k < nt; k++)
    {

      kernel_average1 <<< grid, threads >>> (d_qre, d_qim, d_pre, d_pim);
      cudaThreadSynchronize ();

      reset_sourced <<< grid, threads >>> ();
      cudaThreadSynchronize ();

      // read trajectory (note overlap with rhs kernel execution)
  fscanf(fpin,"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", &ptime, &rp, &tp, &tmp1, &tmp2, &tmp3, &phip, &E, &lz, &Q); 

      // adjust radial location to avoid crash hear horizon
      rp = rp + 1e-6;

      d2rdt2 = (tmp1 - tmp11) / dt * 2.0;
      d2thdt2 = (tmp2 - tmp22) / dt * 2.0;
      d2phidt2 = (tmp3 - tmp33) / dt * 2.0;

      sourced <<< grid_source, threads_source >>>
	(d_theta, start, timer, rp, phip, tp, E , lz , Q, 
	 tmp1, d2rdt2, 0.0, tmp2, d2thdt2, 0.0, tmp3, d2phidt2, d_r_h);
      cudaThreadSynchronize ();

      tmp11 = tmp1;
      tmp22 = tmp2;
      tmp33 = tmp3;

      kernel_rhs1 <<< grid, threads >>>
	(d_qre, d_qim, d_pre, d_pim, d_theta, d_r_h);
      cudaThreadSynchronize ();

      kernel_update1 <<< grid, threads >>> ();
      cudaThreadSynchronize ();

      kernel_boundary1 <<< grid_boundary, threads_boundary >>> ();
      cudaThreadSynchronize ();

      kernel_average2 <<< grid, threads >>> ();
      cudaThreadSynchronize ();

      reset_sourced <<< grid, threads >>> ();
      cudaThreadSynchronize ();

      // read trajectory (note overlap with rhs kernel execution)
  fscanf(fpin,"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", &ptime, &rp, &tp, &tmp1, &tmp2, &tmp3, &phip, &E, &lz, &Q); 

      // adjust radial location to avoid crash hear horizon
      rp = rp + 1e-6;

      d2rdt2 = (tmp1 - tmp11) / dt * 2.0;
      d2thdt2 = (tmp2 - tmp22) / dt * 2.0;
      d2phidt2 = (tmp3 - tmp33) / dt * 2.0;

      sourced <<< grid_source, threads_source >>>
	(d_theta, start, timer, rp, phip, tp, E , lz , Q, 
	 tmp1, d2rdt2, 0.0, tmp2, d2thdt2, 0.0, tmp3, d2phidt2, d_r_c);
      cudaThreadSynchronize ();

      tmp11 = tmp1;
      tmp22 = tmp2;
      tmp33 = tmp3;

      kernel_rhs2 <<< grid, threads >>>
	(d_qre, d_qim, d_pre, d_pim, d_theta, d_r_c);
      cudaThreadSynchronize ();

      kernel_update2 <<< grid, threads >>> (d_qre, d_qim, d_pre, d_pim);
      cudaThreadSynchronize ();

      kernel_boundary2a <<< grid_boundary, threads_boundary >>>
	(d_qre, d_qim, d_pre, d_pim);
      cudaThreadSynchronize ();

      kernel_boundary2b <<< grid_boundary_m, threads_boundary_m >>>
	(d_qre, d_qim, d_pre, d_pim, d_qre_buff, d_qim_buff);
      cudaThreadSynchronize ();


      cudaMemcpy (qre_buff, d_qre_buff, M * sizeof (double),
		      cudaMemcpyDeviceToHost);
      cudaMemcpy (qim_buff, d_qim_buff, M * sizeof (double),
		      cudaMemcpyDeviceToHost);

 
         dedt = 0.0;
         rr = r_c[N-1];

         for (int q = 0; q < M; q++){
            int_qre[q] = int_qre[q] + qre_buff[q] * dt; 
            int_qim[q] = int_qim[q] + qim_buff[q] * dt; 
            psi2 = int_qre[q]*int_qre[q]+int_qim[q]*int_qim[q];;
            rho2 = rr * rr + pow( aa * cos(theta[q]), 2);
            dvol = 0.5 * dtheta * sin(theta[q]) * rr * rr;
            dedt = dedt + psi2 * dvol * pow(rr,6) / pow(rho2,4);

            dint_qre[q] = dint_qre[q] + int_qre[q] * dt;
            dint_qim[q] = dint_qim[q] + int_qim[q] * dt;
         }


            hplm[0] = project (dint_qre, abs(mm), mm);
            hclm[0] = -project (dint_qim, abs(mm), mm);
            hplm[1] = project (dint_qre, 1+abs(mm), mm);
            hclm[1] = -project (dint_qim, 1+abs(mm), mm);
            hplm[2] = project (dint_qre, 2+abs(mm), mm);
            hclm[2] = -project (dint_qim, 2+abs(mm), mm);
            hplm[3] = project (dint_qre, 3+abs(mm), mm);
            hclm[3] = -project (dint_qim, 3+abs(mm), mm);
            hplm[4] = project (dint_qre, 4+abs(mm), mm);
            hclm[4] = -project (dint_qim, 4+abs(mm), mm);

      if (!(k % (int) Fr))
	{

         if ( ((k * dt) >= (Xmax + 200.0)) && (rp <= 12.0) )
	 fprintf(fpout,"%lf %g %g %g %g %g %g %g %g %g %g\n", timer-Xmax, hplm[0], hclm[0], hplm[1], hclm[1], hplm[2], hclm[2], hplm[3], hclm[3], hplm[4], hclm[4] );
         if (my_rank==0) 
	 printf("%lf %g \n", timer, dedt );
                  
	}

      timer += dt;
    }

    fclose (fpout);
    fclose (fpin);

    cudaFree (d_r_h);
    cudaFree (d_r_c);
    cudaFree (d_theta);
    cudaFree (d_qre_buff);
    cudaFree (d_qim_buff);
    cudaFree (d_qre);
    cudaFree (d_qim);
    cudaFree (d_pre);
    cudaFree (d_pim);

}

