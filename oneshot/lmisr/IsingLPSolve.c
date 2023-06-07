/* This routine is a modified version of the "original" 
 * routine that was used to solve the LP:
 *
 *     min d'z
 *     s.t.
 *     Bz >= 1
 *
 * where B = (A(\alpha)_inc - A(\alpha)_cor) is the
 * constraint matrix for a fixed ancilla configuration \alpha,
 * d' is the sum of the rows of B divided by the total
 * number of possible "incorrct" output/ancilliary state 
 * configurations, and 1 is the appropriately sized column 
 * vector of all ones. This LP has a solution iff \alpha 
 * is a valid ancilla arrangement.
 * 
 * In view of the "original" routine, the above LP is 
 * reformulated as 
 * 
 *     max b' \lam
 *     s.t.
 *     A' \lam + s = c, s >= 0,
 *
 * where A = -B', b = -d, and c = -1, so that we can 
 * think of this LP as the dual of the primal LP in
 * standard form from (14.1) in Nocedal and Wright's
 * Numerical Optimization textbook; treating this LP as 
 * the dual problem rather than the primal problem reduces
 * the "original" routine's memory requirement. 
 * 
 * The goal of this modified routine is to solve the related 
 * problem:
 *
 *     min 1'\rho
 *     s.t.
 *     B \phi + \rho >= 1, \rho >= 0.
 *
 * This problem has a solution, regardless if the ancilla 
 * configuration \alpha is valid or not.  In fact, if the
 * ancilla configuration is valid, the objective value of 
 * the LP at the optimal solution will be zero, whereas 
 * a positive value at the optimal solution indicates an 
 * invalid ancilla configuration.  Moreoever, the smaller 
 * the objective value at an optimal solution, the "closer"
 * (in some sense) the ancilla configuration is to being valid.
 *
 * Similar to the original LP, we treat the LP for the 
 * modified routine as the dual problem to reduce this routine's
 * memory requirement.  Moreover, we want to solve:
 *
 *     max [0'; -1']'[\phi; \rho]
 *     s.t.
 *     [-B  -I][\phi]  +  [s1]  = [-1]
 *     [ 0  -I][\rho]     [s2]    [ 0],   s1, s2 >= 0
 *
 * which is also of the form
 *
 *     max b' \lam
 *     s.t.
 *     A' \lam + s = c, s >= 0.
 * 
 * We solve this modified problem using the Mehrotra predictor-corrector
 * method (cf. Algorithmm 14.3 Nocedal & Wright) in this routine.  Note 
 * that like the original routine, we never explicitly form the constraint 
 * matrix, but instead use various subroutines to quickly compute products
 * of the constraint matrix (and/or other related matrices) with vectors 
 * by exploiting the structure of this matrix.  Explicitly forming and 
 * computing products with the contraint matrix would require excessive 
 * amounts of memory and greatly increase computational demands.   
 *
 * INPUTS: N1 = argv[1] = no. bits used to represent the 1st input factor
 *         N2 = argv[2] = no. bits used to represent the 2nd input factor
 *	   na = argv[3] = no. of ancillary bits
 *              argv[4] = name of text file containing ancilla values.  
 *              	  In this file, each column of values represents the
 *              	  state of a given ancillary bit for each different product/
 *              	  set of inputs.  Any given column lists these values 
 *              	  in the order such that the values for each product along 
 *              	  the columns in the 2^N1 x 2^N2 multiplication table are 
 *              	  contiguous in the column in this file. 
 *
 *
 * OUTPUTS: This routine prints out the objective value at the 
 * optimal soltuion, along with the fraction of contraints satisfied
 * at the optimal solution to stdout.       
 * */     	   
#include <cblas.h>
#include <string.h>
#include <time.h>
#include "util.h"

void free_ptr(double *ptr){

    free(ptr);

}

double* lmisr(int N1, int N2, int na, int N, int M1, bool *aux){

    int i, j, k, l, q, p;
    int M = N+na;
    int K = M*(M+1)/2;
    int L = (int)round(pow(2.0,N+M));
    int K2 = (int)(pow(2,M));
    int P = (int)(pow(2,na));
    int R1 = (int)(pow(2,N1));
    int R2 = (int)(pow(2,N2));

    double tol  = 1e-12;
    double tol2 = 1e-8;
    double tol3 = 1e-4;
    double eta = 0.9;
    int maxit = 200;

    double normrc, normrb, my_error, my_error0;
    double normrc0, normrb0, ntemp, rel_err;
    double alpha_ap, alpha_ad;
    double mu, mu_aff, sigma;
    double rel_err_old = 2.0;
    double fval = 0.0;
    double fmin = 1e100;


    openblas_set_num_threads(1);

    bool **a = (bool**)malloc(sizeof(bool*)*M1);
    for(i = 0; i < M1; i++){
        a[i] = (bool*)malloc(sizeof(bool)*na);
    }

    for(i = 0; i < M1; i++){
        for(j = 0; j < na; j++){
            a[i][j] = aux[i+j*M1] ? 1 : 0;
        }
    }


    bool **compute_me1 = (bool **)malloc(sizeof(bool*)*M);
    for(j = 0; j < M; j++){
        compute_me1[j] = (bool*)malloc(sizeof(bool)*((int)round(pow(2.0,j))));
    }
    getSumDiffInds(N1, N2, na, 4, compute_me1);

    bool **compute_me2 = (bool **)malloc(sizeof(bool*)*M);
    for(j = 0; j < M; j++){
        compute_me2[j] = (bool*)malloc(sizeof(bool)*((int)round(pow(2.0,j))));
    }
    getSumDiffInds(N1, N2, na, 2, compute_me2);

		/*
				* Andrew: I think that this variable "sz" may be the number of columns in the original constraint matrix. I can calculate the two and see whether they are equal. The original consraint matrix has a number of columns equal to G + Gc2, and Gc2 = G!/(2 * (G-2)!) = (1/2) (G)(G-1). I can then convert this into N and na, since G = 2N + na. So this number of columns is 2N + na + (1/2) (2N + na)(2N + na - 1) = 2N + na + (1/2)(4N^2 + 2Nna - 2N + 2Nna + na^2 - na 
				* = 2N^2 + 2Nna - N + (1/2)(na)(na-1) + 2N + na
				* = 2N^2 + 2Nna + N + (1/2)(na)(na+1)
				*
				* Now note that "sz" = (1/2) (3N^2 + N) + (2N)na + sum(1 .. na) 
				* = (1/2) (3N^2 + N) + 2Nna + (1/2)(na)(na+1) 
				* = (3/2) N^2 + N/2 + 2Nna + (1/2)(na^2 + na) 
				* = 2N^2 + 2Nna + N + (1/2)(na)(na+1) - (1/2)(N^2 + N)
				*
				* This is almost the same. This suggests that Teresa has already thought of the optimization that we do not need to consider the coefficients of monomials only containing input spins. There are exactly N linear terms and Nc2 quadratics, so these terms amount to N + (1/2)(N)(N-1) = N + (1/2)(N^2 - N) = (1/2)(N^2 + N), which exactly the extra factor.
				*
				* Therefore, sz is in fact the number of columns in the original constraint matrix with the optimization that we do not look for coefficients of input-only term
				* This is almost the same. This suggests that Teresa has already thought of the optimization that we do not need to consider the coefficients of monomials only containing input spins. There are exactly N linear terms and Nc2 quadratics, so these terms amount to N + (1/2)(N)(N-1) = N + (1/2)(N^2 - N) = (1/2)(N^2 + N), which exactly the extra factor.
				*
				* Therefore, sz is in fact the number of columns in the original constraint matrix with the optimization that we do not look for coefficients of input-only terms.
				*
				*/
    int sz = (N*(3*N+1))/2;
    for(i = 0; i < na; i++){
	sz += 2*N+i+1;
    }    

    bool **CC = (bool**)malloc(sizeof(bool*)*M1);
    for(i = 0; i < M1; i++){
	CC[i] = (bool*)malloc(sizeof(bool)*sz);
    }

    double **Grm = (double**)malloc(sizeof(double*)*sz);
    for(i = 0; i < sz; i++){
	Grm[i] = (double*)malloc(sizeof(double)*sz);
    }

    double *rho  = malloc(sizeof(double)*M1);
    double *x1   = malloc(sizeof(double)*L);
    double *x2   = malloc(sizeof(double)*L);
    double *s1   = malloc(sizeof(double)*L);
    double *s2   = malloc(sizeof(double)*L);
    double *rc1  = malloc(sizeof(double)*L);
    double *rc2  = malloc(sizeof(double)*L);
    double *dx1  = malloc(sizeof(double)*L);
    double *dx2  = malloc(sizeof(double)*L);
    double *ds1  = malloc(sizeof(double)*L);
    double *ds2  = malloc(sizeof(double)*L);
    double *xs1  = malloc(sizeof(double)*L);
    double *xs2  = malloc(sizeof(double)*L);
    double *rxs1 = malloc(sizeof(double)*L);
    double *rxs2 = malloc(sizeof(double)*L);
    double *rxst = malloc(sizeof(double)*L);
    double *rb1  = malloc(sizeof(double)*sz);
    double *rb2  = malloc(sizeof(double)*L);
    double *lam1 = malloc(sizeof(double)*sz);
    double *lam2 = malloc(sizeof(double)*L);
    double *dlam1 = malloc(sizeof(double)*sz);
    double *dlam2 = malloc(sizeof(double)*L);
    double *gr  = malloc(sizeof(double)*sz*sz);
    int    *ipiv= malloc(sizeof(int)*sz); 
    double normc_old, normb_old;

    getCorrectColumns(N1, N2, na, a, CC);

    getInitialGuess(N1, N2, na, a, x1, x2, lam1, lam2, s1, s2, CC, Grm,\
		   compute_me1, compute_me2, rc1, rc2, rb1, rb2, xs1, xs2); 


    for(k = 0; k < maxit; k++){

	// Begin computing predictor step
		/*
						* r_c = A^T lambda + s - c
						*/
	computeATxReward(lam1, lam2, N1, N2, na, a, CC, rc1, rc2);
	for(i = 0; i < L; i++){
	    rc1[i] += s1[i] + 1.0;
	}
	for(i = 0; i < L; i++){
	    rc2[i] += s2[i];
	}

				// Remove the constraints on the correct outputs by setting the c values to 0
				// (This is a little opaque, but that's what it's doing)
	q = 0;
	for(j = 0; j < R2; j++){
	    for(i = 0; i < R1; i++){
		p = i*j;
		for(l = 0; l < P; l++){
		    rc1[K2*q+P*p+l] -= 1.0;
		}
		q++;
	    }
	}


		// r_b = Ax - b
	computeAxReward(x1, x2, N1, N2, na, a, CC, rb1, rb2, compute_me2);
	for(i = 0; i < L; i++){
	    rb2[i] += 1.0;
	}

				// Similarly, remove the b coefficients on the correct output lines, which means removing them from the objective.
	q = 0;
        for(j = 0; j < R2; j++){
	    for(i = 0; i < R1; i++){
        	p = i*j;
        	for(l = 0; l < P; l++){
           	    rb2[K2*q+P*p+l] -= 1.0;
        	}
        	q++;
            }
	}

        ntemp   = normTwo(rc1, L);
	normrc  = ntemp*ntemp;
        ntemp   = normTwo(rc2, L);
	normrc += ntemp*ntemp;
	ntemp   = normTwo(rb1, sz);
        normrb  = ntemp*ntemp;
        ntemp   = normTwo(rb2, L);
	normrb += ntemp*ntemp;
	my_error= sqrt(normrc + normrb);

	if(k == 0){
	    my_error0 = my_error;
	    normrb0 = normrb;
	    normrc0 = normrc;
	}
	rel_err = my_error/my_error0;	

	// This is for output that will be printed when done.
				// Since we have figured out that rho = lambda_2, this appears to be calculating the rho result values... not of concern to the actual solver algorithm.
	fval = 0.0;
	for(i = 0; i < L; i++){
	    rxs2[i] = lam2[i];
	}
	q = 0;
        for(j = 0; j < R2; j++){
	    for(i = 0; i < R1; i++){
        	p = i*j;
        	for(l = 0; l < P; l++){
           	    rxs2[K2*q+P*p+l] = 0.0;
        	}
        	q++;
            }
	}
	for(i = 0; i < L; i++){
	    fval += rxs2[i];
	}
	if(fval < fmin){
	    fmin = fval;
	    memcpy(rxst, rxs2, sizeof(double)*L);
            // This is also for output that will be printed when done.
						// NOTE: this rewrites rxs2, which is subsquently reset in the next step. The only time that the following code matters is when the "break" right below this is triggered. The code should be optimized by putting this code within that if statement, so that this code is never run except when it is actually relevant. 
	    computeATx(lam1, N1, N2, na, a, CC, rxs2);
            q = 0;
            for(j = 0; j < R2; j++){
                for(i = 0; i < R1; i++){
                    p = i*j;
                    for(l = 0; l < P; l++){
                        rxs2[K2*q+P*p+l] = -1.0;
                    }
                    q++;
                }
            }
	}

	//printf("On iteration %d, rel_err is %18.16f, fval is %8.6f, 
	// normrc is %8.6e, normrb is %8.6e.\n",
	// k,rel_err,fval,sqrt(normrc),sqrt(normrb));
	//fflush(stdout);

	if((rel_err < tol) || (rel_err != rel_err)){
	   break;
	}
	
	rel_err_old = rel_err;


				// The "rxs" values were used above for some sort of output calculation, but in the next step they are reset to be x*s, which is the third output target of the predictor step. 
				//
				// The predictor system is as follows:
				// A^T dlam + ds = -rc
				// A dx = -rb
				// x*ds + s*dx = -x*s
				//
				// This computes x*s
	for(i = 0; i < L; i++){
	    rxs1[i] = x1[i]*s1[i];
	    rxs2[i] = x2[i]*s2[i];
	}
				// This sets ds = (x*s - x*rc)/s
				//
	for(i = 0; i < L; i++){
	    ds1[i] = -1.0*x1[i]*rc1[i]+rxs1[i];
	    ds2[i] = -1.0*x2[i]*rc2[i]+rxs2[i];
	}
	for(i = 0; i < L; i++){
	    ds1[i] /= s1[i];
	    ds2[i] /= s2[i];
	}

				// dlam = A ds - rb
	computeAxReward(ds1, ds2, N1, N2, na, a, CC, dlam1, dlam2, compute_me2);
	for(i = 0; i < sz; i++){
	    dlam1[i] -= rb1[i];
	}
	for(i = 0; i < L; i++){
	    dlam2[i] -= rb2[i];
	}

		// dx = x/s
	for(i = 0; i < L; i++){
	    dx1[i] = x1[i]/s1[i];
	    dx2[i] = x2[i]/s2[i];
	}


		// ds = dlam
	for(i = 0; i < sz; i++){
	    ds1[i] = dlam1[i];
	}
	for(i = 0; i < L; i++){
	    ds2[i] = dlam2[i];
	}

				/* ?????? */
	solveSysMainReward(ds1, ds2, dlam1, dlam2, N1, N2, na, a, dx1, dx2, CC, Grm, gr, compute_me1, false);


				// ds = - A^T dlam - rc 
	computeATxReward(dlam1, dlam2, N1, N2, na, a, CC, ds1, ds2);
	for(i = 0; i < L; i++){
	    ds1[i] *= -1.0;
	    ds1[i] -= rc1[i];
	    ds2[i] *= -1.0;
	    ds2[i] -= rc2[i];
	}

				// dx = -(x*s + x*ds)/s
	for(i = 0; i < L; i++){
	    dx1[i]  = rxs1[i] + x1[i]*ds1[i];
	    dx1[i] /= -1.0*s1[i];
	    dx2[i]  = rxs2[i] + x2[i]*ds2[i];
	    dx2[i] /= -1.0*s2[i];
	}
	
	// Compute predictor step length
	alpha_ap = 1.0;
	for(i = 0; i < L; i++){
	    if(dx1[i] < 0.0){
		alpha_ap = MIN(alpha_ap,-1.0*x1[i]/dx1[i]);
	    }
	    if(dx2[i] < 0.0){
		alpha_ap = MIN(alpha_ap,-1.0*x2[i]/dx2[i]);
	    }
	}
	alpha_ad = 1.0;
	for(i = 0; i < L; i++){
	    if(ds1[i] < 0.0){
		alpha_ad = MIN(alpha_ad,-1.0*s1[i]/ds1[i]);
	    }
	    if(ds2[i] < 0.0){
		alpha_ad = MIN(alpha_ad,-1.0*s2[i]/ds2[i]);
	    }
	}

	// Compute centering parameter sigma.
	mu = 0.0;
	for(i = 0; i < L; i++){
	    mu += x1[i]*s1[i];
	}
	for(i = 0; i < L; i++){
	    mu += x2[i]*s2[i];
	}
	mu /= (double)(2*L);

	mu_aff = 0.0;
	for(i = 0; i < L; i++){
	    mu_aff += (x1[i] + alpha_ap*dx1[i])*(s1[i] + alpha_ad*ds1[i]);
	}
	for(i = 0; i < L; i++){
	    mu_aff += (x2[i] + alpha_ap*dx2[i])*(s2[i] + alpha_ad*ds2[i]);
	}
	mu_aff /= (double)(2*L);
	sigma = pow(mu_aff/mu,3.0);

	// Begin computing corrector step
	for(i = 0; i < L; i++){
	    rxs1[i] = x1[i]*s1[i] + dx1[i]*ds1[i] - sigma*mu;
	    rxs2[i] = x2[i]*s2[i] + dx2[i]*ds2[i] - sigma*mu;
	}
	for(i = 0; i < L; i++){
	    ds1[i] = -1.0*x1[i]*rc1[i]+rxs1[i];
	    ds2[i] = -1.0*x2[i]*rc2[i]+rxs2[i];
	}
	for(i = 0; i < L; i++){
	    ds1[i] /= s1[i];
	    ds2[i] /= s2[i];
	}
	computeAxReward(ds1, ds2, N1, N2, na, a, CC, dlam1, dlam2, compute_me2);
	for(i = 0; i < sz; i++){
	    dlam1[i] -= rb1[i];
	}
	for(i = 0; i < L; i++){
	    dlam2[i] -= rb2[i];
	}
	for(i = 0; i < L; i++){
	    dx1[i] = x1[i]/s1[i];
	    dx2[i] = x2[i]/s2[i];
	}
	for(i = 0; i < sz; i++){
	    ds1[i] = dlam1[i];
	}
	for(i = 0; i < L; i++){
	    ds2[i] = dlam2[i];
	}

	solveSysMainReward(ds1, ds2, dlam1, dlam2, N1, N2, na, a, dx1, dx2, CC, Grm, gr, compute_me1, true);
	computeATxReward(dlam1, dlam2, N1, N2, na, a, CC, ds1, ds2);

	for(i = 0; i < L; i++){
	    ds1[i] *= -1.0;
	    ds1[i] -= rc1[i];
	    ds2[i] *= -1.0;
	    ds2[i] -= rc2[i];
	}
	for(i = 0; i < L; i++){
	    dx1[i]  = rxs1[i] + x1[i]*ds1[i];
	    dx1[i] /= -1.0*s1[i];
	    dx2[i]  = rxs2[i] + x2[i]*ds2[i];
	    dx2[i] /= -1.0*s2[i];
	}

	// Compute corrector step length.	
	alpha_ap = 1.0;
	for(i = 0; i < L; i++){
	    if(dx1[i] < 0.0){
		alpha_ap = MIN(alpha_ap,-1.0*eta*x1[i]/dx1[i]);
	    }
	    if(dx2[i] < 0.0){
		alpha_ap = MIN(alpha_ap,-1.0*eta*x2[i]/dx2[i]);
	    }
	}

	alpha_ad = 1.0;
	for(i = 0; i < L; i++){
	    if(ds1[i] < 0.0){
		alpha_ad = MIN(alpha_ad,-1.0*eta*s1[i]/ds1[i]);
	    }
	    if(ds2[i] < 0.0){
		alpha_ad = MIN(alpha_ad,-1.0*eta*s2[i]/ds2[i]);
	    }
	}

	for(i = 0; i < L; i++){
	    x1[i] += alpha_ap*dx1[i];
	    x2[i] += alpha_ap*dx2[i];
	}
	for(i = 0; i < L; i++){
	    s1[i] += alpha_ad*ds1[i];
	    s2[i] += alpha_ad*ds2[i];
	}
	for(i = 0; i < sz; i++){
	    lam1[i] += alpha_ad*dlam1[i];
	}
	for(i = 0; i < L; i++){
	    lam2[i] += alpha_ad*dlam2[i];
	}

	eta = 1.0-0.1*pow(0.1,(double)(k+1)/50.0);
    }

    for(i = 0; i < M1; i++){
        rho[i] = 0;
        for(j = 0; j < M1; j++){
            for(k = 0; k < P; k++){
                rho[i] += rxst[K2*i+P*j+k];
            }
        }
    }

    double art_max = 1.0;
    double my_frac = 0.0;

    for(i = 0; i < M1; i++){
        art_max = MAX(art_max, fabs(rho[i]));
    }
    for(i = 0; i < M1; i++){
        if(rho[i] < art_max*tol3){
            my_frac += 1.0;
        }
    }

    my_frac = my_frac/((double)(M1)); 

    double *rtn = (double*)malloc(sizeof(double)*(M1+2));

    rtn[0] = fmin;
    rtn[1] = my_frac;
    for(i = 0; i < M1; i++){
        rtn[i+2] = rho[i];
    }

    free(rho);
    free(ipiv);
    free(gr);
    free(x1);
    free(x2);
    free(s1);
    free(s2);
    free(rc1);
    free(rc2);
    free(dx1);
    free(dx2);
    free(ds1);
    free(ds2);
    free(xs1);
    free(xs2);
    free(rxs1);
    free(rxs2);
    free(rxst);
    free(rb1);
    free(rb2);
    free(lam1);
    free(lam2);
    free(dlam1);
    free(dlam2);

    for(i = 0; i < M1; i++){
	free(a[i]);
    }
    free(a);

    for(i = 0; i < sz; i++){
	free(Grm[i]);
    }
    free(Grm);

    for(i = 0; i < M1; i++){
	free(CC[i]);
    }
    free(CC);

    freeInds(N, na, compute_me1);
    freeInds(N, na, compute_me2);

    return rtn;

}

int main(int argc, char *argv[]){

    double t = get_time_in_seconds();

    int i, j;
    int N1  = atoi(argv[1]); // No. bits of 1st input factor
    int N2  = atoi(argv[2]); // No. bits of 2nd input factor
    int na = atoi(argv[3]);  // No. ancilla
    int N  = N1+N2;
    int M1 = (int)round(pow(2,N));

    double fmin;
    double my_frac;

    bool **a = (bool**)malloc(sizeof(bool*)*M1);
    for(i = 0; i < M1; i++){
        a[i] = (bool*)malloc(sizeof(bool)*na);
    }

    readInAncilla(N1, N2, na, a, argv[4]);

    bool *aux = (bool*)malloc(sizeof(bool)*M1*na);
    for(i = 0; i < M1; i++){
        for(j = 0; j < na; j++){
            aux[i+j*M1] = a[i][j];
        }
    }

    double *result = lmisr(N1, N2, na, N, M1, aux);

    fmin = result[0];

    double *rho = (double*)malloc(sizeof(double)*M1);

    fmin = result[0];
    my_frac = result[1];
    memcpy(rho, &result[2], M1*sizeof(double));

    printf("Sum of artificial variables is %5.3f.\n",fmin);
    printf("Fraction of constraints satisfied is %8.6f.\n",my_frac);
    printf("Penalty vector is");
    for(i = 0; i < M1; i++){
        if ( i == 0 ) {
            printf(" [ %.4f,", rho[i]);
        }
        else if ( i == M1 - 1 ) {
            printf(" %.4f ]\n", rho[i]);
        }
        else {
            printf(" %.4f,", rho[i]);
        }
    }

    free(result);
    free(aux);
    free(rho);

    for(i = 0; i < M1; i++){
	free(a[i]);
    }
    free(a);
  
    printf("Elapsed time is %4.2f seconds.\n",get_time_in_seconds()-t);

    return 0;
}
