#include "include.h"

// Turns off customizable RHS of the original problem, setting it to be equal to the traditional problem.
#define			VI		1.0


// GLOBAL VARIABLES

int m;							// Number of rows in M
int n;							// Number of columns in M
double* M;					// The constraint matrix

// predictor-corrector state variables
double* x;
double* l;
double* s;
double* dx;
double* dl;
double* ds;
double* rc;

// internal variables for system solve
double* d;
double* zinv;
double* L;
double* lu_decomp;
int* pivot;

// generally useful temporary space
double* swap;

// technical stuff to make BLAS work
int INFO;
char TRANS = 'N';
int LD = 1;

void allocate_vars() {
	x					= (double*) malloc(sizeof(double) * 2 * m);
	l					= (double*) malloc(sizeof(double) * (n + m));
	s					= (double*) malloc(sizeof(double) * 2 * m);
	dx				= (double*) malloc(sizeof(double) * 2 * m);
	dl				= (double*) malloc(sizeof(double) * (n + m));
	ds				= (double*) malloc(sizeof(double) * 2 * m);
	rc				= (double*) malloc(sizeof(double) * 2 * m);
	d					= (double*) malloc(sizeof(double) * m);
	zinv			= (double*) malloc(sizeof(double) * m);
	L					= (double*) malloc(sizeof(double) * 2 * m);
	lu_decomp = (double*) malloc(sizeof(double) * (n * n));
	swap			= (double*) malloc(sizeof(double) * imax(n+m, 2*m));
	pivot			= (int*) malloc(sizeof(int) * n);
}

void free_vars() {
	free(x);
	//free(l);			// DON'T FREE LAMBDA --- IT'S THE RETURN!
	free(s);
	free(dx);
	free(dl);
	free(ds);
	free(d);
	free(zinv);
	free(L);
	free(lu_decomp);
	free(swap);
	free(pivot);
}

 /*
	* Main solver function: Assumes that the constraint matrix and so on has already been set up
	*
	* Runs the MPC algorithm, see Nocedal & Wright chapter 14 for algorithm and documentation for 
	* implementation-specific details.
	*/
double* solve(int max_iter, double tolerance) {
	allocate_vars();

  // iteration variables
  int i, iteration;

	// various small variables
	double alpha_primal_affine, alpha_dual_affine, alpha_primal, alpha_dual, affine_mu, mu, sigma;
	double eta = 0.9;
	double initial_error, error, relative_error;

	// Calulate initial guesses
	
	// Prepare the system solution variables
	// Effectively, sets K = I, so that AKA^T = AA^T
	for(i = 0; i < m; i++) {
		d[i] = 1.0;
		zinv[i] = 0.5;
	}
	generate_coefficient_matrix(lu_decomp, d, zinv);
  dgetrf_(&n, &n, lu_decomp, &n, pivot, &INFO);

	
	// Calculate the initial tilde guesses, using the deltas as temp variables
	
	// x <- A^T (AA^T)^-1 b
	for(i = 0; i < n; i++) {
		dl[i] = 0.0;
	}
	for(i = n; i < n+m; i++) {
		dl[i] = -1.0;
	}
	solve_AKAt(swap, dl, ds);
	multiply_by_At(x, swap);

	// ds <- c
	// l <- (AA^T)^-1 A c
	for(i = 0; i < m; i++) {
		ds[i] = -VI;
	}
	for(i = m; i < 2*m; i++) {
		ds[i] = 0.0;
	}
	multiply_by_A(swap, ds);
	solve_AKAt(l, swap, dx);

	// s <- c - A^T l
	multiply_by_At(s, l);
	for(i = 0; i < 2*m; i++) {
		s[i] = ds[i] - s[i];
	}

	// Initial calculation: first delta corrector step
	
	double delta_x = 0.0;
	double delta_s = 0.0;
	for(i = 0; i < 2*m; i++) {
		delta_x = fmin(delta_x, x[i]);
		delta_s = fmin(delta_s, s[i]);
	}
	delta_x = fmax(0, -(3.0/2.0) * delta_x);
	delta_s = fmax(0, -(3.0/2.0) * delta_s);

	for(i = 0; i < 2*m; i++) {
		x[i] += delta_x;
		s[i] += delta_s;
	}

	// Initial calculation: second delta corrector step ("delta-hat")
	
	double inner_product = 0.0;
	double sum_x = 0.0;
	double sum_s = 0.0;
	for(i = 0; i < 2*m; i++) {
		inner_product += x[i] * s[i];
		sum_x += x[i];
		sum_s += s[i];
	}
	delta_x = inner_product / (2.0 * sum_s);
	delta_s = inner_product / (2.0 * sum_x);

	for(i = 0; i < 2*m; i++) {
		x[i] += delta_x;
		s[i] += delta_s;
	}

	/*
	printvec(x, 2*m, "x");
	printvec(s, 2*m, "s");
	printvec(l, n+m, "l");
	return 0.0;
	*/

	// Start the main iteration
  for(iteration = 0; iteration < max_iter; iteration++) {
    
		//printf("iteration %d\n", iteration);


    // rc <- A^t lambda + s - c
    multiply_by_At(rc, l);
    for(i = 0; i < 2*m; i++) {
      rc[i] += s[i];
    }
    for(i = 0; i < m; i++) {
      rc[i] -= -VI;
    }
		
		// rb <- Ax - b
		// This is stored in swap because we do not need it for anything except the error calculation. This is a whole matrix-vector multiply for the error calculation, which I'd like to avoid; look into this later.
		multiply_by_A(swap, x);
		for(i = n; i < n+m; i++) {
			swap[i] += 1;
		}

		// Calculate relative error
		error = 0;
		for(i = 0; i < 2*m; i++) {
			error += rc[i] * rc[i];
		}
		for(i = 0; i < n+m; i++) {
			error += swap[i] * swap[i];
		}
		error = sqrt(error);
		if(iteration == 0) {
			initial_error = error;
		}
		relative_error = error / initial_error;
		
		// Determine break condition
		if(relative_error < tolerance || relative_error != relative_error) {
			break;
		}
		
    // L <- -x*s
    for(i = 0; i < 2*m; i++) {
      L[i] = -x[i] * s[i];
    }

		// d <- x1/s1
		// zinv <- 1/(x1/s1 + x2/s2)
		for(i = 0; i < m; i++) {
			d[i] = x[i]/s[i];
			zinv[i] = 1.0/(d[i] + (x+m)[i]/(s+m)[i]);
		}

    // Generate the LU decomposition for the system solve. Will be used for both the predictor and corrector step.
    generate_coefficient_matrix(lu_decomp, d, zinv);
		//printmat(lu_decomp, n, n, "lu_decomp");
    dgetrf_(&n, &n, lu_decomp, &n, pivot, &INFO);

		/*
		printvec(x, 2*m, "x");
		printvec(rc, 2*m, "rc");
		printvec(L, 2*m, "L");
		printvec(s, 2*m, "s");
	*/

    // Solves the predictor system to calculate affine deltas.
    solve_main_system();

		/*
		printvec(dx, 2*m, "dx");
		printvec(ds, 2*m, "ds");
		printvec(dl, n+m, "dl");
		return 0.0;
*/


    // Calculate affine alphas
    alpha_primal_affine = 1.0;
    alpha_dual_affine = 1.0;
    for(i = 0; i < 2*m; i++) {
      if(dx[i] < 0.0) {
        alpha_primal_affine = fmin(alpha_primal_affine, -x[i]/dx[i]);
      }
      if(ds[i] < 0.0) {
        alpha_dual_affine = fmin(alpha_dual_affine, -s[i]/ds[i]);
      }
    }
    
    // Calculate affine mu
    affine_mu = 0.0;
    for(i = 0; i < 2*m; i++) {
      affine_mu += (x[i] + alpha_primal_affine * dx[i]) * (s[i] + alpha_dual_affine * ds[i]);
    }
    affine_mu /= (double)(2*m);
  
    // Calculate regular mu... note that at this point L = -x*s, so we can take advantage of this to do the inner product with slightly fewer operations.
    mu = 0.0;
    for(i = 0; i < 2*m; i++) {
      mu -= L[i];
    }
    mu /= (double)(2*m);

    sigma = pow(affine_mu / mu, 3.0);

    // Modify L for the corrector step
    for(i = 0; i < 2*m; i++) {
      L[i] -= dx[i] * ds[i] - sigma * mu;
    }

    // Solves the corrector system.
    solve_main_system();
    
    // Calculate alphas
    alpha_primal = 1000.0;
    alpha_dual = 1000.0;
    for(i = 0; i < 2*m; i++) {
      if(dx[i] < 0.0) {
        alpha_primal = fmin(alpha_primal, -x[i]/dx[i]);
      }
      if(ds[i] < 0.0) {
        alpha_dual = fmin(alpha_dual, -s[i]/ds[i]);
      }
    }
    alpha_primal = fmin(1.0, eta * alpha_primal);
    alpha_dual = fmin(1.0, eta * alpha_dual);

    // Update the variables
    for(i = 0; i < 2*m; i++) {
      x[i] += alpha_primal * dx[i];
      s[i] += alpha_dual * ds[i];
    }
    for(i = 0; i < n+m; i++) {
      l[i] += alpha_dual * dl[i];
    }


		//printvec(l, n+m, "lambda");

    // asymptotic modification of step size, formula copied from Teresa. 
    eta = 1.0 - 0.1 * pow(0.1, (double)(iteration + 1)/50.0);
  }

	free_vars();
	//printf("finished\n");
	return l;		
}

void solve_main_system() {
    
  // iteration variables
  int i;

  // u <- b - A(x + S^-1 (X r_c + L))
	// We store u (the right hand side of the system) in swap
	// We use ds as a temporary variable
  double* swap2 = swap+n;
  for(i = 0; i < 2*m; i++) {
    ds[i] = -(x[i] + (x[i]*rc[i] + L[i])/s[i]);
  }
  multiply_by_A(swap, ds);
  for(i = 0; i < m; i++) {
    swap2[i] -= 1;
  }

	solve_AKAt(dl, swap, ds);

  // After this point, ds is no longer temporary.

  // ds <- -rc - A^t dl
  multiply_by_At(ds, dl);
  for(i = 0; i < 2*m; i++) {
    ds[i] = -rc[i] - ds[i];
  }

  // dx <- S^-1 (L - X ds)
  for(i = 0; i < 2*m; i++) {
    dx[i] = (L[i] - x[i] * ds[i])/s[i];
  }
}

 /*
	* Implements the algorithm for solving system of the form (AKA^T)p = q. 
	* See documentation for an explanation of how it works. 
	* This routine expects to have a temporary array provided for intermediate computations, 
	* as well as for the LU decomposition of M^T(D-D^2Z^-1)M to be pre-computed.
	*/
void solve_AKAt(double* p, double* q, double* tmp) {
	int i;

	double* q2 = q+n;
	double* p2 = p+n;

	// Calculate right hand side for system solve
	// i.e. p <- q1 - M^T DZ^-1 q2
	for(i = 0; i < m; i++) {
		tmp[i] = -d[i] * zinv[i] * q2[i];
	}
	multiply_by_Mt(p, tmp);
	for(i = 0; i < n; i++) {
		p[i] += q[i];
	}

	// Solve the system using precomputed LU decomposition
	dgetrs_(&TRANS, &n, &LD, lu_decomp, &n, pivot, p, &n, &INFO);

	// Calculate p2 from the result
	multiply_by_M(p2, p);
	for(i = 0; i < m; i++) {
		p2[i] = zinv[i] * (q2[i] - d[i] * p2[i]);	
	}
}

// ------------------ OPERATORS ------------------ 

double fmin(double a, double b) {
	if(a > b) {
		return b;
	}
	return a;
}

double fmax(double a, double b) {
	if(a > b) {
		return a;
	}
	return b;
}

int imax(int a, int b) {
	if(a > b) {
		return a;
	}
	return b;
}
void generate_coefficient_matrix(double* target, double* d, double* zinv) {
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			target[i*n + j] = 0.0;
			for(int k = 0; k < m; k++) {
				target[i*n + j] += M[k*n + i] * (d[k] - d[k]*d[k]*zinv[k]) * M[k*n + j];
			}
		}
	}
}

void multiply_by_A(double* target, double* vector) {
	multiply_by_Mt(target, vector);
	for(int i = 0; i < n; i++) {
		target[i] *= -1.0;
	}
	for(int i = 0; i < m; i++) {
		target[i+n] = -vector[i] - vector[i+m];
	}
}

void multiply_by_At(double* target, double* vector) {
	multiply_by_M(target, vector);
	for(int i = 0; i < m; i++) {
		target[i] = -vector[i+n] - target[i];
		target[i+m] = -vector[i+n];
	}
}

void multiply_by_M(double* target, double* vector) {
	for(int i = 0; i < m; i++) {
		target[i] = 0.0;
		for(int j = 0; j < n; j++) {
			target[i] += M[i*n + j] * vector[j];
		}
	}
}

void multiply_by_Mt(double* target, double* vector) {
	for(int i = 0; i < n; i++) {
		target[i] = 0.0;
		for(int j = 0; j < m; j++) {
			target[i] += M[j*n + i] * vector[j];
		}
	}
}


// Debug functions for printing matrices and vectors

// prints a contigous matrix
void printmat(double* mat, int rows, int cols, char* name) {
	printf("%s = ", name);
	for(int i=0; i< rows ; i++) {
		for(int j=0; j<cols; j++) {
			printf("%lf ", mat[i*cols + j]);
		}
		printf("\n");
	}
	printf("\n");
}

// prints a non-contiguous matrix
void printmat2(double** mat, int rows, int cols, char* name) {
	printf("non-contigous\n");
	printf("%s = ", name);
	for(int i=0; i< rows ; i++) {
		for(int j=0; j<cols; j++) {
			printf("%lf ", mat[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

void printvec(double* vec, int length, char* name) {
	printf("%s = ", name);
	for(int i=0; i< length ; i++) {
		printf("%lf ", vec[i]);
	}
	printf("\n");
}

// -------------------------- INTERFACE -------------------------


double* interface(double* constraints, int num_rows, int num_cols, int num_workers) {
	M = constraints;
	m = num_rows;
	n = num_cols;

	openblas_set_num_threads(num_workers); 
	return solve(200, 1e-6);
}

double* free_ptr(void* ptr) {
	free(ptr);
}

 /*
	* Main entry point for running stand-alone---this is basically just a test function designed to see if the solver works on a very small problem.
	*/
int main() {
	/*
	int m = 3;
	int n = 2;
	double** mat = (double**) malloc(sizeof(double*) * m);
	for(int i=0; i<m; i++) {
		mat[i] = (double*) malloc(sizeof(double) * n);
	}
	mat[0][0] = 1;
	mat[0][1] = 1;
	mat[1][0] = -1;
	mat[1][1] = 0;
	mat[2][0] = 1;
	mat[2][1] = -1;

	double result = interface(mat, m, n, 1);
	printf("%lf\n", result);

	for(int i=0; i<m; i++) {
		free(mat[i]);
	}
	free(mat);
*/
	return 0;
}












