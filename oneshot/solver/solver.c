#include "include.h"
#include <unistd.h>
#include <sys/time.h>

// Turns off customizable RHS of the original problem, setting it to be equal to the traditional problem.
#define			VI						1.0
#define			NUM_THREADS		20

typedef struct {
	int8_t* values;	
	int* row_index;
	int* col_ptr;
} CSC;

typedef struct {
	int8_t* values;
	int* col_index;
	int* row_ptr;
} CSR;

// GLOBAL VARIABLES

int m;							// Number of rows in M
int n;							// Number of columns in M
double* M;					// The constraint matrix

// sparse accelerator stuff
CSC M_csc;
CSR M_csr;
int use_csr;
int* coeff_matrix_num_matches;
int** coeff_matrix_matches;
int8_t** coeff_matrix_match_values;

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
double perturbation = 1e-7;


//hyperparameters
double initial_eta;
double eta_decay_param;
double* saved_initial_lam;
int keep_initial;

// thread pool variables
int num_started;
int num_done;
enum THREADPOOL_STATUS{STATUS_RUNNING, STATUS_DESTROY, STATUS_COEFF, STATUS_MT, STATUS_M};
enum THREADPOOL_STATUS threadpool_status;
pthread_mutex_t job_queue_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_t threads[NUM_THREADS];
pthread_barrier_t ready_barrier;
pthread_barrier_t start_barrier;
pthread_barrier_t done_barrier;
int current_num_jobs;
void (*current_job)(int);

// precomputed values for the coefficient matrix generation job
int coeff_num_jobs;
int* coeff_job_indices;						// indexes (i*n + j) of col-col multiplication jobs
int* coeff_job_transpose;

double* temp_pointer;				// pointer to the diagonal entries for the threads to work with
double* target_pointer;


struct timeval tv_start, tv_end;

void start_threadpool() {
	coeff_num_jobs = (n * n + n)/2;
	num_started = 0;
	num_done = 0;
	coeff_job_indices = malloc(sizeof(int) * coeff_num_jobs);
	coeff_job_transpose = malloc(sizeof(int) * coeff_num_jobs);
	int q = 0;
	for(int i=0; i<n; i++) {
		for(int j=0; j<=i; j++) {
			coeff_job_indices[q] = i*n + j;
			coeff_job_transpose[q] = j*n + i;
			q++;
		}
	}
	
	pthread_barrier_init(&ready_barrier, NULL, NUM_THREADS + 1);
	pthread_barrier_init(&done_barrier, NULL, NUM_THREADS + 1);

	threadpool_status = STATUS_RUNNING;
	for(int i=0; i<NUM_THREADS; i++) {
		pthread_create(&threads[i], NULL, &threadloop, NULL);
	}
}

void destroy_threadpool() {
	threadpool_status = STATUS_DESTROY;
	pthread_barrier_wait(&ready_barrier);
	for(int i=0; i<NUM_THREADS; i++) {
		pthread_join(threads[i], NULL);
	}

	pthread_barrier_destroy(&ready_barrier);
	pthread_barrier_destroy(&done_barrier);
	free(coeff_job_indices);
	free(coeff_job_transpose);
}

void* threadloop() {
	while(1) {
		pthread_barrier_wait(&ready_barrier);

		if(threadpool_status == STATUS_DESTROY) {
			break;
		}

		thread_exec();
		pthread_barrier_wait(&done_barrier);
	}
}

void thread_exec() {
	int task;
	while(1) {
		// check the queue to see if there are tasks to do
		pthread_mutex_lock(&job_queue_mutex);
		if(num_started >= current_num_jobs) {
			// no more jobs to do
			pthread_mutex_unlock(&job_queue_mutex);
			break;
		}

		task = num_started;
		num_started++;
		pthread_mutex_unlock(&job_queue_mutex);

		// execute task
		(*current_job)(task);
	}
}

void coeff_task(int index) {
	int task = coeff_job_indices[index];
	int transpose = coeff_job_transpose[index];
	int match, k;
	target_pointer[task] = 0.0;
	for(k=0; k<coeff_matrix_num_matches[task]; k++) {
		match = coeff_matrix_matches[task][k];
		target_pointer[task] += temp_pointer[match] * ((int)coeff_matrix_match_values[task][k]);
	}
	target_pointer[transpose] = target_pointer[task];
	target_pointer[task] += ((float)rand()/(float)(RAND_MAX)) * perturbation;
	target_pointer[transpose] += ((float)rand()/(float)(RAND_MAX)) * perturbation;
}

void mt_task(int col) {
	target_pointer[col] = 0.0;
	int i;
	for(i = M_csc.col_ptr[col]; i < M_csc.col_ptr[col+1]; i++) {
		target_pointer[col] += M_csc.values[i] * temp_pointer[M_csc.row_index[i]];
	}
}

void m_task(int row) {
	target_pointer[row] = 0.0;
	int i;
	for(i = M_csr.row_ptr[row]; i < M_csr.row_ptr[row+1]; i++) {
		target_pointer[row] += M_csr.values[i] * temp_pointer[M_csr.col_index[i]];
	}
}

void threadpool_run_jobs(double* target, double* tmp, int num_jobs, void (*job)(int)) {
	target_pointer = target;
	temp_pointer = tmp;
	num_done = 0;
	num_started = 0;
	current_job = job;
	current_num_jobs = num_jobs;
	pthread_barrier_wait(&ready_barrier);
	pthread_barrier_wait(&done_barrier);
}

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

	coeff_matrix_num_matches = malloc(sizeof(int) * n * n);
	coeff_matrix_matches = malloc(sizeof(int*) * n * n);
	coeff_matrix_match_values = malloc(sizeof(uint8_t*) * n * n);
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

	for(int i=0; i<n; i++) {
		for(int j=0; j<=i; j++) {
			free(coeff_matrix_matches[i*n + j]);
			free(coeff_matrix_match_values[i*n + j]);
		}
	}
	free(coeff_matrix_matches);
	free(coeff_matrix_match_values);
	free(coeff_matrix_num_matches);
}

void safe_make_coeff_matrix() {
	INFO = 1;
	while(INFO > 0) {
		threaded_coefficient_matrix(lu_decomp, d, zinv, swap);
		dgetrf_(&n, &n, lu_decomp, &n, pivot, &INFO);
		
		if(INFO != 0) {
			printf("WARNING: LU-Decomposition is singular at %d\n", INFO);
		}else{
			break;
		}

		for(int i=0; i<n; i++) {
			for(int j=0; j<n; j++) {
				lu_decomp[i*n+j] += ((float)rand()/(float)(RAND_MAX)) * perturbation;
			}
		}
	}
}

 /*
	* Main solver function: Assumes that the constraint matrix and so on has already been set up
	*
	* Runs the MPC algorithm, see Nocedal & Wright chapter 14 for algorithm and documentation for 
	* implementation-specific details.
	*/
double* solve(int max_iter, double tolerance) {
	srand((unsigned int) time(NULL));
	allocate_vars();
	start_threadpool();
	compute_coeff_matrix_matches();

  // iteration variables
  int i, iteration;

	// various small variables
	double alpha_primal_affine, alpha_dual_affine, alpha_primal, alpha_dual, affine_mu, mu, sigma;
	double eta = initial_eta; //0.9;
	double initial_error, error, relative_error;

	// Calulate initial guesses
	
	// Prepare the system solution variables
	// Effectively, sets K = I, so that AKA^T = AA^T
	for(i = 0; i < m; i++) {
		d[i] = 1.0;
		zinv[i] = 0.5;
	}

	safe_make_coeff_matrix();
	
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

	if(keep_initial != 0) {
		for(i=0; i < m+n; i++) {
			saved_initial_lam[i] = l[i];
		}
	}

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
		safe_make_coeff_matrix();

    // Solves the predictor system to calculate affine deltas.
    solve_main_system();

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

		/*
		// check for stopping condition---rhos are not moving much
		error = 0.0;
		for(i=n; i<n+m; i++) {
			initial_error = dl[i] > 0 ? dl[i] : -dl[i];
			error = fmax(initial_error, error);
		}

		if(error < tolerance) {
			break;
		}
*/

		//debug statement which prints current solution
		//printvec(l, n+m, "lambda");

    // asymptotic modification of step size, formula copied from Teresa. 
    eta = initial_eta - eta_decay_param * pow(0.1, (double)(iteration + 1)/50.0);
  }

	if(iteration == max_iter) {
		//printf("WARNING: Solver failed to converge!\n");
	}

	free_vars();
	destroy_threadpool();
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

/* dense triangular coeff matrix generation
void generate_coefficient_matrix(double* target, double* d, double* zinv, double* tmp) {
	for(int i = 0; i < n; i++) {
		for(int j = 0; j <= i; j++) {
			target[i*n + j] = 0.0;
			for(int k = 0; k < m; k++) {
				target[i*n + j] += M[k*n + i] * (d[k] - d[k]*d[k]*zinv[k]) * M[k*n + j];
			}
			target[j*n + i] = target[i*n + j];
		}
	}
}*/



double M_col_dot_prod(int A, int B, double* coeffs) {
	double total = 0.0;
	int i = M_csc.col_ptr[A];
	int i_end = M_csc.col_ptr[A+1];
	int j = M_csc.col_ptr[B];
	int j_end = M_csc.col_ptr[B+1];
	int row_i = M_csc.row_index[i];
	int row_j = M_csc.row_index[j];
	row_i = M_csc.row_index[i];
	row_j = M_csc.row_index[j];
	while(1) {
		if(row_i == row_j) {
			total += ((double)(M_csc.values[i] * M_csc.values[j])) * coeffs[row_j];
			i++;
			j++;
			if(i >= i_end || j >= j_end) {
				return total;
			}
			row_i = M_csc.row_index[i];
			row_j = M_csc.row_index[j];
			continue;
		}
		if(row_i < row_j) {
			i++;
			if(i >= i_end) {
				return total;
			}
			row_i = M_csc.row_index[i];
			continue;
		}
		if(row_i > row_j) {
			j++;
			if(j >= j_end) {
				return total;
			}
			row_j = M_csc.row_index[j];
		}
	}
}

void compute_col_matches(int row, int col) {
	coeff_matrix_num_matches[row*n + col] = 0;
	int memory_allocated = 16;
	coeff_matrix_matches[row*n + col] = malloc(sizeof(int) * memory_allocated);
	coeff_matrix_match_values[row*n + col] = malloc(memory_allocated);
	int matches_found = 0;

	int i = M_csc.col_ptr[row];
	int i_end = M_csc.col_ptr[row+1];
	int j = M_csc.col_ptr[col];
	int j_end = M_csc.col_ptr[col+1];
	// check for all-zero columns
	if(i >= i_end || j >= j_end) {
		return;
	}
	int row_i = M_csc.row_index[i];
	int row_j = M_csc.row_index[j];
	row_i = M_csc.row_index[i];
	row_j = M_csc.row_index[j];
	while(1) {
		if(row_i == row_j) {
			coeff_matrix_matches[row*n + col][matches_found] = row_i;
			coeff_matrix_match_values[row*n + col][matches_found] = M_csc.values[i] * M_csc.values[j];
			matches_found++;
			if(matches_found >= memory_allocated) {
				memory_allocated *= 2;
				coeff_matrix_matches[row*n + col] = realloc(coeff_matrix_matches[row*n + col], sizeof(int) * memory_allocated);
				coeff_matrix_match_values[row*n + col] = realloc(coeff_matrix_match_values[row*n + col], memory_allocated);
			}
	
			i++;
			j++;
			if(i >= i_end || j >= j_end) {
				break;
			}
			row_i = M_csc.row_index[i];
			row_j = M_csc.row_index[j];
			continue;
		}
		if(row_i < row_j) {
			i++;
			if(i >= i_end) {
				break;
			}
			row_i = M_csc.row_index[i];
			continue;
		}
		if(row_i > row_j) {
			j++;
			if(j >= j_end) {
				break;
			}
			row_j = M_csc.row_index[j];
		}
	}
	coeff_matrix_num_matches[row*n + col] = matches_found;
}

void compute_coeff_matrix_matches() {
	for(int i=0; i < n; i++) {
		for(int j=0; j <= i; j++) {
			compute_col_matches(i, j);
		}
	}
}

void threaded_coefficient_matrix(double* target, double* d, double* zinv, double* tmp) {
	for(int i=0; i<m; i++) {
		tmp[i] = d[i] - d[i] * d[i] * zinv[i];
	}

	threadpool_run_jobs(target, tmp, coeff_num_jobs, &coeff_task);
}

void generate_coefficient_matrix(double* target, double* d, double* zinv, double* tmp) {
	int i, j, k, match;

	for(i=0; i<m; i++) {
		tmp[i] = d[i] - d[i] * d[i] * zinv[i];
	}


	for(i = 0; i < n; i++) {
		for(j = 0; j <= i; j++) {
			target[i*n + j] = 0.0;
			for(k=0; k<coeff_matrix_num_matches[i*n + j]; k++) {
				match = coeff_matrix_matches[i*n + j][k];
				target[i*n + j] += tmp[match] * ((int)coeff_matrix_match_values[i*n + j][k]);
			}
			//target[i*n + j] = M_col_dot_prod(i, j, tmp);
			target[j*n + i] = target[i*n + j];
			//printf("%.0lf ", target[i*n + j]);
		}
		//printf("\n");
	}
	//exit(0);
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

/*
void multiply_by_M(double* target, double* vector) {
	for(int i = 0; i < m; i++) {
		target[i] = 0.0;
		for(int j = 0; j < n; j++) {
			target[i] += M[i*n + j] * vector[j];
		}
	}
}*/


void multiply_by_M(double* target, double* vector) {
	if(use_csr) {
		threadpool_run_jobs(target, vector, m, &m_task);
		return;
	}
	int i, col, row_index;

	for(i = 0; i < m; i++) {
		target[i] = 0.0;
	}

	for(col = 0; col < n; col++) {
		for(i = M_csc.col_ptr[col]; i < M_csc.col_ptr[col+1]; i++) {
			row_index = M_csc.row_index[i];
			target[row_index] += M_csc.values[i] * vector[col];
		}
	}
}

/*
void multiply_by_M(double* target, double* vector) {
	for(int i = 0; i < m; i++) {
		target[i] = 0.0;
	}
	threadpool_run_jobs(STATUS_M, target, vector);
}
*/

/*
void multiply_by_Mt(double* target, double* vector) {
	int i, col;

	for(col = 0; col < n; col++) {
		target[col] = 0.0;
		for(i = M_csc.col_ptr[col]; i < M_csc.col_ptr[col+1]; i++) {
			target[col] += M_csc.values[i] * vector[M_csc.row_index[i]];
		}
	}
}
*/

void multiply_by_Mt(double* target, double* vector) {
	threadpool_run_jobs(target, vector, n, &mt_task);
}

/* dense matrix multiply method
void multiply_by_Mt(double* target, double* vector) {
	for(int i = 0; i < n; i++) {
		target[i] = 0.0;
		for(int j = 0; j < m; j++) {
			target[i] += M[j*n + i] * vector[j];
		}
	}
}
*/

void printbin(unsigned long val) {
	for(int i=0; i<sizeof(unsigned long) * 8; i++) {
		printf("%b", val & 1);
		val = val >> 1;
	}
	printf("\n");
}

void generate_CSC_constraints(int n1, int n2, int8_t* aux_array, int num_aux) {
	int N = n1 + n2;
	int G = 2*N + num_aux;
	m = (1 << G) - (1 << (N + num_aux));
	n = N + num_aux + (G * (G-1))/2 - (N * (N-1))/2;
	int num_input_levels = 1 << N;
	int* correct_answers = (int*) malloc(sizeof(int) * num_input_levels);
	int* correct_outaux = (int*) malloc(sizeof(int) * num_input_levels);
	int i, j;
	int input2_mask = (1 << n2) - 1;
	M_csc.col_ptr = (int*) malloc(sizeof(int) * (n+1));

	for(i = 0; i < num_input_levels; i++) {
		correct_answers[i] = ((i & input2_mask) * (i >> n2));
		correct_outaux[i] = correct_answers[i] << num_aux;
		for(j = 0; j < num_aux; j++) {
			if(aux_array[j * num_aux + i] == 1) {
				correct_outaux[i] += 1 << (num_aux-j-1);
			}
		}
	}


	// initial guess for the amount of memory needed--will expand or reduce later as necessary
	int num_vals_added = 0;
	int current_memory_size = (n * m) / 2;
	M_csc.values = (int8_t*) malloc(sizeof(int8_t) * current_memory_size);
	M_csc.row_index = (int*) malloc(sizeof(int) * current_memory_size);

	int inp, out, aux;
	int inp_part, out_part;

	int state;
	int row_index, col_index;

	int8_t wrong_bit, right_bit, val;
	int8_t other_wrong_bit, other_right_bit;


	col_index = 0;
	// generate the columns corresponding to linear terms
	for(i = N; i < G; i++) {
		M_csc.col_ptr[col_index] = num_vals_added;
		row_index = 0;
		for(inp = 0; inp < (1 << N); inp++) {
			inp_part = inp << (N + num_aux);
			right_bit = (int8_t) (((inp_part + correct_outaux[inp]) >> (G-i-1)) & 1);
			for(out = 0; out < (1 << N); out++) {
				if(out == correct_answers[inp]) {
					continue;
				}else{
				}
				out_part = out << num_aux;
				for(aux = 0; aux < (1 << num_aux); aux++) {
					state = inp_part + out_part + aux;
					wrong_bit = (int8_t) ((state >> (G-i-1)) & 1);
					val = wrong_bit - right_bit;
					if(val != 0) {
						//printf("%d %d \n", col_index, row_index);

						M_csc.values[num_vals_added] = val;
						M_csc.row_index[num_vals_added] = row_index;
						num_vals_added++;

						if(num_vals_added >= current_memory_size) {
							printf("reallocating\n");
							// give me another thousand entries!
							current_memory_size += (1 << 10);
							M_csc.values = realloc(M_csc.values, sizeof(int8_t) * current_memory_size);
							M_csc.row_index = realloc(M_csc.row_index, sizeof(int) * current_memory_size);
						}
					}

					row_index++;
				}
			}
		}
		col_index++;
	}



	//generate the columns corresponding to quadratic terms
	for(i = N; i < G; i++) {
		for(j = 0; j < i; j++) {
			M_csc.col_ptr[col_index] = num_vals_added;
			
			row_index = 0;
			for(inp = 0; inp < (1 << N); inp++) {
				inp_part = inp << (N + num_aux);
				right_bit = (int8_t) (((inp_part + correct_outaux[inp]) >> (G-i-1)) & 1);
				other_right_bit = (int8_t) (((inp_part + correct_outaux[inp]) >> (G-j-1)) & 1);
				right_bit *= other_right_bit;
				for(out = 0; out < (1 << N); out++) {
					if(out == correct_answers[inp]) {
						continue;
					}
					out_part = out << num_aux;
					for(aux = 0; aux < (1 << num_aux); aux++) {
						state = inp_part + out_part + aux;
						wrong_bit = (int8_t) ((state >> (G-i-1)) & 1);
						other_wrong_bit = (int8_t) ((state >> (G-j-1)) & 1);
						wrong_bit *= other_wrong_bit;
						val = wrong_bit - right_bit;
						if(val != 0) {
							//printf("%d %d \n", col_index, row_index);
							M_csc.values[num_vals_added] = val;
							M_csc.row_index[num_vals_added] = row_index;
							//printf("added %d max %d\n", num_vals_added, current_memory_size);
							num_vals_added++;

							if(num_vals_added >= current_memory_size) {
								// give me another thousand entries!
								printf("reallocating\n");
								current_memory_size += (1 << 10);
								M_csc.values = realloc(M_csc.values, sizeof(int8_t) * current_memory_size);
								M_csc.row_index = realloc(M_csc.row_index, sizeof(int) * current_memory_size);
							}
						}

						row_index++;
					}
				}
			}
			col_index++;
		}
	}
	M_csc.col_ptr[col_index] = num_vals_added;
	free(correct_answers);
	free(correct_outaux);

	/*
	printf("ccol ");
	for(i = 0; i < n+1; i++) {
		printf("%d ", M_csc.col_ptr[i]);
	}
	printf("\n");

	printf("row_indices ");
	for(i = 0; i < n; i++) {
		for(j = M_csc.col_ptr[i]; j < M_csc.col_ptr[i+1]; j++) {
			printf("%d ", M_csc.row_index[j]);
		}
	}
	printf("\n");



	printf("values ");
	for(i = 0; i < n; i++) {
		for(j = M_csc.col_ptr[i]; j < M_csc.col_ptr[i+1]; j++) {
			printf("%d ", M_csc.values[j]);
		}
	}
	printf("\n");
	*/
}

void free_M_CSC() {
	free(M_csc.values);
	free(M_csc.row_index);
	free(M_csc.col_ptr);
}


// Debug functions for printing matrices and vectors

// prints a contigous matrix
void printmat(double* mat, int rows, int cols, char* name) {
	printf("%s = ", name);
	for(int i=0; i< rows ; i++) {
		for(int j=0; j<cols; j++) {
			//printf("%.1lf ", mat[i*cols + j]);
			printf("%d ", (int)mat[i*cols + j]);
		}
		printf("\n");
	}
	printf("\n");
	printf("done\n");
}

// prints a non-contiguous matrix
void printmat2(double** mat, int rows, int cols, char* name) {
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

void printintvec(int* vec, int length, char* name) {
	printf("%s = ", name);
	for(int i=0; i< length ; i++) {
		printf("%ld ", vec[i]);
	}
	printf("\n");
}

// -------------------------- INTERFACE -------------------------


double* interface(double* constraints, int num_rows, int num_cols, int num_workers, double tolerance, int max_iter) {
	M = constraints;
	m = num_rows;
	n = num_cols;

	openblas_set_num_threads(num_workers); 
	return solve(max_iter, tolerance);
}

double* sparse_interface(int num_rows, int num_cols, int8_t* values, int* row_index, int* col_ptr, int num_workers, double tolerance, int max_iter, double init_eta, double eta_decay, int keep_init) {
	m = num_rows;
	n = num_cols;
	initial_eta = init_eta;
	eta_decay_param = eta_decay;
	keep_initial = keep_init;
	M_csc.values = values;
	M_csc.row_index = row_index;
	M_csc.col_ptr = col_ptr;
	/*
	for(int k=0; k<n; k++) {
		for(int l=M_csc.col_ptr[k]; l < M_csc.col_ptr[k+1]; l++) {
			//printf("col %d index %d row %d value %d\n", k, l, M_csc.row_index[l], (int) M_csc.values[l]);
		}	
	}*/

	use_csr = 0;
	if(keep_initial != 0) {
		saved_initial_lam = malloc(sizeof(double) * (n+m));
	}

	openblas_set_num_threads(num_workers); 
	return solve(max_iter, tolerance);
}

double* get_initial_lam() {
	return saved_initial_lam;
}

double* full_sparse_interface(int num_rows, int num_cols, int8_t* csc_values, int* csc_row_index, int* csc_col_ptr, int8_t* csr_values, int* csr_col_index, int* csr_row_ptr, int num_workers, double tolerance, int max_iter) {
	m = num_rows;
	n = num_cols;
	M_csc.values = csc_values;
	M_csc.row_index = csc_row_index;
	M_csc.col_ptr = csc_col_ptr;
	M_csr.values = csr_values;
	M_csr.col_index = csr_col_index;
	M_csr.row_ptr = csr_row_ptr;
	use_csr = 1;

	openblas_set_num_threads(num_workers);
	return solve(max_iter, tolerance);
}

double* IMul_interface(int n1, int n2, int8_t* aux_array, int num_aux, int num_workers, double tolerance, int max_iter) {
	generate_CSC_constraints(n1, n2, aux_array, num_aux);
	openblas_set_num_threads(num_workers);
	double* result = solve(max_iter, tolerance);
	free_M_CSC();
	return result;
}

double* free_ptr(void* ptr) {
	free(ptr);
}

 /*
	* Main entry point for running stand-alone---this is basically just a test function designed to see if the solver works on a very small problem.
	*/
int main() {
	srand((unsigned int) time(NULL));
	int n1 = 3;
	int n2 = 4;
	int A = 3;
	int aux_array_size = (1 << (n1+n2)) * A;
	int8_t* aux_array = malloc(aux_array_size);
	for(int i=0; i<aux_array_size; i++) {
		aux_array[i] = rand() & 1;
	}
	generate_CSC_constraints(n1, n2, aux_array, A);
	openblas_set_num_threads(2);
	solve(200, 1e-8);
	free(aux_array);

	return 0;
}












