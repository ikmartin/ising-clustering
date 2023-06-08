#include <stdlib.h>

typedef struct {
  char n1;
  char n2;
  char num_aux_spins;
  char* aux_array;
  long num_constraint_cols;
  long num_constraint_rows;
  long num_A_rows;
  long num_A_cols;
} Circuit;

typedef struct {
  double* x;
  double* x2;
  double* s;
  dobule* s2;
  double* l;
  double* l2;
  double* dx;
  double* dx2;
  double* ds;
  double* ds2;
  double* dl;
  double* dl2;
  double* rc;
  double* rc2;
  // These are for the solver step (see documentation)
  double* D;
  double* Zinv;
  double* L;
  double* lu_decomp;
  int* pivot;
} OptimizerVariables;

void allocate_variables(OptimizerVariables* vars, Circuit* circuit) {
  vars->x = (double*) malloc(sizeof(double) * circuit->num_A_cols);
  // etc
}

void free_variables(OptimizerVariables* vars) {
  free(vars->x);
  // etc
}


double solve(Circuit* circuit, int max_iter) {
  OptimizerVariables vars;
  allocate_variables(&vars, circuit);


  // iteration variables
  int i, iteration;

  for(iteration = 0; iteration < max_iter; iteration++) {
    
    // r_c <- A^t lambda + s - c
    multiply_by_At(r_c, l);
    for(i = 0; i < 2*m; i++) {
      r_c[i] += s[i];
    }
    for(i = 0; i < m; i++) {
      r_c[i] -= v[i];
    }

    // L <- -x*s
    for(i = 0; i < 2*m; i++) {
      L[i] = -x[i] * s[i];
    }

    // d <- x1/s1
    // zinv <- 1/(x1/s1 + x2/s2)
    for(i = 0; i < m; i++) {
      d[i] = x[i]/s[i];
      zinv[i] = 1.0/(d[i] + x2[i]/s2[i]);
    }

    // Generate the LU decomposition for the system solve. Will be used for both the predictor and corrector step.
    get_coefficient_matrix(lu_decomp, d, zinv);
    dgetrf_(n, n, lu_decomp, n, pivot, info);

    // Solves the predictor system to calculate affine deltas.
    solve_system(dx, dl, ds, x, s, L, d, zinv, rc, lu_decomp, pivot, m, n);

    // Calculate affine alphas
    alpha_primal_affine = 1.0;
    alpha_dual_affine = 1.0;
    for(i = 0; i < 2*m; i++) {
      if(dx[i] < 0.0) {
        alpha_primal_affine = MIN(alpha_primal_affine, -x[i]/dx[i]);
      }
      if(ds[i] < 0.0) {
        alpha_dual_affine = MIN(alpha_dual_affine, -s[i]/ds[i]);
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
      mu += L[i];
    }
    mu /= (double)(-2*m);

    sigma = pow(affine_mu / mu, 3.0);

    // Modify L for the corrector step
    for(i = 0; i < 2*m; i++) {
      L[i] -= dx[i] * ds[i] - sigma * mu;
    }

    // Solves the corrector system.
    solve_system(dx, dl, ds, x, s, L, d, zinv, rc, lu_decomp, pivot, m, n);
    
    // Calculate alphas
    alpha_primal = 1000.0;
    alpha_dual = 1000.0;
    for(i = 0; i < 2*m; i++) {
      if(dx[i] < 0.0) {
        alpha_primal = MIN(alpha_primal, -x[i]/dx[i]);
      }
      if(ds[i] < 0.0) {
        alpha_dual = MIN(alpha_dual, -s[i]/ds[i]);
      }
    }
    alpha_primal = MIN(1, eta * alpha_primal);
    alpha_dual = MIN(1, eta * alpha_dual);

    // Update the variables
    for(i = 0; i < 2*m; i++) {
      x[i] += alpha_primal * dx[i];
      s[i] += alpha_dual * ds[i];
    }
    for(i = 0; i < n+m; i++) {
      l[i] += alpha_dual * dl[i];
    }

    // asymptotic modification of step size, formula copied from Teresa. 
    eta = 1.0 - 0.1 * pow(0.1, (double)(iteration + 1)/50.0);
  }
}

void solve_system(double* dx, double* dl, double* ds, double* x, double* s, double* L, double* d, double* zinv, double* rc, double* lu_decomp, int* pivot, int m, int n) {
    
  // iteration variables
  int i;

  // u <- b - A(x + S^-1 (X r_c + L))
  // We use ds as a temporary variable since it will be overwritten anyway
  // u is actually stored in dl so that the system solve can be done in place
  double* dl2 = dl+n;
  for(i = 0; i < 2*m; i++) {
    ds[i] = -(x[i] + (x[i]*rc[i] + L[i])/s[i]);
  }
  multiply_by_A(dl, ds);
  for(i = 0; i < m; i++) {
    dl2[i] -= 1;
  }

  // calculate RHS for system solve
  for(i = 0; i < m; i++) {
    dl2[i] *= d[i] * zinv[i];
  }
  multiply_by_Mt(ds, dl2);
  for(i = 0; i < n; i++) {
    dl[i] -= ds[i];
  }

  dgetrs_("No transpose", n, 1, lu_decomp, n, pivot, dl, 1, info);

  // dl2 <- zinv * (u2 - d * M dl1)
  multiply_by_M(ds, dl);
  for(i = 0; i < m; i++) {
    dl2[i] = zinv[i] * (dl2[i] - d[i] * ds[i]);
  }

  // After this point, ds is no longer a temporary variable!

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
