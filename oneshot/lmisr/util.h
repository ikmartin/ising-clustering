#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>


#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))


/* Need this LAPACK routine for dense linear system solve.
 * Use -lopenblas Library flag when compiling.*/
int dgesv_(int*, int*, double*, int*, int*, double*, int*, int*);

/* Get number of seconds since Unix Epoch */
double get_time_in_seconds(void);

/* Store length n bit-wise representation of the integer   
 * k in bit_array */
void int2bit(int k, int n, bool *bit_array);

/* Read in values for ancilla spins. Spin value for
 * ith product of jth ancilla is stored in CC[i][j].  
 * True indicated by 1; false indicated by -1. 
 * Multiplication table is traversed such that products
 * in same column are adjacent in memory.*/
void readInAncilla(int N1, int N2, int na, bool **a, const char *filename);

/* Get collection of indices needed to compute outputs
 * for computeSumsAndDiffs. */
void getSumDiffInds(int N1, int N2, int na, int k, bool **compute_me);

/* Routine to free the index arrays from the previous
 * function. */
void freeInds(int N, int na, bool **compute_me);

/* For each product, get "correct" answer associated
 * with each column of original constraint matrix. */
void getCorrectColumns(int N1, int N2, int na, bool **a, bool **CC);


/* Compute quantities to be used to construct matrix for main
 * linear system solve.*/
void computeSumsAndDiffs(double *v0, double *v1, double **v2,
			 double ***v3, double ****v4, double *u, 
			int M, bool all_info, bool **compute_me);

/* Get submatrix of matrix needed for solving main linear 
 * system solve (for original problem) */
void getSumDiffMatrix(int N1, int N2, int na, bool **a, double *d2,
		      int ind, bool **CC, double **Mat, bool **compute_me);

/* Solve main linear system for new/modified problem. */
void solveSysMainReward(double *rhs1, double *rhs2, double *y1, double *y2,
                        int N1, int N2, int na, bool **a, double *d21, double *d22,
                        bool **CC, double **Grm, double *gr, 
			bool **compute_me, bool quick);

/* Solve main linear system for original problem.*/
void solveSysMain(double *x, int N1, int N2, int na, bool **a, double *d2,
		  bool **CC, double **Grm, bool **compute_me);

/* Compute [y1; y2] = [-B', 0; -I, -I] [x1; x2], where B = -A' is the 
 * constraint matrix in the original problem. */
void computeAxReward(double *x1, double *x2, int N1, int N2, int na, bool **a, bool **CC,
                     double *y1, double *y2, bool **compute_me);

/* Compute y = Ax, where B = -A' is the constraint matrix 
 * in the original problem. */
void computeAx(double *x, int N1, int N2, int na, bool **a, bool **CC,
	       double *y, bool **compute_me);

/* Compute [y1; y2] = [-B, -I; 0, -I] [x1; x2], where B = -A' is the 
 * constraint matrix in the original problem. */
void computeATxReward(double *x1, double *x2, int N1, int N2, int na, bool **a, bool **CC,
                      double *y1, double *y2);

/* Compute y = A'x, where B = -A' is the constraint matrix 
 * in the original problem. */
void computeATx(double *x, int N1, int N2, int na, bool **a, bool **CC,
	       double *y);

/* Computes initial guess/interior point for the new/modified problem. */
void getInitialGuess(int N1, int N2, int na, bool **a, double *x01, double *x02,
		     double *lam01, double *lam02, double *s01, double *s02, bool **CC,
		     double **Grm, bool **compute_me1,
		     bool **compute_me2, double *c1, double *c2,
		     double *b1, double *b2, double *xs1, double *xs2);

/* Euclidean norm */
double normTwo(double *x, int n);
