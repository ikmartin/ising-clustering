#include <stdlib.h>
#include <cblas.h>
#include <stdio.h>
#include <math.h>

typedef struct {
  char n1;
  char n2;
  char num_aux_spins;
  char* aux_array;
} Circuit;

// LAPACK routines
extern int dgetrf_(int* M, int* N, double* A, int* LDA, int* IPIV, int* INFO);
extern int dgetrs_(char* TRANS, int* N, int* NRHS, double* A, int* LDA, int* IPIV, double* B, int* LDB, int* INFO);

// External interface functions
double interface(double** constraints, int num_rows, int num_cols, int num_workers);

// Utility functions for dealing with the constraint matrix
void generate_coefficient_matrix(double* target, double* d, double* zinv);
void multiply_by_A(double* target, double* vector);
void multiply_by_At(double* target, double* vector);
void multiply_by_M(double* target, double* vector);
void multiply_by_Mt(double* target, double* vector);

// Main solver functions
double solve(int max_iter);
void solve_main_system();
void solve_AKAt(double* target, double* rhs, double* tmp);

// basic functions
double fmin(double a, double b);
double fmax(double a, double b);
int imax(int a, int b);
