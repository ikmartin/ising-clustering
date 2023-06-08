#include "include.h"

extern double** M;
extern int m;
extern int n;

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
				target[i*n + j] += M[k][i] * (d[k] - d[k]*d[k]*zinv[k]) * M[k][j];
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
			target[i] += M[i][j] * vector[j];
		}
	}
}

void multiply_by_Mt(double* target, double* vector) {
	for(int i = 0; i < n; i++) {
		target[i] = 0.0;
		for(int j = 0; j < m; j++) {
			target[i] += M[j][i] * vector[j];
		}
	}
}

