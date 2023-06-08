#include "include.h"

extern double** M;
extern int m;
extern int n;


double interface(double** constraints, int num_rows, int num_cols, int num_workers) {
	M = constraints;
	m = num_rows;
	n = num_cols;

	openblas_set_num_threads(num_workers); 
	return solve(200);
}

 /*
	* Main entry point for running stand-alone---this is basically just a test function designed to see if the solver works on a very small problem.
	*/
int main() {
	return 0;
}
