#include "util.h"

/* Get number of seconds since Unix Epoch */
double get_time_in_seconds(void) {
    struct timeval tv;
    int _tod = gettimeofday(&tv, NULL);
    if (_tod != 0) abort();
    return (double) tv.tv_sec + tv.tv_usec * 1e-6;
}


/* Store length n bit-wise representation of the integer
 *  * k in bit_array */
void int2bit(int k, int n, bool *bit_array){

    int i;
    for(i = 0; i < n; i++){
	bit_array[i] = ((k >> (n-1-i)) & 1) != 0;
    }
}


/* Read in values for ancilla spins. Spin value for
 * ith product of jth ancilla is stored in CC[i][j].
 * True indicated by 1; false indicated by -1. 
 * Multiplication table is traversed such that products
 * in same column are adjacent in memory. */
void readInAncilla(int N1, int N2, int na, bool **a, const char *filename){

    int i, j;
    int N = N1+N2;
    int M = (int)round(pow(2,N));
    FILE *fp;
    int input;

    fp = fopen(filename,"r");
    for(i = 0; i < M; i++){
	for(j = 0; j < na; j++){
	    fscanf(fp,"%d",&input);
	    if(input == 1){
		a[i][j] = true;
	    }else if(input == -1){
		a[i][j] = false;
	    }else{
		fprintf(stderr,\
		"Error: Ancilla file contains value that is not +/-1.\n");
	    }
	}
    }
    fclose(fp);
}

/* Get collection of indices needed to compute outputs
 * for computeSumsAndDiffs. */
void getSumDiffInds(int N1, int N2, int na, int k, bool **compute_me){

    int i, j, l, p;
    int N = N1+N2;
    int M = N+na;
    int Imax;

    bool *ibit = (bool*)malloc(sizeof(bool)*M);

    for(j = 0; j < M; j++){

	Imax = (int)round(pow(2,j));
	
	if(j <= k){
	    for(i = 0; i < Imax; i++){
		compute_me[j][i] = true;
	    }
	}else{

	    for(i = 0; i < Imax; i++){
	
	        int2bit(i,j,ibit);
	        p = 0;
	        for(l = 0; l < j; l++){
		    if(ibit[l]) p++;
	        }

	        if(p > k){
		    compute_me[j][i] = false;
	        }else{
		    compute_me[j][i] = true;
	        }
	    }
	}
    }
    free(ibit);

}

/* Routine to free the index arrays from the previous
 * function. */
void freeInds(int N, int na, bool **compute_me){

    int i;
    int M = N+na;

    for(i = 0; i < M; i++){
	free(compute_me[i]);

    }
    free(compute_me);
}


/* For each product, get "correct" answer associated
 * with each column of original constraint matrix. */
void getCorrectColumns(int N1, int N2, int na, bool **a, bool **CC){

    int i, j, k, p, q;
    int N = N1+N2;
    int sz =  (N*(3*N+1))/2;
    int n1 = (int)round(pow(2,N));
    int n2 = (int)round(2*N+na);
    int n3 = (int)round(pow(2,N1));
    int n4 = (int)round(pow(2,N2));

    bool *i_bit = (bool*)malloc(sizeof(bool)*N1);
    bool *j_bit = (bool*)malloc(sizeof(bool)*N2);
    bool *p_bit = (bool*)malloc(sizeof(bool)*N);

    bool **C = (bool**)malloc(sizeof(bool*)*n1);
    for(i = 0; i < n1; i++){
	C[i] = (bool*)malloc(sizeof(bool)*n2);
    }

    q = 0;
    for(j = 0; j < n4; j++){

	int2bit(j,N2,j_bit);    

	for(i = 0; i < n3; i++){

	    int2bit(i,N1,i_bit);    

	    p = i*j;
	    int2bit(p,N,p_bit);

	    for(k = 0; k < N1; k++){
		C[q][    k] = i_bit[k];
	    }
	    for(k = 0; k < N2; k++){
		C[q][N1+k] = j_bit[k];
	    }
	    for(k = 0; k < N; k++){
		C[q][N+k] = p_bit[k];
	    }
	    for(k = 0; k < na; k++){
		C[q][2*N+k] = a[q][k];
	    }
	    q++;
	}
    }

    for(q = 0; q < n1; q++){

	for(i = 0; i < N+na; i++){
	    CC[q][i] = C[q][N+i]; 
	}

	k = N+na;
        for(i = 0; i < N; i++){
	    for(j = N; j < 2*N+na; j++){
		CC[q][k] = C[q][i] == C[q][j];
		k++;
	    }
	}

        for(i = N; i < 2*N+na; i++){
	    for(j = i+1; j < 2*N+na; j++){
		CC[q][k] = C[q][i] == C[q][j];
		k++;
	    }
	}
    }


    for(i = 0; i < n1; i++){
	free(C[i]);
    }
    free(C);
    free(i_bit);
    free(j_bit);
    free(p_bit);
}

/* Compute quantities to be used to construct matrix for main
 * linear system solve.*/
void computeSumsAndDiffs(double *v0, double *v1, double **v2,
                         double ***v3, double ****v4, double *u, int M,
			 bool all_info, bool **compute_me){

    int i, j, k, l, p;
    int Imax, Jmax;
    int iter;
    double *w = (double*)malloc(sizeof(double)*round(pow(2.0,(double)M)));

    for(iter = 0; iter < M; iter++){

	Imax = (int)round(pow(2.0,(double)(iter)));
	Jmax = (int)round(pow(2.0,(double)(M-iter-1)));

	for(i = 0; i < Imax; i++){
	    if(compute_me[iter][i]){
	    p = 2*Jmax*i;
	    for(j = 0; j < Jmax; j++){
		w[p+j] = u[p+j];
	    }
	    for(j = 0; j < Jmax; j++){
		w[p+j] += u[p+Jmax+j];
	    }
	    for(j = 0; j < Jmax; j++){
		w[p+Jmax+j] = -u[p+j];
	    }
	    for(j = 0; j < Jmax; j++){
		w[p+Jmax+j] += u[p+Jmax+j];
	    }
	    }
	}	
	Imax = (int)round(pow(2.0,(double)M));
	for(i = 0; i < Imax; i++){
	    u[i] = w[i];
	}
    }
    (*v0) = w[0];
   
    for(i = 0; i < M; i++){
	v1[i] = w[(int)round(pow(2.0,(double)(M-i-1)))];
    }

    for(i = 0; i < M; i++){
	for(j = i+1; j < M; j++){
	    v2[i][j-i-1] = w[(int)round(pow(2.0,(double)(M-i-1))) + \
                             (int)round(pow(2.0,(double)(M-j-1)))];
	}
    }

    if(all_info){
    for(i = 0; i < M; i++){
	for(j = i+1; j < M; j++){
	    for(k = j+1; k < M; k++){
		v3[i][j-i-1][k-j-1] = \
			      w[(int)round(pow(2.0,(double)(M-i-1))) + \
                 	        (int)round(pow(2.0,(double)(M-j-1))) + \
                 	        (int)round(pow(2.0,(double)(M-k-1)))];
	    }
	}
    }

    for(i = 0; i < M; i++){
	for(j = i+1; j < M; j++){
	    for(k = j+1; k < M; k++){
		for(l = k+1; l < M; l++){
		v4[i][j-i-1][k-j-1][l-k-1] = \
				 w[(int)round(pow(2.0,(double)(M-i-1))) + \
                 	           (int)round(pow(2.0,(double)(M-j-1))) + \
                 	           (int)round(pow(2.0,(double)(M-k-1))) + \
                 	           (int)round(pow(2.0,(double)(M-l-1)))];
		}
	    }
	}
    }
    }

    free(w);
}

/* Compute quantities to be used to construct matrix for main
 * linear system solve.*/
void getSumDiffMatrix(int N1, int N2, int na, bool **a, double *d2,
                      int ind, bool **CC, double **Mat, bool **compute_me){


    int i, j, k, l, q;
    int N = N1+N2;
    int iq, kq;
    int sz = (N*(3*N+1))/2;
    for(i = 0; i < na; i++){
	sz += 2*N+i+1;
    }
 
    int M = N+na;
    int K = (M*(M+1))/2;
    int *mat_set = (int*)malloc(sizeof(int)*(sz-N*M));

    for(i = 0; i < M; i++){
	mat_set[i] = i;
    }
    for(i = M*(N+1); i < sz; i++){
	mat_set[i-M*N] = i;
    }

    double **Mat1 = (double**)malloc(sizeof(double*)*K);
    for(i = 0; i < K; i++){
	Mat1[i] = (double*)malloc(sizeof(double)*K);
    }

    double **Mat2 = (double**)malloc(sizeof(double*)*K);
    for(i = 0; i < K; i++){
	Mat2[i] = (double*)malloc(sizeof(double)*K);
    }

    double **Mat3 = (double**)malloc(sizeof(double*)*K);
    for(i = 0; i < K; i++){
	Mat3[i] = (double*)malloc(sizeof(double)*K);
    }

    double     v0;
    double    *v1 = (double   *)malloc(sizeof(double   )*M);
    double   **v2 = (double  **)malloc(sizeof(double  *)*(M-1));

    for(i = 0; i < M-1; i++){
	v2[i] = (double*)malloc(sizeof(double)*(M-i-1));
    } 

    double  ***v3 = (double ***)malloc(sizeof(double **)*(M-2));

    for(i = 0; i < M-2; i++){
	v3[i] = (double**)malloc(sizeof(double*)*(M-i-2));
    } 

    for(i = 0; i < M-2; i++){
	for(j = 0; j < (M-i-2); j++){
	    v3[i][j] = (double*)malloc(sizeof(double)*(M-i-j-2));
	}
    } 

    double ****v4 = (double****)malloc(sizeof(double***)*(M-3));


    for(i = 0; i < M-3; i++){
	v4[i] = (double***)malloc(sizeof(double**)*(M-i-3));
    } 

    for(i = 0; i < M-3; i++){
	for(j = 0; j < (M-i-3); j++){
	    v4[i][j] = (double**)malloc(sizeof(double*)*(M-i-j-3));
	}
    } 
    for(i = 0; i < M-3; i++){
	for(j = 0; j < (M-i-3); j++){
	    for(k = 0; k < (M-i-j-3); k++){
		v4[i][j][k] = (double*)malloc(sizeof(double)*(M-i-j-k-3));
	    }
	}
    } 


    computeSumsAndDiffs(&v0, v1, v2, v3, v4, d2, M, true, compute_me);

    for(i = 0; i < K; i++){
	for(j = 0; j < K; j++){
	    Mat1[i][j] = i ==j ? v0 : 0;
	}
    }

    for(i = 0; i < M; i++){

	for(j = i+1; j < M; j++){
	    Mat1[i][j] = v2[i][j-i-1]; 
	    if((i >= N) && !(a[ind][i-N])){
		Mat1[i][j] *= -1.0;
	    }
	    if((j >= N) && !(a[ind][j-N])){
		Mat1[i][j] *= -1.0;
	    }
	}

	q = M;
	for(j = 0; j < M; j++){
	    for(k = j+1; k < M; k++){
		if(i == j){
		    Mat1[i][q] = v1[k];
		}else if(i == k){
		    Mat1[i][q] = v1[j];
		}else{
		    Mat1[i][q] = \
		    v3[MIN(i,j)][MIN(MAX(i,j),k)-MIN(i,j)-1][MAX(i,k)-MIN(MAX(i,j),k)-1];
		}
		if((i >= N) && !(a[ind][i-N])){
		    Mat1[i][q] *= -1.0;
		}
		if((j >= N) && !(a[ind][j-N])){
		    Mat1[i][q] *= -1.0;
		}
		if((k >= N) && !(a[ind][k-N])){
		    Mat1[i][q] *= -1.0;
		}
		q++;
	    }
	}
    }

    iq = M;
    for(i = 0; i < M; i++){
    for(j = i+1; j < M; j++){

	kq = iq+1;

	k = i;
	for(l = j+1; l < M; l++){
	    Mat1[iq][kq] = v2[j][l-j-1];
	    if((j >= N) && !(a[ind][j-N])){
		Mat1[iq][kq] *= -1.0;
	    }
	    if((l >= N) && !(a[ind][l-N])){
		Mat1[iq][kq] *= -1.0;
	    }
	    kq++;
	}

	for(k = i+1; k < M; k++){
	    for(l = k+1; l < M; l++){
		if(j == k){
		    Mat1[iq][kq] = v2[i][l-i-1];
		}else if(j == l){
		    Mat1[iq][kq] = v2[i][k-i-1];
		}else{
		    Mat1[iq][kq] = \
		    v4[i][MIN(j,k)-i-1][MIN(l,MAX(j,k))-MIN(j,k)-1]\
		      [MAX(j,l)-MIN(l,MAX(j,k))-1];
		}
		if((i >= N) && !(a[ind][i-N])){
		    Mat1[iq][kq] *= -1.0;
		}
		if((j >= N) && !(a[ind][j-N])){
		    Mat1[iq][kq] *= -1.0;
		}
		if((k >= N) && !(a[ind][k-N])){
		    Mat1[iq][kq] *= -1.0;
		}
		if((l >= N) && !(a[ind][l-N])){
		    Mat1[iq][kq] *= -1.0;
		}

		kq++;
	    }
	}

	iq++;
    }
    }
    for(i = 0; i < K; i++){
	for(j = i+1; j < K; j++){
	    Mat1[j][i] = Mat1[i][j]; 
	}
    }

    // Set up Mat2
    for(i = 0; i < M; i++){
	for(j = 0; j < K; j++){
	    Mat2[i][j] = v1[i]; 
	}
	if((i >= N) && !a[ind][i-N]){
	    for(j = 0; j < K; j++){
		Mat2[i][j] *= -1.0; 
	    }
	}
    }
    
    q = M;
    for(i = 0; i < M; i++){
    for(j = i+1; j < M; j++){
	
	for(l = 0; l < K; l++){
	    Mat2[q][l] = v2[i][j-i-1];
	}
	if((i >= N) && !(a[ind][i-N])){
	    for(l = 0; l < K; l++){
		Mat2[q][l] *= -1.0;
	    }
	}
	if((j >= N) && !(a[ind][j-N])){
	    for(l = 0; l < K; l++){
		Mat2[q][l] *= -1.0;
	    }
	}
	q++;
    }
    }
    for(i = 0; i < K; i++){
	for(j = 0; j < K; j++){
	    if(!CC[ind][mat_set[j]]){
		Mat2[i][j] *= -1.0;
	    }
	}
    }

    // Set up Mat3
    for(i = 0; i < K; i++){
	for(j = 0; j < K; j++){
	    Mat3[i][j] = v0;
	    if(CC[ind][mat_set[i]] != CC[ind][mat_set[j]]){
		Mat3[i][j] *= -1.0;
	    }
	}
    }

    for(i = 0; i < K; i++){
	for(j = 0; j < K; j++){
	    Mat[i][j]  = 1.0/4.0*(Mat1[i][j]+Mat3[i][j]);
	    Mat[i][j] -= 1.0/4.0*(Mat2[i][j]+Mat2[j][i]);
	}
    }

    for(i = 0; i < M-3; i++){
        for(j = 0; j < M-i-3; j++){
            for(k = 0; k < M-i-j-3; k++){
                free(v4[i][j][k]);
            }
            free(v4[i][j]);
        }
        free(v4[i]);
    }
    free(v4);


    for(i = 0; i < M-2; i++){
        for(j = 0; j < M-i-2; j++){
            free(v3[i][j]);
        }
        free(v3[i]);
    }
    free(v3);

    for(i = 0; i < M-1; i++){
        free(v2[i]);
    }
    free(v2);

    free(v1);



    free(mat_set);

    for(i = 0; i < K; i++){
	free(Mat1[i]);
	free(Mat2[i]);
	free(Mat3[i]);
    }
    free(Mat1);
    free(Mat2);
    free(Mat3);
}

/* Solve main linear system for new/modified problem. */
void solveSysMainReward(double *rhs1, double *rhs2, double *y1, double *y2, 
                        int N1, int N2, int na, bool **a, double *d21, double *d22, 
			bool **CC, double **Grm, double *gr, 
			bool **compute_me, bool quick){

		int i, j, q;
		int info;
		int N = N1 + N2;
		int M = 2*N+na;
		int L = (int)round(pow(2,M));
		int sz = (N*(3*N+1))/2;

		for(i = 0; i < na; i++){
				sz += 2*N+i+1;
		}

		int *ipiv= malloc(sizeof(int)*sz);

		for(i = 0; i < L; i++){
				rhs2[i] /= (d21[i] + d22[i]); 
		}

		for(i = 0; i < L; i++){
				y2[i] = rhs2[i]*d21[i];
		}

		
		// B is the constraint matrix in the original problem (or is it -B? unclear)
		// D is apparently the diagonal matrix (d21 + d22) / d21
		computeAx(y2, N1, N2, na, a, CC, y1, compute_me); // y1 = -B D^-1 rhs2

		if(!quick){
				for(i = 0; i < L; i++){
						y2[i] = d21[i] - d21[i]*d21[i]/(d21[i] + d22[i]);
				}
				solveSysMain(rhs1, N1, N2, na, a, y2, CC, Grm, compute_me); // rhs1 <- (A - B D^-1 C)^-1 rhs1
		}else{
				q = 0;
				for(i = 0; i < sz; i++){
						for(j = 0; j < sz; j++){
								gr[q] = Grm[i][j];
								q++;
						}
				}
				i = 1;
				dgesv_(&sz, &i, gr, &sz, ipiv, rhs1, &sz, &info);
		}
		q = 0;
		for(i = 0; i < sz; i++){
				for(j = 0; j < sz; j++){
						gr[q] = Grm[i][j];
						q++;
				}
		}
		i = 1;
		dgesv_(&sz, &i, gr, &sz, ipiv, y1, &sz, &info); // y1 <- -(A - B D^-1 C)^-1 B D^-1 rhs2

		for(i = 0; i < sz; i++){
				y1[i] += rhs1[i];  // y1 = (A - B D^-1 C)^-1 rhs1 - (A - B D^-1 C)^-1 B D^-1 rhs2
		}

		computeATx(y1, N1, N2, na, a, CC, y2); 

		for(i = 0; i < L; i++){
				y2[i] *= d21[i]/(d21[i]+d22[i]); // y2 = - D^-1 C (A - B D^-1 C)^-1 rhs1 
							 //      + D^-1 C (A - B D^-1 C)^-1 B D^-1 rhs2
		}

		for(i = 0; i < L; i++){
				y2[i] += rhs2[i];
		}

		free(ipiv);

}


/* Solve main linear system for original problem.*/
void solveSysMain(double *x, int N1, int N2, int na, bool **a, double *d2,
                  bool **CC, double **Grm, bool **compute_me){

    int i, j, k, l, p, q;
    int info;
    int N = N1+N2;
    int sz = (N*(3*N+1))/2; 
    int M = N+na;
    int K = (M*(M+1))/2;
    int L = (int)round(pow(2,M));
    int Imax = (int)round(pow(2,N1));
    int Jmax = (int)round(pow(2,N2));
    bool *ibit = (bool*)malloc(sizeof(bool)*N1);
    bool *jbit = (bool*)malloc(sizeof(bool)*N2);
    bool *my_input = (bool*)malloc(sizeof(bool)*(N+1));
    int ind;
    int num = (N+1)*M;
    double value;

    for(i = 0; i < na; i++){   
	sz += 2*N+i+1;
    }

    int *ipiv = (int*)malloc(sizeof(int)*sz);
    double *gr = (double *)malloc(sizeof(double)*sz*sz);
    double *d2_mini = (double*)malloc(sizeof(double)*L);


    double **Mat = (double**)malloc(sizeof(double*)*K);
    for(i = 0; i < K; i++){
	Mat[i] = (double*)malloc(sizeof(double)*K);
    }

    for(i = 0; i < sz; i++){
	for(j = 0; j < sz; j++){
	    Grm[i][j] = 0;
	}
    }


    for(j = 0; j < Jmax; j++){

	int2bit(j,N2,jbit);	

	for(i = 0; i < Imax; i++){

	    int2bit(i,N1,ibit);	
	    my_input[0] = true;

	    for(k = 0; k < N1; k++){
		my_input[   k+1] = ibit[k];
	    }
	    for(k = 0; k < N2; k++){
		my_input[N1+k+1] = jbit[k];
	    }

	    ind = Imax*j+i;

	    for(k = 0; k < L; k++){
		d2_mini[k] = d2[ind*L+k];
	    }

	    getSumDiffMatrix(N1, N2, na, a, d2_mini, ind, CC, Mat, compute_me);

	    for(k = 0; k < (N+1); k++){
		for(l = 0; l < (N+1); l++){

		    if(my_input[k] == my_input[l]){

			for(p = 0; p < M; p++){
			    for(q = 0; q < M; q++){
				Grm[k*M+p][l*M+q] += Mat[p][q];
			    }
			}	

		    }else{

			for(p = 0; p < M; p++){
			    for(q = 0; q < M; q++){
				Grm[k*M+p][l*M+q] -= Mat[p][q];
			    }
			}	

		    }
		}
	    }

	    for(k = 0; k < (N+1); k++){
		for(l = 0; l < M; l++){
		    if(my_input[k]){
			for(p = 0; p < (sz-num); p++){
			    Grm[k*M+l][num+p] += Mat[l][M+p];
			}
		    }else{
			for(p = 0; p < (sz-num); p++){
			    Grm[k*M+l][num+p] -= Mat[l][M+p];
			}
		    }
		}
	    }

	    for(k = 0; k < (sz-num); k++){
		for(l = 0; l < (sz-num); l++){
 		    Grm[num+k][num+l] += Mat[M+k][M+l];
		}
	    }	

	}

    }

    for(i = 0; i < sz; i++){
	for(j = i+1; j < sz; j++){
	    Grm[j][i] = Grm[i][j];
	}
    }



    q = 0;
    for(i = 0; i < sz; i++){
	for(j = 0; j < sz; j++){
	    gr[q] = Grm[i][j];
	    q++; 
	}
    }
    i = 1;
    dgesv_(&sz, &i, gr, &sz, ipiv, x, &sz, &info);	

    for(i = 0; i < K; i++){
	free(Mat[i]);
    }
    free(Mat);
    free(d2_mini);
    free(ibit);
    free(jbit);
    free(my_input);
    free(ipiv);
    free(gr);

}

/* Compute [y1; y2] = [-B', 0; -I, -I] [x1; x2], where B = -A' is the 
 * constraint matrix in the original problem. */
void computeAxReward(double *x1, double *x2, int N1, int N2, int na, bool **a, bool **CC,
		     double *y1, double *y2, bool **compute_me){

    int i;
    int N = N1+N2;
    int M = 2*N+na;
    int L = (int)round(pow(2,M));

    computeAx(x1, N1, N2, na, a, CC, y1, compute_me);

    for(i = 0; i < L; i++){
	y2[i] = -x1[i];
        y2[i] -= x2[i];

    }

}

/* Compute y = Ax, where B = -A' is the constraint matrix 
 * in the original problem. */
void computeAx(double *x, int N1, int N2, int na, bool **a,
	       bool **CC, double *y, bool **compute_me){


    int i, j, k, l, p, q;
    int N = N1+N2;
    int M = N+na;
    int L = (int)round(pow(2,N+na));

    int sz = (N*(3*N+1))/2;
    for(i = 0; i < na; i++){
	sz += 2*N+i+1;
    }
    int Imax = (int)round(pow(2,N1));
    int Jmax = (int)round(pow(2,N2));
    bool *ibit = (bool*)malloc(sizeof(bool)*N1);
    bool *jbit = (bool*)malloc(sizeof(bool)*N2);
    bool *my_input = (bool*)malloc(sizeof(bool)*(N+1));
    int ind;
    int num = (N+1)*M;

    double *x_mini = (double*)malloc(sizeof(double)*L);
    double   v0;
    double  *v1 = (double *)malloc(sizeof(double)*M);
    double **v2 = (double **)malloc(sizeof(double*)*(M-1)); 

    for(i = 0; i < M-1; i++){
        v2[i] = (double*)malloc(sizeof(double)*(M-i-1));
    }
    double ***v3, ****v4;


    for(i = 0; i < sz; i++){
	y[i] = 0;
    }

    for(j = 0; j < Jmax; j++){

        int2bit(j,N2,jbit);
	for(i = 0; i < Imax; i++){

            int2bit(i,N1,ibit);
            my_input[0] = true;

            for(k = 0; k < N1; k++){
                my_input[   k+1] = ibit[k];
            }
            for(k = 0; k < N2; k++){
                my_input[N1+k+1] = jbit[k];
            }
	
	    ind = Imax*j+i;
	    for(k = 0; k < L; k++){
		x_mini[k] = x[ind*L+k]; 
	    }

	    computeSumsAndDiffs(&v0, v1, v2, v3, v4, x_mini,\
				 M, false, compute_me);

	    q = 0;
	    for(k = 0; k < (N+1); k++){
		for(l = 0; l < N; l++){
		    if(my_input[k]){
			if(CC[ind][q]){
			    y[q] += (v1[l]-v0);
			}else{
			    y[q] += (v1[l]+v0);
			}
		    }else{
			if(CC[ind][q]){
			    y[q] -= (v1[l]+v0);
			}else{
			    y[q] -= (v1[l]-v0);
			}
		    }
		    q++;
		}	
		for(l = N; l < M; l++){
		    if(my_input[k] == a[ind][l-N]){
			if(CC[ind][q]){
			    y[q] += (v1[l]-v0);
			}else{
			    y[q] += (v1[l]+v0);
			}
		    }else{
			if(CC[ind][q]){
			    y[q] -= (v1[l]+v0);
			}else{
			    y[q] -= (v1[l]-v0);
			}
		    }
		    q++;
		}	
	    }
	    for(k = 0; k < N; k++){
		for(l = k+1; l < N; l++){
		    if(CC[ind][q]){
                        y[q] += (v2[k][l-k-1]-v0);
                    }else{
                        y[q] += (v2[k][l-k-1]+v0);
                    }
		    q++;
		}
		for(l = N; l < M; l++){
		    if(a[ind][l-N]){
			if(CC[ind][q]){
			    y[q] += (v2[k][l-k-1]-v0);
			}else{
			    y[q] += (v2[k][l-k-1]+v0);
			}
		    }else{
			if(CC[ind][q]){
			    y[q] -= (v2[k][l-k-1]+v0);
			}else{
			    y[q] -= (v2[k][l-k-1]-v0);
			}
		    }
		    q++;
		}	
	    }
	    for(k = N; k < M; k++){
		for(l = k+1; l < M; l++){
		    if(a[ind][k-N] == a[ind][l-N]){
			if(CC[ind][q]){
			    y[q] += (v2[k][l-k-1]-v0);
			}else{
			    y[q] += (v2[k][l-k-1]+v0);
			}
		    }else{
			if(CC[ind][q]){
			    y[q] -= (v2[k][l-k-1]+v0);
			}else{
			    y[q] -= (v2[k][l-k-1]-v0);
			}
		    }
		    q++;
		}
	    }

	}
    }

    for(i = 0; i < sz; i++){
	y[i] /= -2.0;
    }

    for(i = 0; i < M-1; i++){
        free(v2[i]);
    }
    free(v2);
    free(v1);

    free(ibit);
    free(jbit);
    free(my_input);
    
    free(x_mini);
}

/* Compute [y1; y2] = [-B, -I; 0, -I] [x1; x2], where B = -A' is the 
 * constraint matrix in the original problem. */
void computeATxReward(double *x1, double *x2, int N1, int N2, int na, bool **a, bool **CC,
		      double *y1, double *y2){

    int i;
    int N = N1+N2;
    int M = 2*N+na;
    int L = (int)pow(2,M);

    computeATx(x1, N1, N2, na, a, CC, y1);
    for(i = 0; i < L; i++){
	y2[i] = -x2[i];
	y1[i]+=  y2[i];
    } 

}

/* Compute y = A'x, where B = -A' is the constraint matrix
 * in the original problem. */
void computeATx(double *x, int N1, int N2, int na, bool **a, bool **CC,
               double *y){

    int i, j, k, l, m, p, q;
    int Lmax, Pmax, xind;
    int N = N1+N2;
    int M = N+na;
    int M1, M2;
    int L  = (int)pow(2,M);
    int LL = (int)pow(2,N+M);
    int Imax = (int)round(pow(2,N1));
    int Jmax = (int)round(pow(2,N2));
    bool *ibit = (bool*)malloc(sizeof(bool)*N1);
    bool *jbit = (bool*)malloc(sizeof(bool)*N2);
    bool *lbit = (bool*)malloc(sizeof(bool)*(N+na));
    bool *my_input = (bool*)malloc(sizeof(bool)*(N+1));
    int ind;
    int r, s, t;
    bool temp;
    bool **outbits = (bool**)malloc(sizeof(bool*)*M);
    for(i = 0; i < M; i++){
	outbits[i] = (bool*)malloc(sizeof(bool)*L);
    }

    double **ypart = (double**)malloc(sizeof(double*)*(M+1));
    for(i = 0; i <= M; i++){
	ypart[i] = (double*)malloc(sizeof(double)*((int)pow(2.0,i)));
    }

    int sz = (N*(3*N+1))/2;
    for(i = 0; i < na; i++){
        sz += 2*N+i+1;
    }

    for(k = 0; k < M; k++){
	Lmax = (int)pow(2,k);
	Pmax = L/Lmax/2;
	for(l = 0; l < Lmax; l++){
	    q = l*Pmax*2;
    for(p = 0; p < Pmax; p++){
		outbits[k][q+p] = false;	
    	    } 
	    q += Pmax;
	    for(p = 0; p < Pmax; p++){
		outbits[k][q+p] = true;
	    } 
	}
    }

    for(j = 0; j < Jmax; j++){

        int2bit(j,N2,jbit);

        for(i = 0; i < Imax; i++){
            int2bit(i,N1,ibit);
            my_input[0] = true;

            for(k = 0; k < N1; k++){
                my_input[   k+1] = ibit[k];
            }
            for(k = 0; k < N2; k++){
                my_input[N1+k+1] = jbit[k];
            }

            ind = Imax*j+i;
	    ypart[0][0] = 0;
	    for(k = 0; k < sz; k++){
		if(CC[ind][k]){
		    ypart[0][0] -=x[k];
		}else{
		    ypart[0][0] +=x[k];
		}
	    }
	    for(m = 0; m < M; m++){
		M1 = (int)pow(2.0,(double)m);
		M2 = (int)pow(2.0,(double)(m+1));
		r = m+1;
		for(l = 0; l < M1; l++){
		    s = 2*l;
		    ypart[r][s  ] = ypart[m][l];
		    ypart[r][s+1] = ypart[m][l];

		}
		for(k = 0; k < (N+1); k++){

		    xind = M*k+m;
		    if(m < N){
		    if(my_input[k]){
			for(s = 0; s < M2; s = s+2){
			    ypart[r][s  ] -= x[xind];
			    ypart[r][s+1] += x[xind];
			}
		    }else{
			for(s = 0; s < M2; s = s+2){
			    ypart[r][s  ] += x[xind];
			    ypart[r][s+1] -= x[xind];
			}
		    }
		    }else{
		    if(a[ind][m-N] == my_input[k]){
			for(s = 0; s < M2; s = s+2){
			    ypart[r][s  ] -= x[xind];
			    ypart[r][s+1] += x[xind];
			}
		    }else{
			for(s = 0; s < M2; s = s+2){
			    ypart[r][s  ] += x[xind];
			    ypart[r][s+1] -= x[xind];
			}
		    }
	
		    }
		}

		for(k = 0; k < m; k++){
		    xind = (N+1)*M-1;
		    for(l = 0; l < k; l++){
			xind += M-l-1;
		    }
		    xind += (m-k);
		    p = M-1;
		    q = M-(m-k);
		    if(m < N){
			for(l = 0; l < M1; l++){
			    s = 2*l;
			    if(outbits[q][l]){
				ypart[r][s  ] -= x[xind];
				ypart[r][s+1] += x[xind];
			    }else{
				ypart[r][s  ] += x[xind];
				ypart[r][s+1] -= x[xind];
			    }	
			}
		    }else if(k < N){
			temp = a[ind][m-N];
			for(l = 0; l < M1; l++){
			    s = 2*l;
			    if(temp == outbits[q][l]){
				ypart[r][s  ] -= x[xind];
				ypart[r][s+1] += x[xind];
			    }else{
				ypart[r][s  ] += x[xind];
				ypart[r][s+1] -= x[xind];
			    }	
			}
		    }else{
			temp = a[ind][m-N] == a[ind][k-N];
			for(l = 0; l < M1; l++){
			    s = 2*l;
			    if(temp == outbits[q][l]){
				ypart[r][s  ] -= x[xind];
				ypart[r][s+1] += x[xind];
			    }else{
				ypart[r][s  ] += x[xind];
				ypart[r][s+1] -= x[xind];
			    }	
			}
		    }
		}
	    }

	    q = ind*L;
	    for(l = 0; l < L; l++){
		y[q+l] = -ypart[M][l]/2.0;
	    }
	}
    }

    for(i = 0; i < M; i++){
	free(outbits[i]);
	free(ypart[i]);
    }
    free(ypart[M]);
    free(outbits);
    free(ypart);
    free(ibit);
    free(jbit);
    free(my_input);	
    free(lbit);
}

/* Computes initial guess/interior point for the new/modified problem. */
void getInitialGuess(int N1, int N2, int na, bool **a, double *x01, double *x02,
                     double *lam01, double *lam02, double *s01, double *s02, bool **CC,
                     double **Grm, bool **compute_me1,
                     bool **compute_me2, double *c1, double *c2,
                     double *b1, double *b2, double *xs1, double *xs2){

    int i, j, l, q, p;
    int N = N1 + N2;
    int M = (int)(pow(4,N)*pow(2,na));
    int K2 = (int)(pow(2,N+na));
    int R1 = (int)(pow(2,N1)); 
    int R2 = (int)(pow(2,N2));
    int P = (int)(pow(2,na));

    int info;
    double dx, ds, xs_sum, x_sum, s_sum;
    bool quick = false;
    int sz = (N*(3*N+1))/2;
    for(i = 0; i < na; i++){
	sz += 2*N+i+1;
    }
    int *ipiv = (int*)malloc(sizeof(int)*sz);

    double *gr = (double*)malloc(sizeof(double)*sz*sz);

    for(i = 0; i < M; i++){
	c1[i] = -1.0;
    }
    for(i = 0; i < M; i++){
	c2[i] = 0.0;
    }
    for(i = 0; i < sz; i++){
	b1[i] = 0.0;
    }
    for(i = 0; i < M; i++){
	b2[i] = -1.0;
    }
    for(i = 0; i < M; i++){
	xs1[i] = 1.0;
    }
    for(i = 0; i < M; i++){
	xs2[i] = 1.0;
    }

    q = 0;
    for(j = 0; j < R2; j++){
        for(i = 0; i < R1; i++){
            p = i*j;
            for(l = 0; l < P; l++){
                c1[K2*q+P*p+l] = 0.0;
                b2[K2*q+P*p+l] = 0.0;
            }
            q++;
        }
    }
    
    solveSysMainReward(b1, b2, s01, s02, N1, N2, na, a, xs1, xs2, CC, Grm, gr, compute_me1, quick);
    quick = true;

    computeATxReward(s01, s02, N1, N2, na, a, CC, x01, x02);

    computeAxReward(c1, c2, N1, N2, na, a, CC, s01, s02, compute_me2);

    solveSysMainReward(s01, s02, lam01, lam02, N1, N2, na, a, xs1, xs2, CC, Grm, gr, compute_me1, quick);


    for(i = 0; i < sz; i++){
	b1[i] = -1.0*lam01[i];
    }
    for(i = 0; i < M; i++){
	b2[i] = -1.0*lam02[i];
    }
    computeATxReward(b1, b2, N1, N2, na, a, CC, s01, s02);
    for(i = 0; i < M; i++){
	s01[i] += c1[i];
    }

    dx = 1e100;
    for(i = 0; i < M; i++){
	dx = MIN(dx,x01[i]);
    }
    for(i = 0; i < M; i++){
	dx = MIN(dx,x02[i]);
    }
    dx = MAX(-3.0/2.0*dx,0.0);

    if(dx > 0){
        for(i = 0; i < M; i++){
	    x01[i] += dx;
	    x02[i] += dx;
        }
    }

    ds = 1e100;
    for(i = 0; i < M; i++){
	ds  = MIN(ds,s01[i]);
    }
    for(i = 0; i < M; i++){
	ds  = MIN(ds,s02[i]);
    }
    ds = MAX(-3.0/2.0*ds,0.0);
    if(ds > 0){
        for(i = 0; i < M; i++){
	    s01[i] += ds;
	    s02[i] += ds;
        }
    }

    xs_sum = 0.0;
    for(i = 0; i < M; i++){
	xs_sum += x01[i]*s01[i];
    }
    for(i = 0; i < M; i++){
	xs_sum += x02[i]*s02[i];
    }

    x_sum = 0.0;
    for(i = 0; i < M; i++){
	x_sum += x01[i];
    }
    for(i = 0; i < M; i++){
	x_sum += x02[i];
    }

    s_sum = 0.0;
    for(i = 0; i < M; i++){
	s_sum += s01[i];
    }
    for(i = 0; i < M; i++){
	s_sum += s02[i];
    }

    dx = xs_sum/s_sum/2.0;
    ds = xs_sum/x_sum/2.0;

    for(i = 0; i < M; i ++){
	x01[i] += dx;
	x02[i] += dx;
	s01[i] += ds;
	s02[i] += ds;
    }

    free(gr);
    free(ipiv);
}

/* Euclidean norm */
double normTwo(double *x, int n){

    int i;
    double norm2 = 0.0;

    for(i = 0; i < n; i++){
	norm2 += x[i]*x[i];
    }

    norm2 = sqrt(norm2);
    return norm2;
}
