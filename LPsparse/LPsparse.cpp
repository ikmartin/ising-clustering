#include "LPsparse.h"

void exit_with_help(){
	
	cerr << "Usage: ./LPsparse (options) [data_dir]" << endl;
	cerr << "options:" << endl;
	cerr << "-d solve_dual: solve dual to obtain solution of primal problem (default: No)" <<endl;
	cerr << "-c use_cg: use projected-conjugate gradient method to obtain faster asymptotic convergence (default: No)" << endl;
	cerr << "-t tol: tolerance of primal and dual infeasibility for terminiation (default: 1e-3)" << endl;
	cerr << "-e eta: Augmented Lagragnain parameter (default: 1.0)" << endl;
	cerr << "-p tol_phase2: tolerance for changing to phase 2 (default: 1e-2)" << endl;
	cerr << "-s tol_sub: tolerance for solving sub-problem in phase-1 (default: 1e-3)" << endl;
	cerr << "-m max_iter: maximum number of outer iterations (default 1000)" << endl;
	cerr << "-n nnz_tol: truncate a value to 0 if less than nnz_tol for the output (default 1e-4)" << endl;
	exit(0);
}

void parse_cmd_line(int argc, char** argv){

	int i;
	for(i=1;i<argc;i++){
		if( argv[i][0] != '-' )
			break;
		if( ++i >= argc )
			exit_with_help();

		switch(argv[i-1][1]){
			
			case 'd': param.solve_from_dual = true;
				  --i;
				  break;
			case 'c': param.use_CG = true;
				  --i;
				  break;
			case 't': param.tol = atof(argv[i]);
				  break;
			case 'e': param.eta = atof(argv[i]);
				  break;
			case 'p': param.tol_trans = atof(argv[i]);
				  break;
			case 's': param.tol_sub = atof(argv[i]);
				  break;
			case 'm': param.max_iter = atoi(argv[i]);
				  break;
			case 'n': param.nnz_tol = atof(argv[i]);
				  break;
			default:
				  cerr << "unknown option: -" << argv[i-1][1] << endl;
				  exit(0);
		}
	}
	
	if(i>=argc)
		exit_with_help();
	
	param.data_dir = argv[i];
}

int main(int argc, char** argv){
	
	if( argc < 1+1 ){
		exit_with_help();
	}
	parse_cmd_line( argc, argv );

	string data_dir(param.data_dir);
	string fname_meta = data_dir+"/meta";
	string fname_A = data_dir+"/A";
	string fname_b = data_dir+"/b";
	string fname_c = data_dir+"/c";
	string fname_Aeq = data_dir+"/Aeq";
	string fname_beq = data_dir+"/beq";
	
	//read vectors and #variables, #constraints
	int n, nf, m , me;
	readMeta(fname_meta.c_str(), n,nf,m,me);
	double* b = new double[m+me+1]; // [b' be']' (0 can be appended at the end for index)
	double* c = new double[n+nf]; // [c' ce']'
	readVec(fname_c.c_str(), c, 0, n+nf);
	readVec(fname_b.c_str(), b, 0, m);
	readVec(fname_beq.c_str(), b, m, me);
	b[ m+me ] = 0.0;
	
	Constr* A = new Constr[m+me+1]; //(c' can be appended at the end for index)
	readMat(fname_A.c_str(), m, n+nf, A, 0); //offset=0
	readMat(fname_Aeq.c_str(), me, n+nf, A, m); //offset=m
	
	cerr << "n=" << n << endl;
	cerr << "nf=" << nf << endl;
	cerr << "m=" << m << endl;
	cerr << "me=" << me << endl;
	
	ConstrInv* At = new ConstrInv[n+nf];
	transpose(A, m+me, n+nf, At);
	
	double* x = new double[n+nf];   //primal variables
	double* w = new double[m+me];   // dual variables (scaled by eta_t) (might be appended an "1" at end)
	
	double eta_0 = param.eta;
	if( !param.solve_from_dual ){
		for(int i=0;i<m+me;i++)
			w[i] = 0.0; //dual var = 0
		LPsolve(n,nf,m,me, A,At,b,c, x,w);

		for(int i=0;i<m;i++)
			if( w[i] < 0 )
				w[i] = 0.0;
		for(int i=0;i<m+me;i++)
			w[i] *= param.eta; //dual variables y=w*eta_t
	
	}else{
		negate_mat(At, n+nf);
		negate_mat(A, m+me);
		for(int j=0;j<n+nf;j++)
			x[j] = 0.0; //dual var =0
		LPsolve(m,me,n,nf, At, A, c, b, w, x);
		
		for(int j=0;j<n;j++)
			if( x[j] < 0 )
				x[j] = 0.0;
		
		for(int j=0;j<n+nf;j++)
			x[j] *= (param.eta/eta_0);
	}

	string solname = (data_dir + "/sol").c_str();
	string dualsolname = (data_dir + "/sol_dual").c_str();
	cout << solname << endl;
	cout << dualsolname << endl;
	writeVec((char*) solname.c_str(), x, n+nf, param.nnz_tol);
	writeVec((char*) dualsolname.c_str(), w, m+me, param.nnz_tol);
	
	return 0;
}
