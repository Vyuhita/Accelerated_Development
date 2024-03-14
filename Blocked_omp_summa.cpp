/*
 *Matrix Multiplication Impemantation using bocked SUMMA algorithm for 
 *Parallel Execution using openmp
 */
#include<iostream>
#include<cstdlib>
#include<omp.h>
#include<math.h>
#include <chrono>
#include <fstream>
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
// tolerance for validation of matrix multiplication
#define TOL 1e-4

//Global declaration for matrices
double* A;
double* B;
double* C;
int numThreads;
int SIZE;  //Matrix Size

using namespace std;

//Declare and Initilize Matrices with random values 
void Init(){
	//Declaration of matrices
	#pragma omp parallel
	{
		A = new double[SIZE * SIZE];
		B = new double[SIZE * SIZE];
		C = new double[SIZE * SIZE];

		//Initilize Matrices with random values 
		#pragma omp for collapse(2)
		for(int i=0;i < SIZE;i++)
			for(int j=0;j < SIZE;j++){
				A[i*SIZE +j] = double(rand())/double((RAND_MAX));
				B[i*SIZE +j] = double(rand())/double((RAND_MAX));
				//C[i*SIZE +j] = 0.0;
		}
	}/*** End of parallel region ***/

}

double validate(){
	double eps = 0.0;
    double* D = new double[SIZE * SIZE];
    for(int i=0;i<SIZE;++i) {
        for (int j = 0; j < SIZE; ++j) {
            D[i*SIZE+j] = 0.0;
            for(int k=0;k<SIZE;k++) {
                D[i*SIZE+j]=D[i*SIZE+j]+A[i*SIZE+k]*B[k*SIZE+j];
            } 
        }
    }
    
    for(int i=0;i<SIZE;i++){
        for(int j=0;j<SIZE;j++){
            int idx = i* SIZE + j;
            D[idx] = fabs(D[idx] - C[idx]);
            if (eps < D[idx]) {
                eps = D[idx];
            }
        }
    }
    delete D;
    return eps;
}

// block SUMMA Implementation
void BlockedSUMMA(int b){
	//#pragma omp parallel
	//block_size = N/sqrt(num_procs);

   	int i,j,k;
   	int ii, jj, kk;
   	double temp;
   	//cout<<"Theads before parallel : "<<omp_get_num_threads()<<endl;
    //#pragma omp parallel shared(A,B,C) private(i, j, k, ii, jj, kk) num_threads(numThreads)
//	{ 
// 		//#pragma omp for collapse(3) 
// 	 	for (k = 0; k < SIZE; k += b)
// 	 		//#pragma omp for collapse(2)
//            for (i = 0; i < SIZE; i += b)
//                for (j = 0; j < SIZE; j += b){
//                	//#pragma omp for collapse(3)
//                    for (kk = k; kk < MIN(k + b, SIZE); kk++) /* Each kk preparing corresponding block of C*/
//                        //#pragma omp for collapse(2)
//                        for (ii = i; ii < MIN(i + b, SIZE); ii++) {/*Moving accross rows of blockA and blockC*/
//                        	//temp = A[ii * MIN(i + b, SIZE) + kk];
//                            for (jj = j; jj < MIN(j + b, SIZE); jj++) /*Moving Across colums of blockC and blockB*/
//                                C[ii* MIN(i + b, SIZE) +jj] += A[ii * MIN(i + b, SIZE) + kk]*B[kk*MIN(i + b, SIZE)+jj]; /*jj drives frequent updated on C */
//                        }
//                }
//	}/*** End of parallel region ***/


    #pragma omp parallel shared(A,B,C) private(i, j, k, ii, jj) num_threads(numThreads)
    { 
 		//#pragma omp for collapse(3) 
 	 	for (k = 0; k < SIZE; k++){
 	 		#pragma omp for collapse(2)
            for (i = 0; i < SIZE; i += b){
                for (j = 0; j < SIZE; j += b){
                	int max_row = MIN(SIZE, i + b); 
                	int max_col = MIN(SIZE, j + b);
                	for(int ii=i;ii<max_row;ii++) { /*Moving accross rows of blockA and blockC*/
        				temp = A[ii*SIZE + k];
        				for(int jj=j;jj<max_col;jj++) { /*Moving Across colums of blockC and blockB*/
            				C[ii* SIZE + jj] += A[ii*SIZE + k] * B[k*SIZE + jj]; /*jj drives frequent updated on C */
        				}
    				}
                }
            }
        }
	}
}

void print() {
    FILE *fp = fopen("parallel.txt","w");
    for(int i = 0; i < SIZE; i++) {
        for(int j = 0; j < SIZE; j++) {
            fprintf(fp, "%lf\n", C[i*SIZE +j]);
        }
    }
    fclose(fp);
}
int main(int argc, char* argv[]){
	//auto timer = chrono::system_clock::now();
	if(argc != 3) {
		cout<<"Usage: ./ver2_omp NumThreads N"<<endl;
		exit(0);
	}
	srand((double)time(NULL));
	numThreads = atoi(argv[1]);
	SIZE = atoi(argv[2]);
	
	// file pointer
	fstream fout;
  
	// opens an existing csv file or creates a new file.
	fout.open("Blocked_Omp.csv", ios::out | ios::app);
	//fout << "NumThreads" << ", "<< "Input Size" << ", "<< "BlockSize" << ", "<< "time(ms)"<< "\n";

	Init();
	int limit =1024; // BlockSize< n/sqrt(threads)
 	int b;
 	for(b=1024; b<=limit; b*=2 ){
 		#pragma omp for collapse(2)
		for(int i=0;i < SIZE;i++)
			for(int j=0;j < SIZE;j++){
				C[i*SIZE +j] = 0.0;
			}
 		auto now = std::chrono::system_clock::now();
    	BlockedSUMMA(b);
    	auto time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - now).count();
 		cout<<"\nBlocked Multiplication with B="<<b <<" time for n="<<SIZE<<" in millisec="<<time<<" BlockSize :"<<b<<"\n";
 		//print();  /* Writing the result on parallel.txt */
 		//cout<<"Validate: "<<Validate<<endl;

		double eps = validate();
    	if (eps > TOL) {
			printf("SUMMA: NOT VALIDATE: eps = %f\n", eps); 	
 		} else {
			printf("SUMMA: OK: eps = %f\n", eps);
    	}

		// Insert the data to file
        fout << numThreads << ", "
             << SIZE << ", "
             << b << ", "
             << time  
             << "\n";
	}
	
	delete [] A;
	delete [] B;
	delete [] C;
	
	//cout<<"Overall code time in millisec="<<std::chrono::duration_cast<std::chrono::milliseconds>(chrono::system_clock::now() - timer).count()<<"\n";
	return 0;
}

