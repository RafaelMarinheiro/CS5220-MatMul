#include <string.h>
#include "../mm_kernel/kdgemm.c"

const char* dgemm_desc = "My awesome dgemm.";

void square_dgemm(const int M, const double *A, const double *B, double *C)
{

	static double buffer[4*_MAGIC_P_ + 4] __attribute__((aligned(16))); 

	double * tempA, * tempB, * tempC;
	int size;
	int MM;
	
	if(M % 2 == 0){
		size = M*M;
		MM = M;
	} else{
		size = M*(M+1);
		MM = M+1;
	}


	//tempA = buffer + 4*_MAGIC_P_ + 4;
	tempA = (double *) _mm_malloc(3*size*sizeof(double), 16);
	// printf("%d\n", tempA);
	tempB = tempA + size;
	tempC = tempB + size;
	

	to_kdgemm_A(M, A, tempA);
	to_kdgemm_A(M, C, tempC);
	to_kdgemm_B(M, B, tempB);

	int i, j, posC=0, k, posA=0, posB = 0;
	int tB = 0;
	double * pA, * pB, * pC;
	double * startA = buffer + 4;
	double * startB = buffer + 4 + 2*_MAGIC_P_;
	int blockSize = 0;
	pC = tempC;
	int _do_ = (M > _MAGIC_P_);
	for(k = 0; k < ((M + _MAGIC_P_-1)/_MAGIC_P_); k++){			
		int blockSize = M - k*_MAGIC_P_;
		posB = k*_MAGIC_P_*MM;
		for(i = 0; i < M; i += 2){
			if(k*_MAGIC_P_ + _MAGIC_P_ <= M){
				memcpy(startA, tempA + M*i + 2*k*_MAGIC_P_, 2*_MAGIC_P_*sizeof(double));
			} else{
				memcpy(startA, tempA + M*i + 2*k*_MAGIC_P_, 2*blockSize*sizeof(double));
				memset(startA+2*blockSize, 0, 2*(_MAGIC_P_-blockSize)*sizeof(double));
			}
			tB = 0;
			for(j = 0; j < M; j += 2){
				if(j+1 < M){
					buffer[0] = tempC[M*i + 2*j];
					buffer[1] = tempC[M*i + 2*j + 3];
					buffer[2] = tempC[M*i + 2*j + 2];
					buffer[3] = tempC[M*i + 2*j + 1];
				} else{
					buffer[0] = tempC[M*i + 2*j];
					buffer[1] = 0;
					buffer[2] = 0;
					buffer[3] = tempC[M*i + 2*j + 1];	
				}
				if(k*_MAGIC_P_ + _MAGIC_P_ <= M){
					memcpy(startB, tempB + posB + tB, 2*_MAGIC_P_*sizeof(double));
					tB += 2*_MAGIC_P_;
				} else{
					memcpy(startB, tempB + posB + tB, 2*blockSize*sizeof(double));
					memset(startB+2*blockSize, 0, 2*(_MAGIC_P_-blockSize)*sizeof(double));
					tB += 2*(blockSize);
				}
				kdgemm(buffer, startA, startB);
				if(j+1 < M){
					tempC[M*i + 2*j] = buffer[0];
					tempC[M*i + 2*j+3] = buffer[1];
					tempC[M*i + 2*j+2] = buffer[2];
					tempC[M*i + 2*j+1] = buffer[3];
				} else{
					tempC[M*i + 2*j] = buffer[0];
					tempC[M*i + 2*j+1] = buffer[3];	
				}
				
			}
		}
	}

	from_kdgemm_C(M, tempC, C);

	_mm_free(tempA);
	    
}

