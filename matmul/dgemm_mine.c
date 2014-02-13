#include <string.h>
#include "../mm_kernel/kdgemm.c"

const char* dgemm_desc = "My awesome dgemm.";

void square_dgemm(const int M, const double *A, const double *B, double *C)
{

	static double buffer[4*_MAGIC_P_ + 4] __attribute__((aligned(16))); 

	double * tempA, * tempB, * tempC;
	int size;
	
	if(M % 2 == 0){
		size = M*M;
	} else{
		size = M*(M+1);
	}


	//tempA = buffer + 4*_MAGIC_P_ + 4;
	tempA = (double *) _mm_malloc(3*size*sizeof(double), 16);
	// printf("%d\n", tempA);
	tempB = tempA + size;
	tempC = tempB + size;
	

	to_kdgemm_A(M, A, tempA);
	to_kdgemm_A(M, C, tempC);
	to_kdgemm_B(M, B, tempB);

	int i, j, posC=0, k;
	double * pA, * pB, * pC;
	double * startA = buffer + 4;
	double * startB = buffer + 4 + 2*_MAGIC_P_;
	int blockSize = 0;
	pC = tempC;
	for(i = 0; i < M; i += 2){
		for(j = 0; j < M; j += 2){
			if(j + 1 < M){
				buffer[0] = *pC;
				buffer[1] = *(pC+3);
				buffer[2] = *(pC+2);
				buffer[3] = *(pC+1);
			} else{
				buffer[0] = *(pC);
				buffer[1] = 0;
				buffer[2] = 0;
				buffer[3] = *(pC+1);
			}

			pA = tempA + M*i;
			pB = tempB + M*j;
			for(k = 0; k < ((M + _MAGIC_P_-1)/_MAGIC_P_); k++){
				if(k*_MAGIC_P_ + _MAGIC_P_ <= M){
					memcpy(startA, pA, 2*_MAGIC_P_*sizeof(double));
					memcpy(startB, pB, 2*_MAGIC_P_*sizeof(double));

					pA += 2*_MAGIC_P_; 
					pB += 2*_MAGIC_P_;

					kdgemm(buffer, startA, startB);
				} else{
					blockSize = M - k*_MAGIC_P_;
					memcpy(startA, pA, 2*blockSize*sizeof(double));
					memset(startA + 2*blockSize, 0, 2*(_MAGIC_P_-blockSize)*sizeof(double));

					memcpy(startB, pB, 2*blockSize*sizeof(double));
					memset(startB + 2*blockSize, 0, 2*(_MAGIC_P_-blockSize)*sizeof(double));

					pA += 2*blockSize;
					pB += 2*blockSize;
			
					//chamar o kernel
					kdgemm(buffer, startA, startB);
				}
			
			}

			if(j + 1 < M){
				*(pC) = buffer[0];
				*(pC+3) = buffer[1];
				*(pC+2) = buffer[2];
				*(pC+1) = buffer[3];
				pC += 4;
			} else{
				*(pC) = buffer[0];
				*(pC+1) = buffer[3];
				pC += 2;
			}
		}
	}

	from_kdgemm_C(M, tempC, C);

	//_mm_free(tempA);
	    
}
