#include <string.h>

const char* dgemm_desc = "My awesome dgemm.";

void square_dgemm(const int M, const double *A, const double *B, double *C)
{

    static double buffer[4*P + 4] __attribute__((aligned(16))); 

	double * tempA, * tempB, * tempC;
	int size;
	
	if(M % 2 == 0){
		size = M*M;
	} else{
		size = M*(M+1);
	}

	tempA = (double *) _mm_malloc(3*size*sizeof(double), 16);
	tempB = tempA + size*sizeof(double);
	tempC = tempB + size*sizeof(double);
	
	to_kdgemm_A(M, A, tempA);
	to_kdgemm_A(M, C, tempC);
	to_kdgemm_B(M, B, tempB);

	int i, j, posC=0, k;
	int posA = 0, posB = 0;
	int blockSize = 0;
	for(i = 0; i < M; i += 2){
		for(j = 0; j < M; j += 2){
			if(j + 1 < M){
				buffer[0] = tempC[posC];
				buffer[1] = tempC[posC+3];
				buffer[2] = tempC[posC+2];
				buffer[3] = tempC[posC+1];
			} else{
				buffer[0] = tempC[posC];
				buffer[1] = 0;
				buffer[2] = 0;
				buffer[3] = tempC[posC+1];
			}

			posA = 0, posB = 0;
			for(k = 0; k < ((M + P-1)/P); k++){
				if(k*P + P <= M){
					memcpy(buffer + 4, tempA + M*i + 2*posA, 2*P*sizeof(double));
					memcpy(buffer + 4 + 2*P, tempB + M*j + 2*posB, 2*P*sizeof(double));
					posA += P; 
					posB += P;

					//chamar o kernel
					kdgemm(buffer, buffer + 4, buffer + 4 + 2*P);
				} else{
					blockSize = M - k*P;
					memcpy(buffer + 4, tempA + M*i + 2*posA, 2*blockSize*sizeof(double));
					memset(buffer + 4 + 2*blockSize, 0, 2*(M-blockSize)*sizeof(double));

					memcpy(buffer + 4 + 2*P, tempB + M*j + 2*posB, 2*blockSize*sizeof(double));
					memset(buffer + 4 + 2*P + 2*blockSize, 0, 2*(M-blockSize)*sizeof(double));

					posA += blockSize;
					posB += blockSize;

					//chamar o kernel
					kdgemm(buffer, buffer + 4, buffer + 4 + 2*P);
				}
			}

			if(j + 1 < M){
				tempC[posC] = buffer[0];
				tempC[posC+3] = buffer[1];
				tempC[posC+2] = buffer[2];
				tempC[posC+1] = buffer[3];
				posC += 4;
			} else{
				tempC[posC] = buffer[0];
				tempC[posC+1] = buffer[3];
				posC += 2;
			}
		}
	}

	from_kdgemm_C(M, tempC, C);

	_mm_free(tempA);
	    
}