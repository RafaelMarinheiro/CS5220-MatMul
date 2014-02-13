#include <string.h>
#include "../mm_kernel/kdgemm.c"

const char* dgemm_desc = "My awesome dgemm.";

void square_dgemm(const int M, const double *A, const double *B, double *C)
{

	static double buffer[4*_MAGIC_P_ + 4 + 2000000] __attribute__((aligned(16))); 

	double * tempA, * tempB, * tempC;
	int size;
	
	if(M % 2 == 0){
		size = M*M;
	} else{
		size = M*(M+1);
	}


	tempA = buffer + 4*_MAGIC_P_ + 4;
	// tempA = (double *) _mm_malloc(3*size*sizeof(double), 16);
	// printf("%d\n", tempA);
	tempB = tempA + size;
	tempC = tempB + size;
	

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
			for(k = 0; k < ((M + _MAGIC_P_-1)/_MAGIC_P_); k++){
				if(k*_MAGIC_P_ + _MAGIC_P_ <= M){
					memcpy(buffer + 4, tempA + M*i + 2*posA, 2*_MAGIC_P_*sizeof(double));
					memcpy(buffer + 4 + 2*_MAGIC_P_, tempB + M*j + 2*posB, 2*_MAGIC_P_*sizeof(double));

					posA += _MAGIC_P_; 
					posB += _MAGIC_P_;

					//chamar o kernel
					// for(int u = 0; u < 4*_MAGIC_P_ + 4; u++){
					// 	printf("%.0lf ", buffer[u]);
					// }
					// printf("\n");
			
					kdgemm(buffer, buffer + 4, buffer + 4 + 2*_MAGIC_P_);
				} else{
					blockSize = M - k*_MAGIC_P_;
					memcpy(buffer + 4, tempA + M*i + 2*posA, 2*blockSize*sizeof(double));
					memset(buffer + 4 + 2*blockSize, 0, 2*(_MAGIC_P_-blockSize)*sizeof(double));
					// for(int u = 0; u < 2*(M-blockSize); u++){
					// 	buffer[4+2*blockSize + u] = 0;
					// }

					memcpy(buffer + 4 + 2*_MAGIC_P_, tempB + M*j + 2*posB, 2*blockSize*sizeof(double));
					memset(buffer + 4 + 2*_MAGIC_P_ + 2*blockSize, 0, 2*(_MAGIC_P_-blockSize)*sizeof(double));
					// for(int u = 0; u < 2*(M-blockSize); u++){
					// 	buffer[4+2*blockSize + 2*_MAGIC_P_ + u] = 0;
					// }

					posA += blockSize;
					posB += blockSize;
					// for(int u = 0; u < 4*_MAGIC_P_ + 4; u++){
					// 	printf("%.0lf ", buffer[u]);
					// }
					// printf("\n");
			
					//chamar o kernel
					kdgemm(buffer, buffer + 4, buffer + 4 + 2*_MAGIC_P_);
				}
				// 	for(int u = 0; u < 4*_MAGIC_P_ + 4; u++){
				// 	printf("%.0lf ", buffer[u]);
				// }
				// printf("\n");
			
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

	//_mm_free(tempA);
	    
}
