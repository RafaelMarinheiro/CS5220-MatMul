#include <nmmintrin.h>
#include <string.h>

#define _MAGIC_P_ 10

const char* dgemm_desc = "My awesome dgemm.";

/*
 * On the Nehalem architecture, shufpd and multiplication use the same port.
 * 32-bit integer shuffle is a different matter.  If we want to try to make
 * it as easy as possible for the compiler to schedule multiplies along
 * with adds, it therefore makes sense to abuse the integer shuffle
 * instruction.  See also
 *   http://locklessinc.com/articles/interval_arithmetic/
 */
#ifdef USE_SHUFPD
#  define swap_sse_doubles(a) _mm_shuffle_pd(a, a, 1)
#else
#  define swap_sse_doubles(a) (__m128d) _mm_shuffle_epi32((__m128i) a, 0x4e)
#endif

/*
 * Block matrix multiply kernel.
 * Inputs:
 *    A: 2-by-P matrix in column major format.
 *    B: P-by-2 matrix in row major format.
 * Outputs:
 *    C: 2-by-2 matrix with element order [c11, c22, c12, c21]
 *       (diagonals stored first, then off-diagonals)
 */
void kdgemm2P2(const int P, double * restrict C,
               const double * restrict A,
               const double * restrict B)
{
    // This is really implicit in using the aligned ops...
    __assume_aligned(A, 16);
    __assume_aligned(B, 16);
    __assume_aligned(C, 16);

    // Load diagonal and off-diagonals
    __m128d cd = _mm_load_pd(C+0);
    __m128d co = _mm_load_pd(C+2);

    /*
     * Do block dot product.  Each iteration adds the result of a two-by-two
     * matrix multiply into the accumulated 2-by-2 product matrix, which is
     * stored in the registers cd (diagonal part) and co (off-diagonal part).
     */
    for (int k = 0; k < P; k += 2) {

        __m128d a0 = _mm_load_pd(A+2*k+0);
        __m128d b0 = _mm_load_pd(B+2*k+0);
        __m128d td0 = _mm_mul_pd(a0, b0);
        __m128d bs0 = swap_sse_doubles(b0);
        __m128d to0 = _mm_mul_pd(a0, bs0);

        __m128d a1 = _mm_load_pd(A+2*k+2);
        __m128d b1 = _mm_load_pd(B+2*k+2);
        __m128d td1 = _mm_mul_pd(a1, b1);
        __m128d bs1 = swap_sse_doubles(b1);
        __m128d to1 = _mm_mul_pd(a1, bs1);

        __m128d td_sum = _mm_add_pd(td0, td1);
        __m128d to_sum = _mm_add_pd(to0, to1);

        cd = _mm_add_pd(cd, td_sum);
        co = _mm_add_pd(co, to_sum);
    }

    // Write back sum
    _mm_store_pd(C+0, cd);
    _mm_store_pd(C+2, co);
}


void FromColumnMajorToColumnFormat(const int M, const double * restrict A, double * restrict Conv){

	int i=0, j=0, pos = 0;

	for(i = 0; i < M; i+= 2){
		if(i+1 < M){
			for(j = 0; j < M; j++){
				Conv[pos] = A[i + j*M];
				Conv[pos+1] = A[i + j*M + 1];
				pos += 2;
			}
		} else {
			for(j = 0; j < M; j++){
				Conv[pos] = A[i + j*M];
				Conv[pos+1] = 0;
				pos += 2; 
			}
		}
	}
}

void FromColumnFormatToColumnMajor(const int M, const double * restrict Conv, double * restrict A){
	int i=0, j=0, pos = 0;

	for(i = 0; i < M; i+= 2){
		if(i+1 < M){
			for(j = 0; j < M; j++){
				A[i + j*M] = Conv[pos];
				A[i + j*M + 1] = Conv[pos+1];
				pos += 2;  
			}
		} else {
			for(j = 0; j < M; j++){
				A[i + j*M] = Conv[pos];
				pos++;
			}
		}
	}
}

void FromColumnMajorToRowFormat(const int M, const double * restrict A, double * restrict Conv){
	int i=0, j=0, pos=0;
	for(j = 0; j < M; j += 2){
		if(j+1 < M){
			for(i = 0; i < M; i++){
				Conv[pos] = A[i + j*M];
				Conv[pos+1] = A[i + j*M + 1];
				pos += 2;
			}
		} else{
			for(i = 0; i < M; i++){
				Conv[pos] = A[i + j*M];
				Conv[pos+1] = 0;
				pos += 2;
			}
		}
	}
}

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

	tempA = (double *) _mm_malloc(3*size*sizeof(double), 16);
	tempB = tempA + size*sizeof(double);
	tempC = tempB + size*sizeof(double);
	
	FromColumnMajorToColumnFormat(M, A, tempA);
	FromColumnMajorToColumnFormat(M, C, tempC);
	FromColumnMajorToRowFormat(M, B, tempB);

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
					memcpy(buffer + 4, tempA + 2*M*i + 2*posA, 2*_MAGIC_P_*sizeof(double));
					memcpy(buffer + 4 + 2*_MAGIC_P_, tempA + 2*M*i + posB, 2*_MAGIC_P_*sizeof(double));
					posA += _MAGIC_P_; 
					posB += _MAGIC_P_;

					//chamar o kernel
					kdgemm2P2(_MAGIC_P_, buffer, buffer + 4, buffer + 4 + 2*_MAGIC_P_);
				} else{
					blockSize = M - k*_MAGIC_P_;
					memcpy(buffer + 4, tempA + 2*M*i + posA, 2*blockSize*sizeof(double));
					memset(buffer + 4 + 2*blockSize, 0, 2*(M-blockSize)*sizeof(double));

					memcpy(buffer + 4 + 2*_MAGIC_P_, tempA + 2*M*i + posB, 2*blockSize*sizeof(double));
					memset(buffer + 4 + 2*_MAGIC_P_ + 2*blockSize, 0, 2*(M-blockSize)*sizeof(double));

					posA += blockSize;
					posB += blockSize;

					//chamar o kernel
					kdgemm2P2(_MAGIC_P_, buffer, buffer + 4, buffer + 4 + 2*_MAGIC_P_);
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

	FromColumnFormatToColumnMajor(M, tempC, C);

	_mm_free(tempA);
	    
}