#include <nmmintrin.h>

/*
 * Dimensions for a "kernel" multiply.  We use define statements in
 * order to make sure these are treated as compile-time constants
 * (which the optimizer likes)
//  */
// #define M 2
// #define N 2
 #define _MAGIC_P_ 128

/*
 * The ktimer driver expects these variables to be set to whatever
 * the dimensions of a kernel multiply are.  It uses them both for
 * space allocation and for flop rate computations.
 */
const int DIM_M=2;
const int DIM_N=2;
const int DIM_P=_MAGIC_P_;


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

void kdgemm(double * restrict C,
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
    int po = 0;
    for (int k = 0; k < 16; k += 1) {
        __m128d a0 = _mm_load_pd(A+po+0);
        __m128d b0 = _mm_load_pd(B+po+0);
        __m128d td0 = _mm_mul_pd(a0, b0);
        __m128d bs0 = swap_sse_doubles(b0);
        __m128d to0 = _mm_mul_pd(a0, bs0);

        cd = _mm_add_pd(cd, td0);
        co = _mm_add_pd(co, to0);
        

        __m128d a1 = _mm_load_pd(A+po+2);
        __m128d b1 = _mm_load_pd(B+po+2);
        __m128d td1 = _mm_mul_pd(a1, b1);
        __m128d bs1 = swap_sse_doubles(b1);
        __m128d to1 = _mm_mul_pd(a1, bs1);

        cd = _mm_add_pd(cd, td1);
        co = _mm_add_pd(co, to1);

        __m128d a2 = _mm_load_pd(A+po+4);
        __m128d b2 = _mm_load_pd(B+po+4);
        __m128d td2 = _mm_mul_pd(a2, b2);
        __m128d bs2 = swap_sse_doubles(b2);
        __m128d to2 = _mm_mul_pd(a2, bs2);

        cd = _mm_add_pd(cd, td2);
        co = _mm_add_pd(co, to2);
        

        __m128d a3 = _mm_load_pd(A+po+6);
        __m128d b3 = _mm_load_pd(B+po+6);
        __m128d td3 = _mm_mul_pd(a3, b3);
        __m128d bs3 = swap_sse_doubles(b3);
        __m128d to3 = _mm_mul_pd(a3, bs3);

        cd = _mm_add_pd(cd, td3);
        co = _mm_add_pd(co, to3);

        __m128d a4 = _mm_load_pd(A+po+8);
        __m128d b4 = _mm_load_pd(B+po+8);
        __m128d td4 = _mm_mul_pd(a4, b4);
        __m128d bs4 = swap_sse_doubles(b4);
        __m128d to4 = _mm_mul_pd(a4, bs4);

        cd = _mm_add_pd(cd, td4);
        co = _mm_add_pd(co, to4);
        

        __m128d a5 = _mm_load_pd(A+po+10);
        __m128d b5 = _mm_load_pd(B+po+10);
        __m128d td5 = _mm_mul_pd(a5, b5);
        __m128d bs5 = swap_sse_doubles(b5);
        __m128d to5 = _mm_mul_pd(a5, bs5);

        cd = _mm_add_pd(cd, td5);
        co = _mm_add_pd(co, to5);

        __m128d a6 = _mm_load_pd(A+po+12);
        __m128d b6 = _mm_load_pd(B+po+12);
        __m128d td6 = _mm_mul_pd(a6, b6);
        __m128d bs6 = swap_sse_doubles(b6);
        __m128d to6 = _mm_mul_pd(a6, bs6);

        cd = _mm_add_pd(cd, td6);
        co = _mm_add_pd(co, to6);
        

        __m128d a7 = _mm_load_pd(A+po+14);
        __m128d b7 = _mm_load_pd(B+po+14);
        __m128d td7 = _mm_mul_pd(a7, b7);
        __m128d bs7 = swap_sse_doubles(b7);
        __m128d to7 = _mm_mul_pd(a7, bs7);

        cd = _mm_add_pd(cd, td7);
        co = _mm_add_pd(co, to7);
        po += 16;
    }

    // Write back sum
    _mm_store_pd(C+0, cd);
    _mm_store_pd(C+2, co);
}

// /*
//  * Block matrix multiply kernel (simple fixed-size case).
//  * Use restrict to tell the compiler there is no aliasing,
//  * and inform the compiler of alignment constraints.
//  */
// void kdgemm(const double * restrict A,
//             const double * restrict B,
//             double * restrict C)
// {
//     __assume_aligned(A, 16);
//     __assume_aligned(B, 16);
//     __assume_aligned(C, 16);

//     for (int j = 0; j < N; ++j) {
//         for (int k = 0; k < P; ++k) {
//             double bkj = B[k+j*P];
//             for (int i = 0; i < M; ++i) {
//                 C[i+j*M] += A[i+k*M]*bkj;
//             }
//         }
//     }
// }

/*
 * Conversion routines that take a matrix block in column-major form
 * and put it into whatever form the kdgemm routine likes.
 */

void to_kdgemm_A(int ldA, const double* restrict A, double * restrict Ak)
{
    int i=0, j=0, pos = 0;

    for(i = 0; i < ldA; i+= 2){
        if(i+1 < ldA){
            for(j = 0; j < ldA; j++){
                Ak[pos] = A[i + j*ldA];
                Ak[pos+1] = A[i + j*ldA + 1];
                pos += 2;
            }
        } else {
            for(j = 0; j < ldA; j++){
                Ak[pos] = A[i + j*ldA];
                Ak[pos+1] = 0;
                pos += 2; 
            }
        }
    }
}

void from_kdgemm_C(int ldC, const double* restrict Ck, double * restrict C)
{
    int i=0, j=0, pos = 0;

    for(i = 0; i < ldC; i+= 2){
        if(i+1 < ldC){
            for(j = 0; j < ldC; j++){
                C[i + j*ldC] = Ck[pos];
                C[i + j*ldC + 1] = Ck[pos+1];
                pos += 2;  
            }
        } else {
            for(j = 0; j < ldC; j++){
                C[i + j*ldC] = Ck[pos];
                pos += 2;
            }
        }
    }
}

void to_kdgemm_B(int ldB, const double* restrict B, double * restrict Bk)
{
    int i=0, j=0, pos=0;
    int k=0;
    int blocks = (ldB + _MAGIC_P_ - 1)/_MAGIC_P_;
    for(k = 0; k < blocks; k++){
        if(k*_MAGIC_P_ + _MAGIC_P_ <= ldB){
            for(j = 0; j < ldB; j += 2){
                if(j+1 < ldB){
                    for(i = 0; i < _MAGIC_P_; i++){
                        Bk[pos] = B[i + j*ldB + k*_MAGIC_P_];
                        Bk[pos+1] = B[i + j*ldB + ldB + k*_MAGIC_P_];
                        pos += 2;
                    }
                } else{
                    for(i = 0; i < _MAGIC_P_; i++){
                        Bk[pos] = B[i + j*ldB + k*_MAGIC_P_];
                        Bk[pos+1] = 0;
                        pos += 2;
                    }
                }
            }
        } else{
            int rest = ldB - k*_MAGIC_P_;
            for(j = 0; j < ldB; j += 2){
                if(j+1 < ldB){
                    for(i = 0; i < rest; i++){
                        Bk[pos] = B[i + j*ldB + k*_MAGIC_P_];
                        Bk[pos+1] = B[i + j*ldB + ldB + k*_MAGIC_P_];
                        pos += 2;
                    }
                } else{
                    for(i = 0; i < rest; i++){
                        Bk[pos] = B[i + j*ldB + k*_MAGIC_P_];
                        Bk[pos+1] = 0;
                        pos += 2;
                    }
                }
            }
        }
    }
}

