#ifndef MSHABAL_H__
#define MSHABAL_H__

#include <limits.h>

#include <emmintrin.h>
#include <immintrin.h>

/*
* We need an integer type with width 32-bit or more (preferably, with
* a width of exactly 32 bits).
*/
#if defined __STDC__ && __STDC_VERSION__ >= 199901L
#include <stdint.h>
#ifdef UINT32_MAX
typedef uint32_t mshabal_u32;
#else
typedef uint_fast32_t mshabal_u32;
#endif
#else
#if ((UINT_MAX >> 11) >> 11) >= 0x3FF
typedef unsigned int mshabal_u32;
#else
typedef unsigned long mshabal_u32;
#endif
#endif
typedef mshabal_u32 u32;
typedef unsigned long long u64;
typedef unsigned char u8;

/*
* The context structure for a Shabal computation. Contents are
* private. Such a structure should be allocated and released by
* the caller, in any memory area.
*/

#ifdef __MINGW__
#define _ALIGN(x) __attribute__ ((aligned(x)))
#endif

#define C32(x) ((u32)x ## UL)

#ifdef __8WAY__
typedef union {
	u32 w[8];
	u64 q[4];
	__m256i d;
} w256;
#endif

typedef union {
	u32 w[4];
	u64 q[2];
	__m128i d;
} w128;

#ifdef __8WAY__
#ifndef __MINGW__
__declspec(align(ALIGNED_TO))
#endif
typedef struct {
	w256 stateA[12];
	w256 stateB[16];
	w256 stateC[16];
} mshabal8_context; 
#ifdef __MINGW__
_ALIGN(ALIGNED_TO)
#endif
#endif

#ifndef __MINGW__
__declspec(align(ALIGNED_TO))
#endif
typedef struct {
	w128 stateA[12];
	w128 stateB[16];
	w128 stateC[16];
} mshabal4_context; 
#ifdef __MINGW__
_ALIGN(ALIGNED_TO)
#endif

#ifdef __8WAY__
#ifndef __MINGW__
__declspec(align(ALIGNED_TO))
#endif
typedef union {
	u32 words[128];
	u64 quads[64];
	__m256i data[16];
} un; 
#ifdef __MINGW__
_ALIGN(ALIGNED_TO)
#endif
#endif

#ifndef __MINGW__
__declspec(align(ALIGNED_TO))
#endif
typedef union {
	u32 words[64];
	u64 quads[32];
	__m128i data[16];
} un4; 
#ifdef __MINGW__
_ALIGN(ALIGNED_TO)
#endif

#ifdef __8WAY__
#ifndef __MINGW__
__declspec(noinline)
#else
__attribute__((noinline))
#endif
static void mshabal8_compress_fast4_64(mshabal8_context *sc)
{
	size_t j;

	__m256i one = _mm256_set1_epi32(C32(0xFFFFFFFF));
	__m256i eighty = _mm256_setr_epi32(0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
	__m256i zero = _mm256_setzero_si256();

#define PP(xa0, xa1, xb0, xb1, xb2, xb3, xc, xm)   do { \
    __m256i tt; \
    tt = _mm256_or_si256(_mm256_slli_epi32(xa1, 15), \
        _mm256_srli_epi32(xa1, 17)); \
    tt = _mm256_add_epi32(_mm256_slli_epi32(tt, 2), tt); \
    tt = _mm256_xor_si256(_mm256_xor_si256(xa0, tt), xc); \
    tt = _mm256_add_epi32(_mm256_slli_epi32(tt, 1), tt); \
    tt = _mm256_xor_si256( \
        _mm256_xor_si256(tt, xb1), \
        _mm256_xor_si256(_mm256_andnot_si256(xb3, xb2), xm)); \
    xa0 = tt; \
    tt = xb0; \
    tt = _mm256_or_si256(_mm256_slli_epi32(tt, 1), \
        _mm256_srli_epi32(tt, 31)); \
    xb0 = _mm256_xor_si256(tt, _mm256_xor_si256(xa0, one)); \
	        } while (0)

#define PP2(xa0, xa1, xb0, xb1, xb2, xb3, xc)   do { \
    __m256i tt; \
    tt = _mm256_or_si256(_mm256_slli_epi32(xa1, 15), \
        _mm256_srli_epi32(xa1, 17)); \
    tt = _mm256_add_epi32(_mm256_slli_epi32(tt, 2), tt); \
    tt = _mm256_xor_si256(_mm256_xor_si256(xa0, tt), xc); \
    tt = _mm256_add_epi32(_mm256_slli_epi32(tt, 1), tt); \
    tt = _mm256_xor_si256( \
        _mm256_xor_si256(tt, xb1), _mm256_andnot_si256(xb3, xb2)); \
    xa0 = tt; \
    tt = xb0; \
    tt = _mm256_or_si256(_mm256_slli_epi32(tt, 1), \
        _mm256_srli_epi32(tt, 31)); \
    xb0 = _mm256_xor_si256(tt, _mm256_xor_si256(xa0, one)); \
	        } while (0)

#define INIT(A, B, M1, M2) \
    B[0].d = _mm256_add_epi32(B[0].d, M1); \
    B[0].d = _mm256_or_si256(_mm256_slli_epi32(B[0].d, 17), _mm256_srli_epi32(B[0].d, 15)); \
    for (j = 1; j < 16; j++) { \
        B[j].d = _mm256_add_epi32(B[j].d, M2); \
        B[j].d = _mm256_or_si256(_mm256_slli_epi32(B[j].d, 17), _mm256_srli_epi32(B[j].d, 15)); \
	    } \
    A[0].d = _mm256_xor_si256(A[0].d, _mm256_set1_epi32(2)); \
    A[1].d = _mm256_xor_si256(A[1].d, _mm256_set1_epi32(0));

#define PPA(A, B, C, M) \
    PP(A[0x0].d, A[0xB].d, B[0x0].d, B[0xD].d, B[0x9].d, B[0x6].d, C[0x8].d, M); \
    PP2(A[0x1].d, A[0x0].d, B[0x1].d, B[0xE].d, B[0xA].d, B[0x7].d, C[0x7].d); \
    PP2(A[0x2].d, A[0x1].d, B[0x2].d, B[0xF].d, B[0xB].d, B[0x8].d, C[0x6].d); \
    PP2(A[0x3].d, A[0x2].d, B[0x3].d, B[0x0].d, B[0xC].d, B[0x9].d, C[0x5].d); \
    PP2(A[0x4].d, A[0x3].d, B[0x4].d, B[0x1].d, B[0xD].d, B[0xA].d, C[0x4].d); \
    PP2(A[0x5].d, A[0x4].d, B[0x5].d, B[0x2].d, B[0xE].d, B[0xB].d, C[0x3].d); \
    PP2(A[0x6].d, A[0x5].d, B[0x6].d, B[0x3].d, B[0xF].d, B[0xC].d, C[0x2].d); \
    PP2(A[0x7].d, A[0x6].d, B[0x7].d, B[0x4].d, B[0x0].d, B[0xD].d, C[0x1].d); \
    PP2(A[0x8].d, A[0x7].d, B[0x8].d, B[0x5].d, B[0x1].d, B[0xE].d, C[0x0].d); \
    PP2(A[0x9].d, A[0x8].d, B[0x9].d, B[0x6].d, B[0x2].d, B[0xF].d, C[0xF].d); \
    PP2(A[0xA].d, A[0x9].d, B[0xA].d, B[0x7].d, B[0x3].d, B[0x0].d, C[0xE].d); \
    PP2(A[0xB].d, A[0xA].d, B[0xB].d, B[0x8].d, B[0x4].d, B[0x1].d, C[0xD].d); \
    PP2(A[0x0].d, A[0xB].d, B[0xC].d, B[0x9].d, B[0x5].d, B[0x2].d, C[0xC].d); \
    PP2(A[0x1].d, A[0x0].d, B[0xD].d, B[0xA].d, B[0x6].d, B[0x3].d, C[0xB].d); \
    PP2(A[0x2].d, A[0x1].d, B[0xE].d, B[0xB].d, B[0x7].d, B[0x4].d, C[0xA].d); \
    PP2(A[0x3].d, A[0x2].d, B[0xF].d, B[0xC].d, B[0x8].d, B[0x5].d, C[0x9].d); \
    PP(A[0x4].d, A[0x3].d, B[0x0].d, B[0xD].d, B[0x9].d, B[0x6].d, C[0x8].d, M); \
    PP2(A[0x5].d, A[0x4].d, B[0x1].d, B[0xE].d, B[0xA].d, B[0x7].d, C[0x7].d); \
    PP2(A[0x6].d, A[0x5].d, B[0x2].d, B[0xF].d, B[0xB].d, B[0x8].d, C[0x6].d); \
    PP2(A[0x7].d, A[0x6].d, B[0x3].d, B[0x0].d, B[0xC].d, B[0x9].d, C[0x5].d); \
    PP2(A[0x8].d, A[0x7].d, B[0x4].d, B[0x1].d, B[0xD].d, B[0xA].d, C[0x4].d); \
    PP2(A[0x9].d, A[0x8].d, B[0x5].d, B[0x2].d, B[0xE].d, B[0xB].d, C[0x3].d); \
    PP2(A[0xA].d, A[0x9].d, B[0x6].d, B[0x3].d, B[0xF].d, B[0xC].d, C[0x2].d); \
    PP2(A[0xB].d, A[0xA].d, B[0x7].d, B[0x4].d, B[0x0].d, B[0xD].d, C[0x1].d); \
    PP2(A[0x0].d, A[0xB].d, B[0x8].d, B[0x5].d, B[0x1].d, B[0xE].d, C[0x0].d); \
    PP2(A[0x1].d, A[0x0].d, B[0x9].d, B[0x6].d, B[0x2].d, B[0xF].d, C[0xF].d); \
    PP2(A[0x2].d, A[0x1].d, B[0xA].d, B[0x7].d, B[0x3].d, B[0x0].d, C[0xE].d); \
    PP2(A[0x3].d, A[0x2].d, B[0xB].d, B[0x8].d, B[0x4].d, B[0x1].d, C[0xD].d); \
    PP2(A[0x4].d, A[0x3].d, B[0xC].d, B[0x9].d, B[0x5].d, B[0x2].d, C[0xC].d); \
    PP2(A[0x5].d, A[0x4].d, B[0xD].d, B[0xA].d, B[0x6].d, B[0x3].d, C[0xB].d); \
    PP2(A[0x6].d, A[0x5].d, B[0xE].d, B[0xB].d, B[0x7].d, B[0x4].d, C[0xA].d); \
    PP2(A[0x7].d, A[0x6].d, B[0xF].d, B[0xC].d, B[0x8].d, B[0x5].d, C[0x9].d); \
    PP(A[0x8].d, A[0x7].d, B[0x0].d, B[0xD].d, B[0x9].d, B[0x6].d, C[0x8].d, M); \
    PP2(A[0x9].d, A[0x8].d, B[0x1].d, B[0xE].d, B[0xA].d, B[0x7].d, C[0x7].d); \
    PP2(A[0xA].d, A[0x9].d, B[0x2].d, B[0xF].d, B[0xB].d, B[0x8].d, C[0x6].d); \
    PP2(A[0xB].d, A[0xA].d, B[0x3].d, B[0x0].d, B[0xC].d, B[0x9].d, C[0x5].d); \
    PP2(A[0x0].d, A[0xB].d, B[0x4].d, B[0x1].d, B[0xD].d, B[0xA].d, C[0x4].d); \
    PP2(A[0x1].d, A[0x0].d, B[0x5].d, B[0x2].d, B[0xE].d, B[0xB].d, C[0x3].d); \
    PP2(A[0x2].d, A[0x1].d, B[0x6].d, B[0x3].d, B[0xF].d, B[0xC].d, C[0x2].d); \
    PP2(A[0x3].d, A[0x2].d, B[0x7].d, B[0x4].d, B[0x0].d, B[0xD].d, C[0x1].d); \
    PP2(A[0x4].d, A[0x3].d, B[0x8].d, B[0x5].d, B[0x1].d, B[0xE].d, C[0x0].d); \
    PP2(A[0x5].d, A[0x4].d, B[0x9].d, B[0x6].d, B[0x2].d, B[0xF].d, C[0xF].d); \
    PP2(A[0x6].d, A[0x5].d, B[0xA].d, B[0x7].d, B[0x3].d, B[0x0].d, C[0xE].d); \
    PP2(A[0x7].d, A[0x6].d, B[0xB].d, B[0x8].d, B[0x4].d, B[0x1].d, C[0xD].d); \
    PP2(A[0x8].d, A[0x7].d, B[0xC].d, B[0x9].d, B[0x5].d, B[0x2].d, C[0xC].d); \
    PP2(A[0x9].d, A[0x8].d, B[0xD].d, B[0xA].d, B[0x6].d, B[0x3].d, C[0xB].d); \
    PP2(A[0xA].d, A[0x9].d, B[0xE].d, B[0xB].d, B[0x7].d, B[0x4].d, C[0xA].d); \
    PP2(A[0xB].d, A[0xA].d, B[0xF].d, B[0xC].d, B[0x8].d, B[0x5].d, C[0x9].d);

#define ADDA(A, C) \
    A[0x0].d = _mm256_add_epi32(A[0x0].d, C[0xB].d); \
    A[0x0].d = _mm256_add_epi32(A[0x0].d, C[0xF].d); \
    A[0x0].d = _mm256_add_epi32(A[0x0].d, C[0x3].d); \
    A[0x1].d = _mm256_add_epi32(A[0x1].d, C[0xC].d); \
    A[0x1].d = _mm256_add_epi32(A[0x1].d, C[0x0].d); \
    A[0x1].d = _mm256_add_epi32(A[0x1].d, C[0x4].d); \
    A[0x2].d = _mm256_add_epi32(A[0x2].d, C[0xD].d); \
    A[0x2].d = _mm256_add_epi32(A[0x2].d, C[0x1].d); \
    A[0x2].d = _mm256_add_epi32(A[0x2].d, C[0x5].d); \
    A[0x3].d = _mm256_add_epi32(A[0x3].d, C[0xE].d); \
    A[0x3].d = _mm256_add_epi32(A[0x3].d, C[0x2].d); \
    A[0x3].d = _mm256_add_epi32(A[0x3].d, C[0x6].d); \
    A[0x4].d = _mm256_add_epi32(A[0x4].d, C[0xF].d); \
    A[0x4].d = _mm256_add_epi32(A[0x4].d, C[0x3].d); \
    A[0x4].d = _mm256_add_epi32(A[0x4].d, C[0x7].d); \
    A[0x5].d = _mm256_add_epi32(A[0x5].d, C[0x0].d); \
    A[0x5].d = _mm256_add_epi32(A[0x5].d, C[0x4].d); \
    A[0x5].d = _mm256_add_epi32(A[0x5].d, C[0x8].d); \
    A[0x6].d = _mm256_add_epi32(A[0x6].d, C[0x1].d); \
    A[0x6].d = _mm256_add_epi32(A[0x6].d, C[0x5].d); \
    A[0x6].d = _mm256_add_epi32(A[0x6].d, C[0x9].d); \
    A[0x7].d = _mm256_add_epi32(A[0x7].d, C[0x2].d); \
    A[0x7].d = _mm256_add_epi32(A[0x7].d, C[0x6].d); \
    A[0x7].d = _mm256_add_epi32(A[0x7].d, C[0xA].d); \
    A[0x8].d = _mm256_add_epi32(A[0x8].d, C[0x3].d); \
    A[0x8].d = _mm256_add_epi32(A[0x8].d, C[0x7].d); \
    A[0x8].d = _mm256_add_epi32(A[0x8].d, C[0xB].d); \
    A[0x9].d = _mm256_add_epi32(A[0x9].d, C[0x4].d); \
    A[0x9].d = _mm256_add_epi32(A[0x9].d, C[0x8].d); \
    A[0x9].d = _mm256_add_epi32(A[0x9].d, C[0xC].d); \
    A[0xA].d = _mm256_add_epi32(A[0xA].d, C[0x5].d); \
    A[0xA].d = _mm256_add_epi32(A[0xA].d, C[0x9].d); \
    A[0xA].d = _mm256_add_epi32(A[0xA].d, C[0xD].d); \
    A[0xB].d = _mm256_add_epi32(A[0xB].d, C[0x6].d); \
    A[0xB].d = _mm256_add_epi32(A[0xB].d, C[0xA].d); \
    A[0xB].d = _mm256_add_epi32(A[0xB].d, C[0xE].d);

#define SWAPA(X, M1) \
    X[0x0].d = _mm256_sub_epi32(X[0x0].d, M1);

	INIT(sc->stateA, sc->stateB, eighty, zero);
	PPA(sc->stateA, sc->stateB, sc->stateC, eighty);
	ADDA(sc->stateA, sc->stateC);
	SWAPA(sc->stateC, eighty);

	INIT(sc->stateA, sc->stateC, eighty, zero);
	PPA(sc->stateA, sc->stateC, sc->stateB, eighty);
	ADDA(sc->stateA, sc->stateB);
	SWAPA(sc->stateB, eighty);

	INIT(sc->stateA, sc->stateB, eighty, zero);
	PPA(sc->stateA, sc->stateB, sc->stateC, eighty);
	ADDA(sc->stateA, sc->stateC);
	SWAPA(sc->stateC, eighty);

	INIT(sc->stateA, sc->stateC, eighty, zero);
	PPA(sc->stateA, sc->stateC, sc->stateB, eighty);

#undef M
#undef PP
#undef PP2
#undef INIT
#undef PPA
#undef ADDA
#undef SWAPA
}
#endif

#ifndef __MINGW__
__declspec(noinline)
#else
__attribute__((noinline))
#endif
static void mshabal4_compress_fast4_64(mshabal4_context *sc)
{
	size_t j;

	__m128i one = _mm_set1_epi32(C32(0xFFFFFFFF));
	__m128i eighty = _mm_setr_epi32(0x80, 0x80, 0x80, 0x80);
	__m128i zero = _mm_setzero_si128();

#define PP(xa0, xa1, xb0, xb1, xb2, xb3, xc, xm)   do { \
    __m128i tt; \
    tt = _mm_or_si128(_mm_slli_epi32(xa1, 15), \
        _mm_srli_epi32(xa1, 17)); \
    tt = _mm_add_epi32(_mm_slli_epi32(tt, 2), tt); \
    tt = _mm_xor_si128(_mm_xor_si128(xa0, tt), xc); \
    tt = _mm_add_epi32(_mm_slli_epi32(tt, 1), tt); \
    tt = _mm_xor_si128( \
        _mm_xor_si128(tt, xb1), \
        _mm_xor_si128(_mm_andnot_si128(xb3, xb2), xm)); \
    xa0 = tt; \
    tt = xb0; \
    tt = _mm_or_si128(_mm_slli_epi32(tt, 1), \
        _mm_srli_epi32(tt, 31)); \
    xb0 = _mm_xor_si128(tt, _mm_xor_si128(xa0, one)); \
		        } while (0)

#define PP2(xa0, xa1, xb0, xb1, xb2, xb3, xc)   do { \
    __m128i tt; \
    tt = _mm_or_si128(_mm_slli_epi32(xa1, 15), \
        _mm_srli_epi32(xa1, 17)); \
    tt = _mm_add_epi32(_mm_slli_epi32(tt, 2), tt); \
    tt = _mm_xor_si128(_mm_xor_si128(xa0, tt), xc); \
    tt = _mm_add_epi32(_mm_slli_epi32(tt, 1), tt); \
    tt = _mm_xor_si128( \
        _mm_xor_si128(tt, xb1), _mm_andnot_si128(xb3, xb2)); \
    xa0 = tt; \
    tt = xb0; \
    tt = _mm_or_si128(_mm_slli_epi32(tt, 1), \
        _mm_srli_epi32(tt, 31)); \
    xb0 = _mm_xor_si128(tt, _mm_xor_si128(xa0, one)); \
		        } while (0)

#define INIT(A, B, M1, M2) \
    B[0].d = _mm_add_epi32(B[0].d, M1); \
    B[0].d = _mm_or_si128(_mm_slli_epi32(B[0].d, 17), _mm_srli_epi32(B[0].d, 15)); \
    for (j = 1; j < 16; j++) { \
        B[j].d = _mm_add_epi32(B[j].d, M2); \
        B[j].d = _mm_or_si128(_mm_slli_epi32(B[j].d, 17), _mm_srli_epi32(B[j].d, 15)); \
		    } \
    A[0].d = _mm_xor_si128(A[0].d, _mm_set1_epi32(2)); \
    A[1].d = _mm_xor_si128(A[1].d, _mm_set1_epi32(0));

#define PPA(A, B, C, M) \
    PP(A[0x0].d, A[0xB].d, B[0x0].d, B[0xD].d, B[0x9].d, B[0x6].d, C[0x8].d, M); \
    PP2(A[0x1].d, A[0x0].d, B[0x1].d, B[0xE].d, B[0xA].d, B[0x7].d, C[0x7].d); \
    PP2(A[0x2].d, A[0x1].d, B[0x2].d, B[0xF].d, B[0xB].d, B[0x8].d, C[0x6].d); \
    PP2(A[0x3].d, A[0x2].d, B[0x3].d, B[0x0].d, B[0xC].d, B[0x9].d, C[0x5].d); \
    PP2(A[0x4].d, A[0x3].d, B[0x4].d, B[0x1].d, B[0xD].d, B[0xA].d, C[0x4].d); \
    PP2(A[0x5].d, A[0x4].d, B[0x5].d, B[0x2].d, B[0xE].d, B[0xB].d, C[0x3].d); \
    PP2(A[0x6].d, A[0x5].d, B[0x6].d, B[0x3].d, B[0xF].d, B[0xC].d, C[0x2].d); \
    PP2(A[0x7].d, A[0x6].d, B[0x7].d, B[0x4].d, B[0x0].d, B[0xD].d, C[0x1].d); \
    PP2(A[0x8].d, A[0x7].d, B[0x8].d, B[0x5].d, B[0x1].d, B[0xE].d, C[0x0].d); \
    PP2(A[0x9].d, A[0x8].d, B[0x9].d, B[0x6].d, B[0x2].d, B[0xF].d, C[0xF].d); \
    PP2(A[0xA].d, A[0x9].d, B[0xA].d, B[0x7].d, B[0x3].d, B[0x0].d, C[0xE].d); \
    PP2(A[0xB].d, A[0xA].d, B[0xB].d, B[0x8].d, B[0x4].d, B[0x1].d, C[0xD].d); \
    PP2(A[0x0].d, A[0xB].d, B[0xC].d, B[0x9].d, B[0x5].d, B[0x2].d, C[0xC].d); \
    PP2(A[0x1].d, A[0x0].d, B[0xD].d, B[0xA].d, B[0x6].d, B[0x3].d, C[0xB].d); \
    PP2(A[0x2].d, A[0x1].d, B[0xE].d, B[0xB].d, B[0x7].d, B[0x4].d, C[0xA].d); \
    PP2(A[0x3].d, A[0x2].d, B[0xF].d, B[0xC].d, B[0x8].d, B[0x5].d, C[0x9].d); \
    PP(A[0x4].d, A[0x3].d, B[0x0].d, B[0xD].d, B[0x9].d, B[0x6].d, C[0x8].d, M); \
    PP2(A[0x5].d, A[0x4].d, B[0x1].d, B[0xE].d, B[0xA].d, B[0x7].d, C[0x7].d); \
    PP2(A[0x6].d, A[0x5].d, B[0x2].d, B[0xF].d, B[0xB].d, B[0x8].d, C[0x6].d); \
    PP2(A[0x7].d, A[0x6].d, B[0x3].d, B[0x0].d, B[0xC].d, B[0x9].d, C[0x5].d); \
    PP2(A[0x8].d, A[0x7].d, B[0x4].d, B[0x1].d, B[0xD].d, B[0xA].d, C[0x4].d); \
    PP2(A[0x9].d, A[0x8].d, B[0x5].d, B[0x2].d, B[0xE].d, B[0xB].d, C[0x3].d); \
    PP2(A[0xA].d, A[0x9].d, B[0x6].d, B[0x3].d, B[0xF].d, B[0xC].d, C[0x2].d); \
    PP2(A[0xB].d, A[0xA].d, B[0x7].d, B[0x4].d, B[0x0].d, B[0xD].d, C[0x1].d); \
    PP2(A[0x0].d, A[0xB].d, B[0x8].d, B[0x5].d, B[0x1].d, B[0xE].d, C[0x0].d); \
    PP2(A[0x1].d, A[0x0].d, B[0x9].d, B[0x6].d, B[0x2].d, B[0xF].d, C[0xF].d); \
    PP2(A[0x2].d, A[0x1].d, B[0xA].d, B[0x7].d, B[0x3].d, B[0x0].d, C[0xE].d); \
    PP2(A[0x3].d, A[0x2].d, B[0xB].d, B[0x8].d, B[0x4].d, B[0x1].d, C[0xD].d); \
    PP2(A[0x4].d, A[0x3].d, B[0xC].d, B[0x9].d, B[0x5].d, B[0x2].d, C[0xC].d); \
    PP2(A[0x5].d, A[0x4].d, B[0xD].d, B[0xA].d, B[0x6].d, B[0x3].d, C[0xB].d); \
    PP2(A[0x6].d, A[0x5].d, B[0xE].d, B[0xB].d, B[0x7].d, B[0x4].d, C[0xA].d); \
    PP2(A[0x7].d, A[0x6].d, B[0xF].d, B[0xC].d, B[0x8].d, B[0x5].d, C[0x9].d); \
    PP(A[0x8].d, A[0x7].d, B[0x0].d, B[0xD].d, B[0x9].d, B[0x6].d, C[0x8].d, M); \
    PP2(A[0x9].d, A[0x8].d, B[0x1].d, B[0xE].d, B[0xA].d, B[0x7].d, C[0x7].d); \
    PP2(A[0xA].d, A[0x9].d, B[0x2].d, B[0xF].d, B[0xB].d, B[0x8].d, C[0x6].d); \
    PP2(A[0xB].d, A[0xA].d, B[0x3].d, B[0x0].d, B[0xC].d, B[0x9].d, C[0x5].d); \
    PP2(A[0x0].d, A[0xB].d, B[0x4].d, B[0x1].d, B[0xD].d, B[0xA].d, C[0x4].d); \
    PP2(A[0x1].d, A[0x0].d, B[0x5].d, B[0x2].d, B[0xE].d, B[0xB].d, C[0x3].d); \
    PP2(A[0x2].d, A[0x1].d, B[0x6].d, B[0x3].d, B[0xF].d, B[0xC].d, C[0x2].d); \
    PP2(A[0x3].d, A[0x2].d, B[0x7].d, B[0x4].d, B[0x0].d, B[0xD].d, C[0x1].d); \
    PP2(A[0x4].d, A[0x3].d, B[0x8].d, B[0x5].d, B[0x1].d, B[0xE].d, C[0x0].d); \
    PP2(A[0x5].d, A[0x4].d, B[0x9].d, B[0x6].d, B[0x2].d, B[0xF].d, C[0xF].d); \
    PP2(A[0x6].d, A[0x5].d, B[0xA].d, B[0x7].d, B[0x3].d, B[0x0].d, C[0xE].d); \
    PP2(A[0x7].d, A[0x6].d, B[0xB].d, B[0x8].d, B[0x4].d, B[0x1].d, C[0xD].d); \
    PP2(A[0x8].d, A[0x7].d, B[0xC].d, B[0x9].d, B[0x5].d, B[0x2].d, C[0xC].d); \
    PP2(A[0x9].d, A[0x8].d, B[0xD].d, B[0xA].d, B[0x6].d, B[0x3].d, C[0xB].d); \
    PP2(A[0xA].d, A[0x9].d, B[0xE].d, B[0xB].d, B[0x7].d, B[0x4].d, C[0xA].d); \
    PP2(A[0xB].d, A[0xA].d, B[0xF].d, B[0xC].d, B[0x8].d, B[0x5].d, C[0x9].d);

#define ADDA(A, C) \
    A[0x0].d = _mm_add_epi32(A[0x0].d, C[0xB].d); \
    A[0x0].d = _mm_add_epi32(A[0x0].d, C[0xF].d); \
    A[0x0].d = _mm_add_epi32(A[0x0].d, C[0x3].d); \
    A[0x1].d = _mm_add_epi32(A[0x1].d, C[0xC].d); \
    A[0x1].d = _mm_add_epi32(A[0x1].d, C[0x0].d); \
    A[0x1].d = _mm_add_epi32(A[0x1].d, C[0x4].d); \
    A[0x2].d = _mm_add_epi32(A[0x2].d, C[0xD].d); \
    A[0x2].d = _mm_add_epi32(A[0x2].d, C[0x1].d); \
    A[0x2].d = _mm_add_epi32(A[0x2].d, C[0x5].d); \
    A[0x3].d = _mm_add_epi32(A[0x3].d, C[0xE].d); \
    A[0x3].d = _mm_add_epi32(A[0x3].d, C[0x2].d); \
    A[0x3].d = _mm_add_epi32(A[0x3].d, C[0x6].d); \
    A[0x4].d = _mm_add_epi32(A[0x4].d, C[0xF].d); \
    A[0x4].d = _mm_add_epi32(A[0x4].d, C[0x3].d); \
    A[0x4].d = _mm_add_epi32(A[0x4].d, C[0x7].d); \
    A[0x5].d = _mm_add_epi32(A[0x5].d, C[0x0].d); \
    A[0x5].d = _mm_add_epi32(A[0x5].d, C[0x4].d); \
    A[0x5].d = _mm_add_epi32(A[0x5].d, C[0x8].d); \
    A[0x6].d = _mm_add_epi32(A[0x6].d, C[0x1].d); \
    A[0x6].d = _mm_add_epi32(A[0x6].d, C[0x5].d); \
    A[0x6].d = _mm_add_epi32(A[0x6].d, C[0x9].d); \
    A[0x7].d = _mm_add_epi32(A[0x7].d, C[0x2].d); \
    A[0x7].d = _mm_add_epi32(A[0x7].d, C[0x6].d); \
    A[0x7].d = _mm_add_epi32(A[0x7].d, C[0xA].d); \
    A[0x8].d = _mm_add_epi32(A[0x8].d, C[0x3].d); \
    A[0x8].d = _mm_add_epi32(A[0x8].d, C[0x7].d); \
    A[0x8].d = _mm_add_epi32(A[0x8].d, C[0xB].d); \
    A[0x9].d = _mm_add_epi32(A[0x9].d, C[0x4].d); \
    A[0x9].d = _mm_add_epi32(A[0x9].d, C[0x8].d); \
    A[0x9].d = _mm_add_epi32(A[0x9].d, C[0xC].d); \
    A[0xA].d = _mm_add_epi32(A[0xA].d, C[0x5].d); \
    A[0xA].d = _mm_add_epi32(A[0xA].d, C[0x9].d); \
    A[0xA].d = _mm_add_epi32(A[0xA].d, C[0xD].d); \
    A[0xB].d = _mm_add_epi32(A[0xB].d, C[0x6].d); \
    A[0xB].d = _mm_add_epi32(A[0xB].d, C[0xA].d); \
    A[0xB].d = _mm_add_epi32(A[0xB].d, C[0xE].d);

#define SWAPA(X, M1) \
    X[0x0].d = _mm_sub_epi32(X[0x0].d, M1);

	INIT(sc->stateA, sc->stateB, eighty, zero);
	PPA(sc->stateA, sc->stateB, sc->stateC, eighty);
	ADDA(sc->stateA, sc->stateC);
	SWAPA(sc->stateC, eighty);

	INIT(sc->stateA, sc->stateC, eighty, zero);
	PPA(sc->stateA, sc->stateC, sc->stateB, eighty);
	ADDA(sc->stateA, sc->stateB);
	SWAPA(sc->stateB, eighty);

	INIT(sc->stateA, sc->stateB, eighty, zero);
	PPA(sc->stateA, sc->stateB, sc->stateC, eighty);
	ADDA(sc->stateA, sc->stateC);
	SWAPA(sc->stateC, eighty);

	INIT(sc->stateA, sc->stateC, eighty, zero);
	PPA(sc->stateA, sc->stateC, sc->stateB, eighty);

#undef M
#undef PP
#undef PP2
#undef INIT
#undef PPA
#undef ADDA
#undef SWAPA
}

#ifdef __8WAY__
#ifndef __MINGW__
__declspec(noinline)
#else
__attribute__((noinline))
#endif
static void mshabal8_compress_fast4_32(mshabal8_context *sc, const void *data)
{
	size_t j;
	un u;

	u.quads[0] = ((u64*)data)[0];
	u.quads[1] = ((u64*)data)[1];
	u.quads[2] = ((u64*)data)[2];
	u.quads[3] = ((u64*)data)[3];
	u.quads[4] = ((u64*)data)[4];
	u.quads[5] = ((u64*)data)[5];
	u.quads[6] = ((u64*)data)[6];
	u.quads[7] = ((u64*)data)[7];
	u.quads[8] = ((u64*)data)[8];
	u.quads[9] = ((u64*)data)[9];
	u.quads[10] = ((u64*)data)[10];
	u.quads[11] = ((u64*)data)[11];
	u.quads[12] = ((u64*)data)[12];
	u.quads[13] = ((u64*)data)[13];
	u.quads[14] = ((u64*)data)[14];
	u.quads[15] = ((u64*)data)[15];
	u.quads[16] = ((u64*)data)[16];
	u.quads[17] = ((u64*)data)[17];
	u.quads[18] = ((u64*)data)[18];
	u.quads[19] = ((u64*)data)[19];
	u.quads[20] = ((u64*)data)[20];
	u.quads[21] = ((u64*)data)[21];
	u.quads[22] = ((u64*)data)[22];
	u.quads[23] = ((u64*)data)[23];
	u.quads[24] = ((u64*)data)[24];
	u.quads[25] = ((u64*)data)[25];
	u.quads[26] = ((u64*)data)[26];
	u.quads[27] = ((u64*)data)[27];
	u.quads[28] = ((u64*)data)[28];
	u.quads[29] = ((u64*)data)[29];
	u.quads[30] = ((u64*)data)[30];
	u.quads[31] = ((u64*)data)[31];

	__m256i one = _mm256_set1_epi32(C32(0xFFFFFFFF));
	__m256i eighty = _mm256_setr_epi32(0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
	__m256i zero = _mm256_setzero_si256();

#define M(i) u.data[i]

#define PP(xa0, xa1, xb0, xb1, xb2, xb3, xc, xm)   do { \
    __m256i tt; \
    tt = _mm256_or_si256(_mm256_slli_epi32(xa1, 15), \
        _mm256_srli_epi32(xa1, 17)); \
    tt = _mm256_add_epi32(_mm256_slli_epi32(tt, 2), tt); \
    tt = _mm256_xor_si256(_mm256_xor_si256(xa0, tt), xc); \
    tt = _mm256_add_epi32(_mm256_slli_epi32(tt, 1), tt); \
    tt = _mm256_xor_si256( \
        _mm256_xor_si256(tt, xb1), \
        _mm256_xor_si256(_mm256_andnot_si256(xb3, xb2), xm)); \
    xa0 = tt; \
    tt = xb0; \
    tt = _mm256_or_si256(_mm256_slli_epi32(tt, 1), \
        _mm256_srli_epi32(tt, 31)); \
    xb0 = _mm256_xor_si256(tt, _mm256_xor_si256(xa0, one)); \
	        } while (0)

#define PP2(xa0, xa1, xb0, xb1, xb2, xb3, xc)   do { \
    __m256i tt; \
    tt = _mm256_or_si256(_mm256_slli_epi32(xa1, 15), \
        _mm256_srli_epi32(xa1, 17)); \
    tt = _mm256_add_epi32(_mm256_slli_epi32(tt, 2), tt); \
    tt = _mm256_xor_si256(_mm256_xor_si256(xa0, tt), xc); \
    tt = _mm256_add_epi32(_mm256_slli_epi32(tt, 1), tt); \
    tt = _mm256_xor_si256( \
        _mm256_xor_si256(tt, xb1), _mm256_andnot_si256(xb3, xb2)); \
    xa0 = tt; \
    tt = xb0; \
    tt = _mm256_or_si256(_mm256_slli_epi32(tt, 1), \
        _mm256_srli_epi32(tt, 31)); \
    xb0 = _mm256_xor_si256(tt, _mm256_xor_si256(xa0, one)); \
	        } while (0)

#define INIT(A, B, M, M1, M2) \
	B[0].d = _mm256_add_epi32(B[0].d, M(0)); \
	B[0].d = _mm256_or_si256(_mm256_slli_epi32(B[0].d, 17), _mm256_srli_epi32(B[0].d, 15)); \
	B[1].d = _mm256_add_epi32(B[1].d, M(1)); \
	B[1].d = _mm256_or_si256(_mm256_slli_epi32(B[1].d, 17), _mm256_srli_epi32(B[1].d, 15)); \
	B[2].d = _mm256_add_epi32(B[2].d, M(2)); \
	B[2].d = _mm256_or_si256(_mm256_slli_epi32(B[2].d, 17), _mm256_srli_epi32(B[2].d, 15)); \
	B[3].d = _mm256_add_epi32(B[3].d, M(3)); \
	B[3].d = _mm256_or_si256(_mm256_slli_epi32(B[3].d, 17), _mm256_srli_epi32(B[3].d, 15)); \
	B[4].d = _mm256_add_epi32(B[4].d, M(4)); \
	B[4].d = _mm256_or_si256(_mm256_slli_epi32(B[4].d, 17), _mm256_srli_epi32(B[4].d, 15)); \
	B[5].d = _mm256_add_epi32(B[5].d, M(5)); \
	B[5].d = _mm256_or_si256(_mm256_slli_epi32(B[5].d, 17), _mm256_srli_epi32(B[5].d, 15)); \
	B[6].d = _mm256_add_epi32(B[6].d, M(6)); \
	B[6].d = _mm256_or_si256(_mm256_slli_epi32(B[6].d, 17), _mm256_srli_epi32(B[6].d, 15)); \
	B[7].d = _mm256_add_epi32(B[7].d, M(7)); \
	B[7].d = _mm256_or_si256(_mm256_slli_epi32(B[7].d, 17), _mm256_srli_epi32(B[7].d, 15)); \
	B[8].d = _mm256_add_epi32(B[8].d, M1); \
	B[8].d = _mm256_or_si256(_mm256_slli_epi32(B[8].d, 17), _mm256_srli_epi32(B[8].d, 15)); \
	for (j = 9; j < 16; j++) { \
        B[j].d = _mm256_add_epi32(B[j].d, M2); \
        B[j].d = _mm256_or_si256(_mm256_slli_epi32(B[j].d, 17), _mm256_srli_epi32(B[j].d, 15)); \
		    } \
    A[0].d = _mm256_xor_si256(A[0].d, _mm256_set1_epi32(1)); \
    A[1].d = _mm256_xor_si256(A[1].d, _mm256_set1_epi32(0));

#define PPA(A, B, C, M, M1) \
    PP(A[0x0].d, A[0xB].d, B[0x0].d, B[0xD].d, B[0x9].d, B[0x6].d, C[0x8].d, M(0x0)); \
    PP(A[0x1].d, A[0x0].d, B[0x1].d, B[0xE].d, B[0xA].d, B[0x7].d, C[0x7].d, M(0x1)); \
    PP(A[0x2].d, A[0x1].d, B[0x2].d, B[0xF].d, B[0xB].d, B[0x8].d, C[0x6].d, M(0x2)); \
    PP(A[0x3].d, A[0x2].d, B[0x3].d, B[0x0].d, B[0xC].d, B[0x9].d, C[0x5].d, M(0x3)); \
    PP(A[0x4].d, A[0x3].d, B[0x4].d, B[0x1].d, B[0xD].d, B[0xA].d, C[0x4].d, M(0x4)); \
    PP(A[0x5].d, A[0x4].d, B[0x5].d, B[0x2].d, B[0xE].d, B[0xB].d, C[0x3].d, M(0x5)); \
    PP(A[0x6].d, A[0x5].d, B[0x6].d, B[0x3].d, B[0xF].d, B[0xC].d, C[0x2].d, M(0x6)); \
    PP(A[0x7].d, A[0x6].d, B[0x7].d, B[0x4].d, B[0x0].d, B[0xD].d, C[0x1].d, M(0x7)); \
    PP(A[0x8].d, A[0x7].d, B[0x8].d, B[0x5].d, B[0x1].d, B[0xE].d, C[0x0].d, M1); \
    PP2(A[0x9].d, A[0x8].d, B[0x9].d, B[0x6].d, B[0x2].d, B[0xF].d, C[0xF].d); \
    PP2(A[0xA].d, A[0x9].d, B[0xA].d, B[0x7].d, B[0x3].d, B[0x0].d, C[0xE].d); \
    PP2(A[0xB].d, A[0xA].d, B[0xB].d, B[0x8].d, B[0x4].d, B[0x1].d, C[0xD].d); \
    PP2(A[0x0].d, A[0xB].d, B[0xC].d, B[0x9].d, B[0x5].d, B[0x2].d, C[0xC].d); \
    PP2(A[0x1].d, A[0x0].d, B[0xD].d, B[0xA].d, B[0x6].d, B[0x3].d, C[0xB].d); \
    PP2(A[0x2].d, A[0x1].d, B[0xE].d, B[0xB].d, B[0x7].d, B[0x4].d, C[0xA].d); \
    PP2(A[0x3].d, A[0x2].d, B[0xF].d, B[0xC].d, B[0x8].d, B[0x5].d, C[0x9].d); \
    PP(A[0x4].d, A[0x3].d, B[0x0].d, B[0xD].d, B[0x9].d, B[0x6].d, C[0x8].d, M(0x0)); \
    PP(A[0x5].d, A[0x4].d, B[0x1].d, B[0xE].d, B[0xA].d, B[0x7].d, C[0x7].d, M(0x1)); \
    PP(A[0x6].d, A[0x5].d, B[0x2].d, B[0xF].d, B[0xB].d, B[0x8].d, C[0x6].d, M(0x2)); \
    PP(A[0x7].d, A[0x6].d, B[0x3].d, B[0x0].d, B[0xC].d, B[0x9].d, C[0x5].d, M(0x3)); \
    PP(A[0x8].d, A[0x7].d, B[0x4].d, B[0x1].d, B[0xD].d, B[0xA].d, C[0x4].d, M(0x4)); \
    PP(A[0x9].d, A[0x8].d, B[0x5].d, B[0x2].d, B[0xE].d, B[0xB].d, C[0x3].d, M(0x5)); \
    PP(A[0xA].d, A[0x9].d, B[0x6].d, B[0x3].d, B[0xF].d, B[0xC].d, C[0x2].d, M(0x6)); \
    PP(A[0xB].d, A[0xA].d, B[0x7].d, B[0x4].d, B[0x0].d, B[0xD].d, C[0x1].d, M(0x7)); \
    PP(A[0x0].d, A[0xB].d, B[0x8].d, B[0x5].d, B[0x1].d, B[0xE].d, C[0x0].d, M1); \
    PP2(A[0x1].d, A[0x0].d, B[0x9].d, B[0x6].d, B[0x2].d, B[0xF].d, C[0xF].d); \
    PP2(A[0x2].d, A[0x1].d, B[0xA].d, B[0x7].d, B[0x3].d, B[0x0].d, C[0xE].d); \
    PP2(A[0x3].d, A[0x2].d, B[0xB].d, B[0x8].d, B[0x4].d, B[0x1].d, C[0xD].d); \
    PP2(A[0x4].d, A[0x3].d, B[0xC].d, B[0x9].d, B[0x5].d, B[0x2].d, C[0xC].d); \
    PP2(A[0x5].d, A[0x4].d, B[0xD].d, B[0xA].d, B[0x6].d, B[0x3].d, C[0xB].d); \
    PP2(A[0x6].d, A[0x5].d, B[0xE].d, B[0xB].d, B[0x7].d, B[0x4].d, C[0xA].d); \
    PP2(A[0x7].d, A[0x6].d, B[0xF].d, B[0xC].d, B[0x8].d, B[0x5].d, C[0x9].d); \
    PP(A[0x8].d, A[0x7].d, B[0x0].d, B[0xD].d, B[0x9].d, B[0x6].d, C[0x8].d, M(0x0)); \
    PP(A[0x9].d, A[0x8].d, B[0x1].d, B[0xE].d, B[0xA].d, B[0x7].d, C[0x7].d, M(0x1)); \
    PP(A[0xA].d, A[0x9].d, B[0x2].d, B[0xF].d, B[0xB].d, B[0x8].d, C[0x6].d, M(0x2)); \
    PP(A[0xB].d, A[0xA].d, B[0x3].d, B[0x0].d, B[0xC].d, B[0x9].d, C[0x5].d, M(0x3)); \
    PP(A[0x0].d, A[0xB].d, B[0x4].d, B[0x1].d, B[0xD].d, B[0xA].d, C[0x4].d, M(0x4)); \
    PP(A[0x1].d, A[0x0].d, B[0x5].d, B[0x2].d, B[0xE].d, B[0xB].d, C[0x3].d, M(0x5)); \
    PP(A[0x2].d, A[0x1].d, B[0x6].d, B[0x3].d, B[0xF].d, B[0xC].d, C[0x2].d, M(0x6)); \
    PP(A[0x3].d, A[0x2].d, B[0x7].d, B[0x4].d, B[0x0].d, B[0xD].d, C[0x1].d, M(0x7)); \
    PP(A[0x4].d, A[0x3].d, B[0x8].d, B[0x5].d, B[0x1].d, B[0xE].d, C[0x0].d, M1); \
    PP2(A[0x5].d, A[0x4].d, B[0x9].d, B[0x6].d, B[0x2].d, B[0xF].d, C[0xF].d); \
    PP2(A[0x6].d, A[0x5].d, B[0xA].d, B[0x7].d, B[0x3].d, B[0x0].d, C[0xE].d); \
    PP2(A[0x7].d, A[0x6].d, B[0xB].d, B[0x8].d, B[0x4].d, B[0x1].d, C[0xD].d); \
    PP2(A[0x8].d, A[0x7].d, B[0xC].d, B[0x9].d, B[0x5].d, B[0x2].d, C[0xC].d); \
    PP2(A[0x9].d, A[0x8].d, B[0xD].d, B[0xA].d, B[0x6].d, B[0x3].d, C[0xB].d); \
    PP2(A[0xA].d, A[0x9].d, B[0xE].d, B[0xB].d, B[0x7].d, B[0x4].d, C[0xA].d); \
    PP2(A[0xB].d, A[0xA].d, B[0xF].d, B[0xC].d, B[0x8].d, B[0x5].d, C[0x9].d);

#define ADDA(A, C) \
    A[0x0].d = _mm256_add_epi32(A[0x0].d, C[0xB].d); \
    A[0x0].d = _mm256_add_epi32(A[0x0].d, C[0xF].d); \
    A[0x0].d = _mm256_add_epi32(A[0x0].d, C[0x3].d); \
    A[0x1].d = _mm256_add_epi32(A[0x1].d, C[0xC].d); \
    A[0x1].d = _mm256_add_epi32(A[0x1].d, C[0x0].d); \
    A[0x1].d = _mm256_add_epi32(A[0x1].d, C[0x4].d); \
    A[0x2].d = _mm256_add_epi32(A[0x2].d, C[0xD].d); \
    A[0x2].d = _mm256_add_epi32(A[0x2].d, C[0x1].d); \
    A[0x2].d = _mm256_add_epi32(A[0x2].d, C[0x5].d); \
    A[0x3].d = _mm256_add_epi32(A[0x3].d, C[0xE].d); \
    A[0x3].d = _mm256_add_epi32(A[0x3].d, C[0x2].d); \
    A[0x3].d = _mm256_add_epi32(A[0x3].d, C[0x6].d); \
    A[0x4].d = _mm256_add_epi32(A[0x4].d, C[0xF].d); \
    A[0x4].d = _mm256_add_epi32(A[0x4].d, C[0x3].d); \
    A[0x4].d = _mm256_add_epi32(A[0x4].d, C[0x7].d); \
    A[0x5].d = _mm256_add_epi32(A[0x5].d, C[0x0].d); \
    A[0x5].d = _mm256_add_epi32(A[0x5].d, C[0x4].d); \
    A[0x5].d = _mm256_add_epi32(A[0x5].d, C[0x8].d); \
    A[0x6].d = _mm256_add_epi32(A[0x6].d, C[0x1].d); \
    A[0x6].d = _mm256_add_epi32(A[0x6].d, C[0x5].d); \
    A[0x6].d = _mm256_add_epi32(A[0x6].d, C[0x9].d); \
    A[0x7].d = _mm256_add_epi32(A[0x7].d, C[0x2].d); \
    A[0x7].d = _mm256_add_epi32(A[0x7].d, C[0x6].d); \
    A[0x7].d = _mm256_add_epi32(A[0x7].d, C[0xA].d); \
    A[0x8].d = _mm256_add_epi32(A[0x8].d, C[0x3].d); \
    A[0x8].d = _mm256_add_epi32(A[0x8].d, C[0x7].d); \
    A[0x8].d = _mm256_add_epi32(A[0x8].d, C[0xB].d); \
    A[0x9].d = _mm256_add_epi32(A[0x9].d, C[0x4].d); \
    A[0x9].d = _mm256_add_epi32(A[0x9].d, C[0x8].d); \
    A[0x9].d = _mm256_add_epi32(A[0x9].d, C[0xC].d); \
    A[0xA].d = _mm256_add_epi32(A[0xA].d, C[0x5].d); \
    A[0xA].d = _mm256_add_epi32(A[0xA].d, C[0x9].d); \
    A[0xA].d = _mm256_add_epi32(A[0xA].d, C[0xD].d); \
    A[0xB].d = _mm256_add_epi32(A[0xB].d, C[0x6].d); \
    A[0xB].d = _mm256_add_epi32(A[0xB].d, C[0xA].d); \
    A[0xB].d = _mm256_add_epi32(A[0xB].d, C[0xE].d);

#define SWAPA(X, M, M1) \
    X[0x0].d = _mm256_sub_epi32(X[0x0].d, M(0x0)); \
    X[0x1].d = _mm256_sub_epi32(X[0x1].d, M(0x1)); \
    X[0x2].d = _mm256_sub_epi32(X[0x2].d, M(0x2)); \
    X[0x3].d = _mm256_sub_epi32(X[0x3].d, M(0x3)); \
    X[0x4].d = _mm256_sub_epi32(X[0x4].d, M(0x4)); \
    X[0x5].d = _mm256_sub_epi32(X[0x5].d, M(0x5)); \
    X[0x6].d = _mm256_sub_epi32(X[0x6].d, M(0x6)); \
    X[0x7].d = _mm256_sub_epi32(X[0x7].d, M(0x7)); \
    X[0x8].d = _mm256_sub_epi32(X[0x8].d, M1); \

	INIT(sc->stateA, sc->stateB, M, eighty, zero);
	PPA(sc->stateA, sc->stateB, sc->stateC, M, eighty);
	ADDA(sc->stateA, sc->stateC);
	SWAPA(sc->stateC, M, eighty);

	INIT(sc->stateA, sc->stateC, M, eighty, zero);
	PPA(sc->stateA, sc->stateC, sc->stateB, M, eighty);
	ADDA(sc->stateA, sc->stateB);
	SWAPA(sc->stateB, M, eighty);

	INIT(sc->stateA, sc->stateB, M, eighty, zero);
	PPA(sc->stateA, sc->stateB, sc->stateC, M, eighty);
	ADDA(sc->stateA, sc->stateC);
	SWAPA(sc->stateC, M, eighty);

	INIT(sc->stateA, sc->stateC, M, eighty, zero);
	PPA(sc->stateA, sc->stateC, sc->stateB, M, eighty);

#undef M
#undef PP
#undef PP2
#undef INIT
#undef PPA
#undef ADDA
#undef SWAPA
}
#endif

#ifndef __MINGW__
__declspec(noinline)
#else
__attribute__((noinline))
#endif
static void mshabal4_compress_fast4_32(mshabal4_context *sc, const void *data)
{
	size_t j;
	un4 u;

	u.quads[0] = ((u64*)data)[0];
	u.quads[1] = ((u64*)data)[1];
	u.quads[2] = ((u64*)data)[2];
	u.quads[3] = ((u64*)data)[3];
	u.quads[4] = ((u64*)data)[4];
	u.quads[5] = ((u64*)data)[5];
	u.quads[6] = ((u64*)data)[6];
	u.quads[7] = ((u64*)data)[7];
	u.quads[8] = ((u64*)data)[8];
	u.quads[9] = ((u64*)data)[9];
	u.quads[10] = ((u64*)data)[10];
	u.quads[11] = ((u64*)data)[11];
	u.quads[12] = ((u64*)data)[12];
	u.quads[13] = ((u64*)data)[13];
	u.quads[14] = ((u64*)data)[14];
	u.quads[15] = ((u64*)data)[15];

	__m128i one = _mm_set1_epi32(C32(0xFFFFFFFF));
	__m128i eighty = _mm_setr_epi32(0x80, 0x80, 0x80, 0x80);
	__m128i zero = _mm_setzero_si128();

#define M(i) u.data[i]

#define PP(xa0, xa1, xb0, xb1, xb2, xb3, xc, xm)   do { \
    __m128i tt; \
    tt = _mm_or_si128(_mm_slli_epi32(xa1, 15), \
        _mm_srli_epi32(xa1, 17)); \
    tt = _mm_add_epi32(_mm_slli_epi32(tt, 2), tt); \
    tt = _mm_xor_si128(_mm_xor_si128(xa0, tt), xc); \
    tt = _mm_add_epi32(_mm_slli_epi32(tt, 1), tt); \
    tt = _mm_xor_si128( \
        _mm_xor_si128(tt, xb1), \
        _mm_xor_si128(_mm_andnot_si128(xb3, xb2), xm)); \
    xa0 = tt; \
    tt = xb0; \
    tt = _mm_or_si128(_mm_slli_epi32(tt, 1), \
        _mm_srli_epi32(tt, 31)); \
    xb0 = _mm_xor_si128(tt, _mm_xor_si128(xa0, one)); \
			        } while (0)

#define PP2(xa0, xa1, xb0, xb1, xb2, xb3, xc)   do { \
    __m128i tt; \
    tt = _mm_or_si128(_mm_slli_epi32(xa1, 15), \
        _mm_srli_epi32(xa1, 17)); \
    tt = _mm_add_epi32(_mm_slli_epi32(tt, 2), tt); \
    tt = _mm_xor_si128(_mm_xor_si128(xa0, tt), xc); \
    tt = _mm_add_epi32(_mm_slli_epi32(tt, 1), tt); \
    tt = _mm_xor_si128( \
        _mm_xor_si128(tt, xb1), _mm_andnot_si128(xb3, xb2)); \
    xa0 = tt; \
    tt = xb0; \
    tt = _mm_or_si128(_mm_slli_epi32(tt, 1), \
        _mm_srli_epi32(tt, 31)); \
    xb0 = _mm_xor_si128(tt, _mm_xor_si128(xa0, one)); \
			        } while (0)

#define INIT(A, B, M, M1, M2) \
	B[0].d = _mm_add_epi32(B[0].d, M(0)); \
	B[0].d = _mm_or_si128(_mm_slli_epi32(B[0].d, 17), _mm_srli_epi32(B[0].d, 15)); \
	B[1].d = _mm_add_epi32(B[1].d, M(1)); \
	B[1].d = _mm_or_si128(_mm_slli_epi32(B[1].d, 17), _mm_srli_epi32(B[1].d, 15)); \
	B[2].d = _mm_add_epi32(B[2].d, M(2)); \
	B[2].d = _mm_or_si128(_mm_slli_epi32(B[2].d, 17), _mm_srli_epi32(B[2].d, 15)); \
	B[3].d = _mm_add_epi32(B[3].d, M(3)); \
	B[3].d = _mm_or_si128(_mm_slli_epi32(B[3].d, 17), _mm_srli_epi32(B[3].d, 15)); \
	B[4].d = _mm_add_epi32(B[4].d, M(4)); \
	B[4].d = _mm_or_si128(_mm_slli_epi32(B[4].d, 17), _mm_srli_epi32(B[4].d, 15)); \
	B[5].d = _mm_add_epi32(B[5].d, M(5)); \
	B[5].d = _mm_or_si128(_mm_slli_epi32(B[5].d, 17), _mm_srli_epi32(B[5].d, 15)); \
	B[6].d = _mm_add_epi32(B[6].d, M(6)); \
	B[6].d = _mm_or_si128(_mm_slli_epi32(B[6].d, 17), _mm_srli_epi32(B[6].d, 15)); \
	B[7].d = _mm_add_epi32(B[7].d, M(7)); \
	B[7].d = _mm_or_si128(_mm_slli_epi32(B[7].d, 17), _mm_srli_epi32(B[7].d, 15)); \
	B[8].d = _mm_add_epi32(B[8].d, M1); \
	B[8].d = _mm_or_si128(_mm_slli_epi32(B[8].d, 17), _mm_srli_epi32(B[8].d, 15)); \
	for (j = 9; j < 16; j++) { \
        B[j].d = _mm_add_epi32(B[j].d, M2); \
        B[j].d = _mm_or_si128(_mm_slli_epi32(B[j].d, 17), _mm_srli_epi32(B[j].d, 15)); \
			    } \
    A[0].d = _mm_xor_si128(A[0].d, _mm_set1_epi32(1)); \
    A[1].d = _mm_xor_si128(A[1].d, _mm_set1_epi32(0));

#define PPA(A, B, C, M, M1) \
    PP(A[0x0].d, A[0xB].d, B[0x0].d, B[0xD].d, B[0x9].d, B[0x6].d, C[0x8].d, M(0x0)); \
    PP(A[0x1].d, A[0x0].d, B[0x1].d, B[0xE].d, B[0xA].d, B[0x7].d, C[0x7].d, M(0x1)); \
    PP(A[0x2].d, A[0x1].d, B[0x2].d, B[0xF].d, B[0xB].d, B[0x8].d, C[0x6].d, M(0x2)); \
    PP(A[0x3].d, A[0x2].d, B[0x3].d, B[0x0].d, B[0xC].d, B[0x9].d, C[0x5].d, M(0x3)); \
    PP(A[0x4].d, A[0x3].d, B[0x4].d, B[0x1].d, B[0xD].d, B[0xA].d, C[0x4].d, M(0x4)); \
    PP(A[0x5].d, A[0x4].d, B[0x5].d, B[0x2].d, B[0xE].d, B[0xB].d, C[0x3].d, M(0x5)); \
    PP(A[0x6].d, A[0x5].d, B[0x6].d, B[0x3].d, B[0xF].d, B[0xC].d, C[0x2].d, M(0x6)); \
    PP(A[0x7].d, A[0x6].d, B[0x7].d, B[0x4].d, B[0x0].d, B[0xD].d, C[0x1].d, M(0x7)); \
    PP(A[0x8].d, A[0x7].d, B[0x8].d, B[0x5].d, B[0x1].d, B[0xE].d, C[0x0].d, M1); \
    PP2(A[0x9].d, A[0x8].d, B[0x9].d, B[0x6].d, B[0x2].d, B[0xF].d, C[0xF].d); \
    PP2(A[0xA].d, A[0x9].d, B[0xA].d, B[0x7].d, B[0x3].d, B[0x0].d, C[0xE].d); \
    PP2(A[0xB].d, A[0xA].d, B[0xB].d, B[0x8].d, B[0x4].d, B[0x1].d, C[0xD].d); \
    PP2(A[0x0].d, A[0xB].d, B[0xC].d, B[0x9].d, B[0x5].d, B[0x2].d, C[0xC].d); \
    PP2(A[0x1].d, A[0x0].d, B[0xD].d, B[0xA].d, B[0x6].d, B[0x3].d, C[0xB].d); \
    PP2(A[0x2].d, A[0x1].d, B[0xE].d, B[0xB].d, B[0x7].d, B[0x4].d, C[0xA].d); \
    PP2(A[0x3].d, A[0x2].d, B[0xF].d, B[0xC].d, B[0x8].d, B[0x5].d, C[0x9].d); \
    PP(A[0x4].d, A[0x3].d, B[0x0].d, B[0xD].d, B[0x9].d, B[0x6].d, C[0x8].d, M(0x0)); \
    PP(A[0x5].d, A[0x4].d, B[0x1].d, B[0xE].d, B[0xA].d, B[0x7].d, C[0x7].d, M(0x1)); \
    PP(A[0x6].d, A[0x5].d, B[0x2].d, B[0xF].d, B[0xB].d, B[0x8].d, C[0x6].d, M(0x2)); \
    PP(A[0x7].d, A[0x6].d, B[0x3].d, B[0x0].d, B[0xC].d, B[0x9].d, C[0x5].d, M(0x3)); \
    PP(A[0x8].d, A[0x7].d, B[0x4].d, B[0x1].d, B[0xD].d, B[0xA].d, C[0x4].d, M(0x4)); \
    PP(A[0x9].d, A[0x8].d, B[0x5].d, B[0x2].d, B[0xE].d, B[0xB].d, C[0x3].d, M(0x5)); \
    PP(A[0xA].d, A[0x9].d, B[0x6].d, B[0x3].d, B[0xF].d, B[0xC].d, C[0x2].d, M(0x6)); \
    PP(A[0xB].d, A[0xA].d, B[0x7].d, B[0x4].d, B[0x0].d, B[0xD].d, C[0x1].d, M(0x7)); \
    PP(A[0x0].d, A[0xB].d, B[0x8].d, B[0x5].d, B[0x1].d, B[0xE].d, C[0x0].d, M1); \
    PP2(A[0x1].d, A[0x0].d, B[0x9].d, B[0x6].d, B[0x2].d, B[0xF].d, C[0xF].d); \
    PP2(A[0x2].d, A[0x1].d, B[0xA].d, B[0x7].d, B[0x3].d, B[0x0].d, C[0xE].d); \
    PP2(A[0x3].d, A[0x2].d, B[0xB].d, B[0x8].d, B[0x4].d, B[0x1].d, C[0xD].d); \
    PP2(A[0x4].d, A[0x3].d, B[0xC].d, B[0x9].d, B[0x5].d, B[0x2].d, C[0xC].d); \
    PP2(A[0x5].d, A[0x4].d, B[0xD].d, B[0xA].d, B[0x6].d, B[0x3].d, C[0xB].d); \
    PP2(A[0x6].d, A[0x5].d, B[0xE].d, B[0xB].d, B[0x7].d, B[0x4].d, C[0xA].d); \
    PP2(A[0x7].d, A[0x6].d, B[0xF].d, B[0xC].d, B[0x8].d, B[0x5].d, C[0x9].d); \
    PP(A[0x8].d, A[0x7].d, B[0x0].d, B[0xD].d, B[0x9].d, B[0x6].d, C[0x8].d, M(0x0)); \
    PP(A[0x9].d, A[0x8].d, B[0x1].d, B[0xE].d, B[0xA].d, B[0x7].d, C[0x7].d, M(0x1)); \
    PP(A[0xA].d, A[0x9].d, B[0x2].d, B[0xF].d, B[0xB].d, B[0x8].d, C[0x6].d, M(0x2)); \
    PP(A[0xB].d, A[0xA].d, B[0x3].d, B[0x0].d, B[0xC].d, B[0x9].d, C[0x5].d, M(0x3)); \
    PP(A[0x0].d, A[0xB].d, B[0x4].d, B[0x1].d, B[0xD].d, B[0xA].d, C[0x4].d, M(0x4)); \
    PP(A[0x1].d, A[0x0].d, B[0x5].d, B[0x2].d, B[0xE].d, B[0xB].d, C[0x3].d, M(0x5)); \
    PP(A[0x2].d, A[0x1].d, B[0x6].d, B[0x3].d, B[0xF].d, B[0xC].d, C[0x2].d, M(0x6)); \
    PP(A[0x3].d, A[0x2].d, B[0x7].d, B[0x4].d, B[0x0].d, B[0xD].d, C[0x1].d, M(0x7)); \
    PP(A[0x4].d, A[0x3].d, B[0x8].d, B[0x5].d, B[0x1].d, B[0xE].d, C[0x0].d, M1); \
    PP2(A[0x5].d, A[0x4].d, B[0x9].d, B[0x6].d, B[0x2].d, B[0xF].d, C[0xF].d); \
    PP2(A[0x6].d, A[0x5].d, B[0xA].d, B[0x7].d, B[0x3].d, B[0x0].d, C[0xE].d); \
    PP2(A[0x7].d, A[0x6].d, B[0xB].d, B[0x8].d, B[0x4].d, B[0x1].d, C[0xD].d); \
    PP2(A[0x8].d, A[0x7].d, B[0xC].d, B[0x9].d, B[0x5].d, B[0x2].d, C[0xC].d); \
    PP2(A[0x9].d, A[0x8].d, B[0xD].d, B[0xA].d, B[0x6].d, B[0x3].d, C[0xB].d); \
    PP2(A[0xA].d, A[0x9].d, B[0xE].d, B[0xB].d, B[0x7].d, B[0x4].d, C[0xA].d); \
    PP2(A[0xB].d, A[0xA].d, B[0xF].d, B[0xC].d, B[0x8].d, B[0x5].d, C[0x9].d);

#define ADDA(A, C) \
    A[0x0].d = _mm_add_epi32(A[0x0].d, C[0xB].d); \
    A[0x0].d = _mm_add_epi32(A[0x0].d, C[0xF].d); \
    A[0x0].d = _mm_add_epi32(A[0x0].d, C[0x3].d); \
    A[0x1].d = _mm_add_epi32(A[0x1].d, C[0xC].d); \
    A[0x1].d = _mm_add_epi32(A[0x1].d, C[0x0].d); \
    A[0x1].d = _mm_add_epi32(A[0x1].d, C[0x4].d); \
    A[0x2].d = _mm_add_epi32(A[0x2].d, C[0xD].d); \
    A[0x2].d = _mm_add_epi32(A[0x2].d, C[0x1].d); \
    A[0x2].d = _mm_add_epi32(A[0x2].d, C[0x5].d); \
    A[0x3].d = _mm_add_epi32(A[0x3].d, C[0xE].d); \
    A[0x3].d = _mm_add_epi32(A[0x3].d, C[0x2].d); \
    A[0x3].d = _mm_add_epi32(A[0x3].d, C[0x6].d); \
    A[0x4].d = _mm_add_epi32(A[0x4].d, C[0xF].d); \
    A[0x4].d = _mm_add_epi32(A[0x4].d, C[0x3].d); \
    A[0x4].d = _mm_add_epi32(A[0x4].d, C[0x7].d); \
    A[0x5].d = _mm_add_epi32(A[0x5].d, C[0x0].d); \
    A[0x5].d = _mm_add_epi32(A[0x5].d, C[0x4].d); \
    A[0x5].d = _mm_add_epi32(A[0x5].d, C[0x8].d); \
    A[0x6].d = _mm_add_epi32(A[0x6].d, C[0x1].d); \
    A[0x6].d = _mm_add_epi32(A[0x6].d, C[0x5].d); \
    A[0x6].d = _mm_add_epi32(A[0x6].d, C[0x9].d); \
    A[0x7].d = _mm_add_epi32(A[0x7].d, C[0x2].d); \
    A[0x7].d = _mm_add_epi32(A[0x7].d, C[0x6].d); \
    A[0x7].d = _mm_add_epi32(A[0x7].d, C[0xA].d); \
    A[0x8].d = _mm_add_epi32(A[0x8].d, C[0x3].d); \
    A[0x8].d = _mm_add_epi32(A[0x8].d, C[0x7].d); \
    A[0x8].d = _mm_add_epi32(A[0x8].d, C[0xB].d); \
    A[0x9].d = _mm_add_epi32(A[0x9].d, C[0x4].d); \
    A[0x9].d = _mm_add_epi32(A[0x9].d, C[0x8].d); \
    A[0x9].d = _mm_add_epi32(A[0x9].d, C[0xC].d); \
    A[0xA].d = _mm_add_epi32(A[0xA].d, C[0x5].d); \
    A[0xA].d = _mm_add_epi32(A[0xA].d, C[0x9].d); \
    A[0xA].d = _mm_add_epi32(A[0xA].d, C[0xD].d); \
    A[0xB].d = _mm_add_epi32(A[0xB].d, C[0x6].d); \
    A[0xB].d = _mm_add_epi32(A[0xB].d, C[0xA].d); \
    A[0xB].d = _mm_add_epi32(A[0xB].d, C[0xE].d);

#define SWAPA(X, M, M1) \
    X[0x0].d = _mm_sub_epi32(X[0x0].d, M(0x0)); \
    X[0x1].d = _mm_sub_epi32(X[0x1].d, M(0x1)); \
    X[0x2].d = _mm_sub_epi32(X[0x2].d, M(0x2)); \
    X[0x3].d = _mm_sub_epi32(X[0x3].d, M(0x3)); \
    X[0x4].d = _mm_sub_epi32(X[0x4].d, M(0x4)); \
    X[0x5].d = _mm_sub_epi32(X[0x5].d, M(0x5)); \
    X[0x6].d = _mm_sub_epi32(X[0x6].d, M(0x6)); \
    X[0x7].d = _mm_sub_epi32(X[0x7].d, M(0x7)); \
    X[0x8].d = _mm_sub_epi32(X[0x8].d, M1); \

	INIT(sc->stateA, sc->stateB, M, eighty, zero);
	PPA(sc->stateA, sc->stateB, sc->stateC, M, eighty);
	ADDA(sc->stateA, sc->stateC);
	SWAPA(sc->stateC, M, eighty);

	INIT(sc->stateA, sc->stateC, M, eighty, zero);
	PPA(sc->stateA, sc->stateC, sc->stateB, M, eighty);
	ADDA(sc->stateA, sc->stateB);
	SWAPA(sc->stateB, M, eighty);

	INIT(sc->stateA, sc->stateB, M, eighty, zero);
	PPA(sc->stateA, sc->stateB, sc->stateC, M, eighty);
	ADDA(sc->stateA, sc->stateC);
	SWAPA(sc->stateC, M, eighty);

	INIT(sc->stateA, sc->stateC, M, eighty, zero);
	PPA(sc->stateA, sc->stateC, sc->stateB, M, eighty);

#undef M
#undef PP
#undef PP2
#undef INIT
#undef PPA
#undef ADDA
#undef SWAPA
}

#ifdef __8WAY__
#ifndef __MINGW__
__declspec(noinline)
#else
__attribute__((noinline))
#endif
static void mshabal8_compress_fast4_80(mshabal8_context *sc,
	const void *buf0, const void *buf1,
	const void *buf2, const void *buf3,
	const void *buf4, const void *buf5,
	const void *buf6, const void *buf7
	)
{
	size_t j;
	un u;

	for (j = 0; j < 4; j++) {
		size_t k = 8 * j;
		u.words[k + 0] = ((u32 *)buf0)[j + 16];
		u.words[k + 1] = ((u32 *)buf1)[j + 16];
		u.words[k + 2] = ((u32 *)buf2)[j + 16];
		u.words[k + 3] = ((u32 *)buf3)[j + 16];
		u.words[k + 4] = ((u32 *)buf4)[j + 16];
		u.words[k + 5] = ((u32 *)buf5)[j + 16];
		u.words[k + 6] = ((u32 *)buf6)[j + 16];
		u.words[k + 7] = ((u32 *)buf7)[j + 16];
	}

	__m256i one = _mm256_set1_epi32(C32(0xFFFFFFFF));
	__m256i eighty = _mm256_setr_epi32(0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
	__m256i zero = _mm256_setzero_si256();

#define M(i) u.data[i]

#define PP(xa0, xa1, xb0, xb1, xb2, xb3, xc, xm)   do { \
    __m256i tt; \
    tt = _mm256_or_si256(_mm256_slli_epi32(xa1, 15), \
        _mm256_srli_epi32(xa1, 17)); \
    tt = _mm256_add_epi32(_mm256_slli_epi32(tt, 2), tt); \
    tt = _mm256_xor_si256(_mm256_xor_si256(xa0, tt), xc); \
    tt = _mm256_add_epi32(_mm256_slli_epi32(tt, 1), tt); \
    tt = _mm256_xor_si256( \
        _mm256_xor_si256(tt, xb1), \
        _mm256_xor_si256(_mm256_andnot_si256(xb3, xb2), xm)); \
    xa0 = tt; \
    tt = xb0; \
    tt = _mm256_or_si256(_mm256_slli_epi32(tt, 1), \
        _mm256_srli_epi32(tt, 31)); \
    xb0 = _mm256_xor_si256(tt, _mm256_xor_si256(xa0, one)); \
	        } while (0)

#define PP2(xa0, xa1, xb0, xb1, xb2, xb3, xc)   do { \
    __m256i tt; \
    tt = _mm256_or_si256(_mm256_slli_epi32(xa1, 15), \
        _mm256_srli_epi32(xa1, 17)); \
    tt = _mm256_add_epi32(_mm256_slli_epi32(tt, 2), tt); \
    tt = _mm256_xor_si256(_mm256_xor_si256(xa0, tt), xc); \
    tt = _mm256_add_epi32(_mm256_slli_epi32(tt, 1), tt); \
    tt = _mm256_xor_si256( \
        _mm256_xor_si256(tt, xb1), _mm256_andnot_si256(xb3, xb2)); \
    xa0 = tt; \
    tt = xb0; \
    tt = _mm256_or_si256(_mm256_slli_epi32(tt, 1), \
        _mm256_srli_epi32(tt, 31)); \
    xb0 = _mm256_xor_si256(tt, _mm256_xor_si256(xa0, one)); \
	        } while (0)

#define INIT(A, B, M, M1, M2) \
    B[0].d = _mm256_add_epi32(B[0].d, M(0)); \
    B[0].d = _mm256_or_si256(_mm256_slli_epi32(B[0].d, 17), _mm256_srli_epi32(B[0].d, 15)); \
	B[1].d = _mm256_add_epi32(B[1].d, M(1)); \
    B[1].d = _mm256_or_si256(_mm256_slli_epi32(B[1].d, 17), _mm256_srli_epi32(B[1].d, 15)); \
	B[2].d = _mm256_add_epi32(B[2].d, M(2)); \
    B[2].d = _mm256_or_si256(_mm256_slli_epi32(B[2].d, 17), _mm256_srli_epi32(B[2].d, 15)); \
	B[3].d = _mm256_add_epi32(B[3].d, M(3)); \
    B[3].d = _mm256_or_si256(_mm256_slli_epi32(B[3].d, 17), _mm256_srli_epi32(B[3].d, 15)); \
    B[4].d = _mm256_add_epi32(B[4].d, M1); \
    B[4].d = _mm256_or_si256(_mm256_slli_epi32(B[4].d, 17), _mm256_srli_epi32(B[4].d, 15)); \
	for (j = 5; j < 16; j++) { \
    B[j].d = _mm256_add_epi32(B[j].d, M2); \
    B[j].d = _mm256_or_si256(_mm256_slli_epi32(B[j].d, 17), _mm256_srli_epi32(B[j].d, 15)); \
			} \
    A[0].d = _mm256_xor_si256(A[0].d, _mm256_set1_epi32(2)); \
    A[1].d = _mm256_xor_si256(A[1].d, _mm256_set1_epi32(0));

#define PPA(A, B, C, M, M1) \
    PP(A[0x0].d, A[0xB].d, B[0x0].d, B[0xD].d, B[0x9].d, B[0x6].d, C[0x8].d, M(0x0)); \
    PP(A[0x1].d, A[0x0].d, B[0x1].d, B[0xE].d, B[0xA].d, B[0x7].d, C[0x7].d, M(0x1)); \
    PP(A[0x2].d, A[0x1].d, B[0x2].d, B[0xF].d, B[0xB].d, B[0x8].d, C[0x6].d, M(0x2)); \
    PP(A[0x3].d, A[0x2].d, B[0x3].d, B[0x0].d, B[0xC].d, B[0x9].d, C[0x5].d, M(0x3)); \
    PP(A[0x4].d, A[0x3].d, B[0x4].d, B[0x1].d, B[0xD].d, B[0xA].d, C[0x4].d, M1); \
    PP2(A[0x5].d, A[0x4].d, B[0x5].d, B[0x2].d, B[0xE].d, B[0xB].d, C[0x3].d); \
    PP2(A[0x6].d, A[0x5].d, B[0x6].d, B[0x3].d, B[0xF].d, B[0xC].d, C[0x2].d); \
    PP2(A[0x7].d, A[0x6].d, B[0x7].d, B[0x4].d, B[0x0].d, B[0xD].d, C[0x1].d); \
    PP2(A[0x8].d, A[0x7].d, B[0x8].d, B[0x5].d, B[0x1].d, B[0xE].d, C[0x0].d); \
    PP2(A[0x9].d, A[0x8].d, B[0x9].d, B[0x6].d, B[0x2].d, B[0xF].d, C[0xF].d); \
    PP2(A[0xA].d, A[0x9].d, B[0xA].d, B[0x7].d, B[0x3].d, B[0x0].d, C[0xE].d); \
    PP2(A[0xB].d, A[0xA].d, B[0xB].d, B[0x8].d, B[0x4].d, B[0x1].d, C[0xD].d); \
    PP2(A[0x0].d, A[0xB].d, B[0xC].d, B[0x9].d, B[0x5].d, B[0x2].d, C[0xC].d); \
    PP2(A[0x1].d, A[0x0].d, B[0xD].d, B[0xA].d, B[0x6].d, B[0x3].d, C[0xB].d); \
    PP2(A[0x2].d, A[0x1].d, B[0xE].d, B[0xB].d, B[0x7].d, B[0x4].d, C[0xA].d); \
    PP2(A[0x3].d, A[0x2].d, B[0xF].d, B[0xC].d, B[0x8].d, B[0x5].d, C[0x9].d); \
    PP(A[0x4].d, A[0x3].d, B[0x0].d, B[0xD].d, B[0x9].d, B[0x6].d, C[0x8].d, M(0x0)); \
    PP(A[0x5].d, A[0x4].d, B[0x1].d, B[0xE].d, B[0xA].d, B[0x7].d, C[0x7].d, M(0x1)); \
    PP(A[0x6].d, A[0x5].d, B[0x2].d, B[0xF].d, B[0xB].d, B[0x8].d, C[0x6].d, M(0x2)); \
    PP(A[0x7].d, A[0x6].d, B[0x3].d, B[0x0].d, B[0xC].d, B[0x9].d, C[0x5].d, M(0x3)); \
    PP(A[0x8].d, A[0x7].d, B[0x4].d, B[0x1].d, B[0xD].d, B[0xA].d, C[0x4].d, M1); \
    PP2(A[0x9].d, A[0x8].d, B[0x5].d, B[0x2].d, B[0xE].d, B[0xB].d, C[0x3].d); \
    PP2(A[0xA].d, A[0x9].d, B[0x6].d, B[0x3].d, B[0xF].d, B[0xC].d, C[0x2].d); \
    PP2(A[0xB].d, A[0xA].d, B[0x7].d, B[0x4].d, B[0x0].d, B[0xD].d, C[0x1].d); \
    PP2(A[0x0].d, A[0xB].d, B[0x8].d, B[0x5].d, B[0x1].d, B[0xE].d, C[0x0].d); \
    PP2(A[0x1].d, A[0x0].d, B[0x9].d, B[0x6].d, B[0x2].d, B[0xF].d, C[0xF].d); \
    PP2(A[0x2].d, A[0x1].d, B[0xA].d, B[0x7].d, B[0x3].d, B[0x0].d, C[0xE].d); \
    PP2(A[0x3].d, A[0x2].d, B[0xB].d, B[0x8].d, B[0x4].d, B[0x1].d, C[0xD].d); \
    PP2(A[0x4].d, A[0x3].d, B[0xC].d, B[0x9].d, B[0x5].d, B[0x2].d, C[0xC].d); \
    PP2(A[0x5].d, A[0x4].d, B[0xD].d, B[0xA].d, B[0x6].d, B[0x3].d, C[0xB].d); \
    PP2(A[0x6].d, A[0x5].d, B[0xE].d, B[0xB].d, B[0x7].d, B[0x4].d, C[0xA].d); \
    PP2(A[0x7].d, A[0x6].d, B[0xF].d, B[0xC].d, B[0x8].d, B[0x5].d, C[0x9].d); \
    PP(A[0x8].d, A[0x7].d, B[0x0].d, B[0xD].d, B[0x9].d, B[0x6].d, C[0x8].d, M(0x0)); \
    PP(A[0x9].d, A[0x8].d, B[0x1].d, B[0xE].d, B[0xA].d, B[0x7].d, C[0x7].d, M(0x1)); \
    PP(A[0xA].d, A[0x9].d, B[0x2].d, B[0xF].d, B[0xB].d, B[0x8].d, C[0x6].d, M(0x2)); \
    PP(A[0xB].d, A[0xA].d, B[0x3].d, B[0x0].d, B[0xC].d, B[0x9].d, C[0x5].d, M(0x3)); \
    PP(A[0x0].d, A[0xB].d, B[0x4].d, B[0x1].d, B[0xD].d, B[0xA].d, C[0x4].d, M1); \
    PP2(A[0x1].d, A[0x0].d, B[0x5].d, B[0x2].d, B[0xE].d, B[0xB].d, C[0x3].d); \
    PP2(A[0x2].d, A[0x1].d, B[0x6].d, B[0x3].d, B[0xF].d, B[0xC].d, C[0x2].d); \
    PP2(A[0x3].d, A[0x2].d, B[0x7].d, B[0x4].d, B[0x0].d, B[0xD].d, C[0x1].d); \
    PP2(A[0x4].d, A[0x3].d, B[0x8].d, B[0x5].d, B[0x1].d, B[0xE].d, C[0x0].d); \
    PP2(A[0x5].d, A[0x4].d, B[0x9].d, B[0x6].d, B[0x2].d, B[0xF].d, C[0xF].d); \
    PP2(A[0x6].d, A[0x5].d, B[0xA].d, B[0x7].d, B[0x3].d, B[0x0].d, C[0xE].d); \
    PP2(A[0x7].d, A[0x6].d, B[0xB].d, B[0x8].d, B[0x4].d, B[0x1].d, C[0xD].d); \
    PP2(A[0x8].d, A[0x7].d, B[0xC].d, B[0x9].d, B[0x5].d, B[0x2].d, C[0xC].d); \
    PP2(A[0x9].d, A[0x8].d, B[0xD].d, B[0xA].d, B[0x6].d, B[0x3].d, C[0xB].d); \
    PP2(A[0xA].d, A[0x9].d, B[0xE].d, B[0xB].d, B[0x7].d, B[0x4].d, C[0xA].d); \
    PP2(A[0xB].d, A[0xA].d, B[0xF].d, B[0xC].d, B[0x8].d, B[0x5].d, C[0x9].d); \

#define ADDA(A, C) \
    A[0x0].d = _mm256_add_epi32(A[0x0].d, C[0xB].d); \
    A[0x0].d = _mm256_add_epi32(A[0x0].d, C[0xF].d); \
    A[0x0].d = _mm256_add_epi32(A[0x0].d, C[0x3].d); \
    A[0x1].d = _mm256_add_epi32(A[0x1].d, C[0xC].d); \
    A[0x1].d = _mm256_add_epi32(A[0x1].d, C[0x0].d); \
    A[0x1].d = _mm256_add_epi32(A[0x1].d, C[0x4].d); \
    A[0x2].d = _mm256_add_epi32(A[0x2].d, C[0xD].d); \
    A[0x2].d = _mm256_add_epi32(A[0x2].d, C[0x1].d); \
    A[0x2].d = _mm256_add_epi32(A[0x2].d, C[0x5].d); \
    A[0x3].d = _mm256_add_epi32(A[0x3].d, C[0xE].d); \
    A[0x3].d = _mm256_add_epi32(A[0x3].d, C[0x2].d); \
    A[0x3].d = _mm256_add_epi32(A[0x3].d, C[0x6].d); \
    A[0x4].d = _mm256_add_epi32(A[0x4].d, C[0xF].d); \
    A[0x4].d = _mm256_add_epi32(A[0x4].d, C[0x3].d); \
    A[0x4].d = _mm256_add_epi32(A[0x4].d, C[0x7].d); \
    A[0x5].d = _mm256_add_epi32(A[0x5].d, C[0x0].d); \
    A[0x5].d = _mm256_add_epi32(A[0x5].d, C[0x4].d); \
    A[0x5].d = _mm256_add_epi32(A[0x5].d, C[0x8].d); \
    A[0x6].d = _mm256_add_epi32(A[0x6].d, C[0x1].d); \
    A[0x6].d = _mm256_add_epi32(A[0x6].d, C[0x5].d); \
    A[0x6].d = _mm256_add_epi32(A[0x6].d, C[0x9].d); \
    A[0x7].d = _mm256_add_epi32(A[0x7].d, C[0x2].d); \
    A[0x7].d = _mm256_add_epi32(A[0x7].d, C[0x6].d); \
    A[0x7].d = _mm256_add_epi32(A[0x7].d, C[0xA].d); \
    A[0x8].d = _mm256_add_epi32(A[0x8].d, C[0x3].d); \
    A[0x8].d = _mm256_add_epi32(A[0x8].d, C[0x7].d); \
    A[0x8].d = _mm256_add_epi32(A[0x8].d, C[0xB].d); \
    A[0x9].d = _mm256_add_epi32(A[0x9].d, C[0x4].d); \
    A[0x9].d = _mm256_add_epi32(A[0x9].d, C[0x8].d); \
    A[0x9].d = _mm256_add_epi32(A[0x9].d, C[0xC].d); \
    A[0xA].d = _mm256_add_epi32(A[0xA].d, C[0x5].d); \
    A[0xA].d = _mm256_add_epi32(A[0xA].d, C[0x9].d); \
    A[0xA].d = _mm256_add_epi32(A[0xA].d, C[0xD].d); \
    A[0xB].d = _mm256_add_epi32(A[0xB].d, C[0x6].d); \
    A[0xB].d = _mm256_add_epi32(A[0xB].d, C[0xA].d); \
    A[0xB].d = _mm256_add_epi32(A[0xB].d, C[0xE].d);

#define SWAPA(X, M, M1) \
    X[0x0].d = _mm256_sub_epi32(X[0x0].d, M(0x0)); \
    X[0x1].d = _mm256_sub_epi32(X[0x1].d, M(0x1)); \
    X[0x2].d = _mm256_sub_epi32(X[0x2].d, M(0x2)); \
    X[0x3].d = _mm256_sub_epi32(X[0x3].d, M(0x3)); \
    X[0x4].d = _mm256_sub_epi32(X[0x4].d, M1); \

	INIT(sc->stateA, sc->stateB, M, eighty, zero);
	PPA(sc->stateA, sc->stateB, sc->stateC, M, eighty);
	ADDA(sc->stateA, sc->stateC);
	SWAPA(sc->stateC, M, eighty);

	INIT(sc->stateA, sc->stateC, M, eighty, zero);
	PPA(sc->stateA, sc->stateC, sc->stateB, M, eighty);
	ADDA(sc->stateA, sc->stateB);
	SWAPA(sc->stateB, M, eighty);

	INIT(sc->stateA, sc->stateB, M, eighty, zero);
	PPA(sc->stateA, sc->stateB, sc->stateC, M, eighty);
	ADDA(sc->stateA, sc->stateC);
	SWAPA(sc->stateC, M, eighty);

	INIT(sc->stateA, sc->stateC, M, eighty, zero);
	PPA(sc->stateA, sc->stateC, sc->stateB, M, eighty);

#undef M
#undef PP
#undef PP2
#undef INIT
#undef PPA
#undef ADDA
#undef SWAPA
}
#endif

#ifndef __MINGW__
__declspec(noinline)
#else
__attribute__((noinline))
#endif
static void mshabal4_compress_fast4_80(mshabal4_context *sc,
const void *buf0, const void *buf1,
const void *buf2, const void *buf3
)
{
	size_t j;
	un4 u;

	for (j = 0; j < 4; j++) {
		size_t k = 4 * j;
		u.words[k + 0] = ((u32 *)buf0)[j + 16];
		u.words[k + 1] = ((u32 *)buf1)[j + 16];
		u.words[k + 2] = ((u32 *)buf2)[j + 16];
		u.words[k + 3] = ((u32 *)buf3)[j + 16];
	}

	__m128i one = _mm_set1_epi32(C32(0xFFFFFFFF));
	__m128i eighty = _mm_setr_epi32(0x80, 0x80, 0x80, 0x80);
	__m128i zero = _mm_setzero_si128();

#define M(i) u.data[i]

#define PP(xa0, xa1, xb0, xb1, xb2, xb3, xc, xm)   do { \
    __m128i tt; \
    tt = _mm_or_si128(_mm_slli_epi32(xa1, 15), \
        _mm_srli_epi32(xa1, 17)); \
    tt = _mm_add_epi32(_mm_slli_epi32(tt, 2), tt); \
    tt = _mm_xor_si128(_mm_xor_si128(xa0, tt), xc); \
    tt = _mm_add_epi32(_mm_slli_epi32(tt, 1), tt); \
    tt = _mm_xor_si128( \
        _mm_xor_si128(tt, xb1), \
        _mm_xor_si128(_mm_andnot_si128(xb3, xb2), xm)); \
    xa0 = tt; \
    tt = xb0; \
    tt = _mm_or_si128(_mm_slli_epi32(tt, 1), \
        _mm_srli_epi32(tt, 31)); \
    xb0 = _mm_xor_si128(tt, _mm_xor_si128(xa0, one)); \
		} while (0)

#define PP2(xa0, xa1, xb0, xb1, xb2, xb3, xc)   do { \
    __m128i tt; \
    tt = _mm_or_si128(_mm_slli_epi32(xa1, 15), \
        _mm_srli_epi32(xa1, 17)); \
    tt = _mm_add_epi32(_mm_slli_epi32(tt, 2), tt); \
    tt = _mm_xor_si128(_mm_xor_si128(xa0, tt), xc); \
    tt = _mm_add_epi32(_mm_slli_epi32(tt, 1), tt); \
    tt = _mm_xor_si128( \
        _mm_xor_si128(tt, xb1), _mm_andnot_si128(xb3, xb2)); \
    xa0 = tt; \
    tt = xb0; \
    tt = _mm_or_si128(_mm_slli_epi32(tt, 1), \
        _mm_srli_epi32(tt, 31)); \
    xb0 = _mm_xor_si128(tt, _mm_xor_si128(xa0, one)); \
		} while (0)

#define INIT(A, B, M, M1, M2) \
    B[0].d = _mm_add_epi32(B[0].d, M(0)); \
    B[0].d = _mm_or_si128(_mm_slli_epi32(B[0].d, 17), _mm_srli_epi32(B[0].d, 15)); \
	B[1].d = _mm_add_epi32(B[1].d, M(1)); \
    B[1].d = _mm_or_si128(_mm_slli_epi32(B[1].d, 17), _mm_srli_epi32(B[1].d, 15)); \
	B[2].d = _mm_add_epi32(B[2].d, M(2)); \
    B[2].d = _mm_or_si128(_mm_slli_epi32(B[2].d, 17), _mm_srli_epi32(B[2].d, 15)); \
	B[3].d = _mm_add_epi32(B[3].d, M(3)); \
    B[3].d = _mm_or_si128(_mm_slli_epi32(B[3].d, 17), _mm_srli_epi32(B[3].d, 15)); \
    B[4].d = _mm_add_epi32(B[4].d, M1); \
    B[4].d = _mm_or_si128(_mm_slli_epi32(B[4].d, 17), _mm_srli_epi32(B[4].d, 15)); \
	for (j = 5; j < 16; j++) { \
    B[j].d = _mm_add_epi32(B[j].d, M2); \
    B[j].d = _mm_or_si128(_mm_slli_epi32(B[j].d, 17), _mm_srli_epi32(B[j].d, 15)); \
				} \
    A[0].d = _mm_xor_si128(A[0].d, _mm_set1_epi32(2)); \
    A[1].d = _mm_xor_si128(A[1].d, _mm_set1_epi32(0));

#define PPA(A, B, C, M, M1) \
    PP(A[0x0].d, A[0xB].d, B[0x0].d, B[0xD].d, B[0x9].d, B[0x6].d, C[0x8].d, M(0x0)); \
    PP(A[0x1].d, A[0x0].d, B[0x1].d, B[0xE].d, B[0xA].d, B[0x7].d, C[0x7].d, M(0x1)); \
    PP(A[0x2].d, A[0x1].d, B[0x2].d, B[0xF].d, B[0xB].d, B[0x8].d, C[0x6].d, M(0x2)); \
    PP(A[0x3].d, A[0x2].d, B[0x3].d, B[0x0].d, B[0xC].d, B[0x9].d, C[0x5].d, M(0x3)); \
    PP(A[0x4].d, A[0x3].d, B[0x4].d, B[0x1].d, B[0xD].d, B[0xA].d, C[0x4].d, M1); \
    PP2(A[0x5].d, A[0x4].d, B[0x5].d, B[0x2].d, B[0xE].d, B[0xB].d, C[0x3].d); \
    PP2(A[0x6].d, A[0x5].d, B[0x6].d, B[0x3].d, B[0xF].d, B[0xC].d, C[0x2].d); \
    PP2(A[0x7].d, A[0x6].d, B[0x7].d, B[0x4].d, B[0x0].d, B[0xD].d, C[0x1].d); \
    PP2(A[0x8].d, A[0x7].d, B[0x8].d, B[0x5].d, B[0x1].d, B[0xE].d, C[0x0].d); \
    PP2(A[0x9].d, A[0x8].d, B[0x9].d, B[0x6].d, B[0x2].d, B[0xF].d, C[0xF].d); \
    PP2(A[0xA].d, A[0x9].d, B[0xA].d, B[0x7].d, B[0x3].d, B[0x0].d, C[0xE].d); \
    PP2(A[0xB].d, A[0xA].d, B[0xB].d, B[0x8].d, B[0x4].d, B[0x1].d, C[0xD].d); \
    PP2(A[0x0].d, A[0xB].d, B[0xC].d, B[0x9].d, B[0x5].d, B[0x2].d, C[0xC].d); \
    PP2(A[0x1].d, A[0x0].d, B[0xD].d, B[0xA].d, B[0x6].d, B[0x3].d, C[0xB].d); \
    PP2(A[0x2].d, A[0x1].d, B[0xE].d, B[0xB].d, B[0x7].d, B[0x4].d, C[0xA].d); \
    PP2(A[0x3].d, A[0x2].d, B[0xF].d, B[0xC].d, B[0x8].d, B[0x5].d, C[0x9].d); \
    PP(A[0x4].d, A[0x3].d, B[0x0].d, B[0xD].d, B[0x9].d, B[0x6].d, C[0x8].d, M(0x0)); \
    PP(A[0x5].d, A[0x4].d, B[0x1].d, B[0xE].d, B[0xA].d, B[0x7].d, C[0x7].d, M(0x1)); \
    PP(A[0x6].d, A[0x5].d, B[0x2].d, B[0xF].d, B[0xB].d, B[0x8].d, C[0x6].d, M(0x2)); \
    PP(A[0x7].d, A[0x6].d, B[0x3].d, B[0x0].d, B[0xC].d, B[0x9].d, C[0x5].d, M(0x3)); \
    PP(A[0x8].d, A[0x7].d, B[0x4].d, B[0x1].d, B[0xD].d, B[0xA].d, C[0x4].d, M1); \
    PP2(A[0x9].d, A[0x8].d, B[0x5].d, B[0x2].d, B[0xE].d, B[0xB].d, C[0x3].d); \
    PP2(A[0xA].d, A[0x9].d, B[0x6].d, B[0x3].d, B[0xF].d, B[0xC].d, C[0x2].d); \
    PP2(A[0xB].d, A[0xA].d, B[0x7].d, B[0x4].d, B[0x0].d, B[0xD].d, C[0x1].d); \
    PP2(A[0x0].d, A[0xB].d, B[0x8].d, B[0x5].d, B[0x1].d, B[0xE].d, C[0x0].d); \
    PP2(A[0x1].d, A[0x0].d, B[0x9].d, B[0x6].d, B[0x2].d, B[0xF].d, C[0xF].d); \
    PP2(A[0x2].d, A[0x1].d, B[0xA].d, B[0x7].d, B[0x3].d, B[0x0].d, C[0xE].d); \
    PP2(A[0x3].d, A[0x2].d, B[0xB].d, B[0x8].d, B[0x4].d, B[0x1].d, C[0xD].d); \
    PP2(A[0x4].d, A[0x3].d, B[0xC].d, B[0x9].d, B[0x5].d, B[0x2].d, C[0xC].d); \
    PP2(A[0x5].d, A[0x4].d, B[0xD].d, B[0xA].d, B[0x6].d, B[0x3].d, C[0xB].d); \
    PP2(A[0x6].d, A[0x5].d, B[0xE].d, B[0xB].d, B[0x7].d, B[0x4].d, C[0xA].d); \
    PP2(A[0x7].d, A[0x6].d, B[0xF].d, B[0xC].d, B[0x8].d, B[0x5].d, C[0x9].d); \
    PP(A[0x8].d, A[0x7].d, B[0x0].d, B[0xD].d, B[0x9].d, B[0x6].d, C[0x8].d, M(0x0)); \
    PP(A[0x9].d, A[0x8].d, B[0x1].d, B[0xE].d, B[0xA].d, B[0x7].d, C[0x7].d, M(0x1)); \
    PP(A[0xA].d, A[0x9].d, B[0x2].d, B[0xF].d, B[0xB].d, B[0x8].d, C[0x6].d, M(0x2)); \
    PP(A[0xB].d, A[0xA].d, B[0x3].d, B[0x0].d, B[0xC].d, B[0x9].d, C[0x5].d, M(0x3)); \
    PP(A[0x0].d, A[0xB].d, B[0x4].d, B[0x1].d, B[0xD].d, B[0xA].d, C[0x4].d, M1); \
    PP2(A[0x1].d, A[0x0].d, B[0x5].d, B[0x2].d, B[0xE].d, B[0xB].d, C[0x3].d); \
    PP2(A[0x2].d, A[0x1].d, B[0x6].d, B[0x3].d, B[0xF].d, B[0xC].d, C[0x2].d); \
    PP2(A[0x3].d, A[0x2].d, B[0x7].d, B[0x4].d, B[0x0].d, B[0xD].d, C[0x1].d); \
    PP2(A[0x4].d, A[0x3].d, B[0x8].d, B[0x5].d, B[0x1].d, B[0xE].d, C[0x0].d); \
    PP2(A[0x5].d, A[0x4].d, B[0x9].d, B[0x6].d, B[0x2].d, B[0xF].d, C[0xF].d); \
    PP2(A[0x6].d, A[0x5].d, B[0xA].d, B[0x7].d, B[0x3].d, B[0x0].d, C[0xE].d); \
    PP2(A[0x7].d, A[0x6].d, B[0xB].d, B[0x8].d, B[0x4].d, B[0x1].d, C[0xD].d); \
    PP2(A[0x8].d, A[0x7].d, B[0xC].d, B[0x9].d, B[0x5].d, B[0x2].d, C[0xC].d); \
    PP2(A[0x9].d, A[0x8].d, B[0xD].d, B[0xA].d, B[0x6].d, B[0x3].d, C[0xB].d); \
    PP2(A[0xA].d, A[0x9].d, B[0xE].d, B[0xB].d, B[0x7].d, B[0x4].d, C[0xA].d); \
    PP2(A[0xB].d, A[0xA].d, B[0xF].d, B[0xC].d, B[0x8].d, B[0x5].d, C[0x9].d); \

#define ADDA(A, C) \
    A[0x0].d = _mm_add_epi32(A[0x0].d, C[0xB].d); \
    A[0x0].d = _mm_add_epi32(A[0x0].d, C[0xF].d); \
    A[0x0].d = _mm_add_epi32(A[0x0].d, C[0x3].d); \
    A[0x1].d = _mm_add_epi32(A[0x1].d, C[0xC].d); \
    A[0x1].d = _mm_add_epi32(A[0x1].d, C[0x0].d); \
    A[0x1].d = _mm_add_epi32(A[0x1].d, C[0x4].d); \
    A[0x2].d = _mm_add_epi32(A[0x2].d, C[0xD].d); \
    A[0x2].d = _mm_add_epi32(A[0x2].d, C[0x1].d); \
    A[0x2].d = _mm_add_epi32(A[0x2].d, C[0x5].d); \
    A[0x3].d = _mm_add_epi32(A[0x3].d, C[0xE].d); \
    A[0x3].d = _mm_add_epi32(A[0x3].d, C[0x2].d); \
    A[0x3].d = _mm_add_epi32(A[0x3].d, C[0x6].d); \
    A[0x4].d = _mm_add_epi32(A[0x4].d, C[0xF].d); \
    A[0x4].d = _mm_add_epi32(A[0x4].d, C[0x3].d); \
    A[0x4].d = _mm_add_epi32(A[0x4].d, C[0x7].d); \
    A[0x5].d = _mm_add_epi32(A[0x5].d, C[0x0].d); \
    A[0x5].d = _mm_add_epi32(A[0x5].d, C[0x4].d); \
    A[0x5].d = _mm_add_epi32(A[0x5].d, C[0x8].d); \
    A[0x6].d = _mm_add_epi32(A[0x6].d, C[0x1].d); \
    A[0x6].d = _mm_add_epi32(A[0x6].d, C[0x5].d); \
    A[0x6].d = _mm_add_epi32(A[0x6].d, C[0x9].d); \
    A[0x7].d = _mm_add_epi32(A[0x7].d, C[0x2].d); \
    A[0x7].d = _mm_add_epi32(A[0x7].d, C[0x6].d); \
    A[0x7].d = _mm_add_epi32(A[0x7].d, C[0xA].d); \
    A[0x8].d = _mm_add_epi32(A[0x8].d, C[0x3].d); \
    A[0x8].d = _mm_add_epi32(A[0x8].d, C[0x7].d); \
    A[0x8].d = _mm_add_epi32(A[0x8].d, C[0xB].d); \
    A[0x9].d = _mm_add_epi32(A[0x9].d, C[0x4].d); \
    A[0x9].d = _mm_add_epi32(A[0x9].d, C[0x8].d); \
    A[0x9].d = _mm_add_epi32(A[0x9].d, C[0xC].d); \
    A[0xA].d = _mm_add_epi32(A[0xA].d, C[0x5].d); \
    A[0xA].d = _mm_add_epi32(A[0xA].d, C[0x9].d); \
    A[0xA].d = _mm_add_epi32(A[0xA].d, C[0xD].d); \
    A[0xB].d = _mm_add_epi32(A[0xB].d, C[0x6].d); \
    A[0xB].d = _mm_add_epi32(A[0xB].d, C[0xA].d); \
    A[0xB].d = _mm_add_epi32(A[0xB].d, C[0xE].d);

#define SWAPA(X, M, M1) \
    X[0x0].d = _mm_sub_epi32(X[0x0].d, M(0x0)); \
    X[0x1].d = _mm_sub_epi32(X[0x1].d, M(0x1)); \
    X[0x2].d = _mm_sub_epi32(X[0x2].d, M(0x2)); \
    X[0x3].d = _mm_sub_epi32(X[0x3].d, M(0x3)); \
    X[0x4].d = _mm_sub_epi32(X[0x4].d, M1); \

	INIT(sc->stateA, sc->stateB, M, eighty, zero);
	PPA(sc->stateA, sc->stateB, sc->stateC, M, eighty);
	ADDA(sc->stateA, sc->stateC);
	SWAPA(sc->stateC, M, eighty);

	INIT(sc->stateA, sc->stateC, M, eighty, zero);
	PPA(sc->stateA, sc->stateC, sc->stateB, M, eighty);
	ADDA(sc->stateA, sc->stateB);
	SWAPA(sc->stateB, M, eighty);

	INIT(sc->stateA, sc->stateB, M, eighty, zero);
	PPA(sc->stateA, sc->stateB, sc->stateC, M, eighty);
	ADDA(sc->stateA, sc->stateC);
	SWAPA(sc->stateC, M, eighty);

	INIT(sc->stateA, sc->stateC, M, eighty, zero);
	PPA(sc->stateA, sc->stateC, sc->stateB, M, eighty);

#undef M
#undef PP
#undef PP2
#undef INIT
#undef PPA
#undef ADDA
#undef SWAPA
}

#ifdef __8WAY__
#ifndef __MINGW__
__declspec(noinline)
#else
__attribute__((noinline))
#endif
static void mshabal8_compress(mshabal8_context *sc,
	const void *buf0, const void *buf1,
	const void *buf2, const void *buf3,
	const void *buf4, const void *buf5,
	const void *buf6, const void *buf7,
	const void *buf8, const void *buf9,
	const void *buf10, const void *buf11,
	const void *buf12, const void *buf13,
	const void *buf14, const void *buf15)
{

	size_t j;
	un u;
	if (buf9 == NULL) {
		for (j = 0; j < 64; j += 4) {
			size_t k = 2 * j;
			u.words[k + 0] = *(u32 *)((u8*)buf0 + j);
			u.words[k + 1] = *(u32 *)((u8*)buf1 + j);
			u.words[k + 2] = *(u32 *)((u8*)buf2 + j);
			u.words[k + 3] = *(u32 *)((u8*)buf3 + j);
			u.words[k + 4] = *(u32 *)((u8*)buf4 + j);
			u.words[k + 5] = *(u32 *)((u8*)buf5 + j);
			u.words[k + 6] = *(u32 *)((u8*)buf6 + j);
			u.words[k + 7] = *(u32 *)((u8*)buf7 + j);
		}
	}
	else {
		for (j = 0; j < 32; j += 4) {
			size_t k = 2 * j;
			u.words[k + 0] = *(u32 *)((u8*)buf0 + j);
			u.words[k + 1] = *(u32 *)((u8*)buf1 + j);
			u.words[k + 2] = *(u32 *)((u8*)buf2 + j);
			u.words[k + 3] = *(u32 *)((u8*)buf3 + j);
			u.words[k + 4] = *(u32 *)((u8*)buf4 + j);
			u.words[k + 5] = *(u32 *)((u8*)buf5 + j);
			u.words[k + 6] = *(u32 *)((u8*)buf6 + j);
			u.words[k + 7] = *(u32 *)((u8*)buf7 + j);
		}
		for (j = 0; j < 32; j += 4) {
			size_t k = 2 * (j + 32);
			u.words[k + 0] = *(u32 *)((u8*)buf8 + j);
			u.words[k + 1] = *(u32 *)((u8*)buf9 + j);
			u.words[k + 2] = *(u32 *)((u8*)buf10 + j);
			u.words[k + 3] = *(u32 *)((u8*)buf11 + j);
			u.words[k + 4] = *(u32 *)((u8*)buf12 + j);
			u.words[k + 5] = *(u32 *)((u8*)buf13 + j);
			u.words[k + 6] = *(u32 *)((u8*)buf14 + j);
			u.words[k + 7] = *(u32 *)((u8*)buf15 + j);
		}
	}
#define U(i) u.data + i
#define U2(i) u.data[i]
#define M(i)   U2(i)

	__m256i one;

	for (j = 0; j < 16; j++) {
		sc->stateB[j].d = _mm256_add_epi32(sc->stateB[j].d, U2(j));
		sc->stateB[j].d = _mm256_or_si256(_mm256_slli_epi32(sc->stateB[j].d, 17),
			_mm256_srli_epi32(sc->stateB[j].d, 15));
	}

	sc->stateA[0].d = _mm256_xor_si256(sc->stateA[0].d, _mm256_set1_epi32(1));
	sc->stateA[1].d = _mm256_xor_si256(sc->stateA[1].d, _mm256_set1_epi32(0));

	one = _mm256_set1_epi32(C32(0xFFFFFFFF));

#define PP(xa0, xa1, xb0, xb1, xb2, xb3, xc, xm)   do { \
    __m256i tt; \
    tt = _mm256_or_si256(_mm256_slli_epi32(xa1, 15), \
        _mm256_srli_epi32(xa1, 17)); \
    tt = _mm256_add_epi32(_mm256_slli_epi32(tt, 2), tt); \
    tt = _mm256_xor_si256(_mm256_xor_si256(xa0, tt), xc); \
    tt = _mm256_add_epi32(_mm256_slli_epi32(tt, 1), tt); \
    tt = _mm256_xor_si256( \
        _mm256_xor_si256(tt, xb1), \
        _mm256_xor_si256(_mm256_andnot_si256(xb3, xb2), xm)); \
    xa0 = tt; \
    tt = xb0; \
    tt = _mm256_or_si256(_mm256_slli_epi32(tt, 1), \
        _mm256_srli_epi32(tt, 31)); \
    xb0 = _mm256_xor_si256(tt, _mm256_xor_si256(xa0, one)); \
	        } while (0)

	PP(sc->stateA[0x0].d, sc->stateA[0xB].d, sc->stateB[0x0].d, sc->stateB[0xD].d, sc->stateB[0x9].d, sc->stateB[0x6].d, sc->stateC[0x8].d, M(0x0));
	PP(sc->stateA[0x1].d, sc->stateA[0x0].d, sc->stateB[0x1].d, sc->stateB[0xE].d, sc->stateB[0xA].d, sc->stateB[0x7].d, sc->stateC[0x7].d, M(0x1));
	PP(sc->stateA[0x2].d, sc->stateA[0x1].d, sc->stateB[0x2].d, sc->stateB[0xF].d, sc->stateB[0xB].d, sc->stateB[0x8].d, sc->stateC[0x6].d, M(0x2));
	PP(sc->stateA[0x3].d, sc->stateA[0x2].d, sc->stateB[0x3].d, sc->stateB[0x0].d, sc->stateB[0xC].d, sc->stateB[0x9].d, sc->stateC[0x5].d, M(0x3));
	PP(sc->stateA[0x4].d, sc->stateA[0x3].d, sc->stateB[0x4].d, sc->stateB[0x1].d, sc->stateB[0xD].d, sc->stateB[0xA].d, sc->stateC[0x4].d, M(0x4));
	PP(sc->stateA[0x5].d, sc->stateA[0x4].d, sc->stateB[0x5].d, sc->stateB[0x2].d, sc->stateB[0xE].d, sc->stateB[0xB].d, sc->stateC[0x3].d, M(0x5));
	PP(sc->stateA[0x6].d, sc->stateA[0x5].d, sc->stateB[0x6].d, sc->stateB[0x3].d, sc->stateB[0xF].d, sc->stateB[0xC].d, sc->stateC[0x2].d, M(0x6));
	PP(sc->stateA[0x7].d, sc->stateA[0x6].d, sc->stateB[0x7].d, sc->stateB[0x4].d, sc->stateB[0x0].d, sc->stateB[0xD].d, sc->stateC[0x1].d, M(0x7));
	PP(sc->stateA[0x8].d, sc->stateA[0x7].d, sc->stateB[0x8].d, sc->stateB[0x5].d, sc->stateB[0x1].d, sc->stateB[0xE].d, sc->stateC[0x0].d, M(0x8));
	PP(sc->stateA[0x9].d, sc->stateA[0x8].d, sc->stateB[0x9].d, sc->stateB[0x6].d, sc->stateB[0x2].d, sc->stateB[0xF].d, sc->stateC[0xF].d, M(0x9));
	PP(sc->stateA[0xA].d, sc->stateA[0x9].d, sc->stateB[0xA].d, sc->stateB[0x7].d, sc->stateB[0x3].d, sc->stateB[0x0].d, sc->stateC[0xE].d, M(0xA));
	PP(sc->stateA[0xB].d, sc->stateA[0xA].d, sc->stateB[0xB].d, sc->stateB[0x8].d, sc->stateB[0x4].d, sc->stateB[0x1].d, sc->stateC[0xD].d, M(0xB));
	PP(sc->stateA[0x0].d, sc->stateA[0xB].d, sc->stateB[0xC].d, sc->stateB[0x9].d, sc->stateB[0x5].d, sc->stateB[0x2].d, sc->stateC[0xC].d, M(0xC));
	PP(sc->stateA[0x1].d, sc->stateA[0x0].d, sc->stateB[0xD].d, sc->stateB[0xA].d, sc->stateB[0x6].d, sc->stateB[0x3].d, sc->stateC[0xB].d, M(0xD));
	PP(sc->stateA[0x2].d, sc->stateA[0x1].d, sc->stateB[0xE].d, sc->stateB[0xB].d, sc->stateB[0x7].d, sc->stateB[0x4].d, sc->stateC[0xA].d, M(0xE));
	PP(sc->stateA[0x3].d, sc->stateA[0x2].d, sc->stateB[0xF].d, sc->stateB[0xC].d, sc->stateB[0x8].d, sc->stateB[0x5].d, sc->stateC[0x9].d, M(0xF));

	PP(sc->stateA[0x4].d, sc->stateA[0x3].d, sc->stateB[0x0].d, sc->stateB[0xD].d, sc->stateB[0x9].d, sc->stateB[0x6].d, sc->stateC[0x8].d, M(0x0));
	PP(sc->stateA[0x5].d, sc->stateA[0x4].d, sc->stateB[0x1].d, sc->stateB[0xE].d, sc->stateB[0xA].d, sc->stateB[0x7].d, sc->stateC[0x7].d, M(0x1));
	PP(sc->stateA[0x6].d, sc->stateA[0x5].d, sc->stateB[0x2].d, sc->stateB[0xF].d, sc->stateB[0xB].d, sc->stateB[0x8].d, sc->stateC[0x6].d, M(0x2));
	PP(sc->stateA[0x7].d, sc->stateA[0x6].d, sc->stateB[0x3].d, sc->stateB[0x0].d, sc->stateB[0xC].d, sc->stateB[0x9].d, sc->stateC[0x5].d, M(0x3));
	PP(sc->stateA[0x8].d, sc->stateA[0x7].d, sc->stateB[0x4].d, sc->stateB[0x1].d, sc->stateB[0xD].d, sc->stateB[0xA].d, sc->stateC[0x4].d, M(0x4));
	PP(sc->stateA[0x9].d, sc->stateA[0x8].d, sc->stateB[0x5].d, sc->stateB[0x2].d, sc->stateB[0xE].d, sc->stateB[0xB].d, sc->stateC[0x3].d, M(0x5));
	PP(sc->stateA[0xA].d, sc->stateA[0x9].d, sc->stateB[0x6].d, sc->stateB[0x3].d, sc->stateB[0xF].d, sc->stateB[0xC].d, sc->stateC[0x2].d, M(0x6));
	PP(sc->stateA[0xB].d, sc->stateA[0xA].d, sc->stateB[0x7].d, sc->stateB[0x4].d, sc->stateB[0x0].d, sc->stateB[0xD].d, sc->stateC[0x1].d, M(0x7));
	PP(sc->stateA[0x0].d, sc->stateA[0xB].d, sc->stateB[0x8].d, sc->stateB[0x5].d, sc->stateB[0x1].d, sc->stateB[0xE].d, sc->stateC[0x0].d, M(0x8));
	PP(sc->stateA[0x1].d, sc->stateA[0x0].d, sc->stateB[0x9].d, sc->stateB[0x6].d, sc->stateB[0x2].d, sc->stateB[0xF].d, sc->stateC[0xF].d, M(0x9));
	PP(sc->stateA[0x2].d, sc->stateA[0x1].d, sc->stateB[0xA].d, sc->stateB[0x7].d, sc->stateB[0x3].d, sc->stateB[0x0].d, sc->stateC[0xE].d, M(0xA));
	PP(sc->stateA[0x3].d, sc->stateA[0x2].d, sc->stateB[0xB].d, sc->stateB[0x8].d, sc->stateB[0x4].d, sc->stateB[0x1].d, sc->stateC[0xD].d, M(0xB));
	PP(sc->stateA[0x4].d, sc->stateA[0x3].d, sc->stateB[0xC].d, sc->stateB[0x9].d, sc->stateB[0x5].d, sc->stateB[0x2].d, sc->stateC[0xC].d, M(0xC));
	PP(sc->stateA[0x5].d, sc->stateA[0x4].d, sc->stateB[0xD].d, sc->stateB[0xA].d, sc->stateB[0x6].d, sc->stateB[0x3].d, sc->stateC[0xB].d, M(0xD));
	PP(sc->stateA[0x6].d, sc->stateA[0x5].d, sc->stateB[0xE].d, sc->stateB[0xB].d, sc->stateB[0x7].d, sc->stateB[0x4].d, sc->stateC[0xA].d, M(0xE));
	PP(sc->stateA[0x7].d, sc->stateA[0x6].d, sc->stateB[0xF].d, sc->stateB[0xC].d, sc->stateB[0x8].d, sc->stateB[0x5].d, sc->stateC[0x9].d, M(0xF));

	PP(sc->stateA[0x8].d, sc->stateA[0x7].d, sc->stateB[0x0].d, sc->stateB[0xD].d, sc->stateB[0x9].d, sc->stateB[0x6].d, sc->stateC[0x8].d, M(0x0));
	PP(sc->stateA[0x9].d, sc->stateA[0x8].d, sc->stateB[0x1].d, sc->stateB[0xE].d, sc->stateB[0xA].d, sc->stateB[0x7].d, sc->stateC[0x7].d, M(0x1));
	PP(sc->stateA[0xA].d, sc->stateA[0x9].d, sc->stateB[0x2].d, sc->stateB[0xF].d, sc->stateB[0xB].d, sc->stateB[0x8].d, sc->stateC[0x6].d, M(0x2));
	PP(sc->stateA[0xB].d, sc->stateA[0xA].d, sc->stateB[0x3].d, sc->stateB[0x0].d, sc->stateB[0xC].d, sc->stateB[0x9].d, sc->stateC[0x5].d, M(0x3));
	PP(sc->stateA[0x0].d, sc->stateA[0xB].d, sc->stateB[0x4].d, sc->stateB[0x1].d, sc->stateB[0xD].d, sc->stateB[0xA].d, sc->stateC[0x4].d, M(0x4));
	PP(sc->stateA[0x1].d, sc->stateA[0x0].d, sc->stateB[0x5].d, sc->stateB[0x2].d, sc->stateB[0xE].d, sc->stateB[0xB].d, sc->stateC[0x3].d, M(0x5));
	PP(sc->stateA[0x2].d, sc->stateA[0x1].d, sc->stateB[0x6].d, sc->stateB[0x3].d, sc->stateB[0xF].d, sc->stateB[0xC].d, sc->stateC[0x2].d, M(0x6));
	PP(sc->stateA[0x3].d, sc->stateA[0x2].d, sc->stateB[0x7].d, sc->stateB[0x4].d, sc->stateB[0x0].d, sc->stateB[0xD].d, sc->stateC[0x1].d, M(0x7));
	PP(sc->stateA[0x4].d, sc->stateA[0x3].d, sc->stateB[0x8].d, sc->stateB[0x5].d, sc->stateB[0x1].d, sc->stateB[0xE].d, sc->stateC[0x0].d, M(0x8));
	PP(sc->stateA[0x5].d, sc->stateA[0x4].d, sc->stateB[0x9].d, sc->stateB[0x6].d, sc->stateB[0x2].d, sc->stateB[0xF].d, sc->stateC[0xF].d, M(0x9));
	PP(sc->stateA[0x6].d, sc->stateA[0x5].d, sc->stateB[0xA].d, sc->stateB[0x7].d, sc->stateB[0x3].d, sc->stateB[0x0].d, sc->stateC[0xE].d, M(0xA));
	PP(sc->stateA[0x7].d, sc->stateA[0x6].d, sc->stateB[0xB].d, sc->stateB[0x8].d, sc->stateB[0x4].d, sc->stateB[0x1].d, sc->stateC[0xD].d, M(0xB));
	PP(sc->stateA[0x8].d, sc->stateA[0x7].d, sc->stateB[0xC].d, sc->stateB[0x9].d, sc->stateB[0x5].d, sc->stateB[0x2].d, sc->stateC[0xC].d, M(0xC));
	PP(sc->stateA[0x9].d, sc->stateA[0x8].d, sc->stateB[0xD].d, sc->stateB[0xA].d, sc->stateB[0x6].d, sc->stateB[0x3].d, sc->stateC[0xB].d, M(0xD));
	PP(sc->stateA[0xA].d, sc->stateA[0x9].d, sc->stateB[0xE].d, sc->stateB[0xB].d, sc->stateB[0x7].d, sc->stateB[0x4].d, sc->stateC[0xA].d, M(0xE));
	PP(sc->stateA[0xB].d, sc->stateA[0xA].d, sc->stateB[0xF].d, sc->stateB[0xC].d, sc->stateB[0x8].d, sc->stateB[0x5].d, sc->stateC[0x9].d, M(0xF));

	sc->stateA[0xB].d = _mm256_add_epi32(sc->stateA[0xB].d, sc->stateC[0x6].d);
	sc->stateA[0xA].d = _mm256_add_epi32(sc->stateA[0xA].d, sc->stateC[0x5].d);
	sc->stateA[0x9].d = _mm256_add_epi32(sc->stateA[0x9].d, sc->stateC[0x4].d);
	sc->stateA[0x8].d = _mm256_add_epi32(sc->stateA[0x8].d, sc->stateC[0x3].d);
	sc->stateA[0x7].d = _mm256_add_epi32(sc->stateA[0x7].d, sc->stateC[0x2].d);
	sc->stateA[0x6].d = _mm256_add_epi32(sc->stateA[0x6].d, sc->stateC[0x1].d);
	sc->stateA[0x5].d = _mm256_add_epi32(sc->stateA[0x5].d, sc->stateC[0x0].d);
	sc->stateA[0x4].d = _mm256_add_epi32(sc->stateA[0x4].d, sc->stateC[0xF].d);
	sc->stateA[0x3].d = _mm256_add_epi32(sc->stateA[0x3].d, sc->stateC[0xE].d);
	sc->stateA[0x2].d = _mm256_add_epi32(sc->stateA[0x2].d, sc->stateC[0xD].d);
	sc->stateA[0x1].d = _mm256_add_epi32(sc->stateA[0x1].d, sc->stateC[0xC].d);
	sc->stateA[0x0].d = _mm256_add_epi32(sc->stateA[0x0].d, sc->stateC[0xB].d);
	sc->stateA[0xB].d = _mm256_add_epi32(sc->stateA[0xB].d, sc->stateC[0xA].d);
	sc->stateA[0xA].d = _mm256_add_epi32(sc->stateA[0xA].d, sc->stateC[0x9].d);
	sc->stateA[0x9].d = _mm256_add_epi32(sc->stateA[0x9].d, sc->stateC[0x8].d);
	sc->stateA[0x8].d = _mm256_add_epi32(sc->stateA[0x8].d, sc->stateC[0x7].d);
	sc->stateA[0x7].d = _mm256_add_epi32(sc->stateA[0x7].d, sc->stateC[0x6].d);
	sc->stateA[0x6].d = _mm256_add_epi32(sc->stateA[0x6].d, sc->stateC[0x5].d);
	sc->stateA[0x5].d = _mm256_add_epi32(sc->stateA[0x5].d, sc->stateC[0x4].d);
	sc->stateA[0x4].d = _mm256_add_epi32(sc->stateA[0x4].d, sc->stateC[0x3].d);
	sc->stateA[0x3].d = _mm256_add_epi32(sc->stateA[0x3].d, sc->stateC[0x2].d);
	sc->stateA[0x2].d = _mm256_add_epi32(sc->stateA[0x2].d, sc->stateC[0x1].d);
	sc->stateA[0x1].d = _mm256_add_epi32(sc->stateA[0x1].d, sc->stateC[0x0].d);
	sc->stateA[0x0].d = _mm256_add_epi32(sc->stateA[0x0].d, sc->stateC[0xF].d);
	sc->stateA[0xB].d = _mm256_add_epi32(sc->stateA[0xB].d, sc->stateC[0xE].d);
	sc->stateA[0xA].d = _mm256_add_epi32(sc->stateA[0xA].d, sc->stateC[0xD].d);
	sc->stateA[0x9].d = _mm256_add_epi32(sc->stateA[0x9].d, sc->stateC[0xC].d);
	sc->stateA[0x8].d = _mm256_add_epi32(sc->stateA[0x8].d, sc->stateC[0xB].d);
	sc->stateA[0x7].d = _mm256_add_epi32(sc->stateA[0x7].d, sc->stateC[0xA].d);
	sc->stateA[0x6].d = _mm256_add_epi32(sc->stateA[0x6].d, sc->stateC[0x9].d);
	sc->stateA[0x5].d = _mm256_add_epi32(sc->stateA[0x5].d, sc->stateC[0x8].d);
	sc->stateA[0x4].d = _mm256_add_epi32(sc->stateA[0x4].d, sc->stateC[0x7].d);
	sc->stateA[0x3].d = _mm256_add_epi32(sc->stateA[0x3].d, sc->stateC[0x6].d);
	sc->stateA[0x2].d = _mm256_add_epi32(sc->stateA[0x2].d, sc->stateC[0x5].d);
	sc->stateA[0x1].d = _mm256_add_epi32(sc->stateA[0x1].d, sc->stateC[0x4].d);
	sc->stateA[0x0].d = _mm256_add_epi32(sc->stateA[0x0].d, sc->stateC[0x3].d);

#define SWAP_AND_SUB(xb, xc, xm)   do { \
    __m256i tmp; \
    tmp = xb; \
    xb = _mm256_sub_epi32(xc, xm); \
    xc = tmp; \
	        } while (0)

	SWAP_AND_SUB(sc->stateB[0x0].d, sc->stateC[0x0].d, M(0x0));
	SWAP_AND_SUB(sc->stateB[0x1].d, sc->stateC[0x1].d, M(0x1));
	SWAP_AND_SUB(sc->stateB[0x2].d, sc->stateC[0x2].d, M(0x2));
	SWAP_AND_SUB(sc->stateB[0x3].d, sc->stateC[0x3].d, M(0x3));
	SWAP_AND_SUB(sc->stateB[0x4].d, sc->stateC[0x4].d, M(0x4));
	SWAP_AND_SUB(sc->stateB[0x5].d, sc->stateC[0x5].d, M(0x5));
	SWAP_AND_SUB(sc->stateB[0x6].d, sc->stateC[0x6].d, M(0x6));
	SWAP_AND_SUB(sc->stateB[0x7].d, sc->stateC[0x7].d, M(0x7));
	SWAP_AND_SUB(sc->stateB[0x8].d, sc->stateC[0x8].d, M(0x8));
	SWAP_AND_SUB(sc->stateB[0x9].d, sc->stateC[0x9].d, M(0x9));
	SWAP_AND_SUB(sc->stateB[0xA].d, sc->stateC[0xA].d, M(0xA));
	SWAP_AND_SUB(sc->stateB[0xB].d, sc->stateC[0xB].d, M(0xB));
	SWAP_AND_SUB(sc->stateB[0xC].d, sc->stateC[0xC].d, M(0xC));
	SWAP_AND_SUB(sc->stateB[0xD].d, sc->stateC[0xD].d, M(0xD));
	SWAP_AND_SUB(sc->stateB[0xE].d, sc->stateC[0xE].d, M(0xE));
	SWAP_AND_SUB(sc->stateB[0xF].d, sc->stateC[0xF].d, M(0xF));

#undef M
#undef SWAP_AND_SUB
#undef PP
#undef U
#undef U2
}
#endif

#ifndef __MINGW__
__declspec(noinline)
#else
__attribute__((noinline))
#endif
static void mshabal4_compress(mshabal4_context *sc,
const void *buf0, const void *buf1,
const void *buf2, const void *buf3,
const void *buf4, const void *buf5,
const void *buf6, const void *buf7)
{

	size_t j;
	un4 u;
	if (buf7 == NULL) {
		for (j = 0; j < 64; j += 4) {
			u.words[j + 0] = *(u32 *)((u8*)buf0 + j);
			u.words[j + 1] = *(u32 *)((u8*)buf1 + j);
			u.words[j + 2] = *(u32 *)((u8*)buf2 + j);
			u.words[j + 3] = *(u32 *)((u8*)buf3 + j);
		}
	}
	else {
		for (j = 0; j < 32; j += 4) {
			u.words[j + 0] = *(u32 *)((u8*)buf0 + j);
			u.words[j + 1] = *(u32 *)((u8*)buf1 + j);
			u.words[j + 2] = *(u32 *)((u8*)buf2 + j);
			u.words[j + 3] = *(u32 *)((u8*)buf3 + j);
		}
		for (j = 0; j < 32; j += 4) {
			size_t k = (j + 32);
			u.words[k + 0] = *(u32 *)((u8*)buf4 + j);
			u.words[k + 1] = *(u32 *)((u8*)buf5 + j);
			u.words[k + 2] = *(u32 *)((u8*)buf6 + j);
			u.words[k + 3] = *(u32 *)((u8*)buf7 + j);
		}
	}
#define U(i) u.data + i
#define U2(i) u.data[i]
#define M(i)   U2(i)

	__m128i one;

	for (j = 0; j < 16; j++) {
		sc->stateB[j].d = _mm_add_epi32(sc->stateB[j].d, U2(j));
		sc->stateB[j].d = _mm_or_si128(_mm_slli_epi32(sc->stateB[j].d, 17),
			_mm_srli_epi32(sc->stateB[j].d, 15));
	}

	sc->stateA[0].d = _mm_xor_si128(sc->stateA[0].d, _mm_set1_epi32(1));
	sc->stateA[1].d = _mm_xor_si128(sc->stateA[1].d, _mm_set1_epi32(0));

	one = _mm_set1_epi32(C32(0xFFFFFFFF));

#define PP(xa0, xa1, xb0, xb1, xb2, xb3, xc, xm)   do { \
    __m128i tt; \
    tt = _mm_or_si128(_mm_slli_epi32(xa1, 15), \
        _mm_srli_epi32(xa1, 17)); \
    tt = _mm_add_epi32(_mm_slli_epi32(tt, 2), tt); \
    tt = _mm_xor_si128(_mm_xor_si128(xa0, tt), xc); \
    tt = _mm_add_epi32(_mm_slli_epi32(tt, 1), tt); \
    tt = _mm_xor_si128( \
        _mm_xor_si128(tt, xb1), \
        _mm_xor_si128(_mm_andnot_si128(xb3, xb2), xm)); \
    xa0 = tt; \
    tt = xb0; \
    tt = _mm_or_si128(_mm_slli_epi32(tt, 1), \
        _mm_srli_epi32(tt, 31)); \
    xb0 = _mm_xor_si128(tt, _mm_xor_si128(xa0, one)); \
			} while (0)

	PP(sc->stateA[0x0].d, sc->stateA[0xB].d, sc->stateB[0x0].d, sc->stateB[0xD].d, sc->stateB[0x9].d, sc->stateB[0x6].d, sc->stateC[0x8].d, M(0x0));
	PP(sc->stateA[0x1].d, sc->stateA[0x0].d, sc->stateB[0x1].d, sc->stateB[0xE].d, sc->stateB[0xA].d, sc->stateB[0x7].d, sc->stateC[0x7].d, M(0x1));
	PP(sc->stateA[0x2].d, sc->stateA[0x1].d, sc->stateB[0x2].d, sc->stateB[0xF].d, sc->stateB[0xB].d, sc->stateB[0x8].d, sc->stateC[0x6].d, M(0x2));
	PP(sc->stateA[0x3].d, sc->stateA[0x2].d, sc->stateB[0x3].d, sc->stateB[0x0].d, sc->stateB[0xC].d, sc->stateB[0x9].d, sc->stateC[0x5].d, M(0x3));
	PP(sc->stateA[0x4].d, sc->stateA[0x3].d, sc->stateB[0x4].d, sc->stateB[0x1].d, sc->stateB[0xD].d, sc->stateB[0xA].d, sc->stateC[0x4].d, M(0x4));
	PP(sc->stateA[0x5].d, sc->stateA[0x4].d, sc->stateB[0x5].d, sc->stateB[0x2].d, sc->stateB[0xE].d, sc->stateB[0xB].d, sc->stateC[0x3].d, M(0x5));
	PP(sc->stateA[0x6].d, sc->stateA[0x5].d, sc->stateB[0x6].d, sc->stateB[0x3].d, sc->stateB[0xF].d, sc->stateB[0xC].d, sc->stateC[0x2].d, M(0x6));
	PP(sc->stateA[0x7].d, sc->stateA[0x6].d, sc->stateB[0x7].d, sc->stateB[0x4].d, sc->stateB[0x0].d, sc->stateB[0xD].d, sc->stateC[0x1].d, M(0x7));
	PP(sc->stateA[0x8].d, sc->stateA[0x7].d, sc->stateB[0x8].d, sc->stateB[0x5].d, sc->stateB[0x1].d, sc->stateB[0xE].d, sc->stateC[0x0].d, M(0x8));
	PP(sc->stateA[0x9].d, sc->stateA[0x8].d, sc->stateB[0x9].d, sc->stateB[0x6].d, sc->stateB[0x2].d, sc->stateB[0xF].d, sc->stateC[0xF].d, M(0x9));
	PP(sc->stateA[0xA].d, sc->stateA[0x9].d, sc->stateB[0xA].d, sc->stateB[0x7].d, sc->stateB[0x3].d, sc->stateB[0x0].d, sc->stateC[0xE].d, M(0xA));
	PP(sc->stateA[0xB].d, sc->stateA[0xA].d, sc->stateB[0xB].d, sc->stateB[0x8].d, sc->stateB[0x4].d, sc->stateB[0x1].d, sc->stateC[0xD].d, M(0xB));
	PP(sc->stateA[0x0].d, sc->stateA[0xB].d, sc->stateB[0xC].d, sc->stateB[0x9].d, sc->stateB[0x5].d, sc->stateB[0x2].d, sc->stateC[0xC].d, M(0xC));
	PP(sc->stateA[0x1].d, sc->stateA[0x0].d, sc->stateB[0xD].d, sc->stateB[0xA].d, sc->stateB[0x6].d, sc->stateB[0x3].d, sc->stateC[0xB].d, M(0xD));
	PP(sc->stateA[0x2].d, sc->stateA[0x1].d, sc->stateB[0xE].d, sc->stateB[0xB].d, sc->stateB[0x7].d, sc->stateB[0x4].d, sc->stateC[0xA].d, M(0xE));
	PP(sc->stateA[0x3].d, sc->stateA[0x2].d, sc->stateB[0xF].d, sc->stateB[0xC].d, sc->stateB[0x8].d, sc->stateB[0x5].d, sc->stateC[0x9].d, M(0xF));

	PP(sc->stateA[0x4].d, sc->stateA[0x3].d, sc->stateB[0x0].d, sc->stateB[0xD].d, sc->stateB[0x9].d, sc->stateB[0x6].d, sc->stateC[0x8].d, M(0x0));
	PP(sc->stateA[0x5].d, sc->stateA[0x4].d, sc->stateB[0x1].d, sc->stateB[0xE].d, sc->stateB[0xA].d, sc->stateB[0x7].d, sc->stateC[0x7].d, M(0x1));
	PP(sc->stateA[0x6].d, sc->stateA[0x5].d, sc->stateB[0x2].d, sc->stateB[0xF].d, sc->stateB[0xB].d, sc->stateB[0x8].d, sc->stateC[0x6].d, M(0x2));
	PP(sc->stateA[0x7].d, sc->stateA[0x6].d, sc->stateB[0x3].d, sc->stateB[0x0].d, sc->stateB[0xC].d, sc->stateB[0x9].d, sc->stateC[0x5].d, M(0x3));
	PP(sc->stateA[0x8].d, sc->stateA[0x7].d, sc->stateB[0x4].d, sc->stateB[0x1].d, sc->stateB[0xD].d, sc->stateB[0xA].d, sc->stateC[0x4].d, M(0x4));
	PP(sc->stateA[0x9].d, sc->stateA[0x8].d, sc->stateB[0x5].d, sc->stateB[0x2].d, sc->stateB[0xE].d, sc->stateB[0xB].d, sc->stateC[0x3].d, M(0x5));
	PP(sc->stateA[0xA].d, sc->stateA[0x9].d, sc->stateB[0x6].d, sc->stateB[0x3].d, sc->stateB[0xF].d, sc->stateB[0xC].d, sc->stateC[0x2].d, M(0x6));
	PP(sc->stateA[0xB].d, sc->stateA[0xA].d, sc->stateB[0x7].d, sc->stateB[0x4].d, sc->stateB[0x0].d, sc->stateB[0xD].d, sc->stateC[0x1].d, M(0x7));
	PP(sc->stateA[0x0].d, sc->stateA[0xB].d, sc->stateB[0x8].d, sc->stateB[0x5].d, sc->stateB[0x1].d, sc->stateB[0xE].d, sc->stateC[0x0].d, M(0x8));
	PP(sc->stateA[0x1].d, sc->stateA[0x0].d, sc->stateB[0x9].d, sc->stateB[0x6].d, sc->stateB[0x2].d, sc->stateB[0xF].d, sc->stateC[0xF].d, M(0x9));
	PP(sc->stateA[0x2].d, sc->stateA[0x1].d, sc->stateB[0xA].d, sc->stateB[0x7].d, sc->stateB[0x3].d, sc->stateB[0x0].d, sc->stateC[0xE].d, M(0xA));
	PP(sc->stateA[0x3].d, sc->stateA[0x2].d, sc->stateB[0xB].d, sc->stateB[0x8].d, sc->stateB[0x4].d, sc->stateB[0x1].d, sc->stateC[0xD].d, M(0xB));
	PP(sc->stateA[0x4].d, sc->stateA[0x3].d, sc->stateB[0xC].d, sc->stateB[0x9].d, sc->stateB[0x5].d, sc->stateB[0x2].d, sc->stateC[0xC].d, M(0xC));
	PP(sc->stateA[0x5].d, sc->stateA[0x4].d, sc->stateB[0xD].d, sc->stateB[0xA].d, sc->stateB[0x6].d, sc->stateB[0x3].d, sc->stateC[0xB].d, M(0xD));
	PP(sc->stateA[0x6].d, sc->stateA[0x5].d, sc->stateB[0xE].d, sc->stateB[0xB].d, sc->stateB[0x7].d, sc->stateB[0x4].d, sc->stateC[0xA].d, M(0xE));
	PP(sc->stateA[0x7].d, sc->stateA[0x6].d, sc->stateB[0xF].d, sc->stateB[0xC].d, sc->stateB[0x8].d, sc->stateB[0x5].d, sc->stateC[0x9].d, M(0xF));

	PP(sc->stateA[0x8].d, sc->stateA[0x7].d, sc->stateB[0x0].d, sc->stateB[0xD].d, sc->stateB[0x9].d, sc->stateB[0x6].d, sc->stateC[0x8].d, M(0x0));
	PP(sc->stateA[0x9].d, sc->stateA[0x8].d, sc->stateB[0x1].d, sc->stateB[0xE].d, sc->stateB[0xA].d, sc->stateB[0x7].d, sc->stateC[0x7].d, M(0x1));
	PP(sc->stateA[0xA].d, sc->stateA[0x9].d, sc->stateB[0x2].d, sc->stateB[0xF].d, sc->stateB[0xB].d, sc->stateB[0x8].d, sc->stateC[0x6].d, M(0x2));
	PP(sc->stateA[0xB].d, sc->stateA[0xA].d, sc->stateB[0x3].d, sc->stateB[0x0].d, sc->stateB[0xC].d, sc->stateB[0x9].d, sc->stateC[0x5].d, M(0x3));
	PP(sc->stateA[0x0].d, sc->stateA[0xB].d, sc->stateB[0x4].d, sc->stateB[0x1].d, sc->stateB[0xD].d, sc->stateB[0xA].d, sc->stateC[0x4].d, M(0x4));
	PP(sc->stateA[0x1].d, sc->stateA[0x0].d, sc->stateB[0x5].d, sc->stateB[0x2].d, sc->stateB[0xE].d, sc->stateB[0xB].d, sc->stateC[0x3].d, M(0x5));
	PP(sc->stateA[0x2].d, sc->stateA[0x1].d, sc->stateB[0x6].d, sc->stateB[0x3].d, sc->stateB[0xF].d, sc->stateB[0xC].d, sc->stateC[0x2].d, M(0x6));
	PP(sc->stateA[0x3].d, sc->stateA[0x2].d, sc->stateB[0x7].d, sc->stateB[0x4].d, sc->stateB[0x0].d, sc->stateB[0xD].d, sc->stateC[0x1].d, M(0x7));
	PP(sc->stateA[0x4].d, sc->stateA[0x3].d, sc->stateB[0x8].d, sc->stateB[0x5].d, sc->stateB[0x1].d, sc->stateB[0xE].d, sc->stateC[0x0].d, M(0x8));
	PP(sc->stateA[0x5].d, sc->stateA[0x4].d, sc->stateB[0x9].d, sc->stateB[0x6].d, sc->stateB[0x2].d, sc->stateB[0xF].d, sc->stateC[0xF].d, M(0x9));
	PP(sc->stateA[0x6].d, sc->stateA[0x5].d, sc->stateB[0xA].d, sc->stateB[0x7].d, sc->stateB[0x3].d, sc->stateB[0x0].d, sc->stateC[0xE].d, M(0xA));
	PP(sc->stateA[0x7].d, sc->stateA[0x6].d, sc->stateB[0xB].d, sc->stateB[0x8].d, sc->stateB[0x4].d, sc->stateB[0x1].d, sc->stateC[0xD].d, M(0xB));
	PP(sc->stateA[0x8].d, sc->stateA[0x7].d, sc->stateB[0xC].d, sc->stateB[0x9].d, sc->stateB[0x5].d, sc->stateB[0x2].d, sc->stateC[0xC].d, M(0xC));
	PP(sc->stateA[0x9].d, sc->stateA[0x8].d, sc->stateB[0xD].d, sc->stateB[0xA].d, sc->stateB[0x6].d, sc->stateB[0x3].d, sc->stateC[0xB].d, M(0xD));
	PP(sc->stateA[0xA].d, sc->stateA[0x9].d, sc->stateB[0xE].d, sc->stateB[0xB].d, sc->stateB[0x7].d, sc->stateB[0x4].d, sc->stateC[0xA].d, M(0xE));
	PP(sc->stateA[0xB].d, sc->stateA[0xA].d, sc->stateB[0xF].d, sc->stateB[0xC].d, sc->stateB[0x8].d, sc->stateB[0x5].d, sc->stateC[0x9].d, M(0xF));

	sc->stateA[0xB].d = _mm_add_epi32(sc->stateA[0xB].d, sc->stateC[0x6].d);
	sc->stateA[0xA].d = _mm_add_epi32(sc->stateA[0xA].d, sc->stateC[0x5].d);
	sc->stateA[0x9].d = _mm_add_epi32(sc->stateA[0x9].d, sc->stateC[0x4].d);
	sc->stateA[0x8].d = _mm_add_epi32(sc->stateA[0x8].d, sc->stateC[0x3].d);
	sc->stateA[0x7].d = _mm_add_epi32(sc->stateA[0x7].d, sc->stateC[0x2].d);
	sc->stateA[0x6].d = _mm_add_epi32(sc->stateA[0x6].d, sc->stateC[0x1].d);
	sc->stateA[0x5].d = _mm_add_epi32(sc->stateA[0x5].d, sc->stateC[0x0].d);
	sc->stateA[0x4].d = _mm_add_epi32(sc->stateA[0x4].d, sc->stateC[0xF].d);
	sc->stateA[0x3].d = _mm_add_epi32(sc->stateA[0x3].d, sc->stateC[0xE].d);
	sc->stateA[0x2].d = _mm_add_epi32(sc->stateA[0x2].d, sc->stateC[0xD].d);
	sc->stateA[0x1].d = _mm_add_epi32(sc->stateA[0x1].d, sc->stateC[0xC].d);
	sc->stateA[0x0].d = _mm_add_epi32(sc->stateA[0x0].d, sc->stateC[0xB].d);
	sc->stateA[0xB].d = _mm_add_epi32(sc->stateA[0xB].d, sc->stateC[0xA].d);
	sc->stateA[0xA].d = _mm_add_epi32(sc->stateA[0xA].d, sc->stateC[0x9].d);
	sc->stateA[0x9].d = _mm_add_epi32(sc->stateA[0x9].d, sc->stateC[0x8].d);
	sc->stateA[0x8].d = _mm_add_epi32(sc->stateA[0x8].d, sc->stateC[0x7].d);
	sc->stateA[0x7].d = _mm_add_epi32(sc->stateA[0x7].d, sc->stateC[0x6].d);
	sc->stateA[0x6].d = _mm_add_epi32(sc->stateA[0x6].d, sc->stateC[0x5].d);
	sc->stateA[0x5].d = _mm_add_epi32(sc->stateA[0x5].d, sc->stateC[0x4].d);
	sc->stateA[0x4].d = _mm_add_epi32(sc->stateA[0x4].d, sc->stateC[0x3].d);
	sc->stateA[0x3].d = _mm_add_epi32(sc->stateA[0x3].d, sc->stateC[0x2].d);
	sc->stateA[0x2].d = _mm_add_epi32(sc->stateA[0x2].d, sc->stateC[0x1].d);
	sc->stateA[0x1].d = _mm_add_epi32(sc->stateA[0x1].d, sc->stateC[0x0].d);
	sc->stateA[0x0].d = _mm_add_epi32(sc->stateA[0x0].d, sc->stateC[0xF].d);
	sc->stateA[0xB].d = _mm_add_epi32(sc->stateA[0xB].d, sc->stateC[0xE].d);
	sc->stateA[0xA].d = _mm_add_epi32(sc->stateA[0xA].d, sc->stateC[0xD].d);
	sc->stateA[0x9].d = _mm_add_epi32(sc->stateA[0x9].d, sc->stateC[0xC].d);
	sc->stateA[0x8].d = _mm_add_epi32(sc->stateA[0x8].d, sc->stateC[0xB].d);
	sc->stateA[0x7].d = _mm_add_epi32(sc->stateA[0x7].d, sc->stateC[0xA].d);
	sc->stateA[0x6].d = _mm_add_epi32(sc->stateA[0x6].d, sc->stateC[0x9].d);
	sc->stateA[0x5].d = _mm_add_epi32(sc->stateA[0x5].d, sc->stateC[0x8].d);
	sc->stateA[0x4].d = _mm_add_epi32(sc->stateA[0x4].d, sc->stateC[0x7].d);
	sc->stateA[0x3].d = _mm_add_epi32(sc->stateA[0x3].d, sc->stateC[0x6].d);
	sc->stateA[0x2].d = _mm_add_epi32(sc->stateA[0x2].d, sc->stateC[0x5].d);
	sc->stateA[0x1].d = _mm_add_epi32(sc->stateA[0x1].d, sc->stateC[0x4].d);
	sc->stateA[0x0].d = _mm_add_epi32(sc->stateA[0x0].d, sc->stateC[0x3].d);

#define SWAP_AND_SUB(xb, xc, xm)   do { \
    __m128i tmp; \
    tmp = xb; \
    xb = _mm_sub_epi32(xc, xm); \
    xc = tmp; \
		        } while (0)

	SWAP_AND_SUB(sc->stateB[0x0].d, sc->stateC[0x0].d, M(0x0));
	SWAP_AND_SUB(sc->stateB[0x1].d, sc->stateC[0x1].d, M(0x1));
	SWAP_AND_SUB(sc->stateB[0x2].d, sc->stateC[0x2].d, M(0x2));
	SWAP_AND_SUB(sc->stateB[0x3].d, sc->stateC[0x3].d, M(0x3));
	SWAP_AND_SUB(sc->stateB[0x4].d, sc->stateC[0x4].d, M(0x4));
	SWAP_AND_SUB(sc->stateB[0x5].d, sc->stateC[0x5].d, M(0x5));
	SWAP_AND_SUB(sc->stateB[0x6].d, sc->stateC[0x6].d, M(0x6));
	SWAP_AND_SUB(sc->stateB[0x7].d, sc->stateC[0x7].d, M(0x7));
	SWAP_AND_SUB(sc->stateB[0x8].d, sc->stateC[0x8].d, M(0x8));
	SWAP_AND_SUB(sc->stateB[0x9].d, sc->stateC[0x9].d, M(0x9));
	SWAP_AND_SUB(sc->stateB[0xA].d, sc->stateC[0xA].d, M(0xA));
	SWAP_AND_SUB(sc->stateB[0xB].d, sc->stateC[0xB].d, M(0xB));
	SWAP_AND_SUB(sc->stateB[0xC].d, sc->stateC[0xC].d, M(0xC));
	SWAP_AND_SUB(sc->stateB[0xD].d, sc->stateC[0xD].d, M(0xD));
	SWAP_AND_SUB(sc->stateB[0xE].d, sc->stateC[0xE].d, M(0xE));
	SWAP_AND_SUB(sc->stateB[0xF].d, sc->stateC[0xF].d, M(0xF));

#undef M
#undef SWAP_AND_SUB
#undef PP
#undef U
#undef U2
}

#ifdef __8WAY__
#ifndef __MINGW__
__forceinline 
#else
inline
#endif
static void mshabal8_32(mshabal8_context *sc, void *dst, void *dst0, void *dst1, void *dst2, void *dst3,
	void *dst4, void *dst5, void *dst6, void *dst7, const size_t i)
{
	size_t y, z;
	mshabal8_compress_fast4_32(sc, (u8*)dst + i - 1);
	for (z = 0; z < 8; z++) {
		for (y = 0; y < 8; y++) {
			size_t k = z * 8;
			((u32*)((u8*)dst + i))[y + k] = sc->stateC[z + 8].w[y];
		}
	}
	for (z = 0; z < 8; z++)
		((u32*)dst0)[z] = sc->stateC[z + 8].w[0];
	for (z = 0; z < 8; z++)
		((u32*)dst1)[z] = sc->stateC[z + 8].w[1];
	for (z = 0; z < 8; z++)
		((u32*)dst2)[z] = sc->stateC[z + 8].w[2];
	for (z = 0; z < 8; z++)
		((u32*)dst3)[z] = sc->stateC[z + 8].w[3];
	for (z = 0; z < 8; z++)
		((u32*)dst4)[z] = sc->stateC[z + 8].w[4];
	for (z = 0; z < 8; z++)
		((u32*)dst5)[z] = sc->stateC[z + 8].w[5];
	for (z = 0; z < 8; z++)
		((u32*)dst6)[z] = sc->stateC[z + 8].w[6];
	for (z = 0; z < 8; z++)
		((u32*)dst7)[z] = sc->stateC[z + 8].w[7];
}
#endif

#ifndef __MINGW__
__forceinline
#else
inline
#endif
static void mshabal4_32(mshabal4_context *sc, void *dst, void *dst0, void *dst1, void *dst2, void *dst3, const size_t i)
{
	size_t y, z;
	mshabal4_compress_fast4_32(sc, (u8*)dst + i - 1);
	for (z = 0; z < 8; z++) {
		for (y = 0; y < 4; y++) {
			size_t k = z * 4;
			((u32*)((u8*)dst + i))[y + k] = sc->stateC[z + 8].w[y];
		}
	}
	for (z = 0; z < 8; z++)
		((u32*)dst0)[z] = sc->stateC[z + 8].w[0];
	for (z = 0; z < 8; z++)
		((u32*)dst1)[z] = sc->stateC[z + 8].w[1];
	for (z = 0; z < 8; z++)
		((u32*)dst2)[z] = sc->stateC[z + 8].w[2];
	for (z = 0; z < 8; z++)
		((u32*)dst3)[z] = sc->stateC[z + 8].w[3];
}

#ifdef __8WAY__
#ifndef __MINGW__
__forceinline 
#else
inline
#endif
static void mshabal8_64(mshabal8_context *sc, const void *data0, const void *data1,
	const void *data2, const void *data3, const void *data4, const void *data5,
	const void *data6, const void *data7, const void *data8, const void *data9,
	const void *data10, const void *data11, const void *data12, const void *data13,
	const void *data14, const void *data15,
	void *dst0, void *dst1, void *dst2, void *dst3,
	void *dst4, void *dst5, void *dst6, void *dst7)
{
	mshabal8_compress(sc, data0, data1, data2, data3,
		data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14, data15);
	size_t z;
	mshabal8_compress_fast4_64(sc);
	for (z = 0; z < 8; z++)
		((u32*)dst0)[z] = sc->stateC[z + 8].w[0];
	for (z = 0; z < 8; z++)
		((u32*)dst1)[z] = sc->stateC[z + 8].w[1];
	for (z = 0; z < 8; z++)
		((u32*)dst2)[z] = sc->stateC[z + 8].w[2];
	for (z = 0; z < 8; z++)
		((u32*)dst3)[z] = sc->stateC[z + 8].w[3];
	for (z = 0; z < 8; z++)
		((u32*)dst4)[z] = sc->stateC[z + 8].w[4];
	for (z = 0; z < 8; z++)
		((u32*)dst5)[z] = sc->stateC[z + 8].w[5];
	for (z = 0; z < 8; z++)
		((u32*)dst6)[z] = sc->stateC[z + 8].w[6];
	for (z = 0; z < 8; z++)
		((u32*)dst7)[z] = sc->stateC[z + 8].w[7];
}
#endif

#ifndef __MINGW__
__forceinline
#else
inline
#endif
static void mshabal4_64(mshabal4_context *sc, const void *data0, const void *data1,
const void *data2, const void *data3, const void *data4, const void *data5,
const void *data6, const void *data7,
void *dst0, void *dst1, void *dst2, void *dst3)
{
	mshabal4_compress(sc, data0, data1, data2, data3, data4, data5, data6, data7);
	size_t z;
	mshabal4_compress_fast4_64(sc);
	for (z = 0; z < 8; z++)
		((u32*)dst0)[z] = sc->stateC[z + 8].w[0];
	for (z = 0; z < 8; z++)
		((u32*)dst1)[z] = sc->stateC[z + 8].w[1];
	for (z = 0; z < 8; z++)
		((u32*)dst2)[z] = sc->stateC[z + 8].w[2];
	for (z = 0; z < 8; z++)
		((u32*)dst3)[z] = sc->stateC[z + 8].w[3];
}

#ifdef __8WAY__
#ifndef __MINGW__
__forceinline 
#else
inline
#endif
static void mshabal8_80(mshabal8_context *sc, const void *data0, const void *data1,
	const void *data2, const void *data3, const void *data4, const void *data5,
	const void *data6, const void *data7,
	void *dst)
{
	mshabal8_compress(sc, data0, data1, data2, data3,
		data4, data5, data6, data7, 0, 0, 0, 0, 0, 0, 0, 0);
	size_t y, z;
	mshabal8_compress_fast4_80(sc, data0, data1, data2, data3, data4, data5, data6, data7);
	for (z = 0; z < 8; z++) {
		for (y = 0; y < 8; y++) {
			size_t k = z * 8;
			((u32*)dst)[y + k] = sc->stateC[z + 8].w[y];
		}
	}
}
#endif

#ifndef __MINGW__
__forceinline
#else
inline
#endif
static void mshabal4_80(mshabal4_context *sc, const void *data0, const void *data1,
const void *data2, const void *data3, void *dst)
{
	mshabal4_compress(sc, data0, data1, data2, data3, 0, 0, 0, 0);
	size_t y, z;
	mshabal4_compress_fast4_80(sc, data0, data1, data2, data3);
	for (z = 0; z < 8; z++) {
		for (y = 0; y < 4; y++) {
			size_t k = z * 4;
			((u32*)dst)[y + k] = sc->stateC[z + 8].w[y];
		}
	}
}

#endif



