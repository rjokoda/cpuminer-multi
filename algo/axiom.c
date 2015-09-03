#include "miner.h"

#include <string.h>
#include <stdint.h>

#ifdef __AVX2__
#define __8WAY__
#endif

#define ALIGNED_TO 128

#ifndef _MSC_VER
#define __MINGW__
#endif

#include "crypto/mshabal.h"

typedef uint32_t hash_t[8];

//void printHash(unsigned int* h)
//{
//	printf("%u-%u-%u-%u-%u-%u-%u-%u\n", h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
//}

#ifndef __MINGW__
__declspec(align(ALIGNED_TO))
#endif
const uint32_t mshabal8_init_state[] = { 1392002386U, 1392002386U, 1392002386U, 1392002386U, 1392002386U, 1392002386U, 1392002386U, 1392002386U, 3846928793U, 3846928793U, 3846928793U, 3846928793U, 3846928793U, 3846928793U, 3846928793U, 3846928793U, 764339180U, 764339180U, 764339180U, 764339180U, 764339180U, 764339180U, 764339180U, 764339180U, 3110359441U, 3110359441U, 3110359441U, 3110359441U, 3110359441U, 3110359441U, 3110359441U, 3110359441U, 3758590854U, 3758590854U, 3758590854U, 3758590854U, 3758590854U, 3758590854U, 3758590854U, 3758590854U, 3145483465U, 3145483465U, 3145483465U, 3145483465U, 3145483465U, 3145483465U, 3145483465U, 3145483465U, 3535126986U, 3535126986U, 3535126986U, 3535126986U, 3535126986U, 3535126986U, 3535126986U, 3535126986U, 2966612876U, 2966612876U, 2966612876U, 2966612876U, 2966612876U, 2966612876U, 2966612876U, 2966612876U, 349067845U, 349067845U, 349067845U, 349067845U, 349067845U, 349067845U, 349067845U, 349067845U, 581914844U, 581914844U, 581914844U, 581914844U, 581914844U, 581914844U, 581914844U, 581914844U, 4026383467U, 4026383467U, 4026383467U, 4026383467U, 4026383467U, 4026383467U, 4026383467U, 4026383467U, 3944855370U, 3944855370U, 3944855370U, 3944855370U, 3944855370U, 3944855370U, 3944855370U, 3944855370U, 3042297582U, 3042297582U, 3042297582U, 3042297582U, 3042297582U, 3042297582U, 3042297582U, 3042297582U, 1047594390U, 1047594390U, 1047594390U, 1047594390U, 1047594390U, 1047594390U, 1047594390U, 1047594390U, 2804573487U, 2804573487U, 2804573487U, 2804573487U, 2804573487U, 2804573487U, 2804573487U, 2804573487U, 2466337119U, 2466337119U, 2466337119U, 2466337119U, 2466337119U, 2466337119U, 2466337119U, 2466337119U, 3660104186U, 3660104186U, 3660104186U, 3660104186U, 3660104186U, 3660104186U, 3660104186U, 3660104186U, 1768937576U, 1768937576U, 1768937576U, 1768937576U, 1768937576U, 1768937576U, 1768937576U, 1768937576U, 2629222258U, 2629222258U, 2629222258U, 2629222258U, 2629222258U, 2629222258U, 2629222258U, 2629222258U, 184434690U, 184434690U, 184434690U, 184434690U, 184434690U, 184434690U, 184434690U, 184434690U, 2799711765U, 2799711765U, 2799711765U, 2799711765U, 2799711765U, 2799711765U, 2799711765U, 2799711765U, 1362674132U, 1362674132U, 1362674132U, 1362674132U, 1362674132U, 1362674132U, 1362674132U, 1362674132U, 3189859078U, 3189859078U, 3189859078U, 3189859078U, 3189859078U, 3189859078U, 3189859078U, 3189859078U, 3012266128U, 3012266128U, 3012266128U, 3012266128U, 3012266128U, 3012266128U, 3012266128U, 3012266128U, 1051244907U, 1051244907U, 1051244907U, 1051244907U, 1051244907U, 1051244907U, 1051244907U, 1051244907U, 848932068U, 848932068U, 848932068U, 848932068U, 848932068U, 848932068U, 848932068U, 848932068U, 814894548U, 814894548U, 814894548U, 814894548U, 814894548U, 814894548U, 814894548U, 814894548U, 1439380645U, 1439380645U, 1439380645U, 1439380645U, 1439380645U, 1439380645U, 1439380645U, 1439380645U, 3020288049U, 3020288049U, 3020288049U, 3020288049U, 3020288049U, 3020288049U, 3020288049U, 3020288049U, 3290644154U, 3290644154U, 3290644154U, 3290644154U, 3290644154U, 3290644154U, 3290644154U, 3290644154U, 3010673017U, 3010673017U, 3010673017U, 3010673017U, 3010673017U, 3010673017U, 3010673017U, 3010673017U, 3235749205U, 3235749205U, 3235749205U, 3235749205U, 3235749205U, 3235749205U, 3235749205U, 3235749205U, 3306956974U, 3306956974U, 3306956974U, 3306956974U, 3306956974U, 3306956974U, 3306956974U, 3306956974U, 2737289441U, 2737289441U, 2737289441U, 2737289441U, 2737289441U, 2737289441U, 2737289441U, 2737289441U, 1455776103U, 1455776103U, 1455776103U, 1455776103U, 1455776103U, 1455776103U, 1455776103U, 1455776103U, 3982574643U, 3982574643U, 3982574643U, 3982574643U, 3982574643U, 3982574643U, 3982574643U, 3982574643U, 2293603680U, 2293603680U, 2293603680U, 2293603680U, 2293603680U, 2293603680U, 2293603680U, 2293603680U, 1625476794U, 1625476794U, 1625476794U, 1625476794U, 1625476794U, 1625476794U, 1625476794U, 1625476794U, 1972063115U, 1972063115U, 1972063115U, 1972063115U, 1972063115U, 1972063115U, 1972063115U, 1972063115U, 2213030527U, 2213030527U, 2213030527U, 2213030527U, 2213030527U, 2213030527U, 2213030527U, 2213030527U, 3163981864U, 3163981864U, 3163981864U, 3163981864U, 3163981864U, 3163981864U, 3163981864U, 3163981864U, 3873442807U, 3873442807U, 3873442807U, 3873442807U, 3873442807U, 3873442807U, 3873442807U, 3873442807U, 3129187925U, 3129187925U, 3129187925U, 3129187925U, 3129187925U, 3129187925U, 3129187925U, 3129187925U, 2605259872U, 2605259872U, 2605259872U, 2605259872U, 2605259872U, 2605259872U, 2605259872U, 2605259872U }; 
#ifdef __MINGW__
__attribute__((aligned(ALIGNED_TO)))
#endif

#ifndef __MINGW__
__declspec(align(ALIGNED_TO))
#endif
const uint32_t mshabal4_init_state[] = { 1392002386U, 1392002386U, 1392002386U, 1392002386U, 
										 3846928793U, 3846928793U, 3846928793U, 3846928793U, 
										 764339180U, 764339180U, 764339180U, 764339180U, 
										 3110359441U, 3110359441U, 3110359441U, 3110359441U, 
										 3758590854U, 3758590854U, 3758590854U, 3758590854U,  
										 3145483465U, 3145483465U, 3145483465U, 3145483465U, 
										 3535126986U, 3535126986U, 3535126986U, 3535126986U,  
										 2966612876U, 2966612876U, 2966612876U, 2966612876U, 
										 349067845U, 349067845U, 349067845U, 349067845U, 
										 581914844U, 581914844U, 581914844U, 581914844U, 
										 4026383467U, 4026383467U, 4026383467U, 4026383467U, 
										 3944855370U, 3944855370U, 3944855370U, 3944855370U, 
										 3042297582U, 3042297582U, 3042297582U, 3042297582U, 
										 1047594390U, 1047594390U, 1047594390U, 1047594390U, 
										 2804573487U, 2804573487U, 2804573487U, 2804573487U, 
										 2466337119U, 2466337119U, 2466337119U, 2466337119U, 
										 3660104186U, 3660104186U, 3660104186U, 3660104186U, 
										 1768937576U, 1768937576U, 1768937576U, 1768937576U, 
										 2629222258U, 2629222258U, 2629222258U, 2629222258U, 
										 184434690U, 184434690U, 184434690U, 184434690U,
										 2799711765U, 2799711765U, 2799711765U, 2799711765U, 
										 1362674132U, 1362674132U, 1362674132U, 1362674132U, 
										 3189859078U, 3189859078U, 3189859078U, 3189859078U, 
										 3012266128U, 3012266128U, 3012266128U, 3012266128U, 
										 1051244907U, 1051244907U, 1051244907U, 1051244907U, 
										 848932068U, 848932068U, 848932068U, 848932068U, 
										 814894548U, 814894548U, 814894548U, 814894548U,
										 1439380645U, 1439380645U, 1439380645U, 1439380645U, 
										 3020288049U, 3020288049U, 3020288049U, 3020288049U, 
										 3290644154U, 3290644154U, 3290644154U, 3290644154U,
										 3010673017U, 3010673017U, 3010673017U, 3010673017U, 
										 3235749205U, 3235749205U, 3235749205U, 3235749205U,
										 3306956974U, 3306956974U, 3306956974U, 3306956974U, 
										 2737289441U, 2737289441U, 2737289441U, 2737289441U,
										 1455776103U, 1455776103U, 1455776103U, 1455776103U,
										 3982574643U, 3982574643U, 3982574643U, 3982574643U, 
										 2293603680U, 2293603680U, 2293603680U, 2293603680U,
										 1625476794U, 1625476794U, 1625476794U, 1625476794U, 
										 1972063115U, 1972063115U, 1972063115U, 1972063115U, 
										 2213030527U, 2213030527U, 2213030527U, 2213030527U,
										 3163981864U, 3163981864U, 3163981864U, 3163981864U,
										 3873442807U, 3873442807U, 3873442807U, 3873442807U,
										 3129187925U, 3129187925U, 3129187925U, 3129187925U,
										 2605259872U, 2605259872U, 2605259872U, 2605259872U };
#ifdef __MINGW__
__attribute__((aligned(ALIGNED_TO)))
#endif

#define MOD65535_INDEX_AVX(a0, a1, a2, a3, a4, a5, a6, a7, b, i) \
__m256i vec = _mm256_setr_epi32(a0, a1, a2, a3, a4, a5, a6, a7); \
__m256i mask = _mm256_set1_epi32(0xffff); \
vec = _mm256_add_epi32(_mm256_srli_epi32(vec, 16), _mm256_and_si256(vec, mask)); \
vec = _mm256_and_si256(_mm256_add_epi32(vec, _mm256_srli_epi32(_mm256_add_epi32(vec, _mm256_set1_epi32(1)), 16)), mask); \
vec = _mm256_and_si256(_mm256_add_epi32(vec, _mm256_set1_epi32(i)), mask); \
_mm256_store_si256((__m256i *)b, vec);

#define MOD65535_INDEX_SSE(a0, a1, a2, a3, b, i) \
__m128i vec = _mm_setr_epi32(a0, a1, a2, a3); \
__m128i mask = _mm_set1_epi32(0xffff); \
vec = _mm_add_epi32(_mm_srli_epi32(vec, 16), _mm_and_si128(vec, mask)); \
vec = _mm_and_si128(_mm_add_epi32(vec, _mm_srli_epi32(_mm_add_epi32(vec, _mm_set1_epi32(1)), 16)), mask); \
vec = _mm_and_si128(_mm_add_epi32(vec, _mm_set1_epi32(i)), mask); \
_mm_store_si128((__m128i *)b, vec);

#define FREE_MEM_AVX \
ALIGNED_FREE(endiandata_1); \
ALIGNED_FREE(endiandata_2); \
ALIGNED_FREE(endiandata_3); \
ALIGNED_FREE(endiandata_4); \
ALIGNED_FREE(endiandata_5); \
ALIGNED_FREE(endiandata_6); \
ALIGNED_FREE(endiandata_7); \
ALIGNED_FREE(endiandata_8); \
ALIGNED_FREE(hash64_1); \
ALIGNED_FREE(hash64_2); \
ALIGNED_FREE(hash64_3); \
ALIGNED_FREE(hash64_4); \
ALIGNED_FREE(hash64_5); \
ALIGNED_FREE(hash64_6); \
ALIGNED_FREE(hash64_7); \
ALIGNED_FREE(hash64_8); \
ALIGNED_FREE(ji); \
ALIGNED_FREE(memspace); \
ALIGNED_FREE(memspace2); \
ALIGNED_FREE(ctx_shabal); \
ALIGNED_FREE(ctx_shabal_org);

#define FREE_MEM_SSE \
ALIGNED_FREE(endiandata_1); \
ALIGNED_FREE(endiandata_2); \
ALIGNED_FREE(endiandata_3); \
ALIGNED_FREE(endiandata_4); \
ALIGNED_FREE(hash64_1); \
ALIGNED_FREE(hash64_2); \
ALIGNED_FREE(hash64_3); \
ALIGNED_FREE(hash64_4); \
ALIGNED_FREE(ji); \
ALIGNED_FREE(memspace); \
ALIGNED_FREE(memspace2); \
ALIGNED_FREE(ctx_shabal); \
ALIGNED_FREE(ctx_shabal_org);


#ifdef __MINGW__
#define CTXCOPY \
    for(size_t j = 0; j < 12; j++) { \
        ctx_shabal->stateA[j].d = ctx_shabal_org->stateA[j].d; \
	    } \
    for(size_t j = 0; j < 16; j++) { \
        ctx_shabal->stateB[j].d = ctx_shabal_org->stateB[j].d; \
	    } \
    for(size_t j = 0; j < 16; j++) { \
        ctx_shabal->stateC[j].d = ctx_shabal_org->stateC[j].d; \
	    }
#else
#define CTXCOPY \
	ctx_shabal->stateA[0].d = ctx_shabal_org->stateA[0].d; \
	ctx_shabal->stateA[1].d = ctx_shabal_org->stateA[1].d; \
	ctx_shabal->stateA[2].d = ctx_shabal_org->stateA[2].d; \
	ctx_shabal->stateA[3].d = ctx_shabal_org->stateA[3].d; \
	ctx_shabal->stateA[4].d = ctx_shabal_org->stateA[4].d; \
	ctx_shabal->stateA[5].d = ctx_shabal_org->stateA[5].d; \
	ctx_shabal->stateA[6].d = ctx_shabal_org->stateA[6].d; \
	ctx_shabal->stateA[7].d = ctx_shabal_org->stateA[7].d; \
	ctx_shabal->stateA[8].d = ctx_shabal_org->stateA[8].d; \
	ctx_shabal->stateA[9].d = ctx_shabal_org->stateA[9].d; \
	ctx_shabal->stateA[10].d = ctx_shabal_org->stateA[10].d; \
	ctx_shabal->stateA[11].d = ctx_shabal_org->stateA[11].d; \
	ctx_shabal->stateB[0].d = ctx_shabal_org->stateB[0].d; \
	ctx_shabal->stateB[1].d = ctx_shabal_org->stateB[1].d; \
	ctx_shabal->stateB[2].d = ctx_shabal_org->stateB[2].d; \
	ctx_shabal->stateB[3].d = ctx_shabal_org->stateB[3].d; \
	ctx_shabal->stateB[4].d = ctx_shabal_org->stateB[4].d; \
	ctx_shabal->stateB[5].d = ctx_shabal_org->stateB[5].d; \
	ctx_shabal->stateB[6].d = ctx_shabal_org->stateB[6].d; \
	ctx_shabal->stateB[7].d = ctx_shabal_org->stateB[7].d; \
	ctx_shabal->stateB[8].d = ctx_shabal_org->stateB[8].d; \
	ctx_shabal->stateB[9].d = ctx_shabal_org->stateB[9].d; \
	ctx_shabal->stateB[10].d = ctx_shabal_org->stateB[10].d; \
	ctx_shabal->stateB[11].d = ctx_shabal_org->stateB[11].d; \
	ctx_shabal->stateB[12].d = ctx_shabal_org->stateB[12].d; \
	ctx_shabal->stateB[13].d = ctx_shabal_org->stateB[13].d; \
	ctx_shabal->stateB[14].d = ctx_shabal_org->stateB[14].d; \
	ctx_shabal->stateB[15].d = ctx_shabal_org->stateB[15].d; \
	ctx_shabal->stateC[0].d = ctx_shabal_org->stateC[0].d; \
	ctx_shabal->stateC[1].d = ctx_shabal_org->stateC[1].d; \
	ctx_shabal->stateC[2].d = ctx_shabal_org->stateC[2].d; \
	ctx_shabal->stateC[3].d = ctx_shabal_org->stateC[3].d; \
	ctx_shabal->stateC[4].d = ctx_shabal_org->stateC[4].d; \
	ctx_shabal->stateC[5].d = ctx_shabal_org->stateC[5].d; \
	ctx_shabal->stateC[6].d = ctx_shabal_org->stateC[6].d; \
	ctx_shabal->stateC[7].d = ctx_shabal_org->stateC[7].d; \
	ctx_shabal->stateC[8].d = ctx_shabal_org->stateC[8].d; \
	ctx_shabal->stateC[9].d = ctx_shabal_org->stateC[9].d; \
	ctx_shabal->stateC[10].d = ctx_shabal_org->stateC[10].d; \
	ctx_shabal->stateC[11].d = ctx_shabal_org->stateC[11].d; \
	ctx_shabal->stateC[12].d = ctx_shabal_org->stateC[12].d; \
	ctx_shabal->stateC[13].d = ctx_shabal_org->stateC[13].d; \
	ctx_shabal->stateC[14].d = ctx_shabal_org->stateC[14].d; \
	ctx_shabal->stateC[15].d = ctx_shabal_org->stateC[15].d;
#endif

#ifdef __8WAY__
void axiomhash8way(mshabal8_context *ctx_shabal, mshabal8_context *ctx_shabal_org, void *memspace, void *memspace2, uint32_t *ji,
const uint32_t *input1, uint32_t *result1,
const uint32_t *input2, uint32_t *result2,
const uint32_t *input3, uint32_t *result3,
const uint32_t *input4, uint32_t *result4,
const uint32_t *input5, uint32_t *result5,
const uint32_t *input6, uint32_t *result6,
const uint32_t *input7, uint32_t *result7,
const uint32_t *input8, uint32_t *result8)
{
	hash_t *hash1, *hash2, *hash3, *hash4, *hash5, *hash6, *hash7, *hash8;

	hash1 = memspace2;
	hash2 = &hash1[65536 * 1];
	hash3 = &hash1[65536 * 2];
	hash4 = &hash1[65536 * 3];
	hash5 = &hash1[65536 * 4];
	hash6 = &hash1[65536 * 5];
	hash7 = &hash1[65536 * 6];
	hash8 = &hash1[65536 * 7];

	CTXCOPY;

	mshabal8_80(ctx_shabal, input1, input2, input3, input4, input5, input6, input7, input8, memspace);

	for (size_t i = 1; i < 65536; i++)
	{
		CTXCOPY;

		mshabal8_32(ctx_shabal, memspace, hash1[i], hash2[i], hash3[i], hash4[i], hash5[i], hash6[i], hash7[i], hash8[i], i);
	}

	for (size_t i = 0; i < 65535; i++)
	{
		uint16_t p = i - 1;

		CTXCOPY;

		MOD65535_INDEX_AVX(hash1[p][0], hash2[p][0], hash3[p][0], hash4[p][0], hash5[p][0], hash6[p][0], hash7[p][0], hash8[p][0], ji, i);

		mshabal8_64(ctx_shabal, hash1[p], hash2[p], hash3[p], hash4[p], hash5[p], hash6[p], hash7[p], hash8[p],
			hash1[ji[0]], hash2[ji[1]], hash3[ji[2]], hash4[ji[3]], hash5[ji[4]], hash6[ji[5]], hash7[ji[6]], hash8[ji[7]],
			hash1[i], hash2[i], hash3[i], hash4[i], hash5[i], hash6[i], hash7[i], hash8[i]);
	}

	{
		CTXCOPY;

		MOD65535_INDEX_AVX(hash1[65534][0], hash2[65534][0], hash3[65534][0], hash4[65534][0], hash5[65534][0], hash6[65534][0], hash7[65534][0], hash8[65534][0], ji, 65535);

		mshabal8_64(ctx_shabal, hash1[65534], hash2[65534], hash3[65534], hash4[65534], hash5[65534], hash6[65534], hash7[65534], hash8[65534],
			hash1[ji[0]], hash2[ji[1]], hash3[ji[2]], hash4[ji[3]], hash5[ji[4]], hash6[ji[5]], hash7[ji[6]], hash8[ji[7]],
			result1, result2, result3, result4, result5, result6, result7, result8);
	}
}
#endif

void axiomhash4way(mshabal4_context *ctx_shabal, mshabal4_context *ctx_shabal_org, void *memspace, void *memspace2, uint32_t *ji,
	const uint32_t *input1, uint32_t *result1,
	const uint32_t *input2, uint32_t *result2,
	const uint32_t *input3, uint32_t *result3,
	const uint32_t *input4, uint32_t *result4)
{
	hash_t *hash1, *hash2, *hash3, *hash4;

	hash1 = memspace2;
	hash2 = &hash1[65536 * 1];
	hash3 = &hash1[65536 * 2];
	hash4 = &hash1[65536 * 3];

	CTXCOPY;

	mshabal4_80(ctx_shabal, input1, input2, input3, input4, memspace);

	for (size_t i = 1; i < 65536; i++)
	{
		CTXCOPY;

		mshabal4_32(ctx_shabal, memspace, hash1[i], hash2[i], hash3[i], hash4[i], i);
	}

	for (size_t i = 0; i < 65535; i++)
	{
		uint16_t p = i - 1;

		CTXCOPY;

		MOD65535_INDEX_SSE(hash1[p][0], hash2[p][0], hash3[p][0], hash4[p][0], ji, i);

		mshabal4_64(ctx_shabal, hash1[p], hash2[p], hash3[p], hash4[p],
			hash1[ji[0]], hash2[ji[1]], hash3[ji[2]], hash4[ji[3]],
			hash1[i], hash2[i], hash3[i], hash4[i]);
	}

	{
		CTXCOPY;

		MOD65535_INDEX_SSE(hash1[65534][0], hash2[65534][0], hash3[65534][0], hash4[65534][0], ji, 65535);

		mshabal4_64(ctx_shabal, hash1[65534], hash2[65534], hash3[65534], hash4[65534],
			hash1[ji[0]], hash2[ji[1]], hash3[ji[2]], hash4[ji[3]],
			result1, result2, result3, result4);
	}
}

int scanhash_axiom(int thr_id, uint32_t *pdata, const uint32_t *ptarget,
	uint32_t max_nonce, uint64_t *hashes_done, uint32_t *nonces, int *nonces_len)
{
	uint32_t *hash64_1, *hash64_2, *hash64_3, *hash64_4
#ifdef __8WAY__
		, *hash64_5, *hash64_6, *hash64_7, *hash64_8
#endif
		;

	uint32_t *endiandata_1, *endiandata_2, *endiandata_3, *endiandata_4
#ifdef __8WAY__
		, *endiandata_5, *endiandata_6, *endiandata_7, *endiandata_8
#endif
		;

#ifdef __8WAY__
	mshabal8_context 
#else
	mshabal4_context
#endif
		*ctx_shabal, *ctx_shabal_org;

	void *memspace, *memspace2;
	uint32_t *ji;

	const uint32_t Htarg = ptarget[7];
	const uint32_t first_nonce = pdata[19];

	uint32_t n = first_nonce;

	*nonces_len = 0;

#ifdef WIN32
#define ALIGNED_ALLOC(alignment, size) \
    _aligned_malloc(size, alignment);
#define ALIGNED_FREE(ptr) \
    _aligned_free(ptr);
#else
#define ALIGNED_ALLOC(alignment, size) \
    aligned_alloc(alignment, size);
#define ALIGNED_FREE(ptr) \
    free(ptr);
#endif

	hash64_1 = ALIGNED_ALLOC(ALIGNED_TO, sizeof(uint32_t) * 8);
	hash64_2 = ALIGNED_ALLOC(ALIGNED_TO, sizeof(uint32_t) * 8);
	hash64_3 = ALIGNED_ALLOC(ALIGNED_TO, sizeof(uint32_t) * 8);
	hash64_4 = ALIGNED_ALLOC(ALIGNED_TO, sizeof(uint32_t) * 8);

	endiandata_1 = ALIGNED_ALLOC(ALIGNED_TO, sizeof(uint32_t) * 20);
	endiandata_2 = ALIGNED_ALLOC(ALIGNED_TO, sizeof(uint32_t) * 20);
	endiandata_3 = ALIGNED_ALLOC(ALIGNED_TO, sizeof(uint32_t) * 20);
	endiandata_4 = ALIGNED_ALLOC(ALIGNED_TO, sizeof(uint32_t) * 20);

#ifdef __8WAY__
	ctx_shabal = ALIGNED_ALLOC(ALIGNED_TO, sizeof(mshabal8_context));
	ctx_shabal_org = ALIGNED_ALLOC(ALIGNED_TO, sizeof(mshabal8_context));

	for (size_t j = 0; j < 12; j++) {
		for (size_t k = 0; k < 8; k++) {
			ctx_shabal_org->stateA[j].w[k] = mshabal8_init_state[j * 8 + k];
		}
	}
	for (size_t j = 0; j < 16; j++) {
		for (size_t k = 0; k < 8; k++) {
			ctx_shabal_org->stateB[j].w[k] = mshabal8_init_state[(j + 12) * 8 + k];
		}
	}
	for (size_t j = 0; j < 16; j++) {
		for (size_t k = 0; k < 8; k++) {
			ctx_shabal_org->stateC[j].w[k] = mshabal8_init_state[(j + 12 + 16) * 8 + k];
		}
	}

	memspace = ALIGNED_ALLOC(ALIGNED_TO, 65536 * sizeof(uint32_t) * 8 * 8);
	memspace2 = ALIGNED_ALLOC(ALIGNED_TO, 65536 * sizeof(uint32_t) * 8 * 8);

	hash64_5 = ALIGNED_ALLOC(ALIGNED_TO, sizeof(uint32_t) * 8);
	hash64_6 = ALIGNED_ALLOC(ALIGNED_TO, sizeof(uint32_t) * 8);
	hash64_7 = ALIGNED_ALLOC(ALIGNED_TO, sizeof(uint32_t) * 8);
	hash64_8 = ALIGNED_ALLOC(ALIGNED_TO, sizeof(uint32_t) * 8);

	endiandata_5 = ALIGNED_ALLOC(ALIGNED_TO, sizeof(uint32_t) * 20);
	endiandata_6 = ALIGNED_ALLOC(ALIGNED_TO, sizeof(uint32_t) * 20);
	endiandata_7 = ALIGNED_ALLOC(ALIGNED_TO, sizeof(uint32_t) * 20);
	endiandata_8 = ALIGNED_ALLOC(ALIGNED_TO, sizeof(uint32_t) * 20);

	ji = ALIGNED_ALLOC(ALIGNED_TO, sizeof(uint32_t) * 8);
#else
	ctx_shabal = ALIGNED_ALLOC(ALIGNED_TO, sizeof(mshabal4_context));
	ctx_shabal_org = ALIGNED_ALLOC(ALIGNED_TO, sizeof(mshabal4_context));

	for (size_t j = 0; j < 12; j++) {
		for (size_t k = 0; k < 4; k++) {
			ctx_shabal_org->stateA[j].w[k] = mshabal4_init_state[j * 4 + k];
		}
	}
	for (size_t j = 0; j < 16; j++) {
		for (size_t k = 0; k < 4; k++) {
			ctx_shabal_org->stateB[j].w[k] = mshabal4_init_state[(j + 12) * 4 + k];
		}
	}
	for (size_t j = 0; j < 16; j++) {
		for (size_t k = 0; k < 4; k++) {
			ctx_shabal_org->stateC[j].w[k] = mshabal4_init_state[(j + 12 + 16) * 4 + k];
		}
	}

	memspace = ALIGNED_ALLOC(ALIGNED_TO, 65536 * sizeof(uint32_t) * 8 * 4);
	memspace2 = ALIGNED_ALLOC(ALIGNED_TO, 65536 * sizeof(uint32_t) * 8 * 4);

	ji = ALIGNED_ALLOC(ALIGNED_TO, sizeof(uint32_t) * 4);
#endif

	for (size_t i = 0; i < 19; i++) {
		be32enc(&endiandata_1[i], pdata[i]);
	}

	for (size_t i = 0; i < 20; i++) {
		endiandata_2[i] = endiandata_1[i];
		endiandata_3[i] = endiandata_1[i];
		endiandata_4[i] = endiandata_1[i];
#ifdef __8WAY__
		endiandata_5[i] = endiandata_1[i];
		endiandata_6[i] = endiandata_1[i];
		endiandata_7[i] = endiandata_1[i];
		endiandata_8[i] = endiandata_1[i];
#endif
	}

	do {
		be32enc(&endiandata_1[19], n);
		be32enc(&endiandata_2[19], n + 1);
		be32enc(&endiandata_3[19], n + 2);
		be32enc(&endiandata_4[19], n + 3);
#ifdef __8WAY__
		be32enc(&endiandata_5[19], n + 4);
		be32enc(&endiandata_6[19], n + 5);
		be32enc(&endiandata_7[19], n + 6);
		be32enc(&endiandata_8[19], n + 7);
#endif

#ifdef __8WAY__
		axiomhash8way(ctx_shabal, ctx_shabal_org, memspace, memspace2, ji, endiandata_1, hash64_1, endiandata_2, hash64_2, endiandata_3, hash64_3, endiandata_4, hash64_4,
			endiandata_5, hash64_5, endiandata_6, hash64_6, endiandata_7, hash64_7, endiandata_8, hash64_8);
#else
		axiomhash4way(ctx_shabal, ctx_shabal_org, memspace, memspace2, ji, endiandata_1, hash64_1, endiandata_2, hash64_2, endiandata_3, hash64_3, endiandata_4, hash64_4);
#endif

#ifdef __8WAY__
		(*hashes_done) += 8;
#else
		(*hashes_done) += 4;
#endif

		if (hash64_1[7] < Htarg && fulltest(hash64_1, ptarget)) {
			nonces[(*nonces_len)++] = n;
		}
		if (hash64_2[7] < Htarg && fulltest(hash64_2, ptarget)) {
			nonces[(*nonces_len)++] = n + 1;
		}
		if (hash64_3[7] < Htarg && fulltest(hash64_3, ptarget)) {
			nonces[(*nonces_len)++] = n + 2;
		}
		if (hash64_4[7] < Htarg && fulltest(hash64_4, ptarget)) {
			nonces[(*nonces_len)++] = n + 3;
		}
#ifdef __8WAY__
		if (hash64_5[7] < Htarg && fulltest(hash64_5, ptarget)) {
			nonces[(*nonces_len)++] = n + 4;
		}
		if (hash64_6[7] < Htarg && fulltest(hash64_6, ptarget)) {
			nonces[(*nonces_len)++] = n + 5;
		}
		if (hash64_7[7] < Htarg && fulltest(hash64_7, ptarget)) {
			nonces[(*nonces_len)++] = n + 6;
		}
		if (hash64_8[7] < Htarg && fulltest(hash64_8, ptarget)) {
			nonces[(*nonces_len)++] = n + 7;
		}
#endif

#ifdef __8WAY__
		n += 8;
#else
		n += 4;
#endif

		if ((*nonces_len) > 0) {
			pdata[19] = n;
#ifdef __8WAY__
			FREE_MEM_AVX;
#else
			FREE_MEM_SSE
#endif
			return true;
		}

	} while (unlikely(n < max_nonce) && !work_restart[thr_id].restart);

	pdata[19] = n;
#ifdef __8WAY__
	FREE_MEM_AVX;
#else
	FREE_MEM_SSE
#endif
	return 0;
}