#include <memory.h>

#include "sha3/sph_blake.h"
#include "sha3/sph_groestl.h"
#include "sha3/sph_skein.h"
#include "sha3/sph_keccak.h"

#include "lyra2/Lyra2.h"

#include "miner.h"

/*void lyra2_hash(void *state, const void *input)
{
	sph_blake256_context     ctx_blake;
	sph_keccak256_context    ctx_keccak;
	sph_skein256_context     ctx_skein;
	sph_groestl256_context   ctx_groestl;

	uint32_t hashA[8], hashB[8];

	sph_blake256_init(&ctx_blake);
	sph_blake256(&ctx_blake, input, 80);
	sph_blake256_close(&ctx_blake, hashA);

	sph_keccak256_init(&ctx_keccak);
	sph_keccak256(&ctx_keccak, hashA, 32);
	sph_keccak256_close(&ctx_keccak, hashB);

	LYRA2(hashA, 32, hashB, 32, hashB, 32, 1, 8, 8);

	sph_skein256_init(&ctx_skein);
	sph_skein256(&ctx_skein, hashA, 32);
	sph_skein256_close(&ctx_skein, hashB);

	sph_groestl256_init(&ctx_groestl);
	sph_groestl256(&ctx_groestl, hashB, 32);
	sph_groestl256_close(&ctx_groestl, hashA);

	memcpy(state, hashA, 32);
}*/


#ifdef __AVX2__
//#define __AES_NI
#endif

#ifdef __AES_NI
#include "algo/aes_ni/hash-groestl256.h"
#endif

typedef struct {
	sph_blake256_context     blake;
	sph_keccak256_context    keccak;
	sph_skein256_context     skein;
#ifdef __AES_NI
	hashState_groestl256     groestl;
#else
	sph_groestl256_context   groestl;
#endif
} lyra2re_ctx_holder;

lyra2re_ctx_holder lyra2re_ctx;

void init_lyra2re_ctx()
{
	sph_blake256_init(&lyra2re_ctx.blake);
	sph_keccak256_init(&lyra2re_ctx.keccak);
	sph_skein256_init(&lyra2re_ctx.skein);
#ifdef __AES_NI
	init_groestl256(&lyra2re_ctx.groestl);
#else
	sph_groestl256_init(&lyra2re_ctx.groestl);
#endif
}

void lyra2_hash(void *state, const void *input)
{
	lyra2re_ctx_holder ctx;
	memcpy(&ctx, &lyra2re_ctx, sizeof(lyra2re_ctx));

	uint32_t hashA[8], hashB[8];

	sph_blake256(&ctx.blake, input, 80);
	sph_blake256_close(&ctx.blake, hashA);

	sph_keccak256(&ctx.keccak, hashA, 32);
	sph_keccak256_close(&ctx.keccak, hashB);

	LYRA2(hashA, 32, hashB, 32, hashB, 32, 1, 8, 8);

	sph_skein256(&ctx.skein, hashA, 32);
	sph_skein256_close(&ctx.skein, hashB);

#ifdef __AES_NI
	update_groestl256(&ctx.groestl, hashB, 256);
	final_groestl256(&ctx.groestl, hashA);
#else
	sph_groestl256(&ctx.groestl, hashB, 32);
	sph_groestl256_close(&ctx.groestl, hashA);
#endif

	memcpy(state, hashA, 32);
}

int scanhash_lyra2(int thr_id, uint32_t *pdata, const uint32_t *ptarget,
	uint32_t max_nonce,	uint64_t *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[20];
	const uint32_t first_nonce = pdata[19];
	uint32_t nonce = first_nonce;

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x0000ff;

	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], ((uint32_t*)pdata)[k]);

	do {
		const uint32_t Htarg = ptarget[7];
		uint32_t hash[8];
		be32enc(&endiandata[19], nonce);
		lyra2_hash(hash, endiandata);

		if (hash[7] <= Htarg && fulltest(hash, ptarget)) {
			pdata[19] = nonce;
			*hashes_done = pdata[19] - first_nonce;
			return 1;
		}
		nonce++;

	} while (nonce < max_nonce && !work_restart[thr_id].restart);

	pdata[19] = nonce;
	*hashes_done = pdata[19] - first_nonce + 1;
	return 0;
}
