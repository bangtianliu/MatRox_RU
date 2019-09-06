//
// Created by Bangtian Liu on 9/1/19.
//

#ifndef PROJECT_H2GEMM_H
#define PROJECT_H2GEMM_H
#include <iostream>
#include <math.h>
#include <float.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <cholUtils.h>
#ifndef HALIDE_ATTRIBUTE_ALIGN
#ifdef _MSC_VER
#define HALIDE_ATTRIBUTE_ALIGN(x) __declspec(align(x))
#else
#define HALIDE_ATTRIBUTE_ALIGN(x) __attribute__((aligned(x)))
#endif
#endif
#ifndef BUFFER_T_DEFINED
#define BUFFER_T_DEFINED
#include <stdbool.h>
#include <stdint.h>
typedef struct buffer_t {
	uint64_t dev;
	uint8_t* host;
	int32_t extent[4];
	int32_t stride[4];
	int32_t min[4];
	int32_t elem_size;
	HALIDE_ATTRIBUTE_ALIGN(1) bool host_dirty;
	HALIDE_ATTRIBUTE_ALIGN(1) bool dev_dirty;
	HALIDE_ATTRIBUTE_ALIGN(1) uint8_t _padding[10 - sizeof(void *)];
} buffer_t;
#endif
#define __user_context_ NULL
struct halide_filter_metadata_t;
extern "C" {
void *sympiler_malloc(void *ctx, size_t s){return(malloc(s));}
void sympiler_free(void *ctx, void *ptr){free(ptr);};
}

#ifdef _WIN32
float roundf(float);
double round(double);
#else
inline float asinh_f32(float x) {return asinhf(x);}
inline float acosh_f32(float x) {return acoshf(x);}
inline float atanh_f32(float x) {return atanhf(x);}
inline double asinh_f64(double x) {return asinh(x);}
inline double acosh_f64(double x) {return acosh(x);}
inline double atanh_f64(double x) {return atanh(x);}
#endif
inline float sqrt_f32(float x) {return sqrtf(x);}
inline float sin_f32(float x) {return sinf(x);}
inline float asin_f32(float x) {return asinf(x);}
inline float cos_f32(float x) {return cosf(x);}
inline float acos_f32(float x) {return acosf(x);}
inline float tan_f32(float x) {return tanf(x);}
inline float atan_f32(float x) {return atanf(x);}
inline float sinh_f32(float x) {return sinhf(x);}
inline float cosh_f32(float x) {return coshf(x);}
inline float tanh_f32(float x) {return tanhf(x);}
inline float hypot_f32(float x, float y) {return hypotf(x, y);}
inline float exp_f32(float x) {return expf(x);}
inline float log_f32(float x) {return logf(x);}
inline float pow_f32(float x, float y) {return powf(x, y);}
inline float floor_f32(float x) {return floorf(x);}
inline float ceil_f32(float x) {return ceilf(x);}
inline float round_f32(float x) {return roundf(x);}

inline double sqrt_f64(double x) {return sqrt(x);}
inline double sin_f64(double x) {return sin(x);}
inline double asin_f64(double x) {return asin(x);}
inline double cos_f64(double x) {return cos(x);}
inline double acos_f64(double x) {return acos(x);}
inline double tan_f64(double x) {return tan(x);}
inline double atan_f64(double x) {return atan(x);}
inline double sinh_f64(double x) {return sinh(x);}
inline double cosh_f64(double x) {return cosh(x);}
inline double tanh_f64(double x) {return tanh(x);}
inline double hypot_f64(double x, double y) {return hypot(x, y);}
inline double exp_f64(double x) {return exp(x);}
inline double log_f64(double x) {return log(x);}
inline double pow_f64(double x, double y) {return pow(x, y);}
inline double floor_f64(double x) {return floor(x);}
inline double ceil_f64(double x) {return ceil(x);}
inline double round_f64(double x) {return round(x);}

inline float nan_f32() {return NAN;}
inline float neg_inf_f32() {return -INFINITY;}
inline float inf_f32() {return INFINITY;}
inline bool is_nan_f32(float x) {return x != x;}
inline bool is_nan_f64(double x) {return x != x;}
//inline float float_from_bits(uint32_t bits) {
//	union {
//		uint32_t as_uint;
//		float as_float;
//	} u;
//	u.as_uint = bits;
//	return u.as_float;
//}
inline int64_t make_int64(int32_t hi, int32_t lo) {
	return (((int64_t)hi) << 32) | (uint32_t)lo;
}
inline double make_float64(int32_t i0, int32_t i1) {
	union {
		int32_t as_int32[2];
		double as_double;
	} u;
	u.as_int32[0] = i0;
	u.as_int32[1] = i1;
	return u.as_double;
}

template<typename A, typename B> A reinterpret(B b) {A a; memcpy(&a, &b, sizeof(a)); return a;}

double one [2]={1.0,0.}, zero [2]={0.,0.};
int sw = false, lb1 = 0, ub1 = 0;
double *cur; int info=0;
#ifdef __cplusplus
extern "C" {
#endif

int32_t seqGEMM(double *D,
				double *B, double *VT, uint64_t *Dptr, uint64_t *Bptr, int32_t *VTptr, int32_t *lchildren,
				int32_t *rchildren, int32_t *levelset, int32_t *idx, double *mrhs,
				double *apres, int32_t nrhs, int32_t *Ddim, int32_t *wptr, int32_t *uptr, double *wskel,
				int32_t *wskeloffset, double *uskel, int32_t *uskeloffset, int32_t *lm,
				int32_t *slen, int32_t *nidx, int32_t *nidy, int32_t ncount, int32_t *fidx, int32_t *fidy,
				int32_t fcount, double *utmp, uint64_t *utmpoffset, double *ftmp,
				uint64_t *ftmpoffset, int depth) {
	for (int i = 0; i < ncount; i++) {
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
					Ddim[nidx[i]], nrhs, Ddim[nidy[i]],
					float_from_bits(1065353216 /* 1 */), &D[Dptr[i]],
					Ddim[nidx[i]], &mrhs[wptr[nidy[i]]], Ddim[nidy[i]], float_from_bits(1065353216/* 0 */),
					&apres[uptr[nidx[i]]], Ddim[nidx[i]]);
	} // for i
//	for (int i = 0; i < ncount; i++) {
//		for (int j = 0; j < nrhs; j++) {
//			for (int k = 0; k < Ddim[nidx[i]]; k++) {
//				int32_t _0 = j * Ddim[nidx[i]];
//				int32_t _1 = _0 + k;
//				int32_t _2 = uptr[nidx[i]] + _1;
//				int32_t _3 = utmpoffset[i] + _1;
//				double _4 = apres[_2] + utmp[_3];
//				apres[_2] = _4;
//			} // for k
//		} // for j
//	} // for i

	for (int i = depth - 1; i > -1; i--) {
		int32_t _0 = i + 1;
//#pragma omp parallel for
		for (int j = levelset[i]; j < levelset[_0]; j++) {
//            int32_t _1 = k + 1;
//            for (int j = wpart[k]; j < wpart[_1]; j++)
//            {
//        printf("idx=%d\n",idx[j]);
			int32_t _2 = (int32_t) (4294967295);
			bool _3 = lchildren[idx[j]] == _2;
			if (_3) {
				cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
							slen[idx[j]], nrhs, Ddim[lm[idx[j]]],
							float_from_bits(1065353216 /* 1 */), &VT[VTptr[idx[j]]],
							slen[idx[j]], &mrhs[wptr[lm[idx[j]]]], Ddim[lm[idx[j]]], float_from_bits(0 /* 0 */),
							&wskel[wskeloffset[idx[j]]], slen[idx[j]]);
			} // if _3
			else {
				cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
							slen[idx[j]], nrhs, slen[lchildren[idx[j]]],
							float_from_bits(1065353216 /* 1 */), &VT[VTptr[idx[j]]],
							slen[idx[j]], &wskel[wskeloffset[lchildren[idx[j]]]], slen[lchildren[idx[j]]],
							float_from_bits(0 /* 0 */),
							&wskel[wskeloffset[idx[j]]], slen[idx[j]]);
				int32_t _4 = slen[idx[j]] * slen[lchildren[idx[j]]];
				int32_t _5 = _4 + VTptr[idx[j]];
				cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
							slen[idx[j]], nrhs, slen[rchildren[idx[j]]],
							float_from_bits(1065353216 /* 1 */), &VT[_5],
							slen[idx[j]], &wskel[wskeloffset[rchildren[idx[j]]]], slen[rchildren[idx[j]]],
							float_from_bits(1065353216 /* 1 */),
							&wskel[wskeloffset[idx[j]]], slen[idx[j]]);
			} // if _3 else
//            } // for j
		} // for k
	} // for i

	for (int i = 0; i < fcount; i++) {
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
					slen[fidx[i]], nrhs, slen[fidy[i]],
					float_from_bits(1065353216 /* 1 */), &B[Bptr[i]],
					slen[fidx[i]], &wskel[wskeloffset[fidy[i]]], slen[fidy[i]], float_from_bits(1065353216 /* 0 */),
					&uskel[uskeloffset[fidx[i]]], slen[fidx[i]]);
	} // for i
//	for (int i = 0; i < fcount; i++) {
//		for (int j = 0; j < nrhs; j++) {
//			for (int k = 0; k < slen[fidx[i]]; k++) {
//				int32_t _12 = j * slen[fidx[i]];
//				int32_t _13 = _12 + k;
//				int32_t _14 = uskeloffset[fidx[i]] + _13;
//				int32_t _15 = ftmpoffset[i] + _13;
//				double _16 = uskel[_14] + ftmp[_15];
//				uskel[_14] = _16;
//			} // for k
//		} // for j
//	} // for i

	for (int i = 0; i < depth; i++) {
		int32_t _17 = i + 1;
//#pragma omp parallel for
		for (int j = levelset[i]; j < levelset[_17]; j++) {
//            int32_t _18 = wpart[k] - 1;
//            int32_t _19 = k + 1;
//            int32_t _20 = wpart[_19] - 1;
//            for (int j = _20; j > _18; j--)
//            {
			int32_t _21 = (int32_t) (4294967295);
			bool _22 = lchildren[idx[j]] == _21;
			if (_22) {
				cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
							Ddim[lm[idx[j]]], nrhs, slen[idx[j]],
							float_from_bits(1065353216 /* 1 */), &VT[VTptr[idx[j]]],
							slen[idx[j]], &uskel[uskeloffset[idx[j]]], slen[idx[j]],
							float_from_bits(1065353216 /* 1 */),
							&apres[uptr[lm[idx[j]]]], Ddim[lm[idx[j]]]);
			} // if _22
			else {
				cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
							slen[lchildren[idx[j]]], nrhs, slen[idx[j]],
							float_from_bits(1065353216 /* 1 */), &VT[VTptr[idx[j]]],
							slen[idx[j]], &uskel[uskeloffset[idx[j]]], slen[idx[j]],
							float_from_bits(1065353216 /* 1 */),
							&uskel[uskeloffset[lchildren[idx[j]]]], slen[lchildren[idx[j]]]);
				int32_t _23 = slen[idx[j]] * slen[lchildren[idx[j]]];
				int32_t _24 = _23 + VTptr[idx[j]];
				cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
							slen[rchildren[idx[j]]], nrhs, slen[idx[j]],
							float_from_bits(1065353216 /* 1 */), &VT[_24],
							slen[idx[j]], &uskel[uskeloffset[idx[j]]], slen[idx[j]],
							float_from_bits(1065353216 /* 1 */),
							&uskel[uskeloffset[rchildren[idx[j]]]], slen[rchildren[idx[j]]]);
			} // if _22 else
//            } // for j
		} // for k
	} // for i

	return 0;
}


int32_t blockGEMM(double *D,
				  double *B, double *VT, uint64_t *Dptr, uint64_t *Bptr, int32_t *VTptr, int32_t *lchildren,
				  int32_t *rchildren, int32_t *levelset, int32_t *idx, double *mrhs,
				  double *apres, int32_t nrhs, int32_t *Ddim, int32_t *wptr, int32_t *uptr, double *wskel,
				  int32_t *wskeloffset, double *uskel, int32_t *uskeloffset, int32_t *lm,
				  int32_t *slen, int32_t *nblockSet, int32_t *nblocks, int32_t *npairx, int32_t *npairy,
				  int32_t *fblockSet, int32_t *fblocks, int32_t *fpairx, int32_t *fpairy, int32_t ncount,
				  int32_t fcount,
				  int32_t depth) {
	mkl_set_dynamic(false);
#pragma omp parallel for
	for (int i = 0; i < ncount; i++) {
		int32_t _0 = i + 1;
		for (int j = nblockSet[i]; j < nblockSet[_0]; j++) {
			int32_t _1 = j + 1;
			for (int k = nblocks[j]; k < nblocks[_1]; k++) {
				cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
							Ddim[npairx[k]], nrhs, Ddim[npairy[k]],
							float_from_bits(1065353216 /* 1 */), &D[Dptr[k]],
							Ddim[npairx[k]], &mrhs[wptr[npairy[k]]], Ddim[npairy[k]],
							float_from_bits(1065353216 /* 1 */),
							&apres[uptr[npairx[k]]], Ddim[npairx[k]]);
			} // for k
		} // for j
	}

	for (int i = depth - 1; i > -1; i--) {
		int32_t _0 = i + 1;
//#pragma omp parallel for
		for (int j = levelset[i]; j < levelset[_0]; j++) {
//            int32_t _1 = k + 1;
//            for (int j = wpart[k]; j < wpart[_1]; j++)
//            {
//        printf("idx=%d\n",idx[j]);
			int32_t _2 = (int32_t) (4294967295);
			bool _3 = lchildren[idx[j]] == _2;
			if (_3) {
				cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
							slen[idx[j]], nrhs, Ddim[lm[idx[j]]],
							float_from_bits(1065353216 /* 1 */), &VT[VTptr[idx[j]]],
							slen[idx[j]], &mrhs[wptr[lm[idx[j]]]], Ddim[lm[idx[j]]], float_from_bits(0 /* 0 */),
							&wskel[wskeloffset[idx[j]]], slen[idx[j]]);
			} // if _3
			else {
				cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
							slen[idx[j]], nrhs, slen[lchildren[idx[j]]],
							float_from_bits(1065353216 /* 1 */), &VT[VTptr[idx[j]]],
							slen[idx[j]], &wskel[wskeloffset[lchildren[idx[j]]]], slen[lchildren[idx[j]]],
							float_from_bits(0 /* 0 */),
							&wskel[wskeloffset[idx[j]]], slen[idx[j]]);
				int32_t _4 = slen[idx[j]] * slen[lchildren[idx[j]]];
				int32_t _5 = _4 + VTptr[idx[j]];
				cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
							slen[idx[j]], nrhs, slen[rchildren[idx[j]]],
							float_from_bits(1065353216 /* 1 */), &VT[_5],
							slen[idx[j]], &wskel[wskeloffset[rchildren[idx[j]]]], slen[rchildren[idx[j]]],
							float_from_bits(1065353216 /* 1 */),
							&wskel[wskeloffset[idx[j]]], slen[idx[j]]);
			} // if _3 else
//            } // for j
		} // for k
	} // for i


#pragma omp parallel for
	for (int i = 0; i < fcount; i++) {
		int32_t _8 = i + 1;
		for (int j = fblockSet[i]; j < fblockSet[_8]; j++) {
			int32_t _9 = j + 1;
			for (int k = fblocks[j]; k < fblocks[_9]; k++) {
				cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
							slen[fpairx[k]], nrhs, slen[fpairy[k]],
							float_from_bits(1065353216 /* 1 */), &B[Bptr[k]],
							slen[fpairx[k]], &wskel[wskeloffset[fpairy[k]]], slen[fpairy[k]],
							float_from_bits(1065353216 /* 1 */),
							&uskel[uskeloffset[fpairx[k]]], slen[fpairx[k]]);
			} // for k
		} // for j
	} // for i


	for (int i = 0; i < depth; i++) {
		int32_t _17 = i + 1;
//#pragma omp parallel for
		for (int j = levelset[i]; j < levelset[_17]; j++) {
//            int32_t _18 = wpart[k] - 1;
//            int32_t _19 = k + 1;
//            int32_t _20 = wpart[_19] - 1;
//            for (int j = _20; j > _18; j--)
//            {
			int32_t _21 = (int32_t) (4294967295);
			bool _22 = lchildren[idx[j]] == _21;
			if (_22) {
				cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
							Ddim[lm[idx[j]]], nrhs, slen[idx[j]],
							float_from_bits(1065353216 /* 1 */), &VT[VTptr[idx[j]]],
							slen[idx[j]], &uskel[uskeloffset[idx[j]]], slen[idx[j]],
							float_from_bits(1065353216 /* 1 */),
							&apres[uptr[lm[idx[j]]]], Ddim[lm[idx[j]]]);
			} // if _22
			else {
				cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
							slen[lchildren[idx[j]]], nrhs, slen[idx[j]],
							float_from_bits(1065353216 /* 1 */), &VT[VTptr[idx[j]]],
							slen[idx[j]], &uskel[uskeloffset[idx[j]]], slen[idx[j]],
							float_from_bits(1065353216 /* 1 */),
							&uskel[uskeloffset[lchildren[idx[j]]]], slen[lchildren[idx[j]]]);
				int32_t _23 = slen[idx[j]] * slen[lchildren[idx[j]]];
				int32_t _24 = _23 + VTptr[idx[j]];
				cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
							slen[rchildren[idx[j]]], nrhs, slen[idx[j]],
							float_from_bits(1065353216 /* 1 */), &VT[_24],
							slen[idx[j]], &uskel[uskeloffset[idx[j]]], slen[idx[j]],
							float_from_bits(1065353216 /* 1 */),
							&uskel[uskeloffset[rchildren[idx[j]]]], slen[rchildren[idx[j]]]);
			} // if _22 else
//            } // for j
		} // for k
	} // for i


	return 0;
}


int32_t CBHGEMM(
		double *D,
		double *B, double *VT, uint64_t *Dptr, uint64_t *Bptr, int32_t *VTptr, int32_t *lchildren,
		int32_t *rchildren, int32_t *levelset, int32_t *idx, double *mrhs,
		double *apres, int32_t nrhs, int32_t *Ddim, int32_t *wptr, int32_t *uptr, double *wskel,
		int32_t *wskeloffset, double *uskel, int32_t *uskeloffset, int32_t *lm,
		int32_t *slen, int32_t *nblockSet, int32_t *nblocks, int32_t *npairx, int32_t *npairy,
		int32_t *fblockSet, int32_t *fblocks, int32_t *fpairx, int32_t *fpairy, int32_t ncount,
		int32_t fcount, int32_t *wpart, int32_t *clevelset, int cdepth)
{
	mkl_set_dynamic(false);
#pragma omp parallel for
	for (int i = 0; i < ncount; i++) {
		int32_t _0 = i + 1;
		for (int j = nblockSet[i]; j < nblockSet[_0]; j++) {
			int32_t _1 = j + 1;
			for (int k = nblocks[j]; k < nblocks[_1]; k++) {
				cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
							Ddim[npairx[k]], nrhs, Ddim[npairy[k]],
							float_from_bits(1065353216 /* 1 */), &D[Dptr[k]],
							Ddim[npairx[k]], &mrhs[wptr[npairy[k]]], Ddim[npairy[k]],
							float_from_bits(1065353216 /* 1 */),
							&apres[uptr[npairx[k]]], Ddim[npairx[k]]);
			} // for k
		} // for j
	}

	for (int i = 0; i < cdepth; i++)
	{
		int32_t _0 = i + 1;
#pragma omp parallel for
		for (int k = clevelset[i]; k < clevelset[_0]; k++)
		{
			int32_t _1 = k + 1;
			for (int j = wpart[k]; j < wpart[_1]; j++)
			{
				int32_t _2 = (int32_t)(4294967295);
				bool _3 = lchildren[idx[j]] == _2;
				if (_3)
				{
					cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
								slen[idx[j]],nrhs,Ddim[lm[idx[j]]],
								float_from_bits(1065353216 /* 1 */), &VT[VTptr[idx[j]]],
								slen[idx[j]], &mrhs[wptr[lm[idx[j]]]], Ddim[lm[idx[j]]], float_from_bits(0 /* 0 */),
								&wskel[wskeloffset[idx[j]]], slen[idx[j]]);
				} // if _3
				else
				{
					cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
								slen[idx[j]],nrhs,slen[lchildren[idx[j]]],
								float_from_bits(1065353216 /* 1 */), &VT[VTptr[idx[j]]],
								slen[idx[j]], &wskel[wskeloffset[lchildren[idx[j]]]], slen[lchildren[idx[j]]], float_from_bits(0 /* 0 */),
								&wskel[wskeloffset[idx[j]]],  slen[idx[j]]);
					int32_t _4 = slen[idx[j]] * slen[lchildren[idx[j]]];
					int32_t _5 = _4 + VTptr[idx[j]];
					cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
								slen[idx[j]],nrhs,slen[rchildren[idx[j]]],
								float_from_bits(1065353216 /* 1 */), &VT[_5],
								slen[idx[j]], &wskel[wskeloffset[rchildren[idx[j]]]], slen[rchildren[idx[j]]], float_from_bits(1065353216 /* 1 */),
								&wskel[wskeloffset[idx[j]]], slen[idx[j]]);
				} // if _3 else
			} // for j
		} // for k
	} // for i


#pragma omp parallel for
	for (int i = 0; i < fcount; i++) {
		int32_t _8 = i + 1;
		for (int j = fblockSet[i]; j < fblockSet[_8]; j++) {
			int32_t _9 = j + 1;
			for (int k = fblocks[j]; k < fblocks[_9]; k++) {
				cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
							slen[fpairx[k]], nrhs, slen[fpairy[k]],
							float_from_bits(1065353216 /* 1 */), &B[Bptr[k]],
							slen[fpairx[k]], &wskel[wskeloffset[fpairy[k]]], slen[fpairy[k]],
							float_from_bits(1065353216 /* 1 */),
							&uskel[uskeloffset[fpairx[k]]], slen[fpairx[k]]);
			} // for k
		} // for j
	} // for i


	for (int i = cdepth-1; i > -1; i--)
	{
		int32_t _17 = i + 1;
#pragma omp parallel for
		for (int k = clevelset[i]; k < clevelset[_17]; k++)
		{
			int32_t _18 = wpart[k] - 1;
			int32_t _19 = k + 1;
			int32_t _20 = wpart[_19] - 1;
			for (int j = _20; j > _18; j--)
			{
				int32_t _21 = (int32_t)(4294967295);
				bool _22 = lchildren[idx[j]] == _21;
				if (_22)
				{
					cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
								Ddim[lm[idx[j]]],nrhs,slen[idx[j]],
								float_from_bits(1065353216 /* 1 */), &VT[VTptr[idx[j]]],
								slen[idx[j]], &uskel[uskeloffset[idx[j]]], slen[idx[j]], float_from_bits(1065353216 /* 1 */),
								&apres[uptr[lm[idx[j]]]], Ddim[lm[idx[j]]]);
				} // if _22
				else
				{
					cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
								slen[lchildren[idx[j]]],nrhs,slen[idx[j]],
								float_from_bits(1065353216 /* 1 */), &VT[VTptr[idx[j]]],
								slen[idx[j]], &uskel[uskeloffset[idx[j]]], slen[idx[j]], float_from_bits(1065353216 /* 1 */),
								&uskel[uskeloffset[lchildren[idx[j]]]], slen[lchildren[idx[j]]]);
					int32_t _23 = slen[idx[j]] * slen[lchildren[idx[j]]];
					int32_t _24 = _23 + VTptr[idx[j]];
					cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
								slen[rchildren[idx[j]]],nrhs,slen[idx[j]],
								float_from_bits(1065353216 /* 1 */), &VT[_24],
								slen[idx[j]], &uskel[uskeloffset[idx[j]]], slen[idx[j]], float_from_bits(1065353216 /* 1 */),
								&uskel[uskeloffset[rchildren[idx[j]]]],  slen[rchildren[idx[j]]]);
				} // if _22 else
			} // for j
		} // for k
	} // for i

	return 0;
}

int32_t lowH2GEMM(
		double *D,
		double *B, double *VT, uint64_t *Dptr, uint64_t *Bptr, int32_t *VTptr, int32_t *lchildren,
		int32_t *rchildren, int32_t *levelset, int32_t *idx, double *mrhs,
		double *apres, int32_t nrhs, int32_t *Ddim, int32_t *wptr, int32_t *uptr, double *wskel,
		int32_t *wskeloffset, double *uskel, int32_t *uskeloffset, int32_t *lm,
		int32_t *slen, int32_t *nblockSet, int32_t *nblocks, int32_t *npairx, int32_t *npairy,
		int32_t *fblockSet, int32_t *fblocks, int32_t *fpairx, int32_t *fpairy, int32_t ncount,
		int32_t fcount, int32_t *wpart, int32_t *clevelset, int cdepth
){
	int nstop = ncount/12*12;
#pragma omp parallel for
	for (int i = 0; i < nstop; i++) {
		int32_t _0 = i + 1;
		for (int j = nblockSet[i]; j < nblockSet[_0]; j++) {
			int32_t _1 = j + 1;
			for (int k = nblocks[j]; k < nblocks[_1]; k++) {
				cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
							Ddim[npairx[k]], nrhs, Ddim[npairy[k]],
							float_from_bits(1065353216 /* 1 */), &D[Dptr[k]],
							Ddim[npairx[k]], &mrhs[wptr[npairy[k]]], Ddim[npairy[k]],
							float_from_bits(1065353216 /* 1 */),
							&apres[uptr[npairx[k]]], Ddim[npairx[k]]);
			} // for k
		} // for j
	}

	mkl_set_dynamic(true);
	mkl_set_num_threads(12);

	if(nstop<ncount){
		for (int i = nstop; i < ncount; i++)
		{
			int32_t _0 = i + 1;
			for (int j = nblockSet[i]; j < nblockSet[_0]; j++) {
				int32_t _1 = j + 1;
				for (int k = nblocks[j]; k < nblocks[_1]; k++) {
					cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
								Ddim[npairx[k]], nrhs, Ddim[npairy[k]],
								float_from_bits(1065353216 /* 1 */), &D[Dptr[k]],
								Ddim[npairx[k]], &mrhs[wptr[npairy[k]]], Ddim[npairy[k]],
								float_from_bits(1065353216 /* 1 */),
								&apres[uptr[npairx[k]]], Ddim[npairx[k]]);
				} // for k
			} // for j
		}
	}

#pragma omp parallel for
	for(int k = clevelset[0]; k<clevelset[1]; k++)
	{
		for(int j = wpart[k]; j<wpart[k+1]; j++)
		{
			int32_t _2 = (int32_t)(4294967295);
			bool _3 = lchildren[idx[j]] == _2;
			if (_3)
			{
				cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
							slen[idx[j]],nrhs,Ddim[lm[idx[j]]],
							float_from_bits(1065353216 /* 1 */), &VT[VTptr[idx[j]]],
							slen[idx[j]], &mrhs[wptr[lm[idx[j]]]], Ddim[lm[idx[j]]], float_from_bits(0 /* 0 */),
							&wskel[wskeloffset[idx[j]]], slen[idx[j]]);
			} // if _3
			else
			{
				cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
							slen[idx[j]],nrhs,slen[lchildren[idx[j]]],
							float_from_bits(1065353216 /* 1 */), &VT[VTptr[idx[j]]],
							slen[idx[j]], &wskel[wskeloffset[lchildren[idx[j]]]], slen[lchildren[idx[j]]], float_from_bits(0 /* 0 */),
							&wskel[wskeloffset[idx[j]]],  slen[idx[j]]);
				int32_t _4 = slen[idx[j]] * slen[lchildren[idx[j]]];
				int32_t _5 = _4 + VTptr[idx[j]];
				cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
							slen[idx[j]],nrhs,slen[rchildren[idx[j]]],
							float_from_bits(1065353216 /* 1 */), &VT[_5],
							slen[idx[j]], &wskel[wskeloffset[rchildren[idx[j]]]], slen[rchildren[idx[j]]], float_from_bits(1065353216 /* 1 */),
							&wskel[wskeloffset[idx[j]]], slen[idx[j]]);
			} // if _3 else
		}
	}

//#pragma omp parallel for
//	for(int k = clevelset[0]; k<clevelset[1]; k++)
//	{
//		for(int j = wpart[k]; j<wpart[k+1]; j++)
//		{
//			int32_t _2 = (int32_t)(4294967295);
//			bool _3 = lchildren[idx[j]] == _2;
//			if (_3)
//			{
//				cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
//							slen[idx[j]],nrhs,Ddim[lm[idx[j]]],
//							float_from_bits(1065353216 /* 1 */), &VT[VTptr[idx[j]]],
//							slen[idx[j]], &mrhs[wptr[lm[idx[j]]]], Ddim[lm[idx[j]]], float_from_bits(0 /* 0 */),
//							&wskel[wskeloffset[idx[j]]], slen[idx[j]]);
//			} // if _3
//			else
//			{
//				cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
//							slen[idx[j]],nrhs,slen[lchildren[idx[j]]],
//							float_from_bits(1065353216 /* 1 */), &VT[VTptr[idx[j]]],
//							slen[idx[j]], &wskel[wskeloffset[lchildren[idx[j]]]], slen[lchildren[idx[j]]], float_from_bits(0 /* 0 */),
//							&wskel[wskeloffset[idx[j]]],  slen[idx[j]]);
//				int32_t _4 = slen[idx[j]] * slen[lchildren[idx[j]]];
//				int32_t _5 = _4 + VTptr[idx[j]];
//				cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
//							slen[idx[j]],nrhs,slen[rchildren[idx[j]]],
//							float_from_bits(1065353216 /* 1 */), &VT[_5],
//							slen[idx[j]], &wskel[wskeloffset[rchildren[idx[j]]]], slen[rchildren[idx[j]]], float_from_bits(1065353216 /* 1 */),
//							&wskel[wskeloffset[idx[j]]], slen[idx[j]]);
//			} // if _3 else
//		}
//	}

	for (int i = 1; i < cdepth-2; i++)
	{
		int32_t _0 = i + 1;
#pragma omp parallel for
		for (int k = clevelset[i]; k < clevelset[_0]; k++)
		{
			int32_t _1 = k + 1;
			for (int j = wpart[k]; j < wpart[_1]; j++)
			{
				{
					cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
								slen[idx[j]],nrhs,slen[lchildren[idx[j]]],
								float_from_bits(1065353216 /* 1 */), &VT[VTptr[idx[j]]],
								slen[idx[j]], &wskel[wskeloffset[lchildren[idx[j]]]], slen[lchildren[idx[j]]], float_from_bits(0 /* 0 */),
								&wskel[wskeloffset[idx[j]]],  slen[idx[j]]);
					int32_t _4 = slen[idx[j]] * slen[lchildren[idx[j]]];
					int32_t _5 = _4 + VTptr[idx[j]];
					cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
								slen[idx[j]],nrhs,slen[rchildren[idx[j]]],
								float_from_bits(1065353216 /* 1 */), &VT[_5],
								slen[idx[j]], &wskel[wskeloffset[rchildren[idx[j]]]], slen[rchildren[idx[j]]], float_from_bits(1065353216 /* 1 */),
								&wskel[wskeloffset[idx[j]]], slen[idx[j]]);
				} // if _3 else
			} // for j
		} // for k
	} // for i

	mkl_set_num_threads(12);
	for (int i = cdepth-2; i < cdepth; i++)
	{
		int32_t _0 = i + 1;
//#pragma omp parallel for
		for (int k = clevelset[i]; k < clevelset[_0]; k++)
		{
			int32_t _1 = k + 1;
			for (int j = wpart[k]; j < wpart[_1]; j++)
			{
				{
					cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
								slen[idx[j]],nrhs,slen[lchildren[idx[j]]],
								float_from_bits(1065353216 /* 1 */), &VT[VTptr[idx[j]]],
								slen[idx[j]], &wskel[wskeloffset[lchildren[idx[j]]]], slen[lchildren[idx[j]]], float_from_bits(0 /* 0 */),
								&wskel[wskeloffset[idx[j]]],  slen[idx[j]]);
					int32_t _4 = slen[idx[j]] * slen[lchildren[idx[j]]];
					int32_t _5 = _4 + VTptr[idx[j]];
					cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
								slen[idx[j]],nrhs,slen[rchildren[idx[j]]],
								float_from_bits(1065353216 /* 1 */), &VT[_5],
								slen[idx[j]], &wskel[wskeloffset[rchildren[idx[j]]]], slen[rchildren[idx[j]]], float_from_bits(1065353216 /* 1 */),
								&wskel[wskeloffset[idx[j]]], slen[idx[j]]);
				} // if _3 else
			} // for j
		} // for k
	} // for i

	int fstop = fcount/12*12;

#pragma omp parallel for
	for (int i = 0; i < fstop; i++) {
		int32_t _8 = i + 1;
		for (int j = fblockSet[i]; j < fblockSet[_8]; j++) {
			int32_t _9 = j + 1;
			for (int k = fblocks[j]; k < fblocks[_9]; k++) {
				cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
							slen[fpairx[k]], nrhs, slen[fpairy[k]],
							float_from_bits(1065353216 /* 1 */), &B[Bptr[k]],
							slen[fpairx[k]], &wskel[wskeloffset[fpairy[k]]], slen[fpairy[k]],
							float_from_bits(1065353216 /* 1 */),
							&uskel[uskeloffset[fpairx[k]]], slen[fpairx[k]]);
			} // for k
		} // for j
	} // for i

	mkl_set_num_threads(12);
	if(fstop<fcount){
		for (int i = fstop; i < fcount; ++i) {
			int32_t _8 = i + 1;
			for (int j = fblockSet[i]; j < fblockSet[_8]; j++) {
				int32_t _9 = j + 1;
				for (int k = fblocks[j]; k < fblocks[_9]; k++) {
					cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
								slen[fpairx[k]], nrhs, slen[fpairy[k]],
								float_from_bits(1065353216 /* 1 */), &B[Bptr[k]],
								slen[fpairx[k]], &wskel[wskeloffset[fpairy[k]]], slen[fpairy[k]],
								float_from_bits(1065353216 /* 1 */),
								&uskel[uskeloffset[fpairx[k]]], slen[fpairx[k]]);
				} // for k
			} // for j
		}
	}


	for (int i = cdepth-1; i > cdepth-3; i--)
	{
		int32_t _17 = i + 1;
//#pragma omp parallel for
		for (int k = clevelset[i]; k < clevelset[_17]; k++)
		{
			int32_t _18 = wpart[k] - 1;
			int32_t _19 = k + 1;
			int32_t _20 = wpart[_19] - 1;
			for (int j = _20; j > _18; j--)
			{
				{
					cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
								slen[lchildren[idx[j]]],nrhs,slen[idx[j]],
								float_from_bits(1065353216 /* 1 */), &VT[VTptr[idx[j]]],
								slen[idx[j]], &uskel[uskeloffset[idx[j]]], slen[idx[j]], float_from_bits(1065353216 /* 1 */),
								&uskel[uskeloffset[lchildren[idx[j]]]], slen[lchildren[idx[j]]]);
					int32_t _23 = slen[idx[j]] * slen[lchildren[idx[j]]];
					int32_t _24 = _23 + VTptr[idx[j]];
					cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
								slen[rchildren[idx[j]]],nrhs,slen[idx[j]],
								float_from_bits(1065353216 /* 1 */), &VT[_24],
								slen[idx[j]], &uskel[uskeloffset[idx[j]]], slen[idx[j]], float_from_bits(1065353216 /* 1 */),
								&uskel[uskeloffset[rchildren[idx[j]]]],  slen[rchildren[idx[j]]]);
				} // if _22 else
			} // for j
		} // for k
	} // for i

	for (int i = cdepth-3; i > 0; i--)
	{
		int32_t _17 = i + 1;
#pragma omp parallel for
		for (int k = clevelset[i]; k < clevelset[_17]; k++)
		{
			int32_t _18 = wpart[k] - 1;
			int32_t _19 = k + 1;
			int32_t _20 = wpart[_19] - 1;
			for (int j = _20; j > _18; j--)
			{
				{
					cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
								slen[lchildren[idx[j]]],nrhs,slen[idx[j]],
								float_from_bits(1065353216 /* 1 */), &VT[VTptr[idx[j]]],
								slen[idx[j]], &uskel[uskeloffset[idx[j]]], slen[idx[j]], float_from_bits(1065353216 /* 1 */),
								&uskel[uskeloffset[lchildren[idx[j]]]], slen[lchildren[idx[j]]]);
					int32_t _23 = slen[idx[j]] * slen[lchildren[idx[j]]];
					int32_t _24 = _23 + VTptr[idx[j]];
					cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
								slen[rchildren[idx[j]]],nrhs,slen[idx[j]],
								float_from_bits(1065353216 /* 1 */), &VT[_24],
								slen[idx[j]], &uskel[uskeloffset[idx[j]]], slen[idx[j]], float_from_bits(1065353216 /* 1 */),
								&uskel[uskeloffset[rchildren[idx[j]]]],  slen[rchildren[idx[j]]]);
				} // if _22 else
			} // for j
		} // for k
	} // for i


	for (int i = 0; i > -1; i--)
	{
		int32_t _17 = i + 1;
#pragma omp parallel for
		for (int k = clevelset[i]; k < clevelset[_17]; k++)
		{
			int32_t _18 = wpart[k] - 1;
			int32_t _19 = k + 1;
			int32_t _20 = wpart[_19] - 1;
			for (int j = _20; j > _18; j--)
			{
				int32_t _21 = (int32_t)(4294967295);
				bool _22 = lchildren[idx[j]] == _21;
				if (_22)
				{
					cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
								Ddim[lm[idx[j]]],nrhs,slen[idx[j]],
								float_from_bits(1065353216 /* 1 */), &VT[VTptr[idx[j]]],
								slen[idx[j]], &uskel[uskeloffset[idx[j]]], slen[idx[j]], float_from_bits(1065353216 /* 1 */),
								&apres[uptr[lm[idx[j]]]], Ddim[lm[idx[j]]]);
				} // if _22
				else
				{
					cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
								slen[lchildren[idx[j]]],nrhs,slen[idx[j]],
								float_from_bits(1065353216 /* 1 */), &VT[VTptr[idx[j]]],
								slen[idx[j]], &uskel[uskeloffset[idx[j]]], slen[idx[j]], float_from_bits(1065353216 /* 1 */),
								&uskel[uskeloffset[lchildren[idx[j]]]], slen[lchildren[idx[j]]]);
					int32_t _23 = slen[idx[j]] * slen[lchildren[idx[j]]];
					int32_t _24 = _23 + VTptr[idx[j]];
					cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
								slen[rchildren[idx[j]]],nrhs,slen[idx[j]],
								float_from_bits(1065353216 /* 1 */), &VT[_24],
								slen[idx[j]], &uskel[uskeloffset[idx[j]]], slen[idx[j]], float_from_bits(1065353216 /* 1 */),
								&uskel[uskeloffset[rchildren[idx[j]]]],  slen[rchildren[idx[j]]]);
				} // if _22 else
			} // for j
		} // for k
	} // for i

	return 0;
}

//int32_t H2GEMM(double *D,
//			  double *B, double *VT, uint64_t *Dptr, uint64_t *Bptr, int32_t *VTptr, int32_t *lchildren, int32_t *rchildren, int32_t *levelset, int32_t *idx, double *mrhs,
//			  double *apres, int32_t nrhs, int32_t *Ddim, int32_t *wptr, int32_t *uptr, double *wskel, int32_t *wskeloffset, double *uskel, int32_t *uskeloffset, int32_t *lm,
//			  int32_t *slen, int32_t *nidx, int32_t *nidy, int32_t ncount, int32_t *fidx, int32_t *fidy, int32_t fcount, double *utmp, uint64_t *utmpoffset, double *ftmp,
//			  uint64_t *ftmpoffset, int cdepth) {
//#pragma omp parallel for
//	for (int i = 0; i < ncount; i++)
//	{
//		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
//					Ddim[nidx[i]],nrhs,Ddim[nidy[i]],
//					float_from_bits(1065353216 /* 1 */), &D[Dptr[i]],
//					Ddim[nidx[i]], &mrhs[wptr[nidy[i]]], Ddim[nidy[i]], float_from_bits(0 /* 0 */),
//					&utmp[utmpoffset[i]],  Ddim[nidx[i]]);
//	} // for i
//	for (int i = 0; i < ncount; i++)
//	{
//#pragma omp parallel for
//		for (int j = 0; j < nrhs; j++)
//		{
//			for (int k = 0; k < Ddim[nidx[i]]; k++)
//			{
//				int32_t _0 = j * Ddim[nidx[i]];
//				int32_t _1 = _0 + k;
//				int32_t _2 = uptr[nidx[i]] + _1;
//				int32_t _3 = utmpoffset[i] + _1;
//				double _4 = apres[_2] + utmp[_3];
//				apres[_2] = _4;
//			} // for k
//		} // for j
//	} // for i
////	int32_t _5 = (int32_t)(4294967295);
////	int32_t _6 = (int32_t)(6);
//	for (int i = 0; i < cdepth; i++)
//	{
//		int32_t _7 = i + 1;
//#pragma omp parallel for
//		for (int k = clevelset[i]; k < clevelset[_7]; k++)
//		{
//			int32_t _1 = k + 1;
//			for (int j = wpart[k]; j < wpart[_1]; j++)
//			{
//				int32_t _2 = (int32_t)(4294967295);
//				bool _3 = lchildren[idx[j]] == _2;
//				if (_3)
//				{
//					cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
//								slen[idx[j]],nrhs,Ddim[lm[idx[j]]],
//								float_from_bits(1065353216 /* 1 */), &VT[VTptr[idx[j]]],
//								slen[idx[j]], &mrhs[wptr[lm[idx[j]]]], Ddim[lm[idx[j]]], float_from_bits(0 /* 0 */),
//								&wskel[wskeloffset[idx[j]]], slen[idx[j]]);
//				} // if _3
//				else
//				{
//					cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
//								slen[idx[j]],nrhs,slen[lchildren[idx[j]]],
//								float_from_bits(1065353216 /* 1 */), &VT[VTptr[idx[j]]],
//								slen[idx[j]], &wskel[wskeloffset[lchildren[idx[j]]]], slen[lchildren[idx[j]]], float_from_bits(0 /* 0 */),
//								&wskel[wskeloffset[idx[j]]],  slen[idx[j]]);
//					int32_t _4 = slen[idx[j]] * slen[lchildren[idx[j]]];
//					int32_t _5 = _4 + VTptr[idx[j]];
//					cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
//								slen[idx[j]],nrhs,slen[rchildren[idx[j]]],
//								float_from_bits(1065353216 /* 1 */), &VT[_5],
//								slen[idx[j]], &wskel[wskeloffset[rchildren[idx[j]]]], slen[rchildren[idx[j]]], float_from_bits(1065353216 /* 1 */),
//								&wskel[wskeloffset[idx[j]]], slen[idx[j]]);
//				} // if _3 else
//			}
//		}
//	} // for i
//#pragma omp parallel for
//	for (int i = 0; i < fcount; i++)
//	{
//		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
//					slen[fidx[i]],nrhs,slen[fidy[i]],
//					float_from_bits(1065353216 /* 1 */), &B[Bptr[i]],
//					slen[fidx[i]], &wskel[wskeloffset[fidy[i]]], slen[fidy[i]], float_from_bits(0 /* 0 */),
//					&ftmp[ftmpoffset[i]],  slen[fidx[i]]);
//	} // for i
//	for (int i = 0; i < fcount; i++)
//	{
//#pragma omp parallel for
//		for (int j = 0; j < nrhs; j++)
//		{
//			for (int k = 0; k < slen[fidx[i]]; k++)
//			{
//				int32_t _12 = j * slen[fidx[i]];
//				int32_t _13 = _12 + k;
//				int32_t _14 = uskeloffset[fidx[i]] + _13;
//				int32_t _15 = ftmpoffset[i] + _13;
//				double _16 = uskel[_14] + ftmp[_15];
//				uskel[_14] = _16;
//			} // for k
//		} // for j
//	} // for i
//
//	for (int i = cdepth-1; i > -1; i--)
//	{
//		int32_t _17 = i + 1;
//#pragma omp parallel for
//		for (int k = clevelset[i]; k < clevelset[_17]; k++)
//		{
//			int32_t _18 = wpart[k] - 1;
//			int32_t _19 = k + 1;
//			int32_t _20 = wpart[_19] - 1;
//			for (int j = _20; j > _18; j--)
//			{
//				int32_t _21 = (int32_t)(4294967295);
//				bool _22 = lchildren[idx[j]] == _21;
//				if (_22)
//				{
//					cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
//								Ddim[lm[idx[j]]],nrhs,slen[idx[j]],
//								float_from_bits(1065353216 /* 1 */), &VT[VTptr[idx[j]]],
//								slen[idx[j]], &uskel[uskeloffset[idx[j]]], slen[idx[j]], float_from_bits(1065353216 /* 1 */),
//								&apres[uptr[lm[idx[j]]]], Ddim[lm[idx[j]]]);
//				} // if _22
//				else
//				{
//					cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
//								slen[lchildren[idx[j]]],nrhs,slen[idx[j]],
//								float_from_bits(1065353216 /* 1 */), &VT[VTptr[idx[j]]],
//								slen[idx[j]], &uskel[uskeloffset[idx[j]]], slen[idx[j]], float_from_bits(1065353216 /* 1 */),
//								&uskel[uskeloffset[lchildren[idx[j]]]], slen[lchildren[idx[j]]]);
//					int32_t _23 = slen[idx[j]] * slen[lchildren[idx[j]]];
//					int32_t _24 = _23 + VTptr[idx[j]];
//					cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
//								slen[rchildren[idx[j]]],nrhs,slen[idx[j]],
//								float_from_bits(1065353216 /* 1 */), &VT[_24],
//								slen[idx[j]], &uskel[uskeloffset[idx[j]]], slen[idx[j]], float_from_bits(1065353216 /* 1 */),
//								&uskel[uskeloffset[rchildren[idx[j]]]],  slen[rchildren[idx[j]]]);
//				} // if _22 else
//			} // for j
//		} // for k
//	} // for i
//
//	return 0;
//}
#ifdef __cplusplus
}  // extern "C"
#endif
#endif //PROJECT_H2GEMM_H
