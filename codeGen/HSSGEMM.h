//
// Created by Bangtian Liu on 8/23/19.
//

#ifndef PROJECT_HSSGEMM_H
#define PROJECT_HSSGEMM_H
#include <stdint.h>
#include <mkl.h>
#include <stdio.h>
inline float float_from_bits(uint32_t bits) {
    union {
        uint32_t as_uint;
        float as_float;
    } u;
    u.as_uint = bits;
    return u.as_float;
}
int32_t seqHSSGEMM(double *D,
              double *B, double *VT, uint64_t *Dptr, uint64_t *Bptr, int32_t *VTptr, int32_t *lchildren, int32_t *rchildren, int32_t *levelset, int32_t *idx, double *mrhs,
              double *apres, int32_t nrhs, int32_t *Ddim, int32_t *wptr, int32_t *uptr, double *wskel, int32_t *wskeloffset, double *uskel, int32_t *uskeloffset, int32_t *lm,
              int32_t *slen, int32_t ncount, int32_t fcount, int32_t depth) {
//#pragma omp parallel for
    for (int i = 0; i < ncount; i++)
    {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    Ddim[i],nrhs,Ddim[i],
                    float_from_bits(1065353216 /* 1 */), &D[Dptr[i]],
                    Ddim[i], &mrhs[wptr[i]], Ddim[i], float_from_bits(0 /* 0 */),
                    &apres[uptr[i]], Ddim[i]);
    } // for i
    for (int i = depth-1; i >-1; i--)
    {
        int32_t _0 = i + 1;
//#pragma omp parallel for
        for (int j = levelset[i]; j < levelset[_0]; j++)
        {
//            int32_t _1 = k + 1;
//            for (int j = wpart[k]; j < wpart[_1]; j++)
//            {
//        printf("idx=%d\n",idx[j]);
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
//            } // for j
        } // for k
    } // for i
    uint32_t _6 = (uint32_t)(1);
    uint32_t _7 = (uint32_t)(fcount);
//#pragma omp parallel for
    for (int i = _6; i < _7+1; i++)
    {
        uint32_t _8 = (uint32_t)(1);
        int32_t _9 = i - _8;
        int32_t _10 = i + _8;
        int32_t _11 = i & 1;
        uint32_t _12 = (uint32_t)(0);
        bool _13 = _11 == _12;
        int32_t _14 = (int32_t)(_13 ? _9 : _10);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    slen[i],nrhs,slen[_14],
                    _8, &B[Bptr[_9]],
                    slen[i], &wskel[wskeloffset[_14]], slen[_14], _12,
                    &uskel[uskeloffset[i]],  slen[i]);
    } // for i
//    int32_t _15 = 0 - 1;
//    int32_t _16 = depth - 1;
    for (int i = 0; i < depth; i++)
    {
        int32_t _17 = i + 1;
//#pragma omp parallel for
        for (int j = levelset[i]; j < levelset[_17]; j++)
        {
//            int32_t _18 = wpart[k] - 1;
//            int32_t _19 = k + 1;
//            int32_t _20 = wpart[_19] - 1;
//            for (int j = _20; j > _18; j--)
//            {
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
//            } // for j
        } // for k
    } // for i
    return 0;
}

int32_t parHSSGEMM(double *D,
                   double *B, double *VT, uint64_t *Dptr, uint64_t *Bptr, int32_t *VTptr, int32_t *lchildren, int32_t *rchildren, int32_t *levelset, int32_t *idx, double *mrhs,
                   double *apres, int32_t nrhs, int32_t *Ddim, int32_t *wptr, int32_t *uptr, double *wskel, int32_t *wskeloffset, double *uskel, int32_t *uskeloffset, int32_t *lm,
                   int32_t *slen, int32_t *wpart, int32_t *clevelset, int ncount, int fcount, int cdepth)
{
    mkl_set_dynamic(false);
    #pragma omp parallel for
    for (int i = 0; i < ncount; i++)
    {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    Ddim[i],nrhs,Ddim[i],
                    float_from_bits(1065353216 /* 1 */), &D[Dptr[i]],
                    Ddim[i], &mrhs[wptr[i]], Ddim[i], float_from_bits(0 /* 0 */),
                    &apres[uptr[i]], Ddim[i]);
    } // for i

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

    uint32_t _6 = (uint32_t)(1);
    uint32_t _7 = (uint32_t)(fcount+1);
#pragma omp parallel for
    for (int i = _6; i < _7; i++)
    {
        uint32_t _8 = (uint32_t)(1);
        int32_t _9 = i - _8;
        int32_t _10 = i + _8;
        int32_t _11 = i & 1;
        uint32_t _12 = (uint32_t)(0);
        bool _13 = _11 == _12;
        int32_t _14 = (int32_t)(_13 ? _9 : _10);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    slen[i],nrhs,slen[_14],
                    _8, &B[Bptr[_9]],
                    slen[i], &wskel[wskeloffset[_14]], slen[_14], _12,
                    &uskel[uskeloffset[i]],  slen[i]);
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
}

int32_t lowHSSGEMM(double *D,
                   double *B, double *VT, uint64_t *Dptr, uint64_t *Bptr, int32_t *VTptr, int32_t *lchildren, int32_t *rchildren, int32_t *levelset, int32_t *idx, double *mrhs,
                       double *apres, int32_t nrhs, int32_t *Ddim, int32_t *wptr, int32_t *uptr, double *wskel, int32_t *wskeloffset, double *uskel, int32_t *uskeloffset, int32_t *lm,
                   int32_t *slen, int32_t *wpart, int32_t *clevelset, int ncount, int fcount, int cdepth)
{
    int nstop = ncount/12*12;
#pragma omp parallel for
    for (int i = 0; i < nstop; i++)
    {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    Ddim[i],nrhs,Ddim[i],
                    float_from_bits(1065353216 /* 1 */), &D[Dptr[i]],
                    Ddim[i], &mrhs[wptr[i]], Ddim[i], float_from_bits(0 /* 0 */),
                    &apres[uptr[i]], Ddim[i]);
    } // for i
    mkl_set_dynamic(true);
    mkl_set_num_threads(12);
    if(nstop<ncount){
        for (int i = nstop; i < ncount; i++)
        {
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        Ddim[i],nrhs,Ddim[i],
                        float_from_bits(1065353216 /* 1 */), &D[Dptr[i]],
                        Ddim[i], &mrhs[wptr[i]], Ddim[i], float_from_bits(0 /* 0 */),
                        &apres[uptr[i]], Ddim[i]);
        } // for i
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


    int _6 = (1);
    int fc = fcount/12*12;

//    int rfc = fcount - fc;

    int _61 = _6 + fc;

    int _7 = (fcount+1);
#pragma omp parallel for
    for (int i = _6; i < _61; i++)
    {
        uint32_t _8 = (uint32_t)(1);
        int32_t _9 = i - _8;
        int32_t _10 = i + _8;
        int32_t _11 = i & 1;
        uint32_t _12 = (uint32_t)(0);
        bool _13 = _11 == _12;
        int32_t _14 = (int32_t)(_13 ? _9 : _10);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    slen[i],nrhs,slen[_14],
                    _8, &B[Bptr[_9]],
                    slen[i], &wskel[wskeloffset[_14]], slen[_14], _12,
                    &uskel[uskeloffset[i]],  slen[i]);
    } // for i

    mkl_set_num_threads(12);

    for(int i=_61; i<_7; i++)
    {
        int32_t _8 = (int32_t)(1);
        int32_t _9 = i - _8;
        int32_t _10 = i + _8;
        int32_t _11 = i & 1;
        int32_t _12 = (int32_t)(0);
        bool _13 = _11 == _12;
        int32_t _14 = (int32_t)(_13 ? _9 : _10);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    slen[i],nrhs,slen[_14],
                    _8, &B[Bptr[_9]],
                    slen[i], &wskel[wskeloffset[_14]], slen[_14], _12,
                    &uskel[uskeloffset[i]],  slen[i]);
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
//                int32_t _21 = (int32_t)(4294967295);
//                bool _22 = lchildren[idx[j]] == _21;
//                if (_22)
//                {
//                    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
//                                Ddim[lm[idx[j]]],nrhs,slen[idx[j]],
//                                float_from_bits(1065353216 /* 1 */), &VT[VTptr[idx[j]]],
//                                slen[idx[j]], &uskel[uskeloffset[idx[j]]], slen[idx[j]], float_from_bits(1065353216 /* 1 */),
//                                &apres[uptr[lm[idx[j]]]], Ddim[lm[idx[j]]]);
//                } // if _22
//                else
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
//                int32_t _21 = (int32_t)(4294967295);
//                bool _22 = lchildren[idx[j]] == _21;
//                if (_22)
//                {
//                    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
//                                Ddim[lm[idx[j]]],nrhs,slen[idx[j]],
//                                float_from_bits(1065353216 /* 1 */), &VT[VTptr[idx[j]]],
//                                slen[idx[j]], &uskel[uskeloffset[idx[j]]], slen[idx[j]], float_from_bits(1065353216 /* 1 */),
//                                &apres[uptr[lm[idx[j]]]], Ddim[lm[idx[j]]]);
//                } // if _22
//                else
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


#endif //PROJECT_HSSGEMM_H
