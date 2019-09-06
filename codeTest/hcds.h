//
// Created by Bangtian Liu on 6/30/19.
//

#ifndef PROJECT_HCDS_H
#define PROJECT_HCDS_H

#include "HTree.h"
#include "compression.h"
//#include "../../../../opt/intel/compilers_and_libraries_2019.4.233/mac/mkl/include/mkl.h"

struct hcds{
	double *CacheNear;
	uint64_t *nearoffset;

	double *CacheFar;
	uint64_t *faroffset;

	int *projoffset;
	double *Proj;

	int *w_skeloffset;
	int *u_skeloffset;

	double *w_skel;
	double *u_skel;

	int *woffset;
	int *uoffset;

	int *skel_length;
	int *proj_column;

	// if non-blocking
	unsigned long int *utmpoffset;
	double *utmp;

	unsigned long int *ftmpoffset;
	double *ftmp;

};

void CacheNearBlock(HTree &tree, hcds &cds)
{
	cds.nearoffset = (uint64_t *) mkl_malloc(sizeof(uint64_t)*tree.ncount, 64);
	memset(cds.nearoffset, 0, sizeof(uint64_t)*tree.ncount);

	unsigned long int index=0;
	uint64_t offset=0;
	//int count=0;
	auto start=omp_get_wtime();
  for(int i=0; i<tree.nrow; i++)
	{
		for(int j=tree.nblockset[i]; j<tree.nblockset[i+1]; j++)
		{
	//		++count;
      for(int k = tree.nblocks[j]; k<tree.nblocks[j+1]; k++)
			{
				auto nx = tree.nxval[k];
				auto ny = tree.nyval[k];

				auto dimx = tree.Dim[nx];
				auto dimy = tree.Dim[ny];
				cds.nearoffset[index++] = offset;
				offset += dimx*dimy;
			}
		}
	}
  	auto stop = omp_get_wtime();
	tree.cdstime += stop - start;
  //printf("nblocks=%d\n", count );

	cds.CacheNear = (double *)mkl_malloc(sizeof(double)*offset, 64);
	memset(cds.CacheNear, 0, sizeof(double)*offset);

	index = 0;
//#pragma omp parallel for
	for(int i=0; i<tree.nrow; i++)
	{
		for(int j=tree.nblockset[i]; j<tree.nblockset[i+1]; j++)
		{
			for(int k = tree.nblocks[j]; k<tree.nblocks[j+1]; k++)
			{
				auto nx = tree.nxval[k];
				auto ny = tree.nyval[k];

				auto idx = tree.leaf[nx];
				auto idy = tree.leaf[ny];

				auto lidx = tree.lids + tree.lidsoffset[idx];
				auto lidy = tree.lids + tree.lidsoffset[idy];

				auto lena = tree.lidslen[idx];
				auto lenb = tree.lidslen[idy];

				auto offset = cds.nearoffset[k];

				Fsubmatrix(lidx, lena, lidy, lenb, cds.CacheNear+offset, tree.ktype, tree.X, tree.dim, tree.h);
			}
		}
	}
}

void CacheFarBlock(HTree &tree, hcds &cds, ret *rtmp)
{
	cds.faroffset = (uint64_t *) mkl_malloc(sizeof(uint64_t)*tree.fcount, 64);
	memset(cds.faroffset, 0, sizeof(uint64_t)*tree.fcount);

	unsigned long int index=0;
	uint64_t offset=0;

 // int count=0;
	auto start=omp_get_wtime();
	for(int i=0; i<tree.frow; i++)
	{
		for(int j=tree.fblockset[i]; j<tree.fblockset[i+1]; j++)
		{
   //   count++;
			for(int k = tree.fblocks[j]; k<tree.fblocks[j+1]; k++)
			{
				auto fx = tree.fxval[k];
				auto fy = tree.fyval[k];

				auto dimx = rtmp[fx].skels_length;
				auto dimy = rtmp[fy].skels_length;
				cds.faroffset[index++] = offset;
				offset += dimx*dimy;
			}
		}
	}
	auto stop = omp_get_wtime();
	tree.cdstime += stop - start;

  //printf("fblock=%d\n", count);

	cds.CacheFar = (double *)mkl_malloc(sizeof(double)*offset, 64);
	memset(cds.CacheFar, 0, sizeof(double)*offset);

	index = 0;

//#pragma omp parallel for
	for(int i=0; i<tree.frow; i++)
	{
		for(int j=tree.fblockset[i]; j<tree.fblockset[i+1]; j++)
		{
			for(int k = tree.fblocks[j]; k<tree.fblocks[j+1]; k++)
			{
				auto fx = tree.fxval[k];
				auto fy = tree.fyval[k];


				auto lidx = rtmp[fx].skels ;
				auto lidy = rtmp[fy].skels;

				auto lena = rtmp[fx].skels_length;
				auto lenb = rtmp[fy].skels_length;

				auto offset = cds.faroffset[k];

				Fsubmatrix(lidx, lena, lidy, lenb, cds.CacheFar+offset, tree.ktype, tree.X, tree.dim, tree.h);
			}
		}
	}
}

void  CacheNear(HTree &tree, hcds &cds)
{
	cds.nearoffset = (uint64_t *) mkl_malloc(sizeof(uint64_t)*tree.ncount, 64);
	memset(cds.nearoffset, 0, sizeof(uint64_t)*tree.ncount);

	cds.utmpoffset = (unsigned long int *) mkl_malloc(sizeof(unsigned long int)*tree.ncount, 64);
	memset(cds.utmpoffset, 0, sizeof(unsigned long int)*tree.ncount);


	unsigned long int index=0;
	uint64_t offset=0;
	unsigned long int tmpoffset=0;

	auto start = omp_get_wtime();
	for(int k=0; k<tree.ncount; k++)
	{
		auto nx = tree.nxval[k];
		auto ny = tree.nyval[k];

		auto dimx = tree.Dim[nx];
		auto dimy = tree.Dim[ny];

		cds.nearoffset[index] = offset;
		cds.utmpoffset[index++] = tmpoffset;
		offset += dimx*dimy;
		tmpoffset += dimx*2048;
	}
	auto stop = omp_get_wtime();

	tree.cdstime += (stop - start);
	cds.CacheNear = (double *)mkl_malloc(sizeof(double)*offset, 64);
	memset(cds.CacheNear, 0, sizeof(double)*offset);

//	cds.utmp = (double *)mkl_malloc(sizeof(double)*tmpoffset, 64);
//	memset(cds.utmp, 0, sizeof(double)*tmpoffset);

	index = 0;
#pragma omp parallel for
	for(int k=0; k<tree.ncount; k++)
	{
		auto nx = tree.nxval[k];
		auto ny = tree.nyval[k];

		auto idx = tree.leaf[nx];
		auto idy = tree.leaf[ny];

		auto lidx = tree.lids + tree.lidsoffset[idx];
		auto lidy = tree.lids + tree.lidsoffset[idy];

		auto lena = tree.lidslen[idx];
		auto lenb = tree.lidslen[idy];

		auto offset = cds.nearoffset[k];

		Fsubmatrix(lidx, lena, lidy, lenb, cds.CacheNear+offset, tree.ktype, tree.X, tree.dim, tree.h);
	}
}


void CacheFar(HTree &tree, hcds &cds, ret *rtmp)
{
	cds.faroffset = (uint64_t *) mkl_malloc(sizeof(uint64_t)*tree.fcount, 64);
	memset(cds.faroffset, 0, sizeof(uint64_t)*tree.fcount);

	cds.ftmpoffset = (unsigned long int *)mkl_malloc(sizeof(unsigned long int)*tree.fcount, 64);
	memset(cds.ftmpoffset, 0, sizeof(unsigned long int)*tree.fcount);


	unsigned long int index=0;
	unsigned long int offset=0;
	unsigned long int tmpoffset=0;

	auto start = omp_get_wtime();
	for(int k = 0; k<tree.fcount; k++)
	{
		auto fx = tree.fxval[k];
		auto fy = tree.fyval[k];

		auto dimx = rtmp[fx].skels_length;
		auto dimy = rtmp[fy].skels_length;
		cds.faroffset[index] = offset;
		cds.ftmpoffset[index++] = tmpoffset;
		offset += dimx*dimy;
		tmpoffset += dimx*2048;
	}
	auto stop = omp_get_wtime();

	tree.cdstime += stop - start;

	cds.CacheFar = (double *)mkl_malloc(sizeof(double)*offset, 64);
	memset(cds.CacheFar, 0, sizeof(double)*offset);

//	cds.ftmp = (double *) mkl_malloc(sizeof(double)*tmpoffset, 64);
//	memset(cds.ftmp, 0, sizeof(double)*tmpoffset);

	index = 0;
#pragma omp parallel for
	for(int k = 0; k<tree.fcount; k++)
	{
		auto fx = tree.fxval[k];
		auto fy = tree.fyval[k];


		auto lidx = rtmp[fx].skels ;
		auto lidy = rtmp[fy].skels;

		auto lena = rtmp[fx].skels_length;
		auto lenb = rtmp[fy].skels_length;

		auto offset = cds.faroffset[k];

		Fsubmatrix(lidx, lena, lidy, lenb, cds.CacheFar+offset, tree.ktype, tree.X, tree.dim, tree.h);
	}
}

void cacheProj(HTree &tree, hcds &cds, ret *rtmp)
{
	int numnodes=tree.numnodes;
	cds.projoffset = (int *)mkl_malloc(sizeof(int)*numnodes,64);
	memset(cds.projoffset,0,sizeof(int)*numnodes);
	int offset=0;

	for(int i=tree.depth-1; i>-1; i--)
	{
		for(int j=tree.levelset[i]; j<tree.levelset[i+1]; j++)
		{
			auto idx = tree.idx[j];

			auto skel = rtmp[idx].skels_length;
			auto proj = rtmp[idx].proj_column;
			cds.projoffset[idx]=offset;
			offset+=skel*proj;
		}
	}

	cds.Proj = (double *)malloc(sizeof(double)*offset);
#pragma omp parallel for
	for(int i=tree.depth-1; i>-1; i--) {
		for (int j = tree.levelset[i]; j < tree.levelset[i + 1]; j++)
		{
			auto idx = tree.idx[j];
			auto proj = cds.Proj + cds.projoffset[idx];
			auto skel = rtmp[idx].skels_length;
			auto sproj = rtmp[idx].proj_column;

			memcpy(proj, rtmp[idx].proj, sizeof(double)*skel*sproj);
		}
	}
}

void coarcacheProj(HTree &tree, hcds &cds, clustertree &ctree, ret *rtmp)
{
	auto &posw = ctree.opostw;

	int numnodes=tree.numnodes;
	cds.projoffset = (int *)mkl_malloc(sizeof(int)*numnodes,64);
	memset(cds.projoffset,0,sizeof(int)*numnodes);

	int offset = 0;

	for(int i=0;i<posw.size();i++)
	{
		auto &lpow=posw[i];
		for(int j=0;j<lpow.size();j++)
		{
			auto wpart=lpow[j];
			for(auto &v:wpart)
			{
				auto skel = rtmp[v].skels_length;
				auto proj = rtmp[v].proj_column;
				cds.projoffset[v]=offset;
				offset+=skel*proj;
			}
		}
	}

	cds.Proj = (double *)malloc(sizeof(double)*offset);

#pragma omp parallel for
	for(int i=0;i<posw.size();i++)
	{
		auto &lpow=posw[i];
		for(int j=0;j<lpow.size();j++)
		{
			auto wpart=lpow[j];
			for(auto &v:wpart)
			{
				auto proj = cds.Proj + cds.projoffset[v];
				auto tsize = rtmp[v].skels_length*rtmp[v].proj_column;
				memcpy(proj,rtmp[v].proj, sizeof(double)*tsize);
			}
		}
	}
}

void cacheWUskel(HTree &tree, hcds &cds, ret *rtmp, int nrhs)
{
	int numnodes=tree.numnodes;
	cds.w_skeloffset=(int *)mkl_malloc(sizeof(int)*numnodes,64);
	memset(cds.w_skeloffset,0,sizeof(int)*numnodes);
	cds.u_skeloffset=(int *)mkl_malloc(sizeof(int)*numnodes,64);
	memset(cds.u_skeloffset,0,sizeof(int)*numnodes);

	int offset=0;
	for(int i=tree.depth-1; i>-1; i--) {
		for (int j = tree.levelset[i]; j < tree.levelset[i + 1]; j++) {
			auto idx = tree.idx[j];

			auto skel=rtmp[idx].skels_length;

			cds.w_skeloffset[idx]=offset;
			cds.u_skeloffset[idx]=offset;
			offset+=skel*nrhs;
		}
	}

	cds.w_skel=(double *)malloc(sizeof(double)*offset);
	memset(cds.w_skel,0,sizeof(double)*offset);
	cds.u_skel=(double *)malloc(sizeof(double)*offset);
	memset(cds.u_skel,0,sizeof(double)*offset);
}


void coarcacheWUskel(HTree &tree, hcds &cds, clustertree &ctree, ret *rtmp, int nrhs)
{
	auto &posw = ctree.opostw;
	int numnodes=tree.numnodes;
	cds.w_skeloffset=(int *)mkl_malloc(sizeof(int)*numnodes,64);
	memset(cds.w_skeloffset,0,sizeof(int)*numnodes);
	cds.u_skeloffset=(int *)mkl_malloc(sizeof(int)*numnodes,64);
	memset(cds.u_skeloffset,0,sizeof(int)*numnodes);
	int offset = 0;

	for(int i=0;i<posw.size();i++)
	{
		auto &lpow=posw[i];
		for(int j=0;j<lpow.size();j++)
		{
			auto wpart=lpow[j];
			for(auto &v:wpart)
			{
				auto skel=rtmp[v].skels_length;
				auto ncol=nrhs;
				cds.w_skeloffset[v]=offset;
				cds.u_skeloffset[v]=offset;
				offset+=skel*ncol;
			}
		}
	}

	cds.w_skel=(double *)malloc(sizeof(double)*offset);
	memset(cds.w_skel,0,sizeof(double)*offset);
	cds.u_skel=(double *)malloc(sizeof(double)*offset);
	memset(cds.u_skel,0,sizeof(double)*offset);
}


void cachewuoffset(HTree &tree, hcds &cds, ret *rtmp, int nrhs)
{
	int nl = tree.nleaf;

	cds.woffset = (int *)mkl_malloc(sizeof(int)*nl,64);
	cds.uoffset = (int *)mkl_malloc(sizeof(int)*nl,64);

	int offset=0;
	for(int i=0; i<tree.nleaf; i++)
	{
		auto idx = tree.leaf[i];
		auto dim = tree.lidslen[idx];

		cds.woffset[i] = offset;
		cds.uoffset[i] = offset;
		offset += dim * nrhs;
	}
}

void cacheSkeldim(HTree &tree, hcds &cds, ret *rtmp)
{
	int numnodes=tree.numnodes;


	cds.skel_length=(int *)mkl_malloc(sizeof(int)*numnodes,64);
	memset(cds.skel_length,0,sizeof(int)*numnodes);

	cds.proj_column=(int *)mkl_malloc(sizeof(int)*numnodes,64);
	memset(cds.proj_column,0,sizeof(int)*numnodes);

	for(int i=1;i<numnodes;i++){
		cds.skel_length[i]=rtmp[i].skels_length;
		cds.proj_column[i]=rtmp[i].proj_column;
	}

}

void DUV2CDS(HTree &tree, hcds &cds, clustertree &ctree, ret *rtmp, int nrhs, bool coarsing, bool cbs)
{
	if(cbs){
		CacheNearBlock(tree,cds);
		CacheFarBlock(tree, cds, rtmp);
	}
	else{
		CacheNear(tree, cds);
		CacheFar(tree, cds, rtmp);
	}

  auto start = omp_get_wtime();

	if(coarsing){
		coarcacheProj(tree, cds, ctree, rtmp);
//		coarcacheWUskel(tree, cds, ctree, rtmp, nrhs);
	}
	else {
		cacheProj(tree, cds, rtmp);
//		cacheWUskel(tree,cds, rtmp, nrhs);
	}

	cachewuoffset(tree, cds, rtmp, nrhs);
	cacheSkeldim(tree, cds, rtmp);
  auto ends = omp_get_wtime();
	tree.cdstime+=(ends - start);
//  printf("%f,", tree.cdstime);
}


void transform(HTree & tree, hcds cds, double *W, int n, int nrhs, double *TW)
{
	for(int i=0; i<tree.nleaf; i++)
	{
		auto idx = tree.leaf[i];
		auto len = tree.lidslen[idx];
		auto lids = tree.lids + tree.lidsoffset[idx];

//		printf("idx=%d len=%d\n", idx, len);

		auto tw = TW + cds.woffset[tree.lm[idx]];

		int toffset=0;

		for(int j = 0; j<nrhs; j++)
		{
			for(int k = 0; k<len; k++)
			{
				tw[toffset++] = W[j*n+lids[k]];
			}
		}
	}
}

void allocatewuskel(HTree &tree, hcds &cds, clustertree &ctree, ret *rtmp, int nrhs, bool coarsing)
{
	if(coarsing){
		coarcacheWUskel(tree, cds, ctree, rtmp, nrhs);
	}
	else {
		cacheWUskel(tree,cds, rtmp, nrhs);
	}
//	cachewuoffset(tree, cds, rtmp, nrhs);
}

#endif //PROJECT_HCDS_H
