//
// Created by Bangtian Liu on 6/28/19.
//
//
// Created by Bangtian Liu on 5/7/19.
//

//#include <HGEMM_v1.h>
//#include <HGEMM_gen.h>
//#include <HGEMM_level.h>

#include <HSSGEMM.h>
#include <H2GEMM.h>
#include "Util.h"
#include "../sympiler/HMatrix.h"
//#include "HTree.h"
#include "compression.h"
#include "hcds.h"

using namespace Sympiler::Internal;
using namespace Sympiler;

#define HSS
//#define Coarsen
int main(int argc, char *argv[])
{
//	std::string path = "../sympiler/";

	std::string pointname=argv[1];

	int n = atoi(argv[2]);
	int dim = atoi(argv[3]);
	double acc = atof(argv[4]);
	auto X = (double *)mkl_malloc(sizeof(double)*n*dim, 64);
	memset(X, 0, sizeof(double)*n*dim);
	int size1 = preprocesbin(pointname);
//	printf("n=%d dim=%d size1=%d\n", n, dim, size1);
	assert(size1 == n*dim);
	bin2read(pointname, X, size1);

	int nrhs = 2048;
//  if(argc>=6) nrhs =atoi(argv[5]);
	double buget = atof(argv[5]);
	int option = atoi(argv[6]);

	auto W = (double *) mkl_malloc(sizeof(double)*n*nrhs, 64);
	memset(W, 0, sizeof(double)*n*nrhs);

	randu(n,nrhs,W, 0.0,1.0);

	auto TW = (double *) mkl_malloc(sizeof(double)*n*nrhs, 64);
	memset(TW, 0, sizeof(double)*n*nrhs);

	auto U = (double *) mkl_malloc(sizeof(double)*n*nrhs, 64);
	memset(U, 0, sizeof(double)*n*nrhs);


    Internal::Ktype kfun=Internal::KS_GAUSSIAN;
	double tau = 3.0;
	mkl_set_dynamic(false);
	mkl_set_num_threads(1);
	bool coarsing= true;

	if(option==0){
		coarsing = false;
	}
	else {
		coarsing = true;
	}

	bool cbs;

	if(buget==0.0)cbs=false;
	else {
		if (option > 0) cbs = true;
		else cbs = false;
	}
//	auto start1=omp_get_wtime();
	HTree tree;
	readHtree(tree, coarsing, cbs);
	tree.X = X;
	tree.ktype =kfun;
	tree.h=5;
	tree.dim = dim;

	clustertree ctree;

	if(coarsing){
		coarloadtree(tree, ctree);
	}
	else {
		loadtree(tree, ctree);
	}

	std::vector<ret> rtmp;
	rtmp.resize(tree.numnodes);

	auto start = omp_get_wtime();
	compression(ctree, tree, rtmp.data(), X, 256, n, kfun, dim, 5, coarsing, acc);
	hcds cds;
	DUV2CDS(tree, cds, ctree, rtmp.data(),nrhs, coarsing, cbs);
	auto end = omp_get_wtime();
//	printf("%f", end-start-tree.cdstime);
	transform(tree, cds, W, n, nrhs, TW);
	allocatewuskel(tree, cds, ctree, rtmp.data(), nrhs, coarsing);

	auto flops = computeFlops(tree, rtmp.data(), 256, nrhs);

//	printf("flops=%lu\n",flops);
	start = omp_get_wtime();
	if(buget==0.0){
		switch (option) {
			case 0: {
				seqHSSGEMM(cds.CacheNear, cds.CacheFar, cds.Proj, cds.nearoffset, cds.faroffset, cds.projoffset,
						   tree.tlchildren, tree.trchildren, tree.levelset, tree.idx, TW, U, nrhs, tree.Dim, cds.woffset, cds.uoffset,
						   cds.w_skel, cds.w_skeloffset, cds.u_skel, cds.u_skeloffset, tree.lm, cds.skel_length, tree.ncount, tree.fcount,tree.depth);
				break;
			}

			case 1: {
				parHSSGEMM(cds.CacheNear, cds.CacheFar, cds.Proj, cds.nearoffset, cds.faroffset, cds.projoffset,
						   tree.tlchildren, tree.trchildren, tree.levelset, tree.cidx, TW, U, nrhs, tree.Dim, cds.woffset, cds.uoffset,
						   cds.w_skel, cds.w_skeloffset, cds.u_skel, cds.u_skeloffset, tree.lm, cds.skel_length, tree.wpart, tree.clevelset, tree.ncount, tree.fcount,tree.cdepth);
				break;
			}

			case 2: {
				lowHSSGEMM(cds.CacheNear, cds.CacheFar, cds.Proj, cds.nearoffset, cds.faroffset, cds.projoffset,
						   tree.tlchildren, tree.trchildren, tree.levelset, tree.cidx, TW, U, nrhs, tree.Dim, cds.woffset, cds.uoffset,
						   cds.w_skel, cds.w_skeloffset, cds.u_skel, cds.u_skeloffset, tree.lm, cds.skel_length, tree.wpart, tree.clevelset, tree.ncount, tree.fcount,tree.cdepth);

				break;
			}
		}
	}
	else {
		switch (option) {
			case 0: {
				seqGEMM(cds.CacheNear, cds.CacheFar, cds.Proj, cds.nearoffset, cds.faroffset, cds.projoffset,
						tree.tlchildren, tree.trchildren, tree.levelset, tree.idx, TW, U, nrhs, tree.Dim, cds.woffset, cds.uoffset,
						cds.w_skel, cds.w_skeloffset, cds.u_skel, cds.u_skeloffset, tree.lm, cds.skel_length, tree.nxval, tree.nyval, tree.ncount,
						tree.fxval, tree.fyval, tree.fcount,cds.utmp, cds.utmpoffset, cds.ftmp, cds.ftmpoffset, tree.depth);
				break;
			}

			case 1: {
				blockGEMM(
						cds.CacheNear, cds.CacheFar, cds.Proj, cds.nearoffset, cds.faroffset, cds.projoffset,
						tree.tlchildren, tree.trchildren, tree.levelset, tree.idx, TW, U, nrhs, tree.Dim, cds.woffset, cds.uoffset,
						cds.w_skel, cds.w_skeloffset, cds.u_skel, cds.u_skeloffset, tree.lm, cds.skel_length, tree.nblockset, tree.nblocks, tree.nxval,tree.nyval,
						tree.fblockset, tree.fblocks, tree.fxval, tree.fyval,tree.nrow, tree.frow,tree.depth);
				break;
			}
			case 2: {
				CBHGEMM(cds.CacheNear, cds.CacheFar, cds.Proj, cds.nearoffset, cds.faroffset, cds.projoffset,
						tree.tlchildren, tree.trchildren, tree.levelset, tree.cidx, TW, U, nrhs, tree.Dim, cds.woffset, cds.uoffset,
						cds.w_skel, cds.w_skeloffset, cds.u_skel, cds.u_skeloffset, tree.lm, cds.skel_length, tree.nblockset, tree.nblocks, tree.nxval,tree.nyval,
						tree.fblockset, tree.fblocks, tree.fxval, tree.fyval,tree.nrow, tree.frow,tree.wpart,tree.clevelset, tree.cdepth);
				break;
			}

			case 3: {
				lowH2GEMM(
						cds.CacheNear, cds.CacheFar, cds.Proj, cds.nearoffset, cds.faroffset, cds.projoffset,
						tree.tlchildren, tree.trchildren, tree.levelset, tree.cidx, TW, U, nrhs, tree.Dim, cds.woffset, cds.uoffset,
						cds.w_skel, cds.w_skeloffset, cds.u_skel, cds.u_skeloffset, tree.lm, cds.skel_length, tree.nblockset, tree.nblocks, tree.nxval,tree.nyval,
						tree.fblockset, tree.fblocks, tree.fxval, tree.fyval,tree.nrow, tree.frow,tree.wpart,tree.clevelset, tree.cdepth);
				break;
			}
		}

	}
//	switch(option)
//    seqHSSGEMM(cds.CacheNear, cds.CacheFar, cds.Proj, cds.nearoffset, cds.faroffset, cds.projoffset,
//    tree.tlchildren, tree.trchildren, tree.levelset, tree.idx, TW, U, nrhs, tree.Dim, cds.woffset, cds.uoffset,
//    cds.w_skel, cds.w_skeloffset, cds.u_skel, cds.u_skeloffset, tree.lm, cds.skel_length, tree.ncount, tree.fcount,tree.depth);

//		parHSSGEMM(cds.CacheNear, cds.CacheFar, cds.Proj, cds.nearoffset, cds.faroffset, cds.projoffset,
//				   tree.tlchildren, tree.trchildren, tree.levelset, tree.cidx, TW, U, nrhs, tree.Dim, cds.woffset, cds.uoffset,
//					cds.w_skel, cds.w_skeloffset, cds.u_skel, cds.u_skeloffset, tree.lm, cds.skel_length, tree.wpart, tree.clevelset, tree.ncount, tree.fcount,tree.cdepth);

//		lowHSSGEMM(cds.CacheNear, cds.CacheFar, cds.Proj, cds.nearoffset, cds.faroffset, cds.projoffset,
//				   tree.tlchildren, tree.trchildren, tree.levelset, tree.cidx, TW, U, nrhs, tree.Dim, cds.woffset, cds.uoffset,
//					cds.w_skel, cds.w_skeloffset, cds.u_skel, cds.u_skeloffset, tree.lm, cds.skel_length, tree.wpart, tree.clevelset, tree.ncount, tree.fcount,tree.cdepth);


//#ifdef HSS
//	//code for HSS case
//	HGEMM(cds.CacheNear, cds.CacheFar, cds.Proj, cds.nearoffset, cds.faroffset, cds.projoffset,
//	tree.tlchildren, tree.trchildren, tree.levelset, tree.cidx, TW, U, nrhs, tree.Dim, cds.woffset, cds.uoffset, cds.w_skel,
// cds.w_skeloffset, cds.u_skel, cds.u_skeloffset,tree.lm, cds.skel_length, tree.wpart, tree.clevelset);
//#else
//	HGEMM(cds.CacheNear, cds.CacheFar, cds.Proj, cds.nearoffset, cds.faroffset, cds.projoffset,
//		  tree.tlchildren, tree.trchildren, tree.levelset, tree.cidx, TW, U, nrhs, tree.Dim, cds.woffset, cds.uoffset, cds.w_skel,
//		  cds.w_skeloffset, cds.u_skel, cds.u_skeloffset,tree.lm,cds.skel_length, tree.nblockset, tree.nblocks, tree.nxval,
//		  tree.nyval, tree.fblockset, tree.fblocks, tree.fxval, tree.fyval, tree.wpart, tree.clevelset);
//#endif
	end = omp_get_wtime();

	auto eval = end-start;
	std::cout<<flops<<","<<flops/eval<<"\n";
	int d=dim;
	double acc1=computeError(tree.lids, tree.Dim[0],nrhs,n,X,d,W,U);
//	printf(",%e\n",acc1);
}
