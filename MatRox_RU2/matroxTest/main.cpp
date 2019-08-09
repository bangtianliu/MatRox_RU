//
// Created by Bangtian Liu on 5/7/19.
//

//#include <HGEMM_gen.h>
#include <HGEMM_gen.h>
//#include <HGEMM_level.h>

#include "Util.h"
#include "../sympiler/HMatrix.h"

using namespace Sympiler::Internal;
using namespace Sympiler;

//#define Coarsen
int main(int argc, char *argv[])
{
//	std::string path = "../sympiler/";

	std::string pointname=argv[1];
	int n = atoi(argv[2]);
	int dim = atoi(argv[3]);
	int nrhs = 64;
	auto W = (double *) mkl_malloc(sizeof(double)*n*nrhs, 64);
	memset(W, 0, sizeof(double)*n*nrhs);

	randu(n,nrhs,W, 0.0,1.0);

	auto U = (double *) mkl_malloc(sizeof(double)*n*nrhs, 64);
	memset(U, 0, sizeof(double)*n*nrhs);


	Internal::Ktype kfun=Internal::KS_GAUSSIAN;
	double tau = 3.0;
	omp_set_num_threads(12);
  
  double acc1 = atof(argv[4]);
	auto start1 = std::chrono::high_resolution_clock::now();
	HMatrix hmat(Float(64), pointname, n, dim, kfun, tau, acc1);
	auto end1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff2 = end1-start1;
	std::cout<<"#clustering time is " << diff2.count()<<std::endl;

	auto start = std::chrono::high_resolution_clock::now();
	hmat.compression();
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff = end-start;
	std::cout<<"#compression time is " << diff.count()<<std::endl;

	if(hmat.setup.use_coarsing){
		hmat.tree->BalanceCoarLevelSet();
	}

	hmat.setup.nrhs = nrhs;
	start = std::chrono::high_resolution_clock::now();
	hmat.savetree();
	end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> rediff = end-start;
	std::cout<<"#tree time is " << rediff.count()<<std::endl;


	start = std::chrono::high_resolution_clock::now();
	hmat.cds = new CDS(*hmat.tree,hmat.setup, hmat.tmpresult);
	end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> rediff1 = end-start;
	std::cout<<"data layout is " << rediff1.count()<<std::endl;

	if(hmat.setup.use_cbs){

//		hmat.tree->NearBlock(2); // tuning parameter
////			tree->FarBlock(4,A->setup.use_coarsing);
//		hmat.tree->FarBlock(4, false);

		hmat.StructureNearBlock(hmat.tree->nearblocks);
		hmat.StructureFarBlock(hmat.tree->farblocks);

	}


	auto &cds = hmat.cds;

	start = std::chrono::high_resolution_clock::now();
	// level by level case for HSS
//	HGEMM(cds->CacheNear, cds->CacheFar, cds->Proj, cds->nearoffset, cds->faroffset, cds->projoffset,
//	hmat.tlchildren, hmat.trchildren, hmat.levelset, hmat.idx, W, U, nrhs, hmat.Dim, hmat.woffset, hmat.uoffset, cds->w_skel,
// cds->w_skeloffset, cds->u_skel, cds->u_skeloffset,hmat.leafmap,cds->skel_length );

// base code for general case

//	HGEMM(cds->CacheNear, cds->CacheFar, cds->Proj, cds->nearoffset, cds->faroffset, cds->projoffset,
//		  hmat.tlchildren, hmat.trchildren, hmat.levelset, hmat.idx, W, U, nrhs, hmat.Dim, hmat.woffset, hmat.uoffset, cds->w_skel,
//		  cds->w_skeloffset, cds->u_skel, cds->u_skeloffset,hmat.leafmap,cds->skel_length, cds->nidx, cds->nidy, cds->ncount,cds->fidx,
//		  cds->fidy, cds->fcount, cds->utmp, cds->utmpoffset, cds->ftmp, cds->ftmpoffset);
	//coarsening code for HSS case
	HGEMM(cds->CacheNear, cds->CacheFar, cds->Proj, cds->nearoffset, cds->faroffset, cds->projoffset,
	hmat.tlchildren, hmat.trchildren, hmat.levelset, hmat.idx, W, U, nrhs, hmat.Dim, hmat.woffset, hmat.uoffset, cds->w_skel,
 cds->w_skeloffset, cds->u_skel, cds->u_skeloffset,hmat.leafmap, cds->skel_length, hmat.wpart, hmat.clevelset);

//	HGEMM(cds->CacheNear, cds->CacheFar, cds->Proj, cds->nearoffset, cds->faroffset, cds->projoffset,
//	hmat.tlchildren, hmat.trchildren, hmat.levelset, hmat.idx, W, U, nrhs, hmat.Dim, hmat.woffset, hmat.uoffset, cds->w_skel,
// cds->w_skeloffset, cds->u_skel, cds->u_skeloffset,hmat.leafmap,cds->skel_length, hmat.nblockset, hmat.nblocks, hmat.nxval,
//	hmat.nyval, hmat.fblockset, hmat.fblocks, hmat.fxval, hmat.fyval, hmat.wpart, hmat.clevelset);


	end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> eval = end-start;
	std::cout<<"evaluation time is " << eval.count()<<std::endl;
//	HGEMM(CacheNear, CacheFar, Proj, nearoffset, faroffset, projoffset,lchildren,rchildren,
//		  W,levelset,idx,U,dim,wptr, uptr,wskel, wskeloffset, uskel, uskeloffset, lm,slen);
#ifdef Coarsen
	HGEMM(CacheNear, CacheFar, Proj, nearoffset, faroffset, projoffset,lchildren,rchildren,
		  W,levelset,cidx,U,dim,wptr, uptr,wskel, wskeloffset, uskel, uskeloffset, lm,slen,wpart,clevelset);
#else
//	HGEMM(CacheNear, CacheFar, Proj, nearoffset, faroffset, projoffset,lchildren,rchildren,
//		  W,levelset,idx,U,dim,wptr, uptr,wskel, wskeloffset, uskel, uskeloffset, lm,slen);
//	HGEMM(CacheNear, CacheFar, Proj, nearoffset, faroffset, projoffset,lchildren,rchildren,
//		  W,levelset,cidx,U,dim,wptr, uptr,wskel, wskeloffset, uskel, uskeloffset, lm,slen, nearx, neary, ncount,
//			farx, fary, fcount, utmp, utmpoffset, ftmp, ftmpoffset,
//		  nblockset, nblocks, nxpair, nypair, fblockset, fblocks, fxpair, fypair,
//		  wpart,clevelset);
#endif
	int d=dim;
//	printf("begining...\n");
	double acc=computeError(hmat.lids.data(),(int)hmat.lids.size(),nrhs,n,hmat.setup.X,d,W,U);
	printf("accuracy is %e\n",acc);
}
