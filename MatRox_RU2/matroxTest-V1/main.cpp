//
// Created by Bangtian Liu on 6/28/19.
//
//
// Created by Bangtian Liu on 5/7/19.
//

//#include <HGEMM_v1.h>
#include <HGEMM_gen.h>
//#include <HGEMM_level.h>

#include "Util.h"
#include "../sympiler/HMatrix.h"
//#include "HTree.h"
#include "compression.h"
#include "hcds.h"

using namespace Sympiler::Internal;
using namespace Sympiler;

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
	assert(size1 == n*dim);
	bin2read(pointname, X, size1);

	int nrhs = 2048;
  if(argc>=6) nrhs =atoi(argv[5]);
	auto W = (double *) mkl_malloc(sizeof(double)*n*nrhs, 64);
	memset(W, 0, sizeof(double)*n*nrhs);

	randu(n,nrhs,W, 0.0,1.0);

	auto TW = (double *) mkl_malloc(sizeof(double)*n*nrhs, 64);
	memset(TW, 0, sizeof(double)*n*nrhs);

	auto U = (double *) mkl_malloc(sizeof(double)*n*nrhs, 64);
	memset(U, 0, sizeof(double)*n*nrhs);


    Internal::Ktype kfun=Internal::KS_GAUSSIAN;
	double tau = 3.0;
	mkl_set_dynamic(true);

	bool coarsing=true;
	bool cbs;
#ifdef HSS
	cbs=false;
#else
	cbs=true;
#endif

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
	compression(ctree, tree, rtmp.data(), X, 256, n, kfun, dim, 5, acc);
	hcds cds;
	DUV2CDS(tree, cds, ctree, rtmp.data(),nrhs, coarsing, cbs);
	auto end = omp_get_wtime();
	printf("%f", end-start-tree.cdstime);


	transform(tree, cds, W, n, nrhs, TW);


	start = omp_get_wtime();


#ifdef HSS
	//code for HSS case
	HGEMM(cds.CacheNear, cds.CacheFar, cds.Proj, cds.nearoffset, cds.faroffset, cds.projoffset,
	tree.tlchildren, tree.trchildren, tree.levelset, tree.cidx, TW, U, nrhs, tree.Dim, cds.woffset, cds.uoffset, cds.w_skel,
 cds.w_skeloffset, cds.u_skel, cds.u_skeloffset,tree.lm, cds.skel_length, tree.wpart, tree.clevelset);
#else
	HGEMM(cds.CacheNear, cds.CacheFar, cds.Proj, cds.nearoffset, cds.faroffset, cds.projoffset,
		  tree.tlchildren, tree.trchildren, tree.levelset, tree.cidx, TW, U, nrhs, tree.Dim, cds.woffset, cds.uoffset, cds.w_skel,
		  cds.w_skeloffset, cds.u_skel, cds.u_skeloffset,tree.lm,cds.skel_length, tree.nblockset, tree.nblocks, tree.nxval,
		  tree.nyval, tree.fblockset, tree.fblocks, tree.fxval, tree.fyval, tree.wpart, tree.clevelset);
#endif
	end = omp_get_wtime();

	auto eval = end-start;
	std::cout<<"," << eval;
	int d=dim;
	double acc1=computeError(tree.lids, tree.Dim[0],nrhs,n,X,d,W,U);
	printf(",%e\n",acc1);
}

