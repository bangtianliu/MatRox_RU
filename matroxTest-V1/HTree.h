//
// Created by Bangtian Liu on 6/28/19.
//

#ifndef PROJECT_HTREE_H
#define PROJECT_HTREE_H


#include "Util.h"
#include "../sympiler/HTree.h"
#include "HTree.h"
//#include "hcds.h"
#include <map>
#include <unordered_set>


struct HTree{
	int *sidlen;
	int *sidoffset;
	int *sids;

	int depth;
	int *levelset;
	int *idx;

	int *clevelset;
	int *wpart;
	int *cidx;
	int cdepth;

	int *nblockset;
	int nrow;
	int *nblocks;
	int nb;
	int ncount;
	int *nxval;
	int *nyval;

	int *fblockset;
	int frow;
	int *fblocks;
	int fb;
	int fcount;
	int *fxval;
	int *fyval;

	int *Dim;
	int *lm;
	int nleaf;
	int numnodes;

	int *lids;
	int *lidsoffset;
	int *lidslen;

	int *tlchildren;
	int *trchildren;

	int *leaf;

	double *X;
	Internal::Ktype ktype;
	double h;
	int dim;

	double cdstime=0;
	std::vector<std::pair<double , int>> NN;

	std::vector<std::unordered_set<int>> pnids;
	std::vector<std::map<int, double>> snids;

	std::vector<std::multimap<double, int>> ordered_snids;

};

void readHtree(HTree &tree, bool coarsing, bool cbs)
{
	std::string nn("../sympiler/NN.bin");
	int len = preprocesoffset(nn);
	int *NN = (int *)malloc(sizeof(int)*len);
	double *dist = (double *)malloc(sizeof(double)*len);

	bin2read(nn, NN, len);

	std::string distNN("../sympiler/distNN.bin");
	bin2read(distNN, dist, len);

	tree.NN.resize(len);
#pragma omp parallel for
	for(int i=0; i<len; i++)
	{
		tree.NN[i] = make_pair(dist[i], NN[i]);
	}

	if(coarsing){
		std::string clevel("../sympiler/clevelset.bin");
		int len = preprocesoffset(clevel);
		tree.clevelset = (int *)malloc(sizeof(int)*len);
		bin2read(clevel, tree.clevelset,len);
		tree.cdepth=len-1;

		std::string cidx("../sympiler/cidx.bin");
		len = preprocesoffset(cidx);
		tree.cidx = (int *)malloc(sizeof(int)*len);
		bin2read(cidx, tree.cidx,len);

		std::string wpart("../sympiler/wpart.bin");
		len = preprocesoffset(wpart);
		tree.wpart = (int *)malloc(sizeof(int)*len);
		bin2read(wpart, tree.wpart, len);

		std::string level("../sympiler/levelset.bin");
		len = preprocesoffset(level);
		tree.levelset = (int *)malloc(sizeof(int)*len);
		bin2read(level, tree.levelset, len);
		tree.depth=len-1;

		std::string idx("../sympiler/idx.bin");
		len = preprocesoffset(idx);
		tree.idx = (int *)malloc(sizeof(int)*len);
		bin2read(idx, tree.idx, len);

	} else{
		std::string level("../sympiler/levelset.bin");
		len = preprocesoffset(level);
		tree.levelset = (int *)malloc(sizeof(int)*len);
		bin2read(level, tree.levelset, len);
		tree.depth=len-1;

		std::string idx("../sympiler/idx.bin");
		len = preprocesoffset(idx);
		tree.idx = (int *)malloc(sizeof(int)*len);
		bin2read(idx, tree.idx, len);
	}

	if(cbs){
		std::string nbset("../sympiler/nblockset.bin");
		len = preprocesoffset(nbset);
		tree.nrow=len;
		tree.nblockset = (int *)malloc(sizeof(int)*len);
		bin2read(nbset, tree.nblockset, len);

		std::string nblock("../sympiler/nblocks.bin");
		len = preprocesoffset(nblock);
		tree.nb=len;
		tree.nblocks = (int *)malloc(sizeof(int)*len);
		bin2read(nblock, tree.nblocks, len);

		std::string nxval("../sympiler/nxval.bin");
		len = preprocesoffset(nxval);
		tree.ncount=len;
		tree.nxval = (int *)malloc(sizeof(int)*len);
		bin2read(nxval, tree.nxval,len);

		std::string nyval("../sympiler/nyval.bin");
		len = preprocesoffset(nyval);
		tree.nyval = (int *)malloc(sizeof(int)*len);
		bin2read(nyval, tree.nyval,len);

		std::string fbset("../sympiler/fblockset.bin");
		len = preprocesoffset(fbset);
		tree.frow=len;
		tree.fblockset = (int *)malloc(sizeof(int)*len);
		bin2read(fbset, tree.fblockset, len);

		std::string fblock("../sympiler/fblocks.bin");
		len = preprocesoffset(fblock);
		tree.fb=len;
		tree.fblocks = (int *)malloc(sizeof(int)*len);
		bin2read(fblock, tree.fblocks, len);

		std::string fxval("../sympiler/fxval.bin");
		len = preprocesoffset(fxval);
		tree.fcount=len;
		tree.fxval = (int *)malloc(sizeof(int)*len);
		bin2read(fxval, tree.fxval,len);

		std::string fyval("../sympiler/fyval.bin");
		len = preprocesoffset(fyval);
		tree.fyval = (int *)malloc(sizeof(int)*len);
		bin2read(fyval, tree.fyval,len);
	}
	else {
		std::string nxval("../sympiler/nxval.bin");
		len = preprocesoffset(nxval);
		tree.ncount=len;
		tree.nxval = (int *)malloc(sizeof(int)*len);
		bin2read(nxval, tree.nxval,len);

		std::string nyval("../sympiler/nyval.bin");
		len = preprocesoffset(nyval);
		tree.nyval = (int *)malloc(sizeof(int)*len);
		bin2read(nyval, tree.nyval,len);

		std::string fxval("../sympiler/fxval.bin");
		len = preprocesoffset(fxval);
		tree.fcount=len;
		tree.fxval = (int *)malloc(sizeof(int)*len);
		bin2read(fxval, tree.fxval,len);

		std::string fyval("../sympiler/fyval.bin");
		len = preprocesoffset(fyval);
		tree.fyval = (int *)malloc(sizeof(int)*len);
		bin2read(fyval, tree.fyval,len);
	}

	std::string dim("../sympiler/dim.bin");
	len = preprocesoffset(dim);
	tree.nleaf = len;
	tree.Dim = (int *)malloc(sizeof(int)*len);
	bin2read(dim, tree.Dim, len);

	tree.numnodes = 2*tree.nleaf-1;

	tree.pnids.resize(tree.numnodes);
	tree.snids.resize(tree.numnodes);
	tree.ordered_snids.resize(tree.numnodes);

	std::string lm("../sympiler/leafmap.bin");
	len = preprocesoffset(lm);
	tree.lm = (int *)malloc(sizeof(int)*len);
	bin2read(lm, tree.lm, len);




	tree.leaf = (int *)malloc(sizeof(int)*tree.nleaf);
	memset(tree.leaf, 0, sizeof(int)*tree.nleaf);
	for(int i=0; i<len; i++)
	{
		if(tree.lm[i]!=-1){
			int idx = tree.lm[i];
			tree.leaf[idx] = i;
		}
	}


	std::string lids("../sympiler/lids.bin");
	len = preprocesoffset(lids);
	tree.lids = (int *)malloc(sizeof(int)*len);
	bin2read(lids, tree.lids, len);

	std::string lidsoffset("../sympiler/lidsoffset.bin");
	tree.lidsoffset = (int *)malloc(sizeof(int)*tree.numnodes);
	bin2read(lidsoffset, tree.lidsoffset, tree.numnodes);

	std::string lidslen("../sympiler/lidslen.bin");
	tree.lidslen = (int *)malloc(sizeof(int)*tree.numnodes);
	bin2read(lidslen, tree.lidslen, tree.numnodes);

	std::string tlc("../sympiler/lchildren.bin");
	len = preprocesoffset(tlc);
	tree.tlchildren = (int *)malloc(sizeof(int)*len);
	bin2read(tlc, tree.tlchildren, len);

	std::string trc("../sympiler/rchildren.bin");
	tree.trchildren = (int *)malloc(sizeof(int)*len);
	bin2read(trc, tree.trchildren, len);

}


void coarloadtree(HTree &tree, clustertree &ctree)
{
	ctree.postw.resize(tree.cdepth);
	for(int i=0; i<tree.cdepth; i++)
	{
		int nwparts=tree.clevelset[i+1]-tree.clevelset[i];
		ctree.postw[i].resize(nwparts);
		for(int j=tree.clevelset[i]; j<tree.clevelset[i+1]; j++)
		{
			int loc=j-tree.clevelset[i];
			int nnodes = tree.wpart[j+1] - tree.wpart[j];
			ctree.postw[i][loc].reserve(nnodes);
			for(int k = tree.wpart[j]; k<tree.wpart[j+1]; k++)
			{
				int idx = tree.idx[k];
				ctree.postw[i][loc].push_back(idx);
			}
		}
	}
}

void loadtree(HTree &tree, clustertree &ctree)
{
	ctree.levelsets.resize(tree.depth);
	for(int i=0; i<tree.depth; i++)
	{
		int nnodes = tree.levelset[i+1] - tree.levelset[i];
		ctree.levelsets[i].reserve(nnodes);
		for(int j = tree.levelset[i]; j<tree.levelset[i+1];j++)
		{
			int idx = tree.idx[j];
			ctree.levelsets[i].push_back(idx);
		}
	}
}





#endif //PROJECT_HTREE_H
