//
// Created by Bangtian Liu on 5/12/19.
//

#include "DSInspector.h"


namespace Sympiler {
	namespace Internal {
		StructureObject::StructureObject() {
				levelPtr = "clevelset";
				wpartPtr = "wpart";
				idxPtr = "cidx";

				nblocksetPtr = "nblockSet";
				nblocksPtr = "nblocks";
				nxvalPtr = "npairx";
				nyvalPtr = "npairy";

				fblocksetPtr = "fblockSet";
				fblocksPtr = "fblocks";
				fxvalPtr = "fpairx";
				fyvalPtr = "fpairy";
		}

		StructureObject::~StructureObject() {
				delete [] (levelSet);
				delete [] (wpart);
				delete [] (idx);

//				delete [] (nblockset);
//				delete [] (nblocks);
//				delete [] (nxval);
//				delete [] (nyval);
//
//				delete [] (fblockset);
//				delete [] (fblocks);
//				delete [] (fxval);
//				delete [] (fyval);
		}

		DSInspector::DSInspector() {
			ds = new StructureObject();
		}

		DSInspector::~DSInspector() {
			delete ds;
		}

		bool DSInspector::isHSS(clustertree *tree) {
			return tree->isHSS();
		}

		bool DSInspector::isPerfect(clustertree *tree) {
			return tree->isPerfect();
		}

		HGEMMInspector::HGEMMInspector(HTree *tree) :
			DSInspector(),htree(tree)
		{

		}

		StructureObject* HGEMMInspector::Inspect() {
			auto start = omp_get_wtime();
      CoarsenLevelSetDetection();
			StructureCoarsen(htree->tree->postw);
      auto end1 = omp_get_wtime();
      //printf("coarsening time is %f\n", end1 - start);
			ds->isHSS = isHSS(htree->tree);
			htree->setup.use_coarsing= true;
			if(!ds->isHSS){
        auto start = omp_get_wtime();
				BlockSetDetection();
				StructureNearBlock(htree->tree->nearblocks);
				StructureFarBlock(htree->tree->farblocks);
				htree->setup.use_cbs=true;
        auto end1 = omp_get_wtime();
        //printf("blocking time is %f\n", end1-start);
			}

			//Data layout transformation
//			A->cds = new CDS(*A->tree, A->setup, A->tmpresult);

			return ds;
		}


		void HGEMMInspector::CoarsenLevelSetDetection() {
			auto tree=htree->tree;
			if(isPerfect(tree)){
				tree->parlevel(1,1);
				tree->genpostw();
			}
			else{
				tree->parlevel(1,1);
				tree->imgenpostw();
			}

//			auto len = tree->postw.size();
//			tree->opostw.resize(len);
//
//			auto &pow=tree->postw;
//			auto &opow=tree->opostw;

//			for(int i=0;i<pow.size();i++){
//				auto &lpow=pow[i];
//				int nw = lpow.size();
//
//				int nparts;
//
//				int nthreads=omp_get_max_threads();
//
//				if(nw>=nthreads){
//					nparts=nthreads;
//				}
//				else {
//					nparts=nw/2;
//				}
//				if(nparts==0)nparts=1;
//				opow[i].resize(nparts);
//
//				A->binpacking(lpow,opow[i],nparts);
//
//			}

//			tree->coarmapskelidx();
		}


		void HGEMMInspector::StructureCoarsen(std::vector<std::vector<std::vector<int>>> &postw) {
			int len = postw.size();
			ds->cdepth=len;

			ds->levelSet = new int[len+1]();
			int numnodes = htree->tree->getnumnodes();

			ds->idx = new int[numnodes-1]();

			len = 0;
			int index=0;

			for(auto &v:postw){
				ds->levelSet[index++]=len;
				len += v.size();
			}
			ds->levelSet[index] = len;
//#ifdef SAVE
//			write2txt("clevelset.txt",ds->levelSet,postw.size()+1);
//#endif
			ds->wpart = new int[len+1]();
			int slenwpart = len + 1;
			len=0;
			index=0;
			int tidx=0;
			for(auto &v:postw)
			{
				for(auto &w:v)
				{
					ds->wpart[index++] = len;
					len += w.size();
					for(auto &t:w){
//						printf("%d\t",t);
						ds->idx[tidx++]=t;
					}
//					printf("\n");
				}
			}
			ds->wpart[index] = len;
//#ifdef SAVE
//			write2txt("wpart.txt",ds->wpart,slenwpart);
//			write2txt("cidx.txt", ds->idx, numnodes-1);
//#endif
		}

		void HGEMMInspector::BlockSetDetection() {
			auto &tree = htree->tree;

			tree->NearBlock(2); // tuning parameter
//			tree->FarBlock(4,A->setup.use_coarsing);
			tree->FarBlock(4, false);
		}

		void HGEMMInspector::StructureNearBlock(std::vector<std::vector<std::vector<pair<int,int>>>> &nblocks) {
			int nb = nblocks.size();
			int nlnodes = htree->tree->ncount;
			ds->nblockset = new int[nb+1]();
			ds->nxval = new int[nlnodes]();
			ds->nyval = new int[nlnodes]();

			ds->nl = nb;

			int len = 0;
			int index = 0;
			for(auto &v:nblocks)
			{
				ds->nblockset[index++] = len;
				len += v.size();
			}
			ds->nblockset[index] = len;
#ifdef SAVE
			write2txt("nblockset.txt",ds->nblockset,nb+1);
#endif
			ds->nblocks = new int[len+1]();


			index = 0;
			int tidx = 0;
			len = 0;
			for(auto &v:nblocks)
			{
				for(auto &w:v)
				{
					ds->nblocks[index++] = len;
					len += w.size();
					for(auto &t:w)
					{
						ds->nxval[tidx] = t.first;
						ds->nyval[tidx++] = t.second;
					}
				}
			}

			ds->nblocks[index] = len;
#ifdef SAVE
			write2txt("nblock.txt",ds->nblocks,index+1);
			write2txt("nxbpair.txt", ds->nxval, nlnodes);
			write2txt("nybpair.txt", ds->nyval, nlnodes);
#endif
		}

		void HGEMMInspector::StructureFarBlock(std::vector<std::vector<std::vector<pair<int,int>>>> &fblocks) {
			int fb = fblocks.size();
			int nnodes = htree->tree->fcount;

			ds->fblockset = new int[fb+1]();
			ds->fxval = new int[nnodes]();
			ds->fyval = new int[nnodes]();
			ds->fl = fb;

			int len = 0;
			int index = 0;

			for(auto &v:fblocks)
			{
				ds->fblockset[index++] = len;
				len += v.size();
			}

			ds->fblockset[index] = len;
#ifdef SAVE
			write2txt("fblockset.txt",ds->fblockset,fb+1);
#endif

			ds->fblocks = new int[len+1]();

			index = 0;
			int tidx = 0;
			len = 0;

			for(auto &v:fblocks)
			{
				for(auto &w:v)
				{
					ds->fblocks[index++] = len;
					len += w.size();
					for(auto &t:w)
					{
						ds->fxval[tidx] = t.first;
						ds->fyval[tidx++] = t.second;
					}
				}
			}
			ds->fblocks[index] = len;
#ifdef SAVE
		write2txt("fblock.txt",ds->fblocks,index+1);
		write2txt("fxbpair.txt",ds->fxval, nnodes);
		write2txt("fybpair.txt",ds->fyval, nnodes);
#endif
		}

		HGEMMInspector::~HGEMMInspector() {

		}
	}
}
