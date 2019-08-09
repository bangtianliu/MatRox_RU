//
// Created by Bangtian Liu on 6/8/19.
//

#include <ios>
#include <stdio.h>
#include <fstream>
#include "HTree.h"
#include "NearSearch.h"

namespace Sympiler{
	namespace Internal{
		HTree::HTree(Type t, std::string path, int n, int dim, double tau) {
			config(n, dim, tau);
			LoadPoints(path);
			double start, end;
			start = omp_get_wtime();
			tree=new clustertree(setup);
			end = omp_get_wtime();
//			printf("%f, ",end-start);
			start = omp_get_wtime();
			this->Sampling();
			end = omp_get_wtime();
//			printf("%f, ", end - start);
//			this->SaveSampling();
		}

		void HTree::config(int n, int dim, double tau) {
			setup.n = n;
			setup.d = dim;
			setup.tau = tau;

			setup.adaptive=true;
			setup.m = 256;
			setup.k = 32;
			setup.maxRank = setup.m;
			setup.h=10;
			setup.nrhs = 1; // configure later
			setup.budget = 0;
			setup.recompress = false;
			setup.use_cbs = false;
			setup.use_coarsing = false;
			setup.use_fmm = true;
			if(setup.use_fmm){
				setup.budget = tau;
			}
			setup.equal=true;
		}


		void HTree::LoadPoints(std::string path) {
			setup.X = (double *) mkl_malloc (sizeof(double)*setup.n*setup.d, 64);
			memset(setup.X, 0, sizeof(double)*setup.n*setup.d);
			std::ifstream file( path.data(), std::ios::in|std::ios::binary|std::ios::ate );
			if ( file.is_open() )
			{
				auto size = file.tellg();
				assert( size == setup.n * setup.d * sizeof(double) );
				file.seekg( 0, std::ios::beg );
				file.read( (char*)setup.X, size );
				file.close();
			}
		}

		void HTree::getDecl(std::vector<Expr> &exprList, std::vector<Argument> &argList) {
			DM = "D";
			BM = "B";
			TVM = "VT";
			DM_ptr =  "Dptr";
			BM_ptr = "Bptr";
			TVM_ptr = "VTptr";
			lchildren = "lchildren";
			rchildren = "rchildren";


			argList.push_back(Argument(DM,Argument::Kind::InputBuffer, halide_type_t(halide_type_float,64),1));
			argList.push_back(Argument(BM,Argument::Kind::InputBuffer, halide_type_t(halide_type_float,64),1));
			argList.push_back(Argument(TVM,Argument::Kind::InputBuffer, halide_type_t(halide_type_float,64),1));

			argList.push_back(Argument(DM_ptr,Argument::Kind::InputBuffer, halide_type_t(halide_type_uint,64),1));
			argList.push_back(Argument(BM_ptr,Argument::Kind::InputBuffer, halide_type_t(halide_type_uint,64),1));
			argList.push_back(Argument(TVM_ptr,Argument::Kind::InputBuffer, halide_type_t(halide_type_int,32),1));

			argList.push_back(Argument(lchildren, Argument::Kind::InputBuffer, halide_type_t(halide_type_int,32),1));
			argList.push_back(Argument(rchildren, Argument::Kind::InputBuffer, halide_type_t(halide_type_int,32),1));
		}

		void HTree::StructureNearBlock() {
			auto &blocks=tree->nearblocks;
			int nb = blocks.size();
			int nlnodes = tree->ncount;
			nblockset = new int[nb+1]();
			nxval = new int[nlnodes]();
			nyval = new int[nlnodes]();
//			ds->nl = nb;
			int len = 0;
			int index = 0;
			for(auto &v:blocks)
			{
				nblockset[index++] = len;
				len += v.size();
			}
			nblockset[index] = len;

			nblocks = new int[len+1]();
			index = 0;
			int tidx = 0;
			len = 0;
			for(auto &v:blocks)
			{
				for(auto &w:v)
				{
					nblocks[index++] = len;
					len += w.size();
					for(auto &t:w)
					{
						nxval[tidx] = t.first;
						nyval[tidx++] = t.second;
					}
				}
			}
			nblocks[index] = len;
		}


		void HTree::StructureFarBlock() {
			auto &blocks = tree->farblocks;

			int fb1 = (int)blocks.size();

			int nnodes = tree->fcount;
			this->fblockset = (int *)malloc(sizeof(int)*(fb1+1));
//					new int[(fb1)]();
			fxval = new int[nnodes]();
			fyval = new int[nnodes]();
//			ds->fl = fb;
//			printf("flll=%d\n",fb);

			int len = 0;
			int index = 0;
			for(auto &v:blocks)
			{
				fblockset[index++] = len;
				len += v.size();
			}
			fblockset[index] = len;

			fblocks = new int[len+1]();
			index = 0;
			int tidx = 0;
			len = 0;
			for(auto &v:blocks)
			{
				for(auto &w:v)
				{
					fblocks[index++] = len;
					len += w.size();
					for(auto &t:w)
					{
						fxval[tidx] = t.first;
						fyval[tidx++] = t.second;
					}
				}
			}
			fblocks[index] = len;
		}

		void HTree::StructureNear()
		{
			auto leaf = tree->leafnodes;
			auto near = tree->nearnodes;

			int count=0;

			for(auto &v:leaf){
				count += near[v].size();
			}

			tree->ncount = count;

			nxval = (int *)mkl_malloc(sizeof(int)*count, 64);
			nyval = (int *)mkl_malloc(sizeof(int)*count, 64);

			int index=0;
			for(auto &v:leaf){
				for(auto &w:near[v])
				{
					nxval[index]=leafmap[v];
					nyval[index]=leafmap[w];
					++index;
				}
			}
		}

		void HTree::StructureFar()
		{
			auto far = tree->farnodes;
			int count=0;
			auto numnodes = tree->getnumnodes();

			for(int i=1; i<numnodes; i++)
			{
				count+=far[i].size();
			}

			tree->fcount = count;

			fxval = (int *)mkl_malloc(sizeof(int)*count, 64);
			fyval = (int *)mkl_malloc(sizeof(int)*count, 64);

			int index = 0;

			for(int i=1; i<numnodes; i++)
			{
				for(auto &w:far[i])
				{
					fxval[index]=i;
					fyval[index]=w;

					++index;
				}
			}
		}

		void HTree::Sampling() {

			auto beg = omp_get_wtime();
			NearSearch(setup);
			auto stop = omp_get_wtime();

		}


		void HTree::SaveSampling() {
			int *NN = (int *)malloc(sizeof(double)*setup.k*setup.n);

			double *distNN = (double *)malloc(sizeof(double)*setup.k*setup.n);

			for(int i=0; i<setup.n; i++)
			{
				for(int j=0; j<setup.k; j++)
				{
					distNN[i*setup.k+j] = setup.NN[i*setup.k+j].first;
					NN[i*setup.k+j]=setup.NN[i*setup.k+j].second;
				}
			}

			write2binary("distNN.bin",distNN, setup.k*setup.n);
			write2binary("NN.bin",NN, setup.k*setup.n);

//			auto &sflevelsets=tree->getsflevelsets();
//			auto numnodes=tree->getnumnodes();
//			auto &tordersnides = tree->getordersnids();
//
//			auto sidoffset = (int *)malloc(sizeof(int)*numnodes);
//			memset(sidoffset, 0, sizeof(int)*numnodes);
//			auto sidlen = (int *)malloc(sizeof(int)*numnodes);
//			memset(sidlen, 0, sizeof(int)*numnodes);
//
//			int offset = 0;
//			for (int i = 0; i < sflevelsets.size()-1; ++i) {
//				auto &levels=sflevelsets[i];
//				for(int j=0; j<levels.size(); j++)
//				{
//					auto idx = levels[j];
////					printf("idx=%d len=%lu\n",idx, tordersnides[idx].size());
//					sidlen[idx]=tordersnides[idx].size();
//					sidoffset[idx]=offset;
//					offset+=tordersnides[idx].size();
//				}
//			}
//
////			write2txt("sidoffset.txt", sidoffset, numnodes);
////			write2txt("sidlen.txt", sidlen, numnodes);
//
//			writeoffset2binary("sidoffset.bin", sidoffset, numnodes);
//			writeoffset2binary("sidlen.bin", sidlen, numnodes);
//
//			auto sids = (int *)malloc(sizeof(int)*offset);
//			memset(sids, 0, sizeof(int)*offset);
//
//			for (int i = 0; i < sflevelsets.size()-1; ++i) {
//				auto &levels=sflevelsets[i];
//				for(int j=0; j<levels.size(); j++)
//				{
//					auto idx = levels[j];
////					auto len = tordersnides[idx].size();
//					auto loc = sidoffset[idx];
//					for(auto cur=tordersnides[idx].begin(); cur!=tordersnides[idx].end(); cur++)
//					{
//						sids[loc++]=cur->second;
//					}
//				}
//			}
//
////			write2txt("sid.txt", sids, offset);
//			writeoffset2binary("sids.bin", sids, offset);
		}
		void HTree::savetree() {
			bool flag = setup.use_coarsing;

			if(flag){
				auto &postw = tree->getpostw();
				int len = postw.size();

				clevelset = (int *)mkl_malloc(sizeof(int)*(len+1), 64);
				int numnodes = tree->getnumnodes();
				cidx = (int *)mkl_malloc(sizeof(int)*(numnodes-1),64);

				len = 0;
				int index=0;
				for(auto &v:postw){
					clevelset[index++]=len;
					len += v.size();
				}
				clevelset[index] = len;

				wpart = (int *)mkl_malloc(sizeof(int)*(len+1), 64);
				len=0;
				index=0;
				int tidx=0;
				for(auto &v:postw)
				{
					for(auto &w:v)
					{
						wpart[index++] = len;
						len += w.size();
						for(auto &t:w){
							cidx[tidx++]=t;
						}
					}
				}
				wpart[index] = len;

				auto &sflevelsets = tree->getsflevelsets();
//				auto numnodes=tree->getnumnodes();
				levelset = (int *)mkl_malloc(sizeof(int)*sflevelsets.size(), 64);
				levelset[0] = 0;
				idx = (int *)mkl_malloc(sizeof(int)*(numnodes-1),64);
				int k=0;
				int l=1;

				for(int i=sflevelsets.size()-2; i>=0;i--)
				{
					auto &levels=sflevelsets[i];
					for(int j=0; j<levels.size(); j++)
					{
						idx[k++]=levels[j];
					}
					levelset[l] = levelset[l-1]+(int)levels.size();
					l++;
				}

			}
			else {
				auto &sflevelsets = tree->getlevelsets();
				auto numnodes=tree->getnumnodes();
				levelset = (int *)mkl_malloc(sizeof(int)*sflevelsets.size(), 64);
				levelset[0] = 0;
				idx = (int *)mkl_malloc(sizeof(int)*(numnodes-1),64);
				int k=0;
				int l=1;

				for(int i=1; i<sflevelsets.size(); i++)
				{
					auto &levels=sflevelsets[i];
					for(int j=0; j<levels.size(); j++)
					{
						idx[k++]=levels[j];
					}
					levelset[l] = levelset[l-1]+(int)levels.size();
					l++;
				}
			}

			int numnodes = tree->getnumnodes();

			tlchildren=(int *)malloc(sizeof(int)*numnodes);
			trchildren=(int *)malloc(sizeof(int)*numnodes);

			for(int i=0;i<numnodes;i++){
				tlchildren[i]=-1;
				trchildren[i]=-1;
			}

			auto sflevelsets = tree->getsflevelsets();
			auto children = tree->getchildren();
			for(int i =sflevelsets.size()-1;i>=0;--i)
			{
				auto &levels = sflevelsets[i];
				for(int j=0; j<levels.size();j++)
				{
					auto v=levels[j];
					if(children[v].size()!=0){
						tlchildren[v]=children[v][0];
						trchildren[v]=children[v][1];
					}
				}
			}

			leafmap = (int *)malloc(sizeof(int)*numnodes);

			for(int i=0; i<numnodes; i++)
			{
				leafmap[i]=-1;
			}

			auto &leaf=tree->getleafoffset();
			auto &levels=sflevelsets[0];

			Dim=(int *)malloc(sizeof(int)*levels.size());
			auto boxes=tree->getboxes();

//			lids = (int *)malloc(sizeof(int)*setup.n);
//			lidsoffset = (int *)malloc(sizeof(int)*levels.size());
//			int offset=0;

			for(int i=0;i<levels.size();i++)
			{
				auto v = levels[i];
				auto idy = leaf.at(v);
				leafmap[v]=idy;
				Dim[idy] = boxes[v].getnum();
//				memcpy(lids+offset, boxes[v].getlids().data(), sizeof(int)*Dim[idy]);
//				lidsoffset[i]=offset;
//				offset+=Dim[idy];
			}

			int count=0;
			int offset=0;
			lidsoffset = (int *)malloc(sizeof(int)*numnodes);
			memset(lidsoffset, 0, sizeof(int)*numnodes);

			lidslen = (int *)malloc(sizeof(int)*numnodes);
			memset(lidslen, 0, sizeof(int)*numnodes);

			for(int i = 0;i<sflevelsets.size()-1;++i) {
				auto &levels = sflevelsets[i];
				for (int j = 0; j < levels.size(); j++) {
					auto v = levels[j];
					lidsoffset[v]=offset;
					lidslen[v] = boxes[v].getnum();
					offset += boxes[v].getnum();
				}
			}

			lids = (int *)malloc(sizeof(int)*offset);
			nlen=offset;
			for(int i = 0;i<sflevelsets.size()-1;++i) {
				auto &levels = sflevelsets[i];
				for (int j = 0; j < levels.size(); j++) {
					auto v = levels[j];
					auto offset = lidsoffset[v];
					memcpy(lids+offset, boxes[v].getlids().data(), sizeof(int)*lidslen[v]);
				}
			}

			flag = setup.use_cbs;
			if(flag){
				StructureNearBlock();
				StructureFarBlock();
			}
			else{
				StructureNear();
				StructureFar();
			}
		}

		void HTree::savetree2disk() {
			bool flag = setup.use_coarsing;

			if(flag){
				auto &postw = tree->getpostw();
				int len = postw.size();
				writeoffset2binary("clevelset.bin",clevelset, len+1);
				writeoffset2binary("cidx.bin", cidx, tree->getnumnodes()-1);

				len = 0;
				for(auto &v:postw)
				{
					len += v.size();
				}
				writeoffset2binary("wpart.bin", wpart, len+1);

				auto &sflevelsets = tree->getlevelsets();
				writeoffset2binary("levelset.bin", levelset, sflevelsets.size());
				writeoffset2binary("idx.bin", idx, tree->getnumnodes()-1);
//				write2txt("levelset.txt",levelset,sflevelsets.size());
			}
			else {
				auto &sflevelsets = tree->getlevelsets();
				writeoffset2binary("levelset.bin", levelset, sflevelsets.size());
				writeoffset2binary("idx.bin", idx, tree->getnumnodes()-1);
			}

			writeoffset2binary("lchildren.bin", tlchildren, tree->getnumnodes());
			writeoffset2binary("rchildren.bin", trchildren, tree->getnumnodes());

			flag = setup.use_cbs;

			if(flag){
				int len=0;
				for(auto &v: tree->nearblocks)
				{
					len += v.size();
				}

				writeoffset2binary("nblockset.bin", nblockset, tree->nearblocks.size()+1);
				writeoffset2binary("nblocks.bin", nblocks, len+1);
				writeoffset2binary("nxval.bin", nxval, tree->ncount);
				writeoffset2binary("nyval.bin", nyval, tree->ncount);

				len=0;
				for(auto &v : tree->farblocks)
				{
					len+=v.size();
				}

//				printf("Far block size=%d len=%d count=%d\n", tree->farblocks.size()+1, len, tree->fcount);

				writeoffset2binary("fblockset.bin", fblockset, tree->farblocks.size()+1);
				writeoffset2binary("fblocks.bin", fblocks, len+1);
				writeoffset2binary("fxval.bin", fxval, tree->fcount);
				writeoffset2binary("fyval.bin", fyval, tree->fcount);
			}
			else {
				writeoffset2binary("nxval.bin", nxval, tree->ncount);
				writeoffset2binary("nyval.bin", nyval, tree->ncount);
				writeoffset2binary("fxval.bin", fxval, tree->fcount);
				writeoffset2binary("fyval.bin", fyval, tree->fcount);
			}

			auto &sflevelset=tree->getsflevelsets();
			auto &levels = sflevelset[0];

			writeoffset2binary("dim.bin", Dim, levels.size());
			writeoffset2binary("leafmap.bin", leafmap, tree->getnumnodes());

			writeoffset2binary("lids.bin", lids, nlen);
			writeoffset2binary("lidsoffset.bin", lidsoffset, tree->getnumnodes());
			writeoffset2binary("lidslen.bin", lidslen, tree->getnumnodes());
		}


		HTree::~HTree() {
			delete tree;
		}

	}
}
