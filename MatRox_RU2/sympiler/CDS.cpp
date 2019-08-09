//
// Created by Bangtian Liu on 5/6/19.
//

#include <fstream>
#include "CDS.h"
#include "nUtil.h"

namespace Sympiler {
	namespace Internal {
		void CDS::cacheNear(clustertree &tree, Setup &setup, std::vector<Ret> &tmp){
			auto leaf = tree.leafnodes;
			auto near = tree.nearnodes;
			//auto setup = obj.setup;
			unsigned long int size=0;
			int count=0;
			unsigned long int tmpsize=0;
			this->totalsize=0;
			for(auto &v:leaf){
				auto box=tree.boxes[v];
				count+=near[v].size();
				int dimx=box.getnum();
				for(auto &w:near[v]) {
					auto nbox=tree.boxes[w];
					int dimy = nbox.getnum();
					size += dimx * dimy;
					tmpsize += dimx * setup.nrhs;
				}
			}

			totalsize+=size;
//			printf("near total size=%lu\n", totalsize);
			this->ncount=count;
			this->CacheNear = (double *)mkl_malloc(sizeof(double)*size,64);
			this->ndimx = (int *)mkl_malloc(sizeof(int)*count,64);
			this->ndimy = (int *)mkl_malloc(sizeof(int)*count,64);
			this->nearoffset = (unsigned long int *)mkl_malloc(sizeof(unsigned long int)*count,64);
			this->nidx = (int *)mkl_malloc(sizeof(int)*count, 64);
			this->nidy = (int *)mkl_malloc(sizeof(int)*count, 64);
			this->utmp = (double *) mkl_malloc(sizeof(double)*tmpsize, 64);
			memset(this->utmp, 0, sizeof(double)*tmpsize);

			this->utmpoffset = (unsigned long int *)mkl_malloc(sizeof(unsigned long int)*count,64);
			memset(this->utmpoffset, 0, sizeof(unsigned long int)*count);

			memset(utmp,0,sizeof(double)*tmpsize);
			memset(CacheNear,0,sizeof(double)*size);


			unsigned long int offset=0;
			unsigned long int tmpoffset=0;
			unsigned long int index =0;
			for(auto &v:leaf){
				auto box=tree.boxes[v];
				int dimx=box.getnum();
				auto lidx=box.getlids();
				for(auto &w:near[v]){
					auto boxy=tree.boxes[w];
					int dimy=boxy.getnum();
					auto lidy=boxy.getlids();
					this->nearoffset[index]=offset;
					this->utmpoffset[index]=tmpoffset;
					this->ndimx[index]=dimx;
					this->ndimy[index]=dimy;

					this->nidx[index]=tree.leafoffset.at(v);
					this->nidy[index]=tree.leafoffset.at(w);

					Fsubmatrix(lidx,  lidy, CacheNear+offset,setup);

					offset+=dimx*dimy;
					tmpoffset+=dimx*setup.nrhs;
					++index;
				}
			}
//#ifdef SAVE
//			write2txt("nearoffset.txt",this->nearoffset, count);
//			write2txt("utmpoffset.txt",this->utmpoffset, count);
//			write2binary("near",this->CacheNear, size);
//#endif
		}


		void CDS::cacheFar(clustertree &tree, Setup &setup, std::vector<Ret> &tmp) {
			auto far = tree.farnodes;
			//auto tmp = obj.tmpresult;
			//auto setup = obj.setup;


			unsigned long int size=0;
			int count=0;
			auto numnodes = tree.getnumnodes();

			unsigned long int  tmpsize=0;


			for(int i=1; i<numnodes; i++)
			{
				count+=far[i].size();
				int dimx=tmp[i].skels_length;
				for(auto &w:far[i]){
					int dimy=tmp[w].skels_length;
					size+=dimx*dimy;
					tmpsize+=dimx*setup.nrhs;
				}
			}
			totalsize+=size;

			this->fcount=count;
			this->CacheFar = (double *) mkl_malloc(sizeof(double)*size, 64);
			this->ftmp = (double *)mkl_malloc(sizeof(double)*tmpsize, 64);
			memset(this->ftmp, 0, sizeof(int)*tmpsize);
			this->fdimx = (int *) mkl_malloc(sizeof(int)*count, 64);
			this->fdimy = (int *) mkl_malloc(sizeof(int)*count, 64);
			this->faroffset = (unsigned long int *) mkl_malloc(sizeof(unsigned long int)*count, 64);
			this->ftmpoffset = (unsigned long int *) mkl_malloc(sizeof(unsigned long int)*count, 64);
			this->fidx = (int *)mkl_malloc(sizeof(int)*count, 64);
			this->fidy = (int *)mkl_malloc(sizeof(int)*count, 64);


			memset(this->ftmpoffset, 0, sizeof(int)*count);
			memset(this->faroffset, 0, sizeof(int)*count);

			unsigned long int offset=0;
			unsigned long int tmpoffset=0;

			int index =0;

			for(int i=1; i<numnodes; i++)
			{
				if(!far[i].empty()){
					for(auto &w:far[i]){

						this->fidx[index]=i;
						this->fidy[index]=w;
						this->faroffset[index]=offset;
						this->ftmpoffset[index]=tmpoffset;
						this->fdimx[index]=tmp[i].skels_length;
						this->fdimy[index]=tmp[w].skels_length;
						this->fidx[index]=i;
						this->fidy[index]=w;

						Fsubmatrix(tmp[i].skels, tmp[i].skels_length,tmp[w].skels,tmp[w].skels_length, CacheFar+offset,setup);

//                cachesubmatrix(tmp[i].skels,tmp[i].skels_length,tmp[w].skels, tmp[w].skels_length,CacheFar+offset,setup);
						offset+=(tmp[i].skels_length*tmp[w].skels_length);
						tmpoffset+=tmp[i].skels_length*setup.nrhs;
						++index;
					}
				}
			}
//#ifdef SAVE
//			write2binary("far",this->CacheFar,size);
//			write2txt("faroffset.txt",this->faroffset,count);
//			write2txt("ftmpoffset.txt", this->ftmpoffset,count);
//#endif
		}


		void CDS::cacheNearBlock(clustertree &tree, Setup &setup, std::vector<Ret> &tmp) {
			auto nblocks = tree.nearblocks;
			auto npairs = tree.ncount;
			auto leafnodes = tree.leafnodes;
			nearoffset = (unsigned long int *) mkl_malloc(sizeof(unsigned long int)*npairs, 64);
			memset(nearoffset, 0, sizeof(unsigned long int)*npairs);

			unsigned long int index=0;
			unsigned long int offset=0;

			for(int i=0;i<nblocks.size();i++)
			{
				auto rBlock = nblocks[i];
				for (int j = 0; j < rBlock.size(); ++j) {
					auto Block = rBlock[j];
					for (int k = 0; k < Block.size(); ++k) {
						auto pairs = Block[k];
						auto nx = pairs.first;
						auto ny = pairs.second;
						int dimx = tree.boxes[leafnodes[nx]].getnum();
						int dimy = tree.boxes[leafnodes[ny]].getnum();
						nearoffset[index++] = offset;
						offset+=dimx*dimy;
					}
				}
			}

			CacheNear = (double *)mkl_malloc(sizeof(double)*offset, 64);
			memset(CacheNear, 0, sizeof(double)*offset);

			int size = offset;

			index=0;


			for(int i=0;i<nblocks.size();i++) {
				auto rBlock = nblocks[i];
				for (int j = 0; j < rBlock.size(); ++j) {
					auto Block = rBlock[j];
					for (int k = 0; k < Block.size(); ++k) {
						auto pairs = Block[k];
						auto nx = pairs.first;
						auto ny = pairs.second;

						auto lidx = tree.boxes[leafnodes[nx]].getlids();
						auto lidy = tree.boxes[leafnodes[ny]].getlids();

						auto offset = nearoffset[index++];

						Fsubmatrix(lidx, lidy, CacheNear + offset, setup);
					}
				}
			}
//#ifdef SAVE
//			write2txt("nearoffset.txt",this->nearoffset, index);
//			write2binary("near",this->CacheNear, size);
//#endif

		}

		void CDS::cacheFarBlock(clustertree &tree, Setup &setup, std::vector<Ret> &tmp) {
			auto fblocks = tree.farblocks;
			auto fpairs = tree.fcount;
			faroffset = (unsigned long int *) mkl_malloc(sizeof(unsigned long int)*fpairs, 64);
			memset(faroffset, 0, sizeof(unsigned long int)*fpairs);

			unsigned long int index = 0;
			unsigned long int offset = 0;

			for(int i = 0; i<fblocks.size(); i++)
			{
				auto rBlock = fblocks[i];
				for(int j = 0; j<rBlock.size(); j++)
				{
					auto Block = rBlock[j];
					for(int k = 0; k < Block.size(); k++)
					{
						auto pairs = Block[k];
						auto fx = pairs.first;
						auto fy = pairs.second;
						int dimx = tmp[fx].skels_length;
						int dimy = tmp[fy].skels_length;
						faroffset[index++] = offset;
						offset += dimx * dimy;
					}
				}
			}

			CacheFar = (double *)mkl_malloc(sizeof(double)*offset, 64);
			memset(CacheFar, 0, sizeof(double)*offset);

			int size = offset;

			index = 0;
			for(int i = 0; i<fblocks.size(); i++)
			{
				auto rBlock = fblocks[i];
				for(int j = 0; j<rBlock.size(); j++)
				{
					auto Block = rBlock[j];
					for(int k = 0; k < Block.size(); k++)
					{
						auto pairs = Block[k];
						auto fx = pairs.first;
						auto fy = pairs.second;

						auto offset = faroffset[index++];

						Fsubmatrix(tmp[fx].skels,tmp[fx].skels_length,tmp[fy].skels,tmp[fy].skels_length, CacheFar+offset, setup);
					}
				}
			}
//#ifdef SAVE
//			write2binary("far",this->CacheFar,size);
//			write2txt("faroffset.txt",this->faroffset,index);
//#endif
		}


		void CDS::cacheProj(clustertree &tree, Setup &setup, std::vector<Ret> &tmp)
		{
			auto numnodes = tree.getnumnodes();
//    auto &omplevelsets=tree.getomplevelsets();
			auto &omplevelsets=tree.getlevelsets();
//			printf("#levels=%ld\n",omplevelsets.size());
			this->projoffset=(int *)mkl_malloc(sizeof(int)*numnodes,64);
			memset(this->projoffset,0,sizeof(int)*numnodes);

			int offset=0;
//	printf("DEBUG:::%s %d\n",__FILE__,__LINE__);

			for (int i= 1; i < omplevelsets.size(); ++i) {
				auto &levels=omplevelsets[i];
//		printf("DEBUG:::%s %d l=%d\n",__FILE__,__LINE__,i);
				for(auto &v:levels){
					auto skel = tmp[v].skels_length;
					auto proj = tmp[v].proj_column;
					this->projoffset[v]=offset;
					offset+=skel*proj;
				}
			}
			totalsize+=offset;
//#ifdef SAVE
//			write2txt("projoffset.txt",this->projoffset,numnodes);
//#endif
//	printf("DEBUG:::%s %d offset=%d\n",__FILE__,__LINE__,offset);
			this->Proj=(double *)mkl_malloc(sizeof(double)*offset,64);
//	printf("DEBUG:::%s %d\n",__FILE__,__LINE__);
#pragma omp parallel for
			for (int i= 1; i < omplevelsets.size(); ++i) {
				auto &levels=omplevelsets[i];
				for(auto &v:levels){
					auto proj = this->Proj + this->projoffset[v];
					auto tsize = tmp[v].skels_length*tmp[v].proj_column;
					memcpy(proj,tmp[v].proj, sizeof(double)*tsize);
				}
			}

//#ifdef SAVE
//		write2binary("proj",this->Proj,offset);
//#endif
		}

		void CDS::CoarcacheProj(clustertree &tree, Setup &setup, std::vector<Ret> &tmp) {
			auto numnodes = tree.getnumnodes();
//			auto &posw = tree.opostw;
			auto &posw = tree.opostw;



			this->projoffset=(int *)mkl_malloc(sizeof(int)*numnodes,64);
			memset(this->projoffset,0,sizeof(int)*numnodes);

			int offset=0;

			for(int i=0;i<posw.size();i++)
			{
				auto &lpow=posw[i];
				for(int j=0;j<lpow.size();j++)
				{
					auto wpart=lpow[j];
					for(auto &v:wpart)
					{
						auto skel = tmp[v].skels_length;
						auto proj = tmp[v].proj_column;
						this->projoffset[v]=offset;
						offset+=skel*proj;
					}
				}
			}
//#ifdef SAVE
//			write2txt("projoffset.txt",this->projoffset,numnodes);
//#endif
			totalsize+=offset;
			this->Proj=(double *)mkl_malloc(sizeof(double)*offset, 64);
#pragma omp parallel for
			for(int i=0;i<posw.size();i++)
			{
				auto &lpow=posw[i];
				for(int j=0;j<lpow.size();j++)
				{
					auto wpart=lpow[j];
					for(auto &v:wpart)
					{
						auto proj = this->Proj + this->projoffset[v];
						auto tsize = tmp[v].skels_length*tmp[v].proj_column;
						memcpy(proj,tmp[v].proj, sizeof(double)*tsize);
					}
				}
			}
//#ifdef SAVE
//			write2binary("proj",this->Proj,offset);
//#endif
		}


		void CDS::cacheWUskel(clustertree &tree, Setup &setup, std::vector<Ret> &tmp){

			int size=0;
			//auto tmp=obj.tmpresult;
			auto numnodes = tree.getnumnodes();
//    auto setup=obj.setup;
			this->w_skeloffset=(int *)mkl_malloc(sizeof(int)*numnodes,64);
			memset(this->w_skeloffset,0,sizeof(int)*numnodes);
			this->u_skeloffset=(int *)mkl_malloc(sizeof(int)*numnodes,64);
			memset(this->u_skeloffset,0,sizeof(int)*numnodes);

			int offset=0;
			auto omplevelsets=tree.getlevelsets();

			for (int i= 1; i < omplevelsets.size(); ++i) {
				auto levels=omplevelsets[i];
				for(auto &v:levels){
					auto skel=tmp[v].skels_length;
					auto ncol=setup.nrhs;
					this->w_skeloffset[v]=offset;
					this->u_skeloffset[v]=offset;
					offset+=skel*ncol;
				}
			}
//	totalsize +=offset;

//			printf("total size=%lu\n", totalsize);
			this->w_skel=(double *)mkl_malloc(sizeof(double)*offset,64);
			memset(this->w_skel,0,sizeof(double)*offset);
			this->u_skel=(double *)mkl_malloc(sizeof(double)*offset,64);
			memset(this->u_skel,0,sizeof(double)*offset);

//#ifdef SAVE
//			ofstream out("wuskel.txt");
//			out<<offset;
//			out.close();
//			write2txt("wskeloffset.txt",this->w_skeloffset,numnodes);
//#endif
		}

/**
 * @brief Coarsing version for caching w_skel and u_skel
 * @param tree
 * @param setup
 * @param tmp
 */
		void CDS::
		CoarcacheWUskel(clustertree &tree, Setup &setup, std::vector<Ret> &tmp)
		{
			auto numnodes = tree.getnumnodes();
//			auto &posw = tree.opostw;
			auto &posw = tree.opostw;
//    auto setup=obj.setup;
			this->w_skeloffset=(int *)mkl_malloc(sizeof(int)*numnodes,64);
			memset(this->w_skeloffset,0,sizeof(int)*numnodes);
			this->u_skeloffset=(int *)mkl_malloc(sizeof(int)*numnodes,64);
			memset(this->u_skeloffset,0,sizeof(int)*numnodes);
			int offset=0;
			for(int i=0;i<posw.size();i++)
			{
				auto &lpow=posw[i];
				for(int j=0;j<lpow.size();j++)
				{
					auto wpart=lpow[j];
					for(auto &v:wpart)
					{
						auto skel=tmp[v].skels_length;
						auto ncol=setup.nrhs;
						this->w_skeloffset[v]=offset;
						this->u_skeloffset[v]=offset;
						offset+=skel*ncol;
					}
				}
			}

			this->w_skel=(double *)mkl_malloc(sizeof(double)*offset,64);
			memset(this->w_skel,0,sizeof(double)*offset);
			this->u_skel=(double *)mkl_malloc(sizeof(double)*offset,64);
			memset(this->u_skel,0,sizeof(double)*offset);
#ifdef SAVE
			ofstream out("wuskel.txt");
			out<<offset;
			out.close();
			write2txt("wskeloffset.txt",this->w_skeloffset,numnodes);
#endif

		}




		void CDS::cacheSkeldim(clustertree &tree, Setup &setup, std::vector<Ret> &tmp){

			auto numnodes = tree.getnumnodes();
			// auto tmp=obj.tmpresult;

			this->skel_length=(int *)mkl_malloc(sizeof(int)*numnodes,64);
			memset(this->skel_length,0,sizeof(int)*numnodes);

			this->proj_column=(int *)mkl_malloc(sizeof(int)*numnodes,64);
			memset(this->proj_column,0,sizeof(int)*numnodes);

			for(int i=1;i<numnodes;i++){
				this->skel_length[i]=tmp[i].skels_length;
				this->proj_column[i]=tmp[i].proj_column;
			}
//#ifdef SAVE
//		write2txt("slen.txt",this->skel_length,numnodes);
//#endif

		}


		CDS::CDS(clustertree &tree, Setup &setup, std::vector<Ret> &tmp) {

			auto ntime=0.0;
			auto ftime=0.0;
			auto ptime = 0.0;
			double beg;

			if(setup.use_cbs){
				beg = omp_get_wtime();
//				NearCBSopt(tree, setup, tmp);
				tree.NearBlock(2);
				ntime += (omp_get_wtime() - beg);
				cacheNearBlock(tree, setup, tmp);
				beg = omp_get_wtime();
				tree.FarBlock(4, false);
				ftime += (omp_get_wtime()-beg);
				cacheFarBlock(tree, setup, tmp);
			}
			else{
				cacheNear(tree, setup,tmp);

				cacheFar(tree, setup,tmp);

			}

//			printf("#near nodes=%d\n",this->ncount);
//			printf("#far nodes=%d\n",this->fcount);

			beg = omp_get_wtime();
			if(setup.use_coarsing){
				CoarcacheProj(tree, setup, tmp);
				CoarcacheWUskel(tree, setup, tmp);
			}
			else{
				cacheProj(tree, setup,tmp);
				cacheWUskel(tree,setup,tmp);
			}

			cacheSkeldim(tree,setup,tmp);

			ptime = omp_get_wtime() - beg;

			overhead = ntime + ftime + ptime;
		}
	}
}