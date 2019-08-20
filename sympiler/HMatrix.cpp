//
// Created by Bangtian Liu on 4/28/19.
//

#include <ios>
#include <fstream>
#include "HMatrix.h"
#include "nUtil.h"
#include "ReNearFar.h"

namespace Sympiler {
	namespace Internal{
		HMatrix::HMatrix(Type t, std::string path, int n, int dim, Ktype ktype, double tau, double acc) : Dense(t, path) {
//			std::string mnumber = std::to_string(mNo);
			config(n,dim,ktype,tau,acc);
			LoadPoints(path);
			tree = new clustertree(setup);
//			this->compression();
//			//will be moved into other Parts.
//			cds = new CDS(*tree, setup, tmpresult);
			if(tree->isHSS()){
				setup.use_coarsing= true;
			}
			else {
				setup.use_coarsing=true;
				setup.use_cbs = true;
			}


			if(setup.use_coarsing){
				if(tree->isPerfect()){
					tree->parlevel(1,1);
					tree->genpostw();
				}
				else{
					tree->parlevel(2,2);
					tree->imgenpostw();
				}
			}


		}



		void HMatrix::config(int n, int dim, Ktype ktype, double tau, double acc) {
			setup.n = n;
			setup.d = dim;
			setup.ktype = ktype;
			setup.tau = tau;
			setup.stol = acc;
			setup.adaptive=true;
			setup.m = 256;
			setup.maxRank = setup.m;
			setup.h=10;
			setup.nrhs = 1; // configure later
			setup.recompress = false;
			setup.use_cbs = false;
			setup.use_coarsing = false;
			setup.use_fmm = false;
			setup.equal = false;
			setup.nthreads = omp_get_max_threads();
		}

		void HMatrix::getDecl(std::vector<Expr> &exprList, std::vector<Argument> &argList) {
			DM = Matrix::name + "D";
			BM = Matrix::name + "B";
			TVM = Matrix::name + "VT";
			DM_ptr = Matrix::name + "Dptr";
			BM_ptr = Matrix::name + "Bptr";
			TVM_ptr = Matrix::name + "VTptr";
			lchildren = "lchildren";
			rchildren = "rchildren";


			argList.push_back(Argument(DM,Argument::Kind::InputBuffer, halide_type_t(halide_type_float,64),1));
			argList.push_back(Argument(BM,Argument::Kind::InputBuffer, halide_type_t(halide_type_float,64),1));
			argList.push_back(Argument(TVM,Argument::Kind::InputBuffer, halide_type_t(halide_type_float,64),1));

			argList.push_back(Argument(DM_ptr,Argument::Kind::InputBuffer, halide_type_t(halide_type_int,32),1));
			argList.push_back(Argument(BM_ptr,Argument::Kind::InputBuffer, halide_type_t(halide_type_int,32),1));
			argList.push_back(Argument(TVM_ptr,Argument::Kind::InputBuffer, halide_type_t(halide_type_int,32),1));

			argList.push_back(Argument(lchildren, Argument::Kind::InputBuffer, halide_type_t(halide_type_int,32),1));
			argList.push_back(Argument(rchildren, Argument::Kind::InputBuffer, halide_type_t(halide_type_int,32),1));
		}

		void HMatrix::LoadPoints(std::string path) {
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

//#ifdef SAVE
//			write2binary("xbin", setup.X,setup.n*setup.d);
//#endif
		}


		void HMatrix::compression() {
			auto &sflevelsets=tree->getsflevelsets();
			auto &boxes=tree->getboxes();
			auto numnodes=tree->getnumnodes();
			auto children=tree->getchildren();
			auto parent=tree->getparent();
			tmpresult.resize(numnodes);
			int *task = new int[numnodes];

//#pragma omp parallel
//#pragma omp single
			{
				auto &levels = sflevelsets[0];
#pragma omp parallel for
				for (int j = 0; j < levels.size(); j++) {
					auto idx = levels[j];
					auto box = boxes[idx];
//#pragma omp task depend(out:task[idx])
					{
						skeletonize(idx, box, tmpresult.data());
					}
				}
				for (int i = 1; i < sflevelsets.size() - 1; ++i) {
					auto &levels = sflevelsets[i];
#pragma omp parallel for
					for (int j = 0; j < levels.size(); j++) {
//            for(std::set<int>::iterator it=levels.begin(); it!=levels.end(); ++it){
						auto idx = levels[j];
						auto box = boxes[idx];
						auto beg = children[idx].front();
						auto end = children[idx].back();
//#pragma omp task depend(in:task[beg], task[end]) depend(out:task[idx])
						skeletonize(idx, box, tmpresult.data());
					}
				}
			}

		}


		void HMatrix::compression(int *sid, int *sidlen, int *sidoffset, int *levelset, int *idx, int len)
		{
			auto &sflevelsets=tree->getsflevelsets();
			auto &boxes=tree->getboxes();
			auto numnodes=tree->getnumnodes();
			auto children=tree->getchildren();
			auto parent=tree->getparent();
			tmpresult.resize(numnodes);
			int *task = new int[numnodes];


			for(int j=levelset[len-1]; j<levelset[len]; j++)
			{
				auto id = idx[j];
				auto box = boxes[id];
				skeletonize(id, box, tmpresult.data(),sid, sidlen, sidoffset);
			}


			for(int i=len-2; i>-1; i--)
			{
				auto ti=i+1;
				for(int j=levelset[i]; j<levelset[ti]; j++)
				{
					auto id = idx[j];
					auto box = boxes[id];
					skeletonize(id, box, tmpresult.data(),sid, sidlen, sidoffset);
				}
			}

//#pragma omp parallel
//#pragma omp single
//			{
//				auto &levels = sflevelsets[0];
////#pragma omp parallel for
//				for (int j = 0; j < levels.size(); j++) {
//					auto idx = levels[j];
//					auto box = boxes[idx];
////#pragma omp task depend(out:task[idx])
//					{
//						skeletonize(idx, box, tmpresult.data(),sid, sidlen, sidoffset);
//					}
//				}
//				for (int i = 1; i < sflevelsets.size() - 1; ++i) {
//					auto &levels = sflevelsets[i];
////#pragma omp parallel for
//					for (int j = 0; j < levels.size(); j++) {
////            for(std::set<int>::iterator it=levels.begin(); it!=levels.end(); ++it){
//						auto idx = levels[j];
//						auto box = boxes[idx];
//						auto beg = children[idx].front();
//						auto end = children[idx].back();
////#pragma omp task depend(in:task[beg], task[end]) depend(out:task[idx])
//						skeletonize(idx, box, tmpresult.data(), sid, sidlen, sidoffset);
//					}
//				}
//			}
		}


		void HMatrix::skeletonize(int idx, boundingbox &box, Ret *rtmp) {
			if(idx==0){
				return;
			}

			auto children = tree->getchildren();

			std::vector<int> amap;
			std::vector<int> bmap;

			if(children[idx].size()==0){
				bmap=box.getlids();
			}
			else{
				for(auto &v:children[idx]){
//					printf("#v=%d\n",v);
					bmap.insert(bmap.end(),rtmp[v].skels,rtmp[v].skels+rtmp[v].skels_length);
				}
			}

			auto nsamples  = 2 * bmap.size();
			auto numpoints = box.getnum();
			auto clids = box.getlids();
			nsamples = (nsamples < 2*setup.m)? 2*setup.m: nsamples; // number of sampling
/**
 * todo: will add other sampling options
 */
			mt19937 generator(idx);
			uniform_int_distribution<> uniform_distribution(0, setup.n-1);
			if(nsamples<(setup.n-numpoints)){
				while(amap.size()<nsamples) {
//					auto sample = rand() % setup.n;
					auto sample = uniform_distribution(generator);
					if (std::find(amap.begin(), amap.end(), sample) == amap.end() &&
						std::find(clids.begin(), clids.end(), sample) == clids.end()) {
						amap.push_back(sample);
					}
				}
			}
			else {
				for ( int sample = 0; sample < setup.n; sample ++ )  // TODO: may can be improved here
				{
					if ( std::find( amap.begin(), amap.end(), sample ) == amap.end() )
					{
						amap.push_back( sample );
					}
				}
			}
			auto Kab = (double *)malloc(sizeof(double)*amap.size()*bmap.size());
			memset(Kab,0,sizeof(double)*amap.size()*bmap.size());

			Fsubmatrix(amap, bmap, Kab, setup);

			auto N = setup.n;
			auto m = amap.size();
			auto n = bmap.size();
			auto q = clids.size();

//			printf("#m=%d n=%d\n",m,n);

			auto tolerance = std::sqrt((m*n*1.0)/(N*(N-q)))*setup.stol;

			int *skels;
			double *proj;
			int *jpvt;

			int s = decomposition(Kab, m, n, tolerance, &skels, &proj, &jpvt, setup);

			for (int i = 0; i < s; ++i) {  // need to check it
				skels[i] = bmap[skels[i]];
			}

			rtmp[idx].skels = skels;
			rtmp[idx].skels_length=s;
			rtmp[idx].proj=proj;
			rtmp[idx].proj_column=(int)bmap.size();
			free(Kab);

		}


		void HMatrix::skeletonize(int idx, boundingbox &box, Ret *rtmp, int *sid, int *sidlen, int *sidoffset) {
			if(idx==0){
				return;
			}

			auto children = tree->getchildren();

//			std::vector<int> amap;
			std::vector<int> bmap;

			if(children[idx].size()==0){
				bmap=box.getlids();
			}
			else{
				for(auto &v:children[idx]){
//					printf("#v=%d\n",v);
					bmap.insert(bmap.end(),rtmp[v].skels,rtmp[v].skels+rtmp[v].skels_length);
				}
			}

			auto nsamples  = 2 * bmap.size();
			auto numpoints = box.getnum();
			auto clids = box.getlids();
			nsamples = (nsamples < 2*setup.m)? 2*setup.m: nsamples; // number of sampling
/**
 * todo: will add other sampling options
 */

			int slen = sidlen[idx];
			int offset = sidoffset[idx];

			std::vector<int> amap(sid+offset, sid+offset+slen);

			mt19937 generator(idx);
			uniform_int_distribution<> uniform_distribution(0, setup.n-1);
			if(nsamples<(setup.n-numpoints)){
				while(amap.size()<nsamples) {
//					auto sample = rand() % setup.n;
					auto sample = uniform_distribution(generator);
					if (std::find(amap.begin(), amap.end(), sample) == amap.end() &&
						std::find(clids.begin(), clids.end(), sample) == clids.end()) {
						amap.push_back(sample);
					}
				}
			}
			else {
				for ( int sample = 0; sample < setup.n; sample ++ )  // TODO: may can be improved here
				{
					if ( std::find( amap.begin(), amap.end(), sample ) == amap.end() )
					{
						amap.push_back( sample );
					}
				}
			}
			auto Kab = (double *)malloc(sizeof(double)*amap.size()*bmap.size());
			memset(Kab,0,sizeof(double)*amap.size()*bmap.size());

			Fsubmatrix(amap, bmap, Kab, setup);

			auto N = setup.n;
			auto m = amap.size();
			auto n = bmap.size();
			auto q = clids.size();

//			printf("#m=%d n=%d\n",m,n);

			auto tolerance = std::sqrt((m*n*1.0)/(N*(N-q)))*setup.stol;

			int *skels;
			double *proj;
			int *jpvt;

			int s = decomposition(Kab, m, n, tolerance, &skels, &proj, &jpvt, setup);

			for (int i = 0; i < s; ++i) {  // need to check it
				skels[i] = bmap[skels[i]];
			}

			rtmp[idx].skels = skels;
			rtmp[idx].skels_length=s;
			rtmp[idx].proj=proj;
			rtmp[idx].proj_column=(int)bmap.size();
			free(Kab);

		}

		void HMatrix::binpacking(std::vector<std::vector<int>> &wpartitions,
								 std::vector<std::vector<int>> &owpartitions, int numofbins)
		{
			auto &children = tree->getchildren();
			auto &leafmap = tree->getleafoffset();
//	auto &boxes = tree.getboxes();
			Dcost *ccost = new Dcost[wpartitions.size()];

			for(auto i = 0; i<wpartitions.size(); i++)
			{
				ccost[i].cost=0;
				ccost[i].index=i;
				for(auto j=0; j<wpartitions[i].size(); j++)
				{
					auto idx = wpartitions[i][j];
					unsigned long cost = 0;
					if(children[idx].empty())
					{
						cost+= 2* tmpresult[idx].skels_length * tmpresult[idx].proj_column * setup.nrhs;
					}
					else {
						for(auto &v : children[idx])
						{
							cost += 2*tmpresult[idx].skels_length*tmpresult[v].skels_length*setup.nrhs;
						}
					}
					ccost[i].cost+=cost;
				}
			}

			std::sort(ccost,ccost+wpartitions.size(),compare);

			uint64_t *ocost = new uint64_t[numofbins];

			memset(ocost, 0, sizeof(uint64_t)*numofbins);

			int partNo = wpartitions.size();

			int minBin = 0;
			for(int i = 0; i < partNo; i++)
			{
				minBin = findMin(ocost,numofbins);
				ocost[minBin] += ccost[i].cost;
				int index = ccost[i].index;
//owpartition

				owpartitions[minBin].insert(owpartitions[minBin].end(),
											wpartitions[index].begin(), wpartitions[index].end());
			}
		}

		void HMatrix::Setnrhs(int nCols) {
//			dPattern->setNrhs(nCols);
			setup.nrhs = nCols;
		}


		void HMatrix::savetree() {
			auto sflevelsets=tree->getsflevelsets();
			auto numnodes=tree->getnumnodes();
			auto boxes=tree->getboxes();
			auto children=tree->getchildren();
			leafmap = (int *)malloc(sizeof(int)*numnodes);
#pragma omp parallel for
			for(int i=0; i<numnodes; i++)
			{
				leafmap[i]=-1;
			}

			auto &leaf=tree->getleafoffset();
			auto &levels=sflevelsets[0];

			Dim=(int *)malloc(sizeof(int)*levels.size());

			for(int i=0;i<levels.size();i++)
			{
				auto v = levels[i];
				auto idy = leaf.at(v);
				leafmap[v]=idy;
				Dim[idy] = boxes[v].getnum();
			}

//			write2txt("lm.txt",leafmap,numnodes);
//			write2txt("dim.txt",Dim, (int)levels.size());

			if(setup.use_coarsing){
				auto &postw=tree->getopostw();
				int len = postw.size();
				clevelset = (int *)mkl_malloc(sizeof(int)*(len+1), 64);
				int numnodes = tree->getnumnodes();
				idx = (int *)mkl_malloc(sizeof(int)*(numnodes-1),64);

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
							idx[tidx++]=t;
						}
					}
				}
				wpart[index] = len;
			}
			else {
				auto &sflevelsets = tree->getlevelsets();
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

			tlchildren=(int *)malloc(sizeof(int)*numnodes);
			trchildren=(int *)malloc(sizeof(int)*numnodes);
			for(int i=0;i<numnodes;i++){
				tlchildren[i]=-1;
				trchildren[i]=-1;
			}

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

			auto leafnodes = tree->getleaf();
//			auto boxes = this->getboxes();
			auto nl = leafnodes.size();

			woffset = (int *)mkl_malloc(sizeof(int)*nl,64);
			uoffset = (int *)mkl_malloc(sizeof(int)*nl,64);

			int offset = 0;

			for (int i = 0; i < leafnodes.size(); ++i) {
				auto idx = leafnodes[i];
				auto box = boxes[idx];
				auto dim = box.getnum();
//				leafdim[i] = dim;
				woffset[i] = offset;
				uoffset[i] = offset;
				offset += dim * setup.nrhs;
			}

			auto idx = leafnodes[0];
			lids = boxes[idx].getlids();
		}


		void HMatrix::StructureNearBlock(std::vector<std::vector<std::vector<pair<int, int>>>> &blocks) {
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

		void HMatrix::StructureFarBlock(std::vector<std::vector<std::vector<pair<int, int>>>> &blocks) {
			int fb = blocks.size();
			int nnodes = tree->fcount;
			fblockset = new int[fb+1]();
			fxval = new int[nnodes]();
			fyval = new int[nnodes]();
//			ds->fl = fb;
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



		HMatrix::~HMatrix() {
			delete  tree;
		}


	}
}
