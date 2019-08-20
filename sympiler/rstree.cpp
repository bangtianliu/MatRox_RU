//
// Created by Bangtian Liu on 5/27/19.
//


#include "rstree.h"

namespace Sympiler {
	namespace Internal {
		void rstree::buildtree(boundingbox &box, Setup &_setup) {
			this->setup = _setup;
			this->lids = box.getlids();
//	printf("lids size:%ld\n", lids.size());
			configure();
//	tree.buildtree(box,setup);
			this->buildbtree(box);
		}

		void rstree::buildbtree(boundingbox &box) {
			int numnodes = (1<<(setup.depth+1))-1;
			boxes.resize(numnodes);
			tmpresult.resize(numnodes);
//	this->depth = setup.depth;
			boxes[0]=box;
			for(int l=0; l<=setup.depth-1; l++)
			{
				for(int index = begin(l); index<= stop(l); ++index)
				{
					boundingbox &box = boxes[index];

					int lc = 2*index+1;
					int rc = 2*index+2;

					box.sbsplit(boxes[lc],boxes[rc],setup);
				}
			}
		}

		void rstree::updatelids(std::vector<boundingbox> boxes, std::vector<int> nearnodes) {
			rlids = boxes[nearnodes[0]].getlids();

			for(int i=1; i<nearnodes.size(); i++)
			{
				auto lids = boxes[nearnodes[i]].getlids();
				rlids.insert(rlids.end(), lids.begin(), lids.end());
			}

			this->m = rlids.size();
		}


		void rstree::configure() {
			setup.binary = true;
			setup.equal = true;
			int m = 32;
			auto tm = lids.size();

			this->n = (int)tm;

//	setup.maxRank = setup.m/2;
//	setup.n = tm;

//	setup.stol = sqrt(m*1.0/setup.n) * sqrt( n*1.0/setup.n) * setup.stol;

			int depth = 0;


			if(tm<=m){
				tm/=2;
				++depth;
			}
			else {
				while (tm > m)
				{
					tm/=2;
					++depth;
				}
			}

			setup.depth=depth;
			setup.m=(int)tm;
			setup.maxRank = setup.m;
		}


		void rstree::compression() {
			auto depth = setup.depth;
#pragma omp parallel
			{
				for(int i = depth; i > 0; --i)
				{
//			auto &levels=sflevelsets[i];
#pragma omp for
					for(int idx=begin(i);idx<=stop(i);idx++){
//				auto idx=levels[j];
						auto box=boxes[idx];
						skeletonize(idx,box, tmpresult.data());
					}
				}
			}
		}


		void rstree::skeletonize(int idx, boundingbox &box, Ret *tmp) {
			if(idx==0) return;

			std::vector<int> amap;
			std::vector<int> bmap;

			if(level(idx)==setup.depth){
				bmap=box.getlids();
			}
			else {
//		for(auto &v:rchildren[idx]){
				auto v = 2*idx+1;
				bmap.insert(bmap.end(),tmp[v].skels,tmp[v].skels+tmp[v].skels_length);
				v=2*idx+2;
				bmap.insert(bmap.end(),tmp[v].skels,tmp[v].skels+tmp[v].skels_length);
// / 	}
			}

			auto nsamples = 2 * bmap.size();

			auto numpoints = box.getnum();
//	auto clids = box.getlids();
			nsamples = (nsamples < 2*setup.m)? 2*setup.m: nsamples; // number of sampling

			mt19937 generator(idx);
			uniform_int_distribution<> uniform_distribution(0, (int)rlids.size()-1);

			if(nsamples<rlids.size()-numpoints){
				while(amap.size()<nsamples){
//					auto sample = rand() % rlids.size();
					auto sample = uniform_distribution(generator);
					if(std::find(amap.begin(), amap.end(), sample)==amap.end())
					{
						amap.push_back(sample);
					}
				}
			}
			else {
				for (int sample = 0; sample < rlids.size(); sample++)
				{
					if ( std::find( amap.begin(), amap.end(), sample ) == amap.end() )
					{
						amap.push_back( sample );
					}
				}
			}


			for(int i=0;i<amap.size();i++)
			{
				amap[i]=rlids[amap[i]];
			}

			auto Kab = (double *)malloc(sizeof(double)*amap.size()*bmap.size());
			memset(Kab,0,sizeof(double)*amap.size()*bmap.size());

//#ifdef POINT
			Fsubmatrix(amap, bmap, Kab, setup);
//#else
//			submatrix(amap,bmap,Kab,setup);
//#endif

			auto m = amap.size();
			auto n = bmap.size();


			auto q = box.getnum();

			auto N = rlids.size();
			auto tolerance = std::sqrt((m*n*1.0)/(N*(N-q)))*setup.stol;

			int *skels;
			double *proj;
			int *jpvt;

			int s = decomposition(Kab, m, n, tolerance, &skels, &proj, &jpvt, setup);

			for (int i = 0; i < s; ++i) {  // need to check it
				skels[i] = bmap[skels[i]];
			}

			tmp[idx].skels = skels;
			tmp[idx].skels_length=s;
			tmp[idx].proj=proj;
			tmp[idx].proj_column=(int)bmap.size();
			free(Kab);
		}

		void rstree::transform(double *wleaf) {
			auto numleafnodes = this->getnumleafs();
			this->w_offset = (int *)mkl_malloc(sizeof(int)*numleafnodes,64);
			memset(w_offset, 0, sizeof(int)*numleafnodes);

			int cn = this->n;
			int nrhs = this->setup.nrhs;

			this->w_leaf = (double *)mkl_malloc(sizeof(double)*cn*nrhs,64);
			memset(w_leaf, 0, sizeof(double)*cn*nrhs);

			int rn = this->n;
			this->u_leaf = (double *)mkl_malloc(sizeof(double)*rn*nrhs, 64);
			memset(u_leaf, 0, sizeof(double)*rn*nrhs);

			int offset=0;

			for(int idx=begin(this->setup.depth); idx<=stop(this->setup.depth); idx++)
			{
				int i = idx-begin(this->setup.depth);
//		auto idx = leaf[i];
				w_offset[i]=offset;

				auto &box=this->boxes[idx];
				auto dim = box.getnum();
				auto tlids = box.getlids();
				auto tw = w_leaf + offset;

				int toffset=0;

				for(int j = 0; j < this->setup.nrhs; j++)
				{
					for(int k = 0; k<tlids.size();k++){
						tw[toffset++] = wleaf[j*this->setup.n + tlids[k]];
					}
				}
				offset += dim*this->setup.nrhs;
			}

			auto rnumleaf = this->getnumleafs();
			this->u_offset = (int *)mkl_malloc(sizeof(int)*rnumleaf,64);
			memset(this->u_offset, 0, sizeof(int)*rnumleaf);

			int uoffset=0;

			for(int idx=begin(this->setup.depth); idx<=stop(this->setup.depth); idx++)
			{
				int i = idx - begin(this->setup.depth);
//		auto idx = leaf[i];
				u_offset[i]=uoffset;
				auto rbox=this->boxes[idx];
				auto rdim=rbox.getnum();
				uoffset += rdim*this->setup.nrhs;
			}
		}

		void rstree::Evalallocation() {
			auto numnodes = this->getnumnodes();

			wskeloffset = (int *)mkl_malloc(sizeof(int)*numnodes,64);
			memset(wskeloffset, 0, sizeof(int)*numnodes);

			int count = 0;
			for(int i=this->setup.depth; i>0; --i) {
//		auto &levels = sflevelsets[i];
				for (int w = begin(i); w <= stop(i); w++)
				{
//			auto w = levels[j];
					wskeloffset[w] = count;
					count += this->tmpresult[w].skels_length*this->setup.nrhs;
				}
			}

			this->w_skels = (double *)mkl_malloc(sizeof(double)*count, 64);
			memset(this->w_skels, 0, sizeof(double)*count);

			count=0;
			numnodes=this->getnumnodes();

			uskeloffset = (int *)mkl_malloc(sizeof(int)*numnodes,64);
			memset(uskeloffset, 0, sizeof(int)*numnodes);


			for(int i=1;i<=this->setup.depth;++i)
			{
				for (int w = begin(i); w <= stop(i); w++)
				{
					uskeloffset[w] = count;
					count += this->tmpresult[w].skels_length*this->setup.nrhs;
				}
			}
			u_skels = (double *)mkl_malloc(sizeof(double)*count, 64);
			memset(u_skels, 0, sizeof(double)*count);
		}

		void rstree::forward() {
			for(int i = this->setup.depth;i>0;i--)
			{
				for(int v = begin(i); v<= stop(i); v++){
					if(i == this->setup.depth){
						auto idy = v-begin(this->setup.depth);
						auto w = w_leaf + w_offset[idy]; // wrong
						auto twskels = w_skels + wskeloffset[v];
						auto proj = this->tmpresult[v].proj;
						auto dimy=this->tmpresult[v].proj_column;
						auto dimx = this->tmpresult[v].skels_length;

						cblas_dgemm(CblasColMajor, CblasNoTrans,CblasNoTrans,
							dimx,this->setup.nrhs,dimy, 1.0f,
							proj, dimx,
							w, dimy,
							1.0f, twskels,dimx);

					}
					else {
						auto proj = this->tmpresult[v].proj;
						auto slen = this->tmpresult[v].skels_length;

						auto twskels = w_skels + wskeloffset[v];
						int offset = 0;

						int w = 2*v+1;
						auto cslen=this->tmpresult[w].skels_length;
						auto ctwskels = w_skels + wskeloffset[w];

						cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
							slen, this->setup.nrhs, cslen, 1.0f,
							proj+offset, slen,
							ctwskels, cslen, 1.0f,
							twskels,slen);

						offset+=slen*cslen;

						w=2*v+2;

						cslen=this->tmpresult[w].skels_length;
						ctwskels = w_skels + wskeloffset[w];

						cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
							slen, this->setup.nrhs, cslen, 1.0f,
							proj+offset, slen,
							ctwskels, cslen, 1.0f,
							twskels,slen);
					}
				}

			}
		}


		void rstree::forward_opt() {
			for(int v = begin(this->setup.depth); v<= stop(this->setup.depth); v++)
			{
				auto idy = v-begin(this->setup.depth);
				auto w = w_leaf + w_offset[idy]; // wrong
				auto twskels = w_skels + wskeloffset[v];
				auto proj = this->tmpresult[v].proj;
				auto dimy=this->tmpresult[v].proj_column;
				auto dimx = this->tmpresult[v].skels_length;

				cblas_dgemm(CblasColMajor, CblasNoTrans,CblasNoTrans,
					dimx,this->setup.nrhs,dimy, 1.0f,
					proj, dimx,
					w, dimy,
					1.0f, twskels,dimx);
			}

			for(int i = this->setup.depth-1;i>0;i--)
			{
				for(int v = begin(i); v<= stop(i); v++)
				{
					auto proj = this->tmpresult[v].proj;
					auto slen = this->tmpresult[v].skels_length;

					auto twskels = w_skels + wskeloffset[v];
					int offset = 0;

					int w = 2*v+1;
					auto cslen=this->tmpresult[w].skels_length;
					auto ctwskels = w_skels + wskeloffset[w];

					cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
						slen, this->setup.nrhs, cslen, 1.0f,
						proj+offset, slen,
						ctwskels, cslen, 1.0f,
						twskels,slen);

					offset+=slen*cslen;

					w=2*v+2;

					cslen=this->tmpresult[w].skels_length;
					ctwskels = w_skels + wskeloffset[w];

					cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
						slen, this->setup.nrhs, cslen, 1.0f,
						proj+offset, slen,
						ctwskels, cslen, 1.0f,
						twskels,slen);

				}
			}
		}

		void rstree::backward() {
			for(int i=1;i<this->setup.depth+1;i++)
			{
				for(int v = begin(i); v<= stop(i); v++)
				{
					if(i==this->setup.depth){
						auto tproj = this->tmpresult[v].proj;

						auto uskels = u_skels + uskeloffset[v];

						auto idx = v-begin(this->setup.depth);

						auto uleaf = u_leaf + u_offset[idx];
						auto dimx = this->tmpresult[v].proj_column;

						auto skels = this->tmpresult[v].skels_length;

						cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
							dimx, this->setup.nrhs, skels, 1.0f,
							tproj, skels,
							uskels,skels, 1.0f,
							uleaf, dimx);

					}
					else {
						auto skel = this->tmpresult[v].skels_length;

						auto proj = this->tmpresult[v].proj;

						auto tuskel = u_skels + uskeloffset[v];

						int offset = 0;
						int w = 2*v + 1;
						auto cskel = this->tmpresult[w].skels_length;
						auto ctuskel = u_skels + uskeloffset[w];


						cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
							cskel, this->setup.nrhs, skel, 1.0f,
							proj + offset, skel,
							tuskel, skel, 1.0f,
							ctuskel, cskel);


						offset += skel * cskel;

						w = 2*v+2;
						cskel = this->tmpresult[w].skels_length;
						ctuskel = u_skels + uskeloffset[w];

						cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
							cskel, this->setup.nrhs, skel, 1.0f,
							proj + offset, skel,
							tuskel, skel, 1.0f,
							ctuskel, cskel);
					}

				}
			}
		}


		void rstree::backward_opt() {
			for(int i=1;i<this->setup.depth;i++)
			{
				for(int v = begin(i); v<= stop(i); v++)
				{
					auto skel = this->tmpresult[v].skels_length;

					auto proj = this->tmpresult[v].proj;

					auto tuskel = u_skels + uskeloffset[v];

					int offset = 0;
					int w = 2*v + 1;
					auto cskel = this->tmpresult[w].skels_length;
					auto ctuskel = u_skels + uskeloffset[w];

					cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
						cskel, this->setup.nrhs, skel, 1.0f,
						proj + offset, skel,
						tuskel, skel, 1.0f,
						ctuskel, cskel);


					offset += skel * cskel;

					w = 2*v+2;
					cskel = this->tmpresult[w].skels_length;
					ctuskel = u_skels + uskeloffset[w];

					cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
						cskel, this->setup.nrhs, skel, 1.0f,
						proj + offset, skel,
						tuskel, skel, 1.0f,
						ctuskel, cskel);

				}
			}


			for(int v = begin(this->setup.depth); v<= stop(this->setup.depth); v++)
			{
				auto tproj = this->tmpresult[v].proj;

				auto uskels = u_skels + uskeloffset[v];

				auto idx = v-begin(this->setup.depth);

				auto uleaf = u_leaf + u_offset[idx];
				auto dimx = this->tmpresult[v].proj_column;

				auto skels = this->tmpresult[v].skels_length;

				cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
					dimx, this->setup.nrhs, skels, 1.0f,
					tproj, skels,
					uskels,skels, 1.0f,
					uleaf, dimx);

			}
		}

		void rstree::backpermute(double *approx_result) {
			auto nleaf = this->getnumleafs();

			for(int i=0;i<nleaf;i++)
			{
				auto v = i + begin(this->setup.depth);
				auto idx = i;

				auto dim = this->boxes[v].getnum();

				auto uleaf = u_leaf + u_offset[idx];
				auto lids = this->boxes[v].getlids();
				for(int j=0; j<dim; ++j){
					for(int i=0; i<this->setup.nrhs;i++){
//				printf("DEBUG: lid=%d uleaf=%f\n", lids[j], uleaf[i*dim+j]);
						approx_result[lids[j]*this->setup.nrhs + i ] += uleaf[i*dim+j];
					}
				}
			}
		}
	}
}
