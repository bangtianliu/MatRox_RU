//
// Created by Bangtian Liu on 5/29/19.
//

#include "ReNearFar.h"

namespace Sympiler {
	namespace Internal {
		void ReNearFar::FindNearFar(const int idx, const int idy, const double tau) {
			findnearfarnodes(idx,idy,tau);
		}

		void ReNearFar::findnearfarnodes(const int idx, const int idy, const double tau) {
			if(well_sperated(rtree->boxes[idx],ctree->boxes[idy],tau) && idx!=0 && idy!=0)
			{
				farnodes[idx].push_back(idy);
			}
			else{
				const int flag = (level(idx)==rtree->setup.depth) << 1 | (level(idy)==ctree->setup.depth);

				switch (flag){
					case 3:
						nearnodes[idx].push_back(idy);
						break;
					case 2:
						findnearfarnodes(idx, 2*idy+1, tau);
						findnearfarnodes(idx, 2*idy+2, tau);
						break;
					case 1:
						findnearfarnodes(2*idx+1,idy,tau);
						findnearfarnodes(2*idx+2,idy,tau);
						break;
					default:
						findnearfarnodes(2*idx+1,2*idy+1,tau);
						findnearfarnodes(2*idx+1,2*idy+2,tau);
						findnearfarnodes(2*idx+2,2*idy+1,tau);
						findnearfarnodes(2*idx+2,2*idy+2,tau);
						break;
				}
			}
		}

		std::vector<int> intersection2(std::vector<int> v1,
									   std::vector<int> v2){
			std::vector<int> v3;

			std::sort(v1.begin(), v1.end());
			std::sort(v2.begin(), v2.end());

			std::set_intersection(v1.begin(),v1.end(),
								  v2.begin(),v2.end(),
								  back_inserter(v3));
			return v3;
		}


		void ReNearFar::MergeFar() {
			int l = rtree->setup.depth;

			for(int i=l-1; i>=1; --i)
			{
				for(int v=begin(i); v<=stop(i); v++)
				{
					auto lc = 2*v+1;
					auto rc = 2*v+2;
					auto tmp = intersection2(farnodes[lc], farnodes[rc]);

					if(!tmp.empty()){
						for(auto &w:tmp)
						{
							auto search = std::find(farnodes[lc].begin(), farnodes[lc].end(), w);
							farnodes[lc].erase(search);

							auto search1 = std::find(farnodes[rc].begin(), farnodes[rc].end(), w);
							farnodes[rc].erase(search1);
							farnodes[v].push_back(w);
						}
					}
				}
			}
		}


		void ReNearFar::CacheNear() {
			int count=0;

			for(int v=begin(rtree->setup.depth); v<=stop(rtree->setup.depth);v++)
			{
//		int i=v-begin(rsetup.depth);
				auto lidx = rtree->boxes[v].getlids();
				for(auto &w:nearnodes[v])
				{
					auto lidy = ctree->boxes[w].getlids();
					count+=lidx.size()*lidy.size();
				}
			}

			Near = (double *)mkl_malloc(sizeof(double)*count, 64);
			memset(Near, 0, sizeof(double)*count);

			auto leafnodes = rtree->getnumleafs();

			this->nearoffset = (int *)mkl_malloc(sizeof(int)*leafnodes, 64);
			memset(nearoffset, 0, sizeof(int)*leafnodes);

			int offset=0;
			for(int v=begin(rtree->setup.depth); v<=stop(rtree->setup.depth);v++)
			{
//		auto v = leafnodes[i];
				int i=v-begin(rtree->setup.depth);
				nearoffset[i]=offset;
				auto lidx = rtree->boxes[v].getlids();

				for(auto &w:nearnodes[v])
				{
					auto lidy = ctree->boxes[w].getlids();

					Fsubmatrix(lidx.data(), (int)lidx.size(), lidy.data(), (int)lidy.size(),Near+offset,rtree->setup);

					offset+=lidx.size()*lidy.size();
				}
			}
		}

		void ReNearFar::CacheFar() {
			auto numnodes = rtree->getnumnodes();

			this->faroffset = (int *)mkl_malloc(sizeof(int)*numnodes,64);
			memset(this->faroffset, 0, sizeof(int)*numnodes);

			int size=0;

			for(int i=1; i<numnodes; i++)
			{
				int dimx=rtree->tmpresult[i].skels_length;
				for(auto &w:farnodes[i])
				{
					int dimy=ctree->tmpresult[w].skels_length;
					size+=dimx*dimy;
				}
			}

			this->Far = (double *)mkl_malloc(sizeof(double)*size, 64);
			memset(Far, 0, sizeof(double)*size);

			int offset = 0;
			for(int i=1; i<numnodes; i++)
			{
				this->faroffset[i]=offset;
				for(auto &w:farnodes[i])
				{
					Fsubmatrix(rtree->tmpresult[i].skels, rtree->tmpresult[i].skels_length, ctree->tmpresult[w].skels, ctree->tmpresult[w].skels_length, Far+offset, rtree->setup);

					offset+=rtree->tmpresult[i].skels_length*ctree->tmpresult[w].skels_length;
				}
			}
		}


		int ReNearFar::getNearNodes() {
			int count=0;

			for(auto &v:nearnodes)
			{
				count +=v.size();
			}
			return count;
		}

		int ReNearFar::getFarNodes() {
			int count=0;

			for(auto &v:farnodes)
			{
				count += v.size();
			}

			return count;
		}

	}
}