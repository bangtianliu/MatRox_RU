//
// Created by Bangtian Liu on 5/29/19.
//

#ifndef PROJECT_RENEARFAR_H
#define PROJECT_RENEARFAR_H


#include "nUtil.h"
#include "rstree.h"

namespace Sympiler {
	namespace Internal {
		class ReNearFar {
			rstree *ctree;
			rstree *rtree;

			std::vector<std::vector<int>> nearnodes;
			std::vector<std::vector<int>> farnodes;

			double *Near;
			double *Far;
			int *nearoffset;
			int *faroffset;

		public:
			int flag;

			ReNearFar(){};
			~ReNearFar(){};

			ReNearFar(rstree &_rtree, rstree &_ctree)
			{
				this->rtree = & _rtree;
				this->ctree = & _ctree;
//		printf("depths are %d %d\n", _rtree.setup.depth, _ctree.setup.depth);
				nearnodes.resize(rtree->getnumnodes());
				farnodes.resize(rtree->getnumnodes());
			}

			void findnearfarnodes(const int idx, const int idy, const double tau);

			void FindNearFar(const int idx, const int idy, const double tau);

			void MergeFar();

			bool isHSS(){
				for(int v=begin(rtree->setup.depth);v<=stop(rtree->setup.depth);v++)
				{
					if(nearnodes[v].size()!=1) return false;
				}
				return true;
			}


			void CacheNear();
			void CacheFar();

			int getNearNodes();

			int getFarNodes();

			void near();

			void forward();

			void forward_opt();
			void far();
			void backward_opt();

			void backward();
			void backpermute(double *approx_result);
		};
	}
}
#endif //PROJECT_RENEARFAR_H
