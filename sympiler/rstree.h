//
// Created by Bangtian Liu on 5/27/19.
//

#ifndef PROJECT_RSTREE_H
#define PROJECT_RSTREE_H

#include "boundingbox.h"
#include "nUtil.h"

using namespace std;

namespace Sympiler {
	namespace Internal {
		class rstree {

			friend class HMatrix;
			friend class ReNearFar;
			std::vector<boundingbox> boxes;

//	std::vector<std::vector<int>> nearnodes;
//	std::vector<std::vector<int>> farnodes;

			std::vector<int> lids;

			std::vector<int> rlids; // from near nodes

			std::vector<Ret> tmpresult;

			Setup setup;

			int m;
			int n;


		public:
			double *w_leaf;
			int *w_offset;
			double *u_leaf;
			int *u_offset;

			double *w_skels;
			double *u_skels;

			int *wskeloffset;
			int *uskeloffset;


			void buildtree(boundingbox &box, Setup &_setup);
			void buildbtree(boundingbox &box);


			void updatelids(std::vector<boundingbox> boxes, std::vector<int> nearnodes);

			void compression();

			void skeletonize(int idx, boundingbox &box, Ret *tmp);

			void configure();

			int getnumleafs(){
				return 1<<setup.depth;
			}

			int getnumnodes(){
				return (1<<(setup.depth+1))-1;
			}

			void transform(double *wleaf);
			void Evalallocation();


			void forward();
			void forward_opt();

			void backward();
			void backward_opt();

			void backpermute(double *approx_result);

			rstree(){};
			~rstree(){};
		};
	}
}
#endif //PROJECT_RSTREE_H
