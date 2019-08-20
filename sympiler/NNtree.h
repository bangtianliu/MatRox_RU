//
// Created by Bangtian Liu on 6/27/19.
//

#ifndef PROJECT_NNTREE_H
#define PROJECT_NNTREE_H

#include <vector>
#include "config.h"

namespace Sympiler{
	namespace Internal{
			/**
 * For nearest point search
 * **/
			class NNtree {
				int depth;
				int n;

				std::vector<std::vector<int>> lids;

//	std::vector<std::pair<T, int>> NN;

				std::vector<int> num_acc;
				std::vector<double > knn_acc;

				const int NUM_TEST=10;


			public:
				NNtree(){};
				void split(const Setup &setup, std::vector<int> lids, std::vector<int> &llids, std::vector<int> &rlids, int idx);
				void buildtree(Setup &setup, int m);

				double search(Setup &setup);

				void searchleaf(Setup &setup, int index);

//    std::vector<std::pair<T, int>> getNN(){
//		return NN;
//	};
			};
	}
}

#endif //PROJECT_NNTREE_H
