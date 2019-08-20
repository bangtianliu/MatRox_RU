//
// Created by Bangtian Liu on 6/27/19.
//

#ifndef PROJECT_NEARSEARCH_H
#define PROJECT_NEARSEARCH_H


#include "config.h"
#include "ClusterTree.h"

namespace Sympiler{
	namespace Internal{
		const int n_tree=10;

		void NearSearch(Setup &setup);
		void BuildNeighbors(Setup &setup, clustertree &tree, int node, int nsamples);

	}
}



#endif //PROJECT_NEARSEARCH_H
