//
// Created by Bangtian Liu on 6/27/19.
//

#include <utility>
#include <vector>
#include <cstdio>
#include <algorithm>
#include "NearSearch.h"
#include "NNtree.h"

namespace Sympiler {
	namespace Internal{
		/**
 * @brief the code is learnt from GOFMM, for nearest neighbour sampling
 * @param setup
 * @return
 * @FIXME BUG
 */
		void NearSearch(Setup &setup)
		{
			int m = 4*setup.k;

			std::pair<double , int> initNN( std::numeric_limits<double >::max(), setup.n);
			std::vector<std::pair<double , int>> tNN(setup.k*setup.n,initNN);


			auto &NN = setup.NN;
			NN.assign(tNN.begin(), tNN.end());

//	NN.resize((size_t)setup.n*setup.k);
//	NN.assign(setup.n*setup.k, initNN);

//	NNtree tree;
			for(int i = 0; i<n_tree; i++)
			{
				NNtree tree = NNtree();
				tree.buildtree(setup,m);
				auto acc=tree.search(setup);

//				printf("\n#ANN iter %2d, average accuracy %.2lf%% (over %4lu samples)\n", i, acc, 10);

				if(acc < 0.8)
				{
					if( 2 * m < 2048) m = 2*m;
				}
				else{
//			setup.NN = tree.getNN();
					break;
				}

			}


			bool sorted = false;

			if(sorted)
			{
				struct
				{
					bool operator () ( std::pair<double , int> a, std::pair<double , int> b )
					{
						return a.first < b.first;
					}
				} ANNLess;


#pragma omp parallel for
				for ( size_t j = 0; j < setup.n; j ++ )
				{
					std::sort( NN.data() + j * setup.k, NN.data() + ( j + 1 ) * setup.k );
				}
			}
			/*the code need to update*/
		};


		void BuildNeighbors(Setup &setup, clustertree &tree, int node, int nsamples)
		{
			auto &boxes = tree.getboxes();
			auto box = boxes[node];
			auto lids = box.getlids();
			auto children = tree.getchildren();
			auto &NN = setup.NN;
			auto k = setup.k;
			auto n = lids.size();
			auto &tpnids = tree.getpnids();
			auto &tsnids = tree.getsnids();
			auto &pnids = tpnids[node];
			auto &snids = tsnids[node];

			if(children[node].empty()){
				pnids = std::unordered_set<int>(); // will modify
				for ( int ii = 0; ii < k / 2; ii ++ ) {
					for (int jj = 0; jj < n; jj++) {
						auto idx = NN[lids[jj] * k + ii].second;
						pnids.insert(idx);
//						printf("%lu;",NN[ lids[jj] * k + ii].second);
					}
				}
//
				for(int i = 0; i<n; i++)
				{
					pnids.erase(lids[i]);
				}

				snids = std::map<int, double >();
				std::vector<std::pair<double , int>> tmp ( k / 2 * n );
				std::set<int> nodeIdx( lids.begin() , lids.end() );
				// Allocate array for sorting
				for ( int ii = (k+1) / 2; ii < k; ii ++ )
				{
					for ( int jj = 0; jj < n; jj ++ )
					{
						tmp [ (ii-(k+1)/2) * n + jj ] = NN[ lids[ jj ] * k + ii ];
					}
				}

				std::sort(tmp.begin(), tmp.end());
				int i = 0;

				while ( snids.size() < nsamples && i <  (k-1) * n / 2 )
				{
					if ( !pnids.count( tmp[i].second ) && !nodeIdx.count( tmp[i].second ) )
					{
						snids.insert( std::pair<int,double >( tmp[i].second , tmp[i].first ) );
					}
					i++;
				}
			}
			else {

				auto &lsnids = tsnids[children[node].front()];
				auto &rsnids = tsnids[children[node].back()];

				auto &lpnids = tpnids[children[node].front()];
				auto &rpnids = tpnids[children[node].back()];

				snids = lsnids;

				for ( auto cur = rsnids.begin(); cur != rsnids.end(); cur ++ )
				{
					auto ret = snids.insert( *cur );
					if ( ret.second == false )
					{
						// Update distance?
						if ( ret.first->second > (*cur).first)
						{
							ret.first->second = (*cur).first;
						}
					}
				}

				// Remove "own" points
				for (int i = 0; i < n; i ++ )
				{
					snids.erase( lids[ i ] );
				}

				// Remove pruning neighbors from left and right
				for (auto cur = lpnids.begin(); cur != lpnids.end(); cur++ )
				{
					snids.erase( *cur );
				}
				for (auto cur = rpnids.begin(); cur != rpnids.end(); cur++ )
				{
					snids.erase( *cur );
				}
			}
		}
	}
}