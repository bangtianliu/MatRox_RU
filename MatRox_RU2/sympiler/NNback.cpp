//
// Created by Bangtian Liu on 6/27/19.
//

#include <random>
#include <algorithm>
#include <set>
#include "NNtree.h"
#include "nUtil.h"

namespace Sympiler{
	namespace Internal{
		void NNtree::split(const Setup &setup, std::vector<int> lids, std::vector<int> &llids, std::vector<int> &rlids)
		{
//	hmlp::Data<T> &X = *Coordinate;
			auto X = setup.X;

			auto d = setup.d;
			auto n = lids.size();

//	std::vector<std::vector<std::size_t> > split( N_SPLIT );

			std::vector<double > direction( d );
			std::vector<double > projection( n, 0.0 );

			// Compute random direction
			static std::default_random_engine generator;
			std::normal_distribution<double > distribution;
			for ( int p = 0; p < d; p ++ )
			{
				direction[ p ] = distribution( generator );
			}

			// Compute projection
			projection.resize( n, 0.0 );
#pragma omp parallel for
			for ( int i = 0; i < n; i ++ )
				for ( int p = 0; p < d; p ++ )
					projection[ i ] += X[ lids[ i ] * d + p ] * direction[ p ];


			// Parallel median search
			// T median = Select( n, n / 2, projection );
			auto proj_copy = projection;
			std::sort( proj_copy.begin(), proj_copy.end() );
			double median = proj_copy[ n / 2 ];

			llids.reserve( n / 2 + 1 );
			rlids.reserve( n / 2 + 1 );

			/** TODO: Can be parallelized */
			std::vector<int> middle;
			for ( int i = 0; i < n; i ++ )
			{
				if      ( projection[ i ] < median ) llids.push_back( lids[i] ); //bugs
				else if ( projection[ i ] > median ) rlids.push_back( lids[i] );
				else                                 middle.push_back( lids[i] );
			}

			for ( int i = 0; i < middle.size(); i ++ )
			{
				if ( llids.size() <= rlids.size() ) llids.push_back( middle[ i ] );
				else                                          rlids.push_back( middle[ i ] );
			}
		}

		void NNtree::buildtree(Setup &setup, int m) {
//     int m = 4*setup.k;

			if(m<32) m =32;

			depth = (int) (log(setup.n * 1.0 / m)/log(2));

			num_acc.resize((int)pow(2,depth));
			knn_acc.resize((int)pow(2,depth));

			num_acc.assign((1<<depth),0);
			knn_acc.assign((1<<depth),0);

			lids.resize((int)pow(2,depth+1)-1);

			lids[0].resize(setup.n);

			setup.lids = (int *)malloc(sizeof(int)*setup.n);

			for(int i=0; i<setup.n; i++)
			{
				setup.lids[i]=i;
			}

			lids[0].assign(setup.lids, setup.lids + setup.n);
//
			for(int i = 0; i<depth; i++)
			{
				int L_beg = begin(i);
				int L_end = stop(i);

				int nodes = L_beg-L_end+1;
				int max_threads = omp_get_max_threads();

				if(nodes >= max_threads || nodes ==1)
				{
					#pragma omp parallel for if (nodes > max_threads)
					for(int j=L_beg;j<=L_end;j++)
					{
						auto lc = 2*j+1;
						auto rc = 2*j+2;
						split(setup,lids[j],lids[lc],lids[rc]);
					}
				}
				else
				{
					#pragma omp parallel for schedule(dynamic)
					for(int j=L_beg;j<=L_end;j++)
					{
						auto lc = 2*j+1;
						auto rc = 2*j+2;
						split(setup,lids[j],lids[lc],lids[rc]);
					}
				}
			}
		}


		double NNtree::search(Setup &setup)
		{
			int l = depth;

			int L_beg = begin(l);
			int L_end = stop(l);


//#pragma omp parallel for
			for(int i=L_beg;i<=L_end;i++)
			{
				searchleaf(setup,i);
			}

			double knn = 0.0;
			int num = 0;
			for(int i = L_beg; i<=L_end;i++)
			{
				auto nidx = i -L_beg;
				knn += knn_acc[nidx];
				num += num_acc[nidx];
			}

			return knn/num;
		}

		void NNtree::searchleaf(Setup &setup, int index) {
			auto X = setup.X;
			auto d = setup.d;
			auto &lid = lids[index];

			auto &NN = setup.NN;

			int offset = index - ((1<<depth)-1);
#pragma omp parallel for
			for (int j = 0; j < lid.size(); ++j) {
				std::set<int> NNset;
				auto idx = lid[j];
				for(int k = 0; k<setup.k; ++k)
				{
					NNset.insert(NN[idx*setup.k+k].second);
				}

				for(int k = 0; k < lid.size(); ++k)
				{
					auto idy = lid[k];

					if(!NNset.count(idy)){
						double dist = 0;
						for ( size_t p = 0; p < d; p ++ )
						{
							double xip = X[ idx * d + p ];
							double xjp = X[ idy * d + p ];
							dist += ( xip - xjp ) * ( xip - xjp );
						}

						std::pair<double , int> query(dist, idy);

						HeapSelect(1, setup.k, &query, &NN[idx*setup.k]);
					}
				}
			}

			for(int j=0; j<lid.size();j++)
			{
				if(lid[j]>=NUM_TEST) continue;
				std::set<int> NNset;

				std::vector<std::pair<double , int>> nn_test(setup.k, std::pair<double ,int>());

				for(int k=0; k<setup.k;k++)
				{
					nn_test[k] = NN[lid[j]*setup.k+k];
					NNset.insert(nn_test[k].second);
				}

				for(int k =0 ; k < setup.n; k++)
				{
					int igid = k;
					int jgid = lid[j];

					if(!NNset.count(igid))
					{
						double dist = 0;
						size_t d = setup.d;
						for ( size_t p = 0; p < d; p ++ )
						{
							double xip = X[ igid * d + p ];
							double xjp = X[ jgid * d + p ];
							dist += ( xip - xjp ) * ( xip - xjp );
						}

						std::pair<double , int> query( dist, igid );
						//hmlp::HeapSelect( 1, NN.row(), &query, NN.data() + jlid * NN.row() );
						HeapSelect( 1, setup.k, &query, nn_test.data() );
						NNset.insert( igid );

					}

				}

				int correct = 0;
				NNset.clear();

				for(int i = 0; i < setup.k; i++){
					NNset.insert(nn_test[i].second);
				}

				for(int i= 0; i < setup.k; i++)
				{
					if(NNset.count(NN[lid[j]*setup.k+i].second)) correct++;
				}

				knn_acc[offset] += correct*1.0/setup.k;
				num_acc[offset]++;
			}

		}
	}
}