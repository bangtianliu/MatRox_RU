//
// Created by Bangtian Liu on 4/30/19.
//

#include <cstdint>

#ifndef PROJECT_CONFIG_H
#define PROJECT_CONFIG_H

namespace Sympiler {
	namespace Internal {
		typedef enum{
			KS_GAUSSIAN,
			KS_POLYNOMIAL,
			KS_LAPLACE,
			KS_GAUSSIAN_VAR_BANDWIDTH,
			KS_TANH,
			KS_QUARTIC,
			KS_MULTIQUADRATIC,
			KS_EPANECHNIKOV,
			KS_LOG,
			KS_EXPONENTIAL,
			KS_NEWTON
		}Ktype;

		struct config{
			uint64_t n;  // number of rows and columns
			int d; // the dimension of input points
			int m; // leaf node block sizes;

			int depth; // the depth of cluster tree.
			int maxRank; // maxRank
			int nrhs; // the number of columns in weight vector w
			double stol; // desired approximation accuracy
			double budget; // control the percentage of off-diagonal full-rank blocks which can not be approximate.

			Ktype ktype;
			double *M; // input matrix
			double *X; // point set
			double *w; // wight vector
			int *lids;
			double h; // bandwidth for Guassian Kernel

			bool symmetric; // guarantee symmtric approximation
			bool adaptive;  // adaptive low-rank approximation.

			bool equal;
			bool binary;
			int nparts; //number of partitions, for clustering using k-means (>2)
			double tau;
			bool use_runtime;
			bool use_cbs=false; // whether to use blocking.

			bool use_fmm; // to compare with GOFMM.

			bool use_coarsing=false; // coarsing

			bool use_pruning; // pruning

			int nthreads;

			int k; // number of nearest neighbour for nearest point search


			std::vector<std::pair<double , int>> NN;

			bool recompress=false;
		};

		typedef config Setup;

		#define MLPOINT
//		#define SAVE
	}
}



#endif //PROJECT_CONFIG_H

