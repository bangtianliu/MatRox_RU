//
// Created by Bangtian Liu on 5/2/19.
//

#ifndef PROJECT_NUTIL_H
#define PROJECT_NUTIL_H

#include <vector>
#include <algorithm>
#include <omp.h>
#include <map>
#include "config.h"
#include <cmath>
#include <random>

using namespace std;
namespace Sympiler {
	namespace Internal {
		struct retvalue{
			int *skels;
			int skels_length;
			double *proj;
			int proj_column;
		};

		typedef retvalue Ret;

		struct Dcost{
			int index;
			unsigned long cost;
		};

		bool compare(Dcost lhs, Dcost rhs);


		template <typename T>
		__inline__ T distance(T *point1, T *point2, int dim)
		{
			T sum=0;
			for(int i=0;i<dim;i++){
				sum+=(point2[i]-point1[i])*(point2[i]-point1[i]);
			}

			return std::sqrt(sum);
		}

		template <typename T>
		__inline__ T *Mean(int *lids, int length, Setup setup)
		{
			int n_split = omp_get_max_threads();
			T *temp = (T *)malloc(sizeof(T)*setup.d*n_split);
			memset(temp,0,sizeof(T)*setup.d*n_split);
			T *mean = (T *)malloc(sizeof(T)*setup.d);
			memset(mean,0,sizeof(T)*setup.d);
			auto X=setup.X;
			int d = setup.d;

//    #pragma omp parallel for num_threads(n_split)
			for ( int j = 0; j < n_split; j ++ )
				for ( int i = j; i < length; i += n_split )
					for ( int p = 0; p < setup.d; p ++ )
						temp[ j * d + p ] += X[ lids[ i ] * d + p ];

			for ( int j = 0; j < n_split; j ++ ) {
//#pragma omp parallel for num_threads(n_split)
				for (int p = 0; p < d; p++)
					mean[p] += temp[j * d + p];
			}
			for ( int p = 0; p < d; p ++ ) mean[ p ] /= length;


			return mean;
		}

		__inline__ int level(int node)
		{
			return (int) floor(log(node+1.0)/log(2));
		}

		inline double dist2(double* x, double* y, int d) {
			double k = 0.;
			for (int i=0; i<d; i++) k += pow(x[i] - y[i], 2.);
			return k;
		}

		inline double dist(double* x, double* y, int d) {
			return sqrt(dist2(x, y, d));
		}

		void write2binary(std::string file, int *matrix, uint64_t len);
//
//void k_means(int k, double* p, int n, int d, int* nc);
		void k_means(int k, double* p, int n, int d, int* nc, std::vector<int> lids, std::vector<std::vector<int>> &clusters);

		int decomposition(double *A, int nRows, int nCols, double tolerance, int **skels, double **proj, int **jpvt, Setup &setup);

		void Fsubmatrix(std::vector<int> &amap, std::vector<int> &bmap, double *submatrix, Setup setup);

		void randn(int nrow, int ncol, double * array, double a, double b);

		void submatrix(std::vector<int> &amap, std::vector<int> &bmap, double *submatrix, Setup &setup);

		void Fsubmatrix(int *amap, int lena, int *bmap, int lenb, double *submatrix, Setup &setup);

		void writepair2txt(std::string file, int *pair, int len);

		int findMin(uint64_t *cost, int size);

		void write2binary(std::string file, double *matrix, uint64_t len);
		void writeoffset2binary(std::string file, int *offset, int len);
		void write2txt(std::string file, unsigned long int *offset, int len);
		void write2txt(std::string file, int *offset, int len);

		__inline__ int begin(int l)
		{
			return (1<<l)- 1;
		}

		__inline__ int stop(int l)
		{
			return (1<<(l+1))-2;
		}

		void HeapAdjust(
						int s, int n,
						std::pair<double , int> *NN);

		void HeapSelect(  int n, int k, std::pair<double , int> *Query, std::pair<double , int> *NN);


		template<typename TA, typename TB>
		std::pair<TB, TA> flip_pair( const std::pair<TA, TB> &p )
		{
			return std::pair<TB, TA>( p.second, p.first );
		}; /** end flip_pair() */

		template<typename TA, typename TB>
		std::multimap<TB, TA> flip_map( const std::map<TA, TB> &src )
		{
			std::multimap<TB, TA> dst;
			std::transform( src.begin(), src.end(), std::inserter( dst, dst.begin() ),
							flip_pair<TA, TB> );
			return dst;
		};

	}
}





#endif //PROJECT_NUTIL_H
