//
// Created by Bangtian Liu on 5/2/19.
//

#include <mkl.h>
#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include "nUtil.h"

namespace Sympiler {
	namespace Internal {
		void k_means(int k, double* p, int n, int d, int* nc, std::vector<int> lids, std::vector<std::vector<int>> &clusters) {
			// pick k random centers
			clusters.resize(k);
			const int kmeans_max_it = 10;
//			random_device rd;
			mt19937 generator(0);
			uniform_int_distribution<> uniform_random(0, n-1);
			double** center = new double*[k];
			for (int c=0; c<k; c++) {
				center[c] = new double[d];
				auto t = uniform_random(generator);
				for (int j=0; j<d; j++)
					center[c][j] = p[lids[t]*d+j];
//            center[c][j] = p[t*d+j];
			}
			int iter = 0;
			int* cluster = new int[n];
			while (iter < kmeans_max_it) {
				// for each point, find the closest cluster center
#pragma omp parallel for
				for (int i=0; i<n; i++) {
					double min_dist = dist2(&p[lids[i]*d], center[0], d);
					cluster[i] = 0;
					for (int c=1; c<k; c++) {
						double dd = dist2(&p[lids[i]*d], center[c], d);
						if (dd < min_dist) {
							min_dist = dd;
							cluster[i] = c;
						}
					}
				}
				// update cluster centers
				for (int c=0; c<k; c++) {
					nc[c] = 0;
					for (int j=0; j<d; j++) center[c][j] = 0.;
				}
				for (int i=0; i<n; i++) {
					auto c = cluster[i];
					nc[c]++;
					for (int j=0; j<d; j++) center[c][j] += p[lids[i]*d+j];
				}
				for (int c=0; c<k; c++)
					for (int j=0; j<d; j++) center[c][j] /= nc[c];
				iter++;
			}

			int* ci = new int[k];
			for (int c=0; c<k; c++) ci[c] = 0;
//    double* p_perm = new double[n*d];
//    int row = 0;
			for (int c=0; c<k; c++) {
				for (int j=0; j<nc[c]; j++) {
					while (cluster[ci[c]] != c) ci[c]++;
//            for (int l=0; l<d; l++) {
//                p_perm[l + row * d] = p[l + lids[ci[c]] * d];
					clusters[c].push_back(lids[ci[c]]);
//            }
					ci[c]++;
//            row++;
				}
			}
//    copy(p_perm, p_perm+n*d, p);
//    delete[] p_perm;
			delete[] ci;

			for (int i=0; i<k; i++) delete[] center[i];
			delete[] center;
			delete[] cluster;
		}

		int decomposition(double *A, int nRows, int nCols, double tolerance, int **skels, double **proj, int **jpvt, Setup &setup)
		{
			assert(nRows > nCols);
			int s;
			int maxRank = setup.maxRank;
//    printf("maxRank=%d\n",maxRank);
			*jpvt = (int *)malloc(sizeof(int)*nCols);
			memset(*jpvt, 0, sizeof(int)*nCols);
//    T *tau = GenMatrix<T>(std::min(nRows,nCols),1);
			double *tau = (double *)mkl_malloc(sizeof(double)*std::min(nRows,nCols),64);

			auto info = LAPACKE_dgeqp3(LAPACK_COL_MAJOR, nRows, nCols, A, nRows, *jpvt, tau);

			if(info!=0){
				printf("%d-th parameter had an illegal value", -info);
			}

#pragma omp parallel for
			for (int i = 0; i < nCols ; ++i) {
				(*jpvt)[i] = (*jpvt)[i]-1;
			}

			for ( s = 1; s < nCols; ++s) {
//        printf("s=%d, a=%f, error=%f nCOls=%d\n",s, A[s*nRows+s],tolerance,nCols);
				if( s>maxRank || std::abs(A[s*nRows+s]) < tolerance) break;
			}

			if(!setup.adaptive)
			{
				s = std::min(maxRank, nCols);
			}

			if(s > maxRank) s = maxRank;
			*skels = (int *)malloc(sizeof(int)*s);
			memcpy(*skels,*jpvt,sizeof(int)*s);
//    memcpy(*skels,*proj, sizeof(int)*s);

//    *proj = GenMatrix<T>(s,nCols);
			*proj = (double *)mkl_malloc(sizeof(double)*s*nCols,64);
			memset(*proj,0, sizeof(double)*s*nCols);

//#pragma omp parallel for
			for ( int j = 0; j < nCols; j ++ )
			{
				for ( int i = 0; i < s; i ++ )
				{
					if ( j < s )
					{
						if ( j >= i ) (*proj)[ j * s + i ] = A[ j * nRows + i ];
						else          (*proj)[ j * s + i ] = 0.0;
					}
					else
					{
						(*proj)[ j * s + i ] = A[ j * nRows + i ];
					}
				}
			}

			if((*proj)[0]==0) return s;  // put on here
			double *R1 = (double *) mkl_malloc(sizeof(double)*s*s, 64);
			memset(R1, 0, sizeof(double)*s*s);  // todo check the segment fault bug here

//#pragma omp parallel for
			for ( int j = 0; j < s; j ++ )
			{
				for ( int i = 0; i < s; i ++ )
				{
					if ( i <= j ) R1[ j * s + i ] = (*proj)[ j * s + i ];
//            if((*proj)[j*s+i]!=(*proj)[j*s+i])printf("NAN FOUND1!!!\n");
				}
			}
//    T *tmp = GenMatrix<T>(s,nCols);
			double *tmp = (double *) mkl_malloc(sizeof(double)*s*nCols,64);
			memcpy(tmp, *proj, sizeof(double)*s*nCols);

			cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper,CblasNoTrans, CblasNonUnit, s, nCols, 1.0, R1, s, tmp, s);

			/** Fill in proj */
			for ( int j = 0; j < nCols; j ++ )
			{
				for ( int i = 0; i < s; i ++ )
				{
					(*proj)[ (*jpvt)[j] * s + i ] = tmp[ j * s + i ];
				}
			}
			return s;
		}

		void Fsubmatrix(std::vector<int> &amap, std::vector<int> &bmap, double *submatrix, Setup setup)
		{
			auto X = setup.X;
			auto d = setup.d;
			auto h = setup.h;

			switch (setup.ktype) {
				case KS_GAUSSIAN: {
#pragma omp parallel for
					for (int j = 0; j < bmap.size(); ++j) {
						for (int i = 0; i < amap.size(); ++i) {
							auto Kij = 0.0;
#pragma omp simd reduction(+:Kij)
							for (int k = 0; k < setup.d; ++k) {
								auto col = bmap[j];
								auto row = amap[i];
								auto tar = X[col * d + k];
								auto src = X[row * d + k];
								Kij += (tar - src) * (tar - src);
							}


							Kij = exp(-Kij / (2 * h * h));
//							Kij = 1;
							submatrix[j * amap.size() + i] = Kij;

						}
					}

					break;
				}

				case KS_LOG: {
					for (int j = 0; j < bmap.size(); j++) {
						for (int i = 0; i < amap.size(); i++) {
							auto Kij = 0.0;
							for (int k = 0; k < setup.d; ++k) {
								auto col = bmap[j];
								auto row = amap[i];
								auto tar = X[col * d + k];
								auto src = X[row * d + k];
								Kij += (tar - src) * (tar - src);
							}

							submatrix[j * amap.size() + i] = -0.5 * log(Kij);

							if (amap[i] == bmap[j])submatrix[j * amap.size() + i] = 1.0;
						}
					}
					break;
				}

				case KS_EXPONENTIAL: {
					for (int j = 0; j < bmap.size(); j++) {
						for (int i = 0; i < amap.size(); i++) {
							auto Kij = 0.0;
							for (int k = 0; k < setup.d; ++k) {
								auto col = bmap[j];
								auto row = amap[i];
								auto tar = X[col * d + k];
								auto src = X[row * d + k];
								Kij += (tar - src) * (tar - src);
							}
							submatrix[j * amap.size() + i] = exp(-sqrt(Kij));
//                    if(i==j)submatrix[j*amap.size()+i]=1.0;
						}
					}
					break;
				}

				case KS_NEWTON: {
					for (int j = 0; j < bmap.size(); j++) {
						for (int i = 0; i < amap.size(); i++) {
							auto Kij = 0.0;
							for (int k = 0; k < setup.d; ++k) {
								auto col = bmap[j];
								auto row = amap[i];
								auto tar = X[col * d + k];
								auto src = X[row * d + k];
								Kij += (tar - src) * (tar - src);
							}
							submatrix[j * amap.size() + i] = sqrt(Kij);
						}
					}

					break;
				}


				default: {
					printf("invalid kernel type\n");
					exit(1);
					break;
				}
			}

		}


		void randn(int nrow, int ncol, double * array, double a, double b)
		{
			std::default_random_engine generator;
			std::normal_distribution<double> distribution( a, b );
//#pragma omp parallel for
			for (int i = 0; i < nrow ; ++i) {
				for (int j = 0; j < ncol; ++j) {
					array[i*ncol + j ] = distribution(generator);
//			array[i*ncol + j ] = 1.0;
//            array[j*ncol + i] =array[i*ncol + j];
				}
			}
		}

		void submatrix(std::vector<int> &amap, std::vector<int> &bmap, double *submatrix, Setup &setup)
		{
			for(int j=0;j<bmap.size();j++)
			{
				for(int i=0;i<amap.size();i++){
					auto col = bmap[j];
					auto row = amap[i];
					submatrix[j*amap.size()+i] = setup.M[col*setup.n+row];
				}
			}
		}



		void Fsubmatrix(int *amap, int lena, int *bmap, int lenb, double *submatrix, Setup &setup)
		{
			auto X = setup.X;
			auto d = setup.d;
//    auto n = setup.n;
			auto h = setup.h;

//    M = (T *)malloc(sizeof(T)*n*n);

			switch (setup.ktype) {
				case KS_GAUSSIAN: {
#pragma omp parallel for collapse(2)
					for (int j = 0; j < lenb; ++j) {
						for (int i = 0; i < lena; ++i) {
							auto Kij = 0.0;
							auto col = bmap[j];
							auto row = amap[i];
#pragma omp simd reduction(+:Kij)
							for (int k = 0; k < setup.d; ++k) {

								auto tar = X[col * d + k];
								auto src = X[row * d + k];
								Kij += (tar - src) * (tar - src);
							}


							Kij = exp(-Kij / (2 * h * h));
//							Kij = 1;
							submatrix[j * lena + i] = Kij;

						}
					}

					break;
				}

				case KS_LOG: {
					for (int j = 0; j < lenb; j++) {
						for (int i = 0; i < lena; i++) {
							auto Kij = 0.0;
							for (int k = 0; k < setup.d; ++k) {
								auto col = bmap[j];
								auto row = amap[i];
								auto tar = X[col * d + k];
								auto src = X[row * d + k];
								Kij += (tar - src) * (tar - src);
							}

							submatrix[j * lena + i] = -0.5 * log(Kij);

							if (amap[i] == bmap[j])submatrix[j * lena + i] = 1.0;
						}
					}
					break;
				}

				case KS_EXPONENTIAL: {
					for (int j = 0; j < lenb; j++) {
						for (int i = 0; i < lena; i++) {
							auto Kij = 0.0;
							for (int k = 0; k < setup.d; ++k) {
								auto col = bmap[j];
								auto row = amap[i];
								auto tar = X[col * d + k];
								auto src = X[row * d + k];
								Kij += (tar - src) * (tar - src);
							}
							submatrix[j * lena + i] = exp(-sqrt(Kij));
//                    if(i==j)submatrix[j*amap.size()+i]=1.0;
						}
					}
					break;
				}

				case KS_NEWTON: {
					for (int j = 0; j < lenb; j++) {
						for (int i = 0; i < lena; i++) {
							auto Kij = 0.0;
							for (int k = 0; k < setup.d; ++k) {
								auto col = bmap[j];
								auto row = amap[i];
								auto tar = X[col * d + k];
								auto src = X[row * d + k];
								Kij += (tar - src) * (tar - src);
							}
							submatrix[j * lena + i] = sqrt(Kij);
//                    if(amap[i]==bmap[j])submatrix[j*amap.size()+i]=0.0;
						}
					}

					break;
				}


				default: {
					printf("invalid kernel type\n");
					exit(1);
					break;
				}
			}
		}


		void write2binary(std::string file, double *matrix, uint64_t len){
			ofstream out(file.data(), ios::out | ios::binary);
 			if(!out){
				std::cout << "Cannot open file.\n";
				return ;
			}
			out.write((char *)matrix, sizeof(double)*len);
			out.close();
		}


		void write2binary(std::string file, int *matrix, uint64_t len){
			ofstream out(file.data(), ios::out | ios::binary);
			if(!out){
				std::cout << "Cannot open file.\n";
				return ;
			}
			out.write((char *)matrix, sizeof(int)*len);
			out.close();
		}


		void writeoffset2binary(std::string file, int *offset, int len)
		{
			ofstream out(file.data(), ios::out | ios::binary);
			if(!out){
				std::cout << "Cannot open file.\n";
				return ;
			}
			out.write((char *)offset, sizeof(int)*len);
			out.close();
		}


		void write2txt(std::string file, unsigned long int *offset, int len)
		{
			ofstream out(file.data());
			for(int i=0; i<len; i++)
			{
				out << offset[i] << "\n";
			}

		}

		void write2txt(std::string file, int *offset, int len)
		{
			ofstream out(file.data());
			for(int i=0; i<len; i++)
			{
				out << offset[i] << "\n";
			}

		}

		void writepair2txt(std::string file, int *pair, int len)
		{
			ofstream out(file.data());
			for(int i=0; i<len; i++)
			{
				out << pair[i*2+0] << " " << pair[i*2+1] << "\n";
			}
		}

		int findMin(uint64_t *cost, int size)
		{
			uint64_t min = std::numeric_limits<uint64_t >::max();
			int minBin=0;

			for(int i=0; i<size; ++i)
			{
				if(cost[i] < min){
					min = cost[i];
					minBin = i;
				}
			}

			return minBin;
		}

		bool compare(Dcost lhs, Dcost rhs)
		{
			return lhs.cost > rhs.cost;
		}

		void HeapAdjust( int s, int n, std::pair<double , int> *NN)
		{
			while ( 2 * s + 1 < n )
			{
				int j = 2 * s + 1;
				if ( ( j + 1 ) < n )
				{
					if ( NN[ j ].first < NN[ j + 1 ].first ) j ++;
				}
				if ( NN[ s ].first < NN[ j ].first )
				{
					std::swap( NN[ s ], NN[ j ] );
					s = j;
				}
				else break;
			}
		}

		void HeapSelect(  int n, int k, std::pair<double , int> *Query, std::pair<double , int> *NN)
		{
			for ( size_t i = 0; i < n; i ++ )
			{
				if ( Query[ i ].first > NN[ 0 ].first )
				{
					continue;
				}
				else // Replace the root with the new query.
				{
					NN[ 0 ] = Query[ i ];
					HeapAdjust( 0, k, NN );
				}
			}

		}
	}
}