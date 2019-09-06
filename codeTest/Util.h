//
// Created by Bangtian Liu on 6/28/19.
//

#ifndef PROJECT_UTIL_H
#define PROJECT_UTIL_H
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <mkl.h>
#include <cstring>

//#include "../sympiler/nUtil.h"

using namespace std;

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

int preprocesbin(std::string name)
{
	ifstream file(name.data(), std::ios::in|std::ios::binary|std::ios::ate);
	auto size = file.tellg();

	return (int)size/sizeof(double);
}

int preprocesoffset(std::string name)
{
	ifstream file(name.data(), std::ios::in|std::ios::binary|std::ios::ate);
	auto size = file.tellg();

	return (int)size/sizeof(int);
}

int preprocesLoffset(std::string name)
{
	ifstream file(name.data(), std::ios::in|std::ios::binary|std::ios::ate);
	auto size = file.tellg();

	return (int)size/sizeof(uint64_t);
}

void bin2read(std::string name, double *mat, int len)
{
	ifstream in(name.data(), ios::in | ios::binary|std::ios::ate);
	in.seekg( 0, std::ios::beg );
	in.read( (char*)mat, len*sizeof(double));
	in.close();
}

void bin2read(std::string name, int *mat, int len)
{
	ifstream in(name.data(), ios::in | ios::binary|std::ios::ate);
	in.seekg( 0, std::ios::beg );
	in.read( (char*)mat, len*sizeof(int));
	in.close();
}

int preprocesstxt(std::string name)
{
	ifstream in(name.data());
	string line;
	int len=0;

	while(getline(in, line)){
		++len;
	}
	return len;
}

void txt2read(std::string name, int *offset)
{
	int index=0;
	ifstream in(name.data());
	string line;
	while(getline(in,line)){
		istringstream liness(line);
		liness >> offset[index];
		++index;
	}
}

void pairtxt2read(std::string name, int *nodex, int *nodey)
{
	int index = 0;
	ifstream in(name.data());
	string line;

	int i, j;
	while(getline(in,line)){
		istringstream liness(line);
		liness >> i >> j;
//		printf("pair (%d, %d)\n",i,j);
		nodex[index]=i;
		nodey[index]=j;
		++index;
	}
	in.close();
}

int processlevelsets(std::string name, int &len)
{
	ifstream in(name.data());
	string line;

	int level, idx;
	int depth=0;
	while(getline(in, line))
	{
		istringstream liness(line);
		liness >> level >> idx ;
		depth = std::max(depth,level);
		++len;
	}
	return depth;
}

void readlevelset(std::string name, int *levelset, int *idx, int len, int n)
{
	ifstream in(name.data());
	string line;
	int l, x;
	int colCnt=0, nnz=0;
	levelset[0]=0;

	for(int i=1; nnz<len;)
	{
		in >> l;
		in >> x;
		if(l==i){
			idx[nnz] = x;
			colCnt++;
			nnz++;
		}
		else {
			levelset[i]=levelset[i-1] + colCnt;
			i++;
			colCnt=1;
			idx[nnz] = x;
			nnz ++;
		}
	}

	levelset[n] = levelset[n-1] + colCnt;
}


void randu(int nrow, int ncol, double * array, double a, double b)
{
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution( a, b );
//#pragma omp parallel for
	for (int i = 0; i < ncol ; ++i) {
		for (int j = 0; j < nrow; ++j) {
			array[i*nrow + j ] = distribution(generator);
			//array[i*nrow + j ] = 1.0;
//            array[j*ncol + i] =array[i*ncol + j];
		}
	}
}

void Fsubmatrix(std::vector<int> &amap, std::vector<int> &bmap, double *submatrix, Ktype ktype, double *X,
				int d, double h
)
{


	switch (ktype) {
		case KS_GAUSSIAN: {
#pragma omp parallel for
			for (int j = 0; j < bmap.size(); ++j) {
				for (int i = 0; i < amap.size(); ++i) {
					auto Kij = 0.0;
#pragma omp simd reduction(+:Kij)
					for (int k = 0; k < d; ++k) {
						auto col = bmap[j];
						auto row = amap[i];
						auto tar = X[col * d + k];
						auto src = X[row * d + k];
						Kij += (tar - src) * (tar - src);
					}


					Kij = exp(-Kij / (2 * h * h));
//					Kij = 1.0;
					submatrix[j * amap.size() + i] = Kij;

				}
			}

			break;
		}

		case KS_LOG: {
			for (int j = 0; j < bmap.size(); j++) {
				for (int i = 0; i < amap.size(); i++) {
					auto Kij = 0.0;
					for (int k = 0; k < d; ++k) {
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
					for (int k = 0; k < d; ++k) {
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
					for (int k = 0; k < d; ++k) {
						auto col = bmap[j];
						auto row = amap[i];
						auto tar = X[col * d + k];
						auto src = X[row * d + k];
						Kij += (tar - src) * (tar - src);
					}
					if(Kij==0)Kij=1;
					submatrix[j * amap.size() + i] = 1/sqrt(Kij);
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


double computeError(int *lids, int len, int nrhs, int n, double *X, int d, double *W, double *U)
{
//	len=100;
	int ntest=10;
//	printf("len=%d\n", len);
	auto amap = std::vector<int>(len);
	auto bmap = std::vector<int>(n);

	for(int i=0;i<len;i++)
	{
		amap[i] = lids[i];
//		printf("idx=%d\n",amap[i]);
	}

	for(int i=0;i<n;i++)
	{
		bmap[i] = i;
	}

	auto Kab = (double *)mkl_malloc(sizeof(double)*len*n,64);
	double *result = (double *)mkl_malloc(sizeof(double)*len*nrhs,64);
	memset(result, 0, sizeof(double)*len*nrhs);

	Ktype ktype = KS_GAUSSIAN;

	Fsubmatrix(amap,bmap,Kab,ktype,X,d,5);

	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
				len,nrhs,n,1.0f,
				Kab,len,
				W, n, 0.0,
				result,len
	);

	double error = 0.0f;
	double nrm2 = 0.0f;
	double averror = 0.0f;
	for(int i=0; i<ntest; ++i)
	{
		error = 0.0f;
		nrm2 = 0.0f;
		for(int j=0; j<nrhs; ++j)
		{
//			printf("i=%d res=%f app=%f\n",i, result[j*len+i], U[j*len+i]);
			error += (result[j*len+i]-U[j*len+i])*(result[j*len+i]-U[j*len+i]);
			nrm2 += (result[j*len+i]*result[j*len+i]);
		}

		error = std::sqrt(error);
		nrm2 = std::sqrt(nrm2);
		averror += error/nrm2;

	}


	return averror/ntest;
}


std::string sadd(string &str)
{
	std::string path = "../sympiler/";
	path.append(str);
	return path;
}

void PrintMatrix(double *mat, int nrow, int ncol, string name)
{
	std::cout<<"Matirx: " << name << "\n";
	for(int i=0;i<nrow;i++)
	{
		for(int j=0;j<ncol;j++)
		{
			printf("%f\t",mat[j*nrow+i]);
		}
		printf("\n");
	}
}


#endif //PROJECT_UTIL_H
