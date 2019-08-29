//
// Created by Bangtian Liu on 6/27/19.
//

#include <stdio.h>
#include <iostream>
#include "HMatrix.h"
#include "HGEMM.h"
#include "time_util.h"

using namespace Sympiler::Internal;
using namespace Sympiler;

//entry to MatRox framework
int main(int argc, char *argv[])
{
	std::string pointname=argv[1];
//	std::cout<<pointname<<std::endl;
	int n = atoi(argv[2]);
	int dim = atoi(argv[3]);
	double tau = atof(argv[4]);
// here we need to define entry for cluster tree
	omp_set_num_threads(12);
	double start, end;
	start = omp_get_wtime();
	HTree ctree(Float(64), pointname, n, dim, tau);
	end = omp_get_wtime();
	auto sec_tc = end - start;
	printf("%f,",sec_tc);
	start = omp_get_wtime();
	HGEMM ker(ctree);
	ker.sympile_to_c("../../symGen/HGEMM");
  auto end1=omp_get_wtime();
	printf("%f,\n",end1-start);


	ctree.savetree();
	ctree.savetree2disk();
	ctree.SaveSampling();
//	end = omp_get_wtime();

	return 0;
}
