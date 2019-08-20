//
// Created by Bangtian Liu on 4/28/19.
//

#include <mkl.h>
#include "DSPattern.h"
#include "nUtil.h"
#include <iostream>
#include <fstream>
#include <omp.h>


namespace Sympiler {
	namespace Internal {
		DSPattern::DSPattern() {

		}

		DSPattern::DSPattern(std::string path, int n, int dim, Ktype ktype, double tau, double acc) {
			setup.n = n;
			setup.m = 256; // leaf node size
			setup.d = dim;
			setup.X = (double *) mkl_malloc (sizeof(double)*n*dim, 64);
			memset(setup.X, 0, sizeof(double)*n*dim);

			setup.tau = tau;
			setup.stol = acc;
			setup.ktype = ktype;

			std::ifstream file( path.data(), std::ios::in|std::ios::binary|std::ios::ate );
			if ( file.is_open() )
			{
				auto size = file.tellg();
				assert( size == setup.n * setup.d * sizeof(double) );
				file.seekg( 0, std::ios::beg );
				file.read( (char*)setup.X, size );
				file.close();
			}
//			randn(setup.n, setup.d, setup.X, -1.0, 1.0);


//			for(int i=0; i<setup.n;i++)
//			{
//				for(int j=0; j<setup.d;j++)
//				{
//					printf("%f\t", setup.X[i*setup.d + j]);
//				}
//				printf("\n");
//			}

#ifdef SAVE
			write2binary("xbin", setup.X,setup.n*setup.d);
#endif
//			setup.depth=6;//maybe comment
			this->config();
			tree = new clustertree(setup);

//			printf("depth=%d\n",setup.depth);
			this->Structure();
			tree->compression();

			cds = new CDS(*tree, setup, tree->tmpresult);

		}

		void DSPattern::config() {
			setup.adaptive=true;
			setup.maxRank = setup.m;
			setup.h=10;
			setup.nrhs = 1; // configure later
			setup.recompress = false;
			setup.use_cbs = false;
			setup.use_coarsing = false;
			setup.nthreads = omp_get_max_threads();
		}

		void DSPattern::Structure() {
			isHSS = tree->isHSS(); // HSS structure?
			perfect = tree->isPerfect(); // perfect or general tree?

		}

		void DSPattern::getDecl(std::vector<Expr> &el, std::vector<Argument> &al) {
			tree->getTreeDecl(el,al);

		}
		DSPattern::~DSPattern() {
			delete tree;
		}
	}
}