//
// Created by Bangtian Liu on 4/28/19.
//

#ifndef PROJECT_DSPATTERN_H
#define PROJECT_DSPATTERN_H

#include "ClusterTree.h"
#include "config.h"
#include "CDS.h"

namespace Sympiler {
	namespace Internal {
		class DSPattern {

			bool isHSS;
			bool perfect;

public:
			clustertree* tree;
			CDS* cds;
			Setup setup; // parameters for H^2

			DSPattern();
			// tau is the separation ratio for admissbility condition
			DSPattern(std::string path, int n, int dim, Ktype ktype, double tau, double acc);

			void config();

			void setNrhs(int nCols){setup.nrhs = nCols;}

			void Structure();

			Setup& getConfiguration(){return setup;}


			void getDecl(std::vector<Expr>& , std::vector<Argument>&);

			~DSPattern();
		};
	}
}



#endif //PROJECT_DSPATTERN_H
