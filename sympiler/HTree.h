//
// Created by Bangtian Liu on 6/8/19.
//

#ifndef PROJECT_HTREE_H
#define PROJECT_HTREE_H

#include "ClusterTree.h"

namespace Sympiler {
	namespace Internal {
		class HTree {
		public:

			Setup setup;

			clustertree *tree;

			//Paramters for Code Generation
			std::string DM, BM, TVM;
			std::string DM_ptr, BM_ptr, TVM_ptr;
			std::string lchildren;
			std::string rchildren;

			int *levelset;
			int *idx;

			int *wpart;
			int *clevelset;
			int *cidx;

			int *tlchildren;
			int *trchildren;

			int *nblockset;
			int *nblocks;
			int *nxval;
			int *nyval;

			int *fblockset;
			int *fblocks;
			int *fxval;
			int *fyval;

			int *leafmap;
			int *Dim;

			int *lids;
			int *lidsoffset;
			int *lidslen;
			int nlen;

			HTree(Type t, std::string path, int n, int dim, double tau);

			void config(int n, int dim, double tau);

			void LoadPoints(std::string path);

			void getDecl(std::vector<Expr>& , std::vector<Argument>&);

			void StructureNearBlock();
			void StructureFarBlock();

			void StructureNear();
			void StructureFar();

			void Sampling();
			void SaveSampling();

			void savetree();

			void savetree2disk();

			~HTree();
		};
	}
}




#endif //PROJECT_HTREE_H
