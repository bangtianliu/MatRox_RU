//
// Created by Bangtian Liu on 4/28/19.
//

#ifndef PROJECT_HMATRIX_H
#define PROJECT_HMATRIX_H

#include <stdio.h>
#include "Triangular.h"
#include "Factorization.h"
#include "DSPattern.h"
#include "ClusterTree.h"
#include "nUtil.h"
#include "rstree.h"
#include "ReNearFar.h"

using namespace Sympiler::Internal;
using namespace Sympiler;


namespace Sympiler{
	namespace Internal {



		class HMatrix: public Dense {

public:
//			DSPattern *dPattern;

			Setup setup;

			clustertree *tree;

			std::vector<rstree> rstrees;

			std::vector<Ret> tmpresult;

			std::vector<ReNearFar> reEvals;
			CDS *cds;

			//Paramters for Code Generation
			std::string DM, BM, TVM;
			std::string DM_ptr, BM_ptr, TVM_ptr;
			std::string lchildren;
			std::string rchildren;


			int *leafmap;
			int *Dim;

			int *levelset;
			int *idx;
			int *tlchildren;
			int *trchildren;

			int *woffset;
			int *uoffset;

			int *wpart;
			int *clevelset;

			int *nblockset;
			int *nblocks;
			int *nxval;
			int *nyval;

			int *fblockset;
			int *fblocks;
			int *fxval;
			int *fyval;

			std::vector<int> lids;

//			int *ncomoffset;
//		Type t;
			HMatrix(Type , std::string, int, int,  Ktype, double, double);

			void config(int n, int dim, Ktype ktype, double tau, double acc);

			void compression();

			void compression(int *sid, int *sidlen, int *sidoffset, int *levelset, int *idx, int len);

			void skeletonize(int idx, boundingbox &box, Ret *rtmp);

			void skeletonize(int idx, boundingbox &box, Ret *rtmp, int *sid, int *sidlen, int *sidoffset);

			void binpacking(std::vector<std::vector<int>> &wpartitions, std::vector<std::vector<int>> &owpartitions, int numofbins);


			void LoadPoints(std::string path);

			~HMatrix();


			virtual void getDecl(std::vector<Expr>& , std::vector<Argument>&);


			Expr paccess(std::string mat, Expr e){
				return Pointer::make(halide_type_t(halide_type_float,64),mat,e);
			};

			Expr Iaccess(std::string ptr, Expr e){
				return Pointer::make(halide_type_t(halide_type_int,32),ptr,e);
			}

			void Setnrhs(int nCols);

			Expr pmatrix(std::string mat, Type t){
				return Variable::make(t, mat);
			}

			void savetree();

			void StructureNearBlock(std::vector<std::vector<std::vector<pair<int,int>>>> &blocks);
			void StructureFarBlock(std::vector<std::vector<std::vector<pair<int,int>>>> &blocks);

		};
	}
}






#endif //PROJECT_HMATRIX_H
