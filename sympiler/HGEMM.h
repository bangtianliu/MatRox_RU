//
// Created by Bangtian Liu on 4/28/19.
//

#ifndef PROJECT_HGEMM_H
#define PROJECT_HGEMM_H
#include "Triangular.h"
#include "Factorization.h"
#include "HMatrix.h"
#include "DSInspector.h"
#include "HTree.h"


using namespace Sympiler::Internal;
using namespace Sympiler;


namespace Sympiler {
	namespace Internal {
		class HGEMM : public Kernel {
//			HMatrix *A;
//			Matrix *rhs, *apres;
			HTree *tree;
			std::string mrhs="mrhs";
			std::string apres="apres";

//			int nrhs;

			std::string nrhs = "nrhs";
			bool isHSS;
	/**
	 * here we define the variable for w and result
	 * maybe I will change it later
	 */
			std::string wptr="wptr";
			std::string uptr="uptr";
			std::string wk="wskel";
			std::string wkptr="wskeloffset";

			std::string leafmap="lm";

			std::string uk="uskel";
			std::string ukptr="uskeloffset";

			std::string Ddim="Ddim";
			std::string slen="slen";

			std::string nidx="nidx";
			std::string nidy="nidy";

			std::string fidx="fidx";
			std::string fidy="fidy";

			std::string utmp = "utmp";
			std::string utmpoffset = "utmpoffset";

			std::string ftmp = "ftmp";
			std::string ftmpoffset = "ftmpoffset";
//
			Stmt Near_HSS();

			Stmt Far_HSS();

			Stmt Forward();

			Stmt Backward();

			Stmt Near();

			Stmt Far();
		public:
			HGEMM();
			HGEMM(HTree &_tree);
//			HGEMM(HMatrix &Amat, Matrix &rmat, Matrix &amat);

			void getDecl(std::vector<Expr> &exprList, std::vector<Argument> &argList);


			void getDeclH2(std::vector<Expr> &exprList, std::vector<Argument> &argList);


			Expr paccess(std::string mat, Expr e){
				return Pointer::make(halide_type_t(halide_type_float,64),mat,e);
			};

			Expr Iaccess(std::string ptr, Expr e){
				return Pointer::make(halide_type_t(halide_type_int,32),ptr,e);
			}


			virtual Stmt baseCode();

			Stmt testcode();
			Stmt MKL_GEMM(Expr alpha, Expr beta, Expr A, Expr B, Expr C,
						  Expr nRow, Expr nCol, Expr k,
						  bool transflag);


			Stmt Traversal(string tname, string lname,
						   string levelset, string idx, Stmt body, Expr min, Expr max, bool flag);


			Stmt hss_evaluation();

			Stmt lower(StructureObject *set);

			Expr access(string name, string idx){
				return Pointer::make(halide_type_t(halide_type_int,32),name,
									 Variable::make(halide_type_t(halide_type_int,32), idx));
			}

			Stmt VICoarsenIG(StructureObject *ds);
			Stmt VIBlockIG(StructureObject *ds);

			Stmt NearBlock(StructureObject *set);
			Stmt FarBlock(StructureObject *set);

			Stmt recomForward();
			Stmt recomBackward();
			Stmt recomNear();
			Stmt recomFar();

			~HGEMM();

			virtual void sympile_to_c(std::string fName, Target t);
			virtual void sympile_to_c(std::string fName);
		};
	}
}




#endif //PROJECT_HGEMM_H
