//
// Created by Bangtian Liu on 5/13/19.
//

#ifndef PROJECT_VICOARSEN_H
#define PROJECT_VICOARSEN_H

#include "IRMutator.h"

namespace Sympiler {
	namespace Internal {

		class VICoarsen : public IRMutator {
			std::string levelset,wpart,idx;
			Expr lb, ub;
			int count=0;
			bool flag;
			void visit(const ForG *for_loop){
				if(for_loop->for_type == ForType::Coarsen) {
					++this->count;
					if(this->count==2)
					{
						Stmt body = for_loop->body;
						Expr imm1 = IntImm::make(halide_type_t(halide_type_int,32),1);
						Expr lbeg = Variable::make(halide_type_t(halide_type_int,32),"i");
						Expr lstop = Add::make(lbeg,imm1);
						Expr wbeg = Pointer::make(halide_type_t(halide_type_int,32), levelset,lbeg);
						Expr wstop = Pointer::make(halide_type_t(halide_type_int,32), levelset, lstop);

						Expr nbeg = Variable::make(halide_type_t(halide_type_int,32), "k");
						Expr nstop = Add::make(nbeg, imm1);

						Expr nbeg1 = Pointer::make(halide_type_t(halide_type_int,32), wpart, nbeg);
						Expr nstop1 = Pointer::make(halide_type_t(halide_type_int,32), wpart, nstop);

						Stmt newbody;
						if(!flag){
							newbody=ForG::make("j", nbeg1,nstop1,ForType::Serial,for_loop->device_api,body,true);
							newbody = ForG::make("k", wbeg, wstop, ForType::Parallel, for_loop->device_api,newbody,true);
							newbody = ForG::make("i",lb,ub,ForType::Serial,for_loop->device_api,newbody,true);
						}
						else {

							Expr imm1 = IntImm::make(halide_type_t(halide_type_int,32),1);
							Expr lb1 = Sub::make(lb,imm1);
							Expr ub1 = Sub::make(ub,imm1);

							Expr nbeg2 = Sub::make(nbeg1, imm1);
							Expr nstop2 = Sub::make(nstop1, imm1);

							newbody = ForG::make("j", nbeg2, nstop2, ForType::Serial,for_loop->device_api,body,false);
							newbody = ForG::make("k", wbeg, wstop, ForType::Parallel, for_loop->device_api,newbody,true);
							newbody = ForG::make("i",lb1,ub1,ForType::Serial,for_loop->device_api,newbody, false);
						}
						stmt = newbody;
						this->count = 0;
					}
					else if(this->count==1) {
						flag = for_loop->flag;
						Stmt body = mutate(for_loop->body);
						stmt = body;
					}
					else {
						IRMutator::visit(for_loop);
					}
				}
				else {
					IRMutator::visit(for_loop);
				}
			}

		public:
			VICoarsen(Expr Lb, Expr Ub, std::string _levelset, std::string _wpart, std::string _idx):
					lb(Lb),ub(Ub), levelset(_levelset), wpart(_wpart), idx(_idx)
			{}
		};

//		int VICoarsen::count = 0;
	}
}



#endif //PROJECT_VICOARSEN_H
