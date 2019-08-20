//
// Created by labuser on 11/25/18.
//

#ifndef PROJECT_VIUNROLL_H
#define PROJECT_VIUNROLL_H
#include "IRMutator.h"
#include "Substitute.h"

namespace Sympiler{
    namespace Internal {
        class VIUnroll : public IRMutator {
                int unroll;
            void visit(const For *for_loop){
                if(for_loop->for_type == ForType::Unrolled)
                {
                    Stmt body = mutate(for_loop->body);
                    std::string loopstr = for_loop->name;
                    Expr index = Variable::make(halide_type_t(halide_type_int,32),loopstr);
                    Expr stride = IntImm::make(halide_type_t(halide_type_int,32),4);

                    Expr times = Div::make(Sub::make(for_loop->extent,for_loop->min),stride);

                    Expr max = Add::make(for_loop->min, Mul::make(times,stride));
                    Stmt s=body;

                    Stmt t;

                    for (int i = 1; i < unroll; ++i) {
                        Expr imm0 = IntImm::make(halide_type_t(halide_type_int,32),i);

                        Expr newindex = Add::make(index,imm0);

                        Stmt newbody = substitute(loopstr,newindex, body);

                        s = Block::make(s,newbody);
                    }

                    s = Foru::make(loopstr,for_loop->min, max, ForType::Unrolled, DeviceAPI::Host, s, stride);

                    t= For::make(loopstr, max, for_loop->extent, ForType::Serial, DeviceAPI::Host, body);

                    stmt = Block::make(s,t);
                }
                else {
                    IRMutator::visit(for_loop);
                }

            }

        public:
            VIUnroll(int ur):
                    unroll(ur){}
        };
    }
}




#endif //PROJECT_VIUNROLL_H
