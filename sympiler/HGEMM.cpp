//
// Created by Bangtian Liu on 4/28/19.
//

#include "HGEMM.h"
#include "Module.h"
#include "Output.h"
#include "DSInspector.h"
#include "VICoarsen.h"
#include "IROperator.h"


namespace Sympiler{
	namespace Internal{

	HGEMM::HGEMM(HTree &_tree) : Kernel("HGEMM"){
		tree = &_tree;

		std::vector<Expr> el;
		std::vector<Argument> al;

		tree->getDecl(el,al);
		args.insert(args.end(), el.begin(), el.end());
		argType.insert(argType.end(), al.begin(), al.end());

		el.clear();
		al.clear();

		tree->tree->getTreeDecl(el,al);
		args.insert(args.end(), el.begin(), el.end());
		argType.insert(argType.end(), al.begin(), al.end());

		el.clear();
		al.clear();

		al.push_back(Argument(mrhs, Argument::Kind::InputBuffer, halide_type_t(halide_type_float, 64),1));
		argType.insert(argType.end(), al.begin(),al.end());

		al.clear();
		al.push_back(Argument(apres, Argument::Kind::InputBuffer, halide_type_t(halide_type_float, 64),1));
		argType.insert(argType.end(), al.begin(),al.end());

		al.clear();
		al.push_back(Argument(nrhs, Argument::Kind::InputScalar, halide_type_t(halide_type_int,32),1));
		argType.insert(argType.end(), al.begin(),al.end());

		el.clear();
		al.clear();
		this->getDecl(el,al);
		args.insert(args.end(),el.begin(), el.end());
		argType.insert(argType.end(),al.begin(), al.end());


		isHSS = tree->tree->isHSS();
		if(!isHSS)
		{
			if(tree->setup.use_cbs==false){
				el.clear();
				al.clear();
				this->getDeclH2(el,al);
				args.insert(args.end(),el.begin(), el.end());
				argType.insert(argType.end(),al.begin(), al.end());
			}
		}
	}

// todo: first make this part works,
	Stmt HGEMM::Near_HSS() {
		Stmt s;

		Expr alpha = FloatImm::make(halide_type_t(halide_type_float,64), 1.0f);
		Expr beta = FloatImm::make(halide_type_t(halide_type_float,64), 0.0f);

		Expr lb = IntImm::make(halide_type_t(halide_type_int,32),0);
		Expr ls = IntImm::make(halide_type_t(halide_type_int,32),tree->tree->getleaf().size());

		Expr min = Iaccess("levelset",ls);
		Expr max = Iaccess("levelset",lb);

		Expr ridx = Variable::make(halide_type_t(halide_type_int, 32),"i");

		Expr imm0 = UIntImm::make(halide_type_t(halide_type_uint, 32), 0);

		Expr offset = Iaccess(tree->DM_ptr, ridx);

		Expr m = Pointer::make(halide_type_t(halide_type_int,32), Ddim, ridx);
		Expr nrhs = Variable::make(halide_type_t(halide_type_int,32),this->nrhs);

		Expr mat = Pointer::make(halide_type_t(halide_type_float, 64),tree->DM,offset);

		Expr woffset = paccess(wptr, ridx);
		Expr tw = Pointer::make(halide_type_t(halide_type_float, 64), mrhs, woffset);

		Expr uoffset = paccess(uptr, ridx);
		Expr tu = Pointer::make(halide_type_t(halide_type_float, 64), apres, uoffset);

		s = MKL_GEMM(alpha,beta, mat, tw, tu, m, nrhs, m, false);

		s= For::make("i", lb, ls, ForType::Parallel, DeviceAPI::Host, s);

//		loweredKer = s;
		return s;
	}

	Stmt HGEMM::Near() {
		Stmt s,t;
		Expr lb=make_const(halide_type_t(halide_type_int, 32),0);
		Expr ub=make_const(halide_type_t(halide_type_int, 32),tree->tree->numNearnodes());

		Expr idx = Variable::make(halide_type_t(halide_type_int, 32),"i");

		Expr nearx = Pointer::make(halide_type_t(halide_type_int,32),nidx,idx);
		Expr neary = Pointer::make(halide_type_t(halide_type_int,32),nidy,idx);

		Expr dimx = Pointer::make(halide_type_t(halide_type_int,32), Ddim, nearx);
		Expr dimy = Pointer::make(halide_type_t(halide_type_int,32), Ddim, neary);

		Expr nrhs = Variable::make(halide_type_t(halide_type_int,32),this->nrhs);
//		Expr nrhs = IntImm::make(halide_type_t(halide_type_int, 32), this->nrhs);

		Expr offset = Iaccess(tree->DM_ptr, idx);
		Expr mat = Pointer::make(halide_type_t(halide_type_float, 64),tree->DM,offset);

		Expr woffset = Iaccess(wptr, neary);
		Expr tw = Pointer::make(halide_type_t(halide_type_float, 64), mrhs, woffset);

		Expr utoffset = Iaccess(utmpoffset, idx);
		Expr utm = Pointer::make(halide_type_t(halide_type_float, 64), utmp,utoffset);

		Expr uoffset = Iaccess(uptr, nearx);

		Expr alpha = FloatImm::make(halide_type_t(halide_type_float,64), 1.0f);
		Expr beta = FloatImm::make(halide_type_t(halide_type_float,64), 0.0f);

		s = MKL_GEMM(alpha, beta, mat, tw, utm, dimx, nrhs, dimy, false);
		s = For::make("i", lb, ub, ForType::Serial, DeviceAPI ::Host, s);

		Expr j = Variable::make(halide_type_t(halide_type_int, 32),"j");
		Expr k = Variable::make(halide_type_t(halide_type_int, 32), "k");

		Expr loc = Add::make(Mul::make(j, dimx), k);

		Expr foffset = Add::make(uoffset, loc);
		Expr ftmpoffset = Add::make(utoffset, loc);

		Expr fu = Pointer::make(halide_type_t(halide_type_float, 64), apres, foffset);
		Expr futmp = Pointer::make(halide_type_t(halide_type_float, 64), utmp, ftmpoffset);

		t = Store::make(fu,"added", Add::make(fu,futmp));

		t = For::make("k", lb, dimx, ForType::Serial, DeviceAPI::Host, t);
		t = For::make("j", lb, nrhs, ForType::Serial, DeviceAPI::Host, t);
		t = For::make("i", lb, ub, ForType::Serial, DeviceAPI::Host, t);

		s = Block::make(s ,t);

		return s;
	}

	Stmt HGEMM::NearBlock(StructureObject *set) {
		argType.push_back(
			Argument(set->nblocksetPtr, Argument::Kind::InputBuffer,
					 halide_type_t(halide_type_int,32),1));

		argType.push_back(
				Argument(set->nblocksPtr, Argument::Kind::InputBuffer,
						 halide_type_t(halide_type_int,32),1));

		argType.push_back(
				Argument(set->nxvalPtr, Argument::Kind::InputBuffer,
						 halide_type_t(halide_type_int,32),1));

		argType.push_back(
				Argument(set->nyvalPtr, Argument::Kind::InputBuffer,
						 halide_type_t(halide_type_int,32),1));

		Stmt s,t;
		Expr lb=make_const(halide_type_t(halide_type_int, 32),0);
		Expr ub=make_const(halide_type_t(halide_type_int, 32),set->nl);

		Expr imm1 = make_one(halide_type_t(halide_type_int, 32));

		Expr tbeg = Variable::make(halide_type_t(halide_type_int,32),"i");
		Expr tstop = Add::make(tbeg, imm1);

		Expr start = Pointer::make(halide_type_t(halide_type_int,32),set->nblocksetPtr,tbeg);
		Expr stop = Pointer::make(halide_type_t(halide_type_int,32),set->nblocksetPtr,tstop);

		Expr secL = Variable::make(halide_type_t(halide_type_int,32),"j");
		Expr secLS = Add::make(secL, imm1);

		Expr Bbeg = Pointer::make(halide_type_t(halide_type_int,32), set->nblocksPtr, secL);
		Expr Bstop = Pointer::make(halide_type_t(halide_type_int,32), set->nblocksPtr, secLS);

		Expr thir = Variable::make(halide_type_t(halide_type_int,32),"k");
//		Expr thirS = Add::make(thirL, imm1);

//		Expr idx = Pointer::make(halide_type_t(halide_type_int,32), set->nvalPtr, thir);

		Expr offset = Pointer::make(halide_type_t(halide_type_int,32), tree->DM_ptr, thir);

//		Expr stride = make_const(halide_type_t(halide_type_int,32),A->tree->getleaf().size());

		Expr nidx = Pointer::make(halide_type_t(halide_type_int,32), set->nxvalPtr,thir);
		Expr nidy = Pointer::make(halide_type_t(halide_type_int,32), set->nyvalPtr,thir);



		Expr mat = Pointer::make(halide_type_t(halide_type_float, 64),tree->DM,offset);

		Expr woffset = paccess(wptr, nidy);
		Expr tw = Pointer::make(halide_type_t(halide_type_float, 64), mrhs, woffset);

		Expr uoffset = paccess(uptr, nidx);
		Expr tu = Pointer::make(halide_type_t(halide_type_float, 64), apres, uoffset);

		Expr alpha = FloatImm::make(halide_type_t(halide_type_float,64), 1.0f);

		Expr dimx = Pointer::make(halide_type_t(halide_type_int,32), Ddim, nidx);
		Expr dimy = Pointer::make(halide_type_t(halide_type_int,32), Ddim, nidy);
		Expr nrhs = Variable::make(halide_type_t(halide_type_int,32),this->nrhs);

		s = MKL_GEMM(alpha, alpha, mat, tw, tu, dimx, nrhs, dimy, false);

		s = For::make("k", Bbeg, Bstop, ForType::Serial, DeviceAPI::Host, s);
		s = For::make("j", start, stop, ForType::Serial, DeviceAPI::Host, s);
		s = For::make("i", lb, ub, ForType::Parallel, DeviceAPI::Host, s);
		return s;
	}

	Stmt HGEMM::FarBlock(StructureObject *set) {
		argType.push_back(
				Argument(set->fblocksetPtr, Argument::Kind::InputBuffer,
						 halide_type_t(halide_type_int,32),1));

		argType.push_back(
				Argument(set->fblocksPtr, Argument::Kind::InputBuffer,
						 halide_type_t(halide_type_int,32),1));

		argType.push_back(
				Argument(set->fxvalPtr, Argument::Kind::InputBuffer,
						 halide_type_t(halide_type_int,32),1));

		argType.push_back(
				Argument(set->fyvalPtr, Argument::Kind::InputBuffer,
						 halide_type_t(halide_type_int,32),1));

		Stmt s,t;
		Expr lb=make_const(halide_type_t(halide_type_int, 32),0);
		Expr ub=make_const(halide_type_t(halide_type_int, 32),set->fl);

		Expr imm1 = make_one(halide_type_t(halide_type_int, 32));

		Expr tbeg = Variable::make(halide_type_t(halide_type_int,32),"i");
		Expr tstop = Add::make(tbeg, imm1);

		Expr start = Pointer::make(halide_type_t(halide_type_int,32),set->fblocksetPtr,tbeg);
		Expr stop = Pointer::make(halide_type_t(halide_type_int,32),set->fblocksetPtr,tstop);

		Expr secL = Variable::make(halide_type_t(halide_type_int,32),"j");
		Expr secLS = Add::make(secL, imm1);

		Expr Bbeg = Pointer::make(halide_type_t(halide_type_int,32), set->fblocksPtr, secL);
		Expr Bstop = Pointer::make(halide_type_t(halide_type_int,32), set->fblocksPtr, secLS);

		Expr thir = Variable::make(halide_type_t(halide_type_int,32),"k");

		Expr offset = Pointer::make(halide_type_t(halide_type_int,32), tree->BM_ptr, thir);


		Expr fidx = Pointer::make(halide_type_t(halide_type_int,32), set->fxvalPtr,thir);
		Expr fidy = Pointer::make(halide_type_t(halide_type_int,32), set->fyvalPtr,thir);

		Expr mat = Pointer::make(halide_type_t(halide_type_float, 64),tree->BM,offset);

		Expr woffset = Pointer::make(halide_type_t(halide_type_int,32),wkptr,fidy);
		Expr tw = Pointer::make(halide_type_t(halide_type_int,32), wk, woffset);

		Expr uoffset = Pointer::make(halide_type_t(halide_type_int,32),ukptr,fidx);
		Expr tu = Pointer::make(halide_type_t(halide_type_int,32),uk, uoffset);

		Expr alpha = FloatImm::make(halide_type_t(halide_type_float,64), 1.0f);

		Expr dimx = Pointer::make(halide_type_t(halide_type_int,32), slen, fidx);
		Expr dimy = Pointer::make(halide_type_t(halide_type_int,32), slen, fidy);
		Expr nrhs = Variable::make(halide_type_t(halide_type_int,32),this->nrhs);

		s = MKL_GEMM(alpha, alpha, mat, tw, tu, dimx, nrhs, dimy, false);

		s = For::make("k", Bbeg, Bstop, ForType::Serial, DeviceAPI::Host, s);
		s = For::make("j", start, stop, ForType::Serial, DeviceAPI::Host, s);
		s = For::make("i", lb, ub, ForType::Parallel, DeviceAPI::Host, s);
		return s;
	}


	/**
	 * @brief Generalized code for farfield blocks, which can deal with both HSS and H^2
	 * @return Statement Node
	 */
	Stmt HGEMM::Far() {
		Stmt s,t;
		Expr lb = make_zero(halide_type_t(halide_type_int, 32));
		Expr ub = make_const(halide_type_t(halide_type_int, 32),tree->tree->numFarnodes());
		Expr idx = Variable::make(halide_type_t(halide_type_int, 32),"i");

		Expr farx = Pointer::make(halide_type_t(halide_type_int, 32), fidx, idx);
		Expr fary = Pointer::make(halide_type_t(halide_type_int, 32), fidy, idx);

		Expr dimx = Pointer::make(halide_type_t(halide_type_int, 32), slen, farx);
		Expr dimy = Pointer::make(halide_type_t(halide_type_int, 32), slen, fary);

		Expr nrhs = Variable::make(halide_type_t(halide_type_int,32),this->nrhs);
		Expr offset = Iaccess(tree->BM_ptr, idx);

		Expr fmat = Pointer::make(halide_type_t(halide_type_float, 64), tree->BM, offset);

		Expr wskeloffset = Iaccess(wkptr,fary);
		Expr wskel = Pointer::make(halide_type_t(halide_type_float, 64), wk, wskeloffset);

		Expr ftoffset = Iaccess(ftmpoffset, idx);
		Expr ftm = Pointer::make(halide_type_t(halide_type_float, 64), ftmp, ftoffset);

		Expr alpha = make_one(halide_type_t(halide_type_float,64));
		Expr beta = make_zero(halide_type_t(halide_type_float,64));

		s = MKL_GEMM(alpha, beta, fmat, wskel, ftm, dimx, nrhs, dimy, false);
		s = For::make("i", lb, ub, ForType::Serial, DeviceAPI::Host, s);

		Expr j = Variable::make(halide_type_t(halide_type_int, 32),"j");
		Expr k = Variable::make(halide_type_t(halide_type_int, 32),"k");

		Expr uskeloffset = Iaccess(ukptr, farx);

		Expr loc = Add::make(Mul::make(j, dimx), k);
		Expr foffset = Add::make(uskeloffset, loc);
		Expr ftmpoffset = Add::make(ftoffset, loc);

		Expr fuskel = Pointer::make(halide_type_t(halide_type_float,64),uk, foffset);
		Expr futmp = Pointer::make(halide_type_t(halide_type_float,64),ftmp, ftmpoffset);

		t = Store::make(fuskel, "added", Add::make(fuskel,futmp));

		t = For::make("k", lb, dimx, ForType::Serial, DeviceAPI::Host, t);
		t = For::make("j", lb, nrhs, ForType::Serial, DeviceAPI::Host, t);
		t = For::make("i", lb, ub, ForType::Serial, DeviceAPI::Host, t);

		s = Block::make(s, t);
		return s;
	}

	Stmt HGEMM::Forward() {
		Stmt s;
		Stmt t;

		Stmt s1, s2;

		Expr imm0 = UIntImm::make(halide_type_t(halide_type_uint, 32), 0);

		Expr tidx = access("idx","j");
		Expr alpha = FloatImm::make(halide_type_t(halide_type_float,64), 1.0f);
		Expr beta = FloatImm::make(halide_type_t(halide_type_float,64), 0.0f);


		Expr rtidx = Iaccess(leafmap,tidx);

		Expr proj = paccess(tree->TVM, Iaccess(tree->TVM_ptr, tidx));
		Expr w = paccess(mrhs, Iaccess(wptr,rtidx));
		Expr wskel = paccess(wk,Iaccess(wkptr,tidx));

		Expr m = Pointer::make(halide_type_t(halide_type_int,32), Ddim,
							  rtidx);
		Expr skel = Pointer::make(halide_type_t(halide_type_int,32),slen,tidx);

		Expr nrhs = Variable::make(halide_type_t(halide_type_int,32),this->nrhs);

		s = MKL_GEMM(alpha, beta, proj,w,wskel, skel, nrhs, m, false);

		Expr min = UIntImm::make(halide_type_t(halide_type_int, 32), -1);
		Expr max = UIntImm::make(halide_type_t(halide_type_int,32), tree->setup.depth-1);

		Expr level = Variable::make(halide_type_t(halide_type_int,32),"i");

		Expr Nimm1 = UIntImm::make(halide_type_t(halide_type_int, 32), -1);

//		Expr condition = EQ::make(level,max);

			Expr lc = Iaccess(tree->lchildren,tidx);
			Expr rc = Iaccess(tree->rchildren,tidx);

		Expr condition = EQ::make(lc, Nimm1);

			Expr lslen = Iaccess(slen,lc);
			Expr rslen = Iaccess(slen,rc);

			Expr lwskel = paccess(wk, Iaccess(wkptr,lc));
			Expr rwskel = paccess(wk, Iaccess(wkptr,rc));

			s1 = MKL_GEMM(alpha, beta, proj, lwskel,wskel, skel, nrhs, lslen,  false);

			Expr offset = Add::make(Mul::make(skel,lslen), Iaccess(tree->TVM_ptr, tidx));
			Expr lproj = paccess(tree->TVM,offset);
			s2 = MKL_GEMM(alpha, alpha, lproj,rwskel,wskel, skel, nrhs, rslen,  false);

		s=IfThenElse::make(condition,s, Block::make(s1,s2));

		s = Traversal("i", "j", "levelset", "idx",s, min, max, false); // --
		return s;
	}

	Stmt HGEMM::Far_HSS() {
		Stmt s;
		Expr imm0 = UIntImm::make(halide_type_t(halide_type_uint, 32), 0);
		Expr imm1 = UIntImm::make(halide_type_t(halide_type_uint, 32), 1);
		Expr imm2 = UIntImm::make(halide_type_t(halide_type_uint, 32), 2);
		Expr imm = UIntImm::make(halide_type_t(halide_type_uint, 32), tree->tree->getnumnodes());

		Expr nrhs = Variable::make(halide_type_t(halide_type_int,32),this->nrhs);

		Expr Variable = Variable::make(halide_type_t(halide_type_int, 32),"i");

		Expr rsib = Add::make(Variable, imm1);
		Expr lsib = Sub::make(Variable, imm1);

		Expr condition = EQ::make(Mod::make(Variable, imm2),imm0);

		Expr faridx = Select::make(condition,lsib,rsib);

		Expr pfaridx = Sub::make(Variable,imm1);


		Expr Far = paccess(tree->BM,Iaccess(tree->BM_ptr,pfaridx));

		Expr wskel = paccess(wk, Iaccess(wkptr,faridx));

		Expr uskel = paccess(uk, Iaccess(ukptr,Variable));

		Expr dimx = Iaccess(slen, Variable);
		Expr dimy = Iaccess(slen, faridx);

		s = MKL_GEMM(imm1, imm0,Far, wskel, uskel, dimx, nrhs,dimy, false);
		s = For::make("i",imm1,imm, ForType::Parallel, DeviceAPI::Host, s);

		return s;
	}


	Stmt HGEMM::Backward() {
		Stmt s,t;
		Stmt s1, s2;

		Expr tidx = access("idx","j");
		Expr alpha = FloatImm::make(halide_type_t(halide_type_float,64), 1.0f);
		Expr beta = FloatImm::make(halide_type_t(halide_type_float,64), 0.0f);
		Expr min = UIntImm::make(halide_type_t(halide_type_uint, 32), 0);
		Expr max = UIntImm::make(halide_type_t(halide_type_uint,32), tree->setup.depth);

		Expr level = Variable::make(halide_type_t(halide_type_int,32),"i");

		Expr imm1 = UIntImm::make(halide_type_t(halide_type_uint, 32), 1);
		Expr Nimm1 = UIntImm::make(halide_type_t(halide_type_int, 32), -1);


		Expr leaf = Sub::make(max,imm1);

		Expr lc = Iaccess(tree->lchildren,tidx);
		Expr rc = Iaccess(tree->rchildren,tidx);

		Expr condition = EQ::make(lc,Nimm1);

		Expr skel = Pointer::make(halide_type_t(halide_type_int,32),slen,tidx);
		Expr nrhs = Variable::make(halide_type_t(halide_type_int,32),this->nrhs);


		Expr uskel = paccess(uk,Iaccess(ukptr,tidx));
		Expr proj = paccess(tree->TVM, Iaccess(tree->TVM_ptr, tidx));

		Expr lslen = Iaccess(slen,lc);
		Expr rslen = Iaccess(slen,rc);

		Expr luskel = paccess(uk, Iaccess(ukptr,lc));
		Expr ruskel = paccess(uk, Iaccess(ukptr,rc));

		s1 = MKL_GEMM(alpha,alpha,proj,uskel,luskel,lslen,nrhs,skel,true);

		Expr offset = Add::make(Mul::make(skel,lslen), Iaccess(tree->TVM_ptr, tidx));
		Expr lproj = paccess(tree->TVM,offset);

		s2 = MKL_GEMM(alpha,alpha,lproj,uskel,ruskel,rslen,nrhs,skel,true);

		Expr rtidx = Iaccess(leafmap,tidx);
		Expr m = Pointer::make(halide_type_t(halide_type_int,32), Ddim,
							   rtidx);
		Expr u = paccess(apres,Iaccess(uptr,rtidx));

		s = MKL_GEMM(alpha,alpha, proj, uskel, u, m, nrhs, skel,true);

		s=IfThenElse::make(condition,s, Block::make(s1,s2));

		s = Traversal("i", "j", "levelset", "idx", s, min, max, true);// ++

		return s;
	}
	Stmt HGEMM::Traversal(string tname, string lname, string levelset, string idx, Stmt body, Expr min, Expr max, bool flag) {
		Stmt s;
		Expr imm1 = IntImm::make(halide_type_t(halide_type_int,32),1);
		Expr imm0 = IntImm::make(halide_type_t(halide_type_int,32),0);
		Expr tbeg = Variable::make(halide_type_t(halide_type_int,32),tname);
		Expr tstop = Add::make(tbeg, imm1);
		Expr start = Pointer::make(halide_type_t(halide_type_int,32),levelset,tbeg);
		Expr stop = Pointer::make(halide_type_t(halide_type_int,32),levelset,tstop);
		s=ForG::make(lname,start,stop, ForType::Coarsen, DeviceAPI::Host, body,true);
		s=ForG::make(tname, min, max, ForType::Coarsen, DeviceAPI::Host, s, flag);

		return s;
	}



	Stmt HGEMM::testcode() {
		Stmt s;
		Stmt t = DeclVariable::make("sum");
		Expr imm0 = UIntImm::make(halide_type_t(halide_type_int,32),0);
		Expr i0 = Variable::make(halide_type_t(halide_type_int, 32),
								 "i");
		Expr vsum = Variable::make(halide_type_t(halide_type_int,32),"sum");
//		Expr sum1 = Let::make("i", imm0, vsum);
		s = Store::make(vsum,"sum2",Add::make(vsum,i0));
		s = Block::make(t,s);


		return s;
	}


	Stmt HGEMM::baseCode() {
		//FIXME: can be replaced by other code, now it is just for testing
		Stmt s,t;

		s=Near();

		t=Forward();

		s=Block::make(s,t);

		s=Block::make(s,Far());

		s=Block::make(s,Backward());

		return s;
	}

	Stmt HGEMM::MKL_GEMM(Expr alpha, Expr beta, Expr A, Expr B, Expr C, Expr nRow, Expr nCol, Expr k,bool transflag) {
		std::vector<Expr> Args;
		std::vector<Argument> Argums;
		Args.push_back(nRow);
		Argums.emplace_back(Argument(false));
		Args.push_back(nCol);
		Argums.emplace_back(Argument(false));
		Args.push_back(k);
		Argums.emplace_back(Argument(false));

		Args.push_back(alpha);
		Argums.emplace_back(Argument(false));

		if(transflag) {
			Args.push_back(A);
			Argums.emplace_back(Argument(true));

			Args.push_back(k);
			Argums.emplace_back(Argument(false));
		}
		else {
			Args.push_back(A);
			Argums.emplace_back(Argument(true));

			Args.push_back(nRow);
			Argums.emplace_back(Argument(false));

		}

		Args.push_back(B);
		Argums.emplace_back(Argument(true));

		Args.push_back(k);
		Argums.emplace_back(Argument(false));

		Args.push_back(beta);
		Argums.emplace_back(Argument(false));

		Args.push_back(C);
		Argums.emplace_back(Argument(true));

		Args.push_back(nRow);
		Argums.emplace_back(Argument(false));

		Stmt s = CallMKL::make("cblas_dgemm", Args, Argums, transflag);

		return s;
	}

	Stmt HGEMM::hss_evaluation() {
		Stmt s,t;
		s = Near_HSS();

		t = Forward();

		s = Block::make(s,t);

		s = Block::make(s, Far_HSS());

		s = Block::make(s, Backward());

		return s;
	}
	void HGEMM::sympile_to_c(std::string fName, Target t) {
		Target target(t);
		HGEMMInspector inspector(tree);
		StructureObject *set = inspector.Inspect();
		loweredKer = this->lower(set);
//				hss_evaluation();
				//Backward();
//				Forward();
//				Forward();
		Module module(fName, target);
		module.append(*this);

		Output::compile_to_source_c(module, fName + "_gen", set->isHSS);
	}

	void HGEMM::sympile_to_c(std::string fName) {
		Target t;
		sympile_to_c(fName, t);
	}

	void HGEMM::getDecl(std::vector<Expr> &el, std::vector<Argument> &al) {
		al.push_back(Argument(Ddim, Argument::Kind::InputBuffer, halide_type_t(halide_type_int,32),1));
		al.push_back(Argument(wptr, Argument::Kind::InputBuffer, halide_type_t(halide_type_int,32),1));
		al.push_back(Argument(uptr, Argument::Kind::InputBuffer, halide_type_t(halide_type_int,32),1));
		al.push_back(Argument(wk, Argument::Kind::InputBuffer, halide_type_t(halide_type_float,64),1));
		al.push_back(Argument(wkptr, Argument::Kind::InputBuffer, halide_type_t(halide_type_int,32),1));
		al.push_back(Argument(uk, Argument::Kind::InputBuffer, halide_type_t(halide_type_float,64),1));
		al.push_back(Argument(ukptr, Argument::Kind::InputBuffer, halide_type_t(halide_type_int,32),1));
		al.push_back(Argument(leafmap, Argument::Kind::InputBuffer, halide_type_t(halide_type_int,32),1));
		al.push_back(Argument(slen, Argument::Kind::InputBuffer, halide_type_t(halide_type_int,32),1));
	}

	void HGEMM::getDeclH2(std::vector<Expr> &el, std::vector<Argument> &al) {
		al.push_back(Argument(nidx, Argument::Kind::InputBuffer, halide_type_t(halide_type_int,32),1));
		al.push_back(Argument(nidy, Argument::Kind::InputBuffer, halide_type_t(halide_type_int,32),1));
		al.push_back(Argument("ncount", Argument::Kind::InputScalar, halide_type_t(halide_type_int,32),1));

		al.push_back(Argument(fidx, Argument::Kind::InputBuffer, halide_type_t(halide_type_int,32),1));
		al.push_back(Argument(fidy, Argument::Kind::InputBuffer, halide_type_t(halide_type_int,32),1));
		al.push_back(Argument("fcount", Argument::Kind::InputScalar, halide_type_t(halide_type_int,32),1));

//		if(!tree->setup.use_cbs)
//		{
			al.push_back(Argument(utmp, Argument::Kind::InputBuffer, halide_type_t(halide_type_float,64),1));
			al.push_back(Argument(utmpoffset, Argument::Kind::InputBuffer, halide_type_t(halide_type_uint,64),1));
			al.push_back(Argument(ftmp, Argument::Kind::InputBuffer, halide_type_t(halide_type_float,64),1));
			al.push_back(Argument(ftmpoffset, Argument::Kind::InputBuffer, halide_type_t(halide_type_uint,64),1));
//		}
	}

	Stmt HGEMM::lower(StructureObject *set){
		Stmt s;
		loweredKer = this->baseCode();

		if(set->isHSS){
			loweredKer = hss_evaluation();
		}


		if(!(set->isHSS)) {
			s = loweredKer;
			s = this->VIBlockIG(set);
			loweredKer = s;
		}

		s = this->VICoarsenIG(set);

//		if(set->isHSS){
//			s = this->VICoarsenIG(set);
//		}
//		if(tree->setup.use_coarsing){
//			loweredKer = this->VICoarsenIG(set);
//		}

		s=loweredKer;
		return s;
	}

	Stmt HGEMM::VICoarsenIG(StructureObject *set) {
		VICoarsen vi_coarsen(make_zero(halide_type_t(halide_type_int, 32)),
							 IntImm::make(halide_type_t(halide_type_int,32),set->cdepth),
							 set->levelPtr, set->wpartPtr, set->idxPtr);
		argType.push_back(
				Argument(set->wpartPtr, Argument::Kind::InputBuffer,
						 halide_type_t(halide_type_int,32),1));

		argType.push_back(
				Argument(set->levelPtr, Argument::Kind::InputBuffer,
						 halide_type_t(halide_type_int,32),1));

//		argType.push_back(
//				Argument(set->idxPtr, Argument::Kind::InputBuffer,
//						 halide_type_t(halide_type_int,32),1));


		loweredKer = vi_coarsen.mutate(loweredKer);

		return loweredKer;
	}

	Stmt HGEMM::VIBlockIG(StructureObject *ds) {
		Stmt s,t;

		s = NearBlock(ds);
		t=Forward();
		s=Block::make(s,t);
		s=Block::make(s,FarBlock(ds));
		s=Block::make(s,Backward());
		return s;
	}

	HGEMM::~HGEMM() {

	}

	}
}
