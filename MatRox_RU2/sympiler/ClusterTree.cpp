//
// Created by Bangtian Liu on 4/29/19.
//

#include "nUtil.h"
#include "boundingbox.h"
#include "config.h"
#include <deque>
#include <fstream>
#include "ClusterTree.h"

namespace Sympiler {
	namespace Internal {

		struct admiss{
			int index;
			double val;
		};

		bool adcompare(admiss lhs, admiss rhs)
		{
			return lhs.val < rhs.val;
		}

		std::vector<int> intersection(std::vector<int> v1,
									  std::vector<int> v2){
			std::vector<int> v3;

			std::sort(v1.begin(), v1.end());
			std::sort(v2.begin(), v2.end());

			std::set_intersection(v1.begin(),v1.end(),
								  v2.begin(),v2.end(),
								  back_inserter(v3));
			return v3;
		}


		clustertree::clustertree(Setup &setup) {
			boundingbox box=boundingbox(setup.X, setup.n, setup.d);

			buildtree(box, setup);

			if(setup.use_fmm){
				// to compare with GOFMM
				this->findnearfarnodes_buget(setup.budget);
			}
			else {
				this->findnearfarnodes(0, 0, setup.tau);
				this->MergeFarNodes();
			}

			this->sortnodes();

			setup.use_coarsing=true;
			if(!isHSS()){
				setup.use_cbs=true;
//				this->NearBlock(2);
//				this->FarBlock(4, false);
			}
		}

		void clustertree::sortnodes() {
			for (int i = 0; i < leafnodes.size(); ++i) {
				if(!nearnodes[leafnodes[i]].empty())
					std::sort(nearnodes[leafnodes[i]].begin(),nearnodes[leafnodes[i]].end());
			}

			for(int i=1;i<this->getnumnodes();i++)
			{
				if(!farnodes[i].empty()){
					std::sort(farnodes[i].begin(),farnodes[i].end());
				}
			}
		}
/**
 * TODO:Fix index problem.
 * @param box
 * @param setup
 */
		void clustertree::buildtree(boundingbox &box, Setup &setup)
		{
//			setup = _setup;
			std::deque<int> d;
			d.push_back(0);
			parents.push_back(-1);
			boxes.push_back(box);
			auto equal=setup.equal;
			int offset=0;
			auto dim=setup.d;
			auto points=setup.X;
			while(d.size()!=0){
				std::vector<int> tmplevel;
				int len=(int)d.size();
//        printf("#LEN=%d\n",len);
				for (int i = 0; i < len; ++i) {
					auto idx=d[i];
					boundingbox *cbox=&boxes[idx];

					tmplevel.push_back(idx);
					auto num=cbox->getnum();
					if(num> setup.m){
							if(setup.equal){
								boundingbox box1;
								boundingbox box2;
#ifdef MLPOINT
								cbox->mlbinarysplit_v2(box1,box2,equal, setup);
#else
								cbox->binarysplit(box1,box2,equal, setup);
#endif
								if(box1.getnum()!=0){
									++offset;

									boxes.push_back(box1);
									d.push_back(offset);
									parents.push_back(idx);
								}
								if(box2.getnum()!=0){
									++offset;
									d.push_back(offset);
									boxes.push_back(box2);
									parents.push_back(idx);
								}
							}
							else {
								auto lids=cbox->getlids();
#ifdef MLPOINT
								auto nc = new int[2];
								std::vector<std::vector<int>> clusters;
								k_means(2,points,num,dim,nc, lids, clusters);
								if(nc[0]!=0){
									boundingbox box1=boundingbox(points,clusters[0],nc[0],dim);
									++offset;
									boxes.push_back(box1);
									d.push_back(offset);
									parents.push_back(idx);
								}
								if(nc[1]!=0){
									boundingbox box2=boundingbox(points,clusters[1],nc[1],dim);
									++offset;
									d.push_back(offset);
									boxes.push_back(box2);
									parents.push_back(idx);
								}
#else
								cbox->binarysplit(box1,box2,equal, setup);
                    			if(box1.getnum()!=0){
                        			++offset;
                        			tmplevel.push_back(offset);
                        			boxes.push_back(box1);
                       				d.push_back(offset);
                        		    parents.push_back(idx);
                    			}
                                if(box2.getnum()!=0){
                        			++offset;
                       			    tmplevel.push_back(offset);
                                    d.push_back(offset);
                                    boxes.push_back(box2);
                                    parents.push_back(idx);
                    			}
#endif
							}

					}
					else{
							leafnodes.push_back(idx);
					}

				}

				for(int i=0;i<len;i++)
				{
					d.pop_front();
				}

				levelsets.push_back(tmplevel);
				++depth;
			}
			--depth;
			setup.depth=depth;
//			_setup.depth=depth;
			findleafmap();

			int internal = parents.size() - leafnodes.size();
			children.resize(parents.size());
			nearnodes.resize(parents.size());
			farnodes.resize(parents.size());
			tmpresult.resize(parents.size());

			std::vector<int> numchildren;
			numchildren.resize(parents.size());

			for (int k = 1; k < parents.size(); ++k) {
				children[parents[k]].push_back(k);
				++numchildren[parents[k]];
			}

			std::vector<int> &numtmp=numchildren;
			for (auto it = leafnodes.begin(); it != leafnodes.end() ; ++it) {
				--numtmp[parents[*it]];
			}
			int index=0;
			for(int i=depth;i>=0;i--)
			{
				if(i==depth){
					std::set<int> tmp(leafnodes.begin(), leafnodes.end());
					omplevelsets.push_back(tmp);
				}
				else {
					auto tmp=omplevelsets[index-1];
					std::set<int> tmp1;
					for(auto it = tmp.begin(); it!=tmp.end(); ++it)
					{
//                    printf("idx=%d parent %d children %d\n",*it, parents[*it],numtmp[parents[*it]]);
						if(numtmp[parents[*it]]==0){
//                        printf("INSERT %d\n",parents[*it]);
							auto retval=tmp1.insert(parents[*it]);
							if(retval.second && parents[parents[*it]]>=0){
								--numtmp[parents[parents[*it]]];
//                            printf("MINUS %d %d\n",parents[parents[*it]],numtmp[parents[parents[*it]]]);
							}
						}
					}
					omplevelsets.push_back(tmp1);
				}
				++index;
			}

			for(int i=0;i<omplevelsets.size();i++){
				std::vector<int> tmp(omplevelsets[i].begin(),omplevelsets[i].end());
				sflevelsets.push_back(tmp);
			}
#ifdef SAVE


#endif
			this->mapskelindex();
//    this->findnearfarnodes(0,0,
// setup.tau);

		}
		/**
		 * @check whether the binary tree is perfect or not
		 * @return if flag is true, HSS structure; otherwise H^2/FMM structure
		 */
		bool clustertree::isHSS() {
			for(auto &v:leafnodes){
				if(nearnodes[v].size()!=1) return false;
			}
			return true;
		}

		bool clustertree::isPerfect() {
			int idx=0;
			for(int i=depth; i>=1; i--)
			{
				auto nNodes = sflevelsets[idx].size();
				if(nNodes!=(1<<nNodes))return false;
				++idx;
			}

			return true;
		}

		int clustertree::numNearnodes() {
			auto &leaf = leafnodes;
			auto &near = nearnodes;

			int count = 0;
			for(auto &v : leaf){
				count += near[v].size();
			}

			return count;
		}

		int clustertree::numFarnodes() {
			auto &far = farnodes;
			auto numnodes = getnumnodes();

			int count=0;

			for(int i=1; i<numnodes; i++){
				count += far[i].size();
			}

			return count;
		}

/**
 * @brief find near and far nodes for each tree node
 * @param idx bounding box idx
 * @param idy bounding box idy
 * @param tau admissbility condition parameter
 */

		void clustertree::findnearfarnodes(const int idx, const int idy, const double tau)
		{

//	printf("idx=%d idy=%d\t",idx,idy);
			if(well_sperated(boxes[idx],boxes[idy],tau))
			{
				farnodes[idx].push_back(idy);
			}
			else{
				const int flag = children[idx].empty() << 1 | children[idy].empty();

				switch (flag){
					case 3:
						nearnodes[idx].push_back(idy);
						break;
					case 2:
						for (auto &v:children[idy]) findnearfarnodes(idx,v,tau);
						break;
					case 1:
						for(auto &v:children[idx]) findnearfarnodes(v,idy,tau);
						break;
					default:
						for(auto &vi: children[idx])
							for(auto &vj: children[idy])
								findnearfarnodes(vi,vj,tau);
						break;
				}
			}
		}

		void clustertree::findleafmap() {
			for(int i=0;i<leafnodes.size();i++){
				auto v=leafnodes[i];
				leafoffset.insert(std::pair<int,int>(v,i));
			}
		}

		void clustertree::allocatenids() {
			int nnodes = this->getnumnodes();
			pnids.resize(nnodes);
			snids.resize(nnodes);

			ordered_snids.resize(nnodes);

		}
/**
 * @brief: prune the nodes which don't have far nodes.
 */
		void clustertree::prune() {

			auto numnodes = getnumnodes();
			pflags.resize((unsigned long)numnodes);
			auto &levels = omplevelsets[omplevelsets.size()-2];

			auto flag = false;

			for(auto &v:levels)
			{
				if(farnodes[v].empty()){
					flag=true;
				}
			}
			if(!flag) return;

			auto len = omplevelsets.size();

			pflags[0]=true;

			{
				auto &level = sflevelsets[len-2];
				int index=0;
				for(auto &v:level){
					pflags[v]=false;
					if(farnodes[v].empty())
					{
						pflags[v]=true;
						level.erase(level.begin()+index,level.begin()+index+1);
						++npnodes;
						--index;
					}
					++index;
				}

//		if(level.size()==omplevelsets[len-2].size())break;
				if(level.empty())++nplevels;
			}


			for(auto i=len-3;i>0;i--)
			{
//		printf("I i=%d\n",i);
				auto &level = sflevelsets[i];
//        int index=0;
				for(int j=0;j<level.size();j++)
				{
					auto v = level[j];
					pflags[v]= false;
					if(farnodes[v].empty() && pflags[parents[v]])
					{
						pflags[v]=true;
						level.erase(level.begin()+j,level.begin()+j+1);
						++npnodes;
						--j;
					}
//            ++index;
				}

				if(level.size()==omplevelsets[i].size())break;
				++nplevels;
			}

			int pos=0;
			int plen=0;
			for(auto i=sflevelsets.size()-2;i>0;--i)
			{
				if(sflevelsets[i].empty()){
					pos=(int)i;
					++plen;
				}
			}
			sflevelsets.erase(sflevelsets.begin()+pos,sflevelsets.begin()+pos+plen);
			nfplvels=plen;
		}

		void clustertree::findnearfarnodes_buget(double budget) {
			findnearnodes_buget(budget);
			for(int i=0; i<sflevelsets.size(); i++)
			{
				auto level = sflevelsets[i];
				for(auto &idx : level)
				{
					MergeFarNodes(idx);
				}
			}
/**
 * @brief: here we use level set to find far nodes and merge far nodes along the tree
 *
 */
//			bfindnearfarnodes(0,0);
//
//			MergeFarNodes();
		}




/**
 * @brief: this is for FMM case to compare with GOFMM
 * @param idx
 * @param idy
 * @return
 */
		bool clustertree::bwell_sperated(const int idx, const int idy) {
			if(idx==idy) return false;
			int beg,end;
			findleafnodes(idx,beg,end);

			bool ret = false;
			for (int i = beg; i <= end; ++i) {
				ret = containany(idy, nearnodes[i]);
				if(ret==true) return !ret;
			}

			return !ret;
		}

		void clustertree::findleafnodes(int idx, int &beg, int &end) {
			auto cl = level(idx);
			auto ll = level(leafnodes[0]);

			beg=idx;
			end=idx;

			for(int i=cl;i<ll;i++)
			{
				beg=2*beg+1;
				end=2*end+2;
			}

		}


		void clustertree::bfindnearfarnodes(const int idx, const int idy)
		{
			if(bwell_sperated(idx,idy))
			{
				farnodes[idx].push_back(idy);
//		printf("idx=%d idy=%d\n",idx,idy);
			}
			else {
				const int flag = children[idx].empty() << 1 | children[idy].empty();
				switch (flag){
					case 3:
//				printf("test idx=%d idy=%d\n",idx,idy);
						break;
					case 2:
						for(auto &v:children[idy]) bfindnearfarnodes(idx,v);
						break;
					case 1:
						for(auto &v:children[idx]) bfindnearfarnodes(v,idy);
						break;
					default:
						for(auto &vi: children[idx])
							for(auto &vj: children[idy])
								bfindnearfarnodes(vi, vj);
				}

			}
		}


		void clustertree::findnearnodes_buget(double budget) {
			auto length = leafnodes.size();

//    admiss ad[length*length];
//	std::vector<admiss> ad;
//	ad.resize(length*length);
			admiss *ad = (admiss *)malloc(sizeof(admiss)*length*length);
			//#pragma omp parallel for
//	printf("begin length=%d\n",length);
			for(int i=0;i<length;i++)
			{
				auto &boxi=boxes[leafnodes[i]];
				auto ri = boxi.getradius();
				auto mpointi=boxi.getmid();
//		printf("v=%d\t",leafnodes[i]);
				for(int j=0;j<length;j++)
				{
//			printf("w=%d\n",leafnodes[j]);
					auto &boxj=boxes[leafnodes[j]];
					auto rj = boxj.getradius();
					auto mpointj=boxj.getmid();
					auto gdist = dist(mpointi,mpointj,boxi.getdim());
					ad[i*length+j].val=gdist/(ri+rj);
					ad[i*length+j].index=leafnodes[j];
				}
			}

			for(int i=0;i<length;i++)
			{
				std::sort(ad+i*length,ad+(i+1)*length, adcompare);

				auto lnode = leafnodes[i];
				nearnodes[lnode].clear();
				nearnodes[lnode].push_back(lnode);
				for(int j=1;j<length;j++)
				{
					int len=nearnodes[lnode].size();
					if(len<(int)(length*budget-1))
					{
//                nearnodes[lnode].push_back(leafnodes[j]);
						nearnodes[lnode].push_back(ad[i*length+j].index);
					}
					else {
						break;
					}
				}
			}

//			bool symmetric = true;
//
//			if(symmetric) {
//				for (int i = 0; i < length; ++i) {
//					auto leafx = leafnodes[i];
//					for (auto it = nearnodes[leafx].begin(); it != nearnodes[leafx].end(); ++it) {
//						auto ret = std::find(nearnodes[*it].begin(),nearnodes[*it].end(),leafx);
//						if(ret==nearnodes[*it].end()){
//							// not found
//							nearnodes[*it].push_back(leafx);
////					 printf("debug: %d %d\n",*it,leafx);
//						}
////                 else {
////                     //found
////                 }
//					}
//				}
//			}


		}

/**
 * @brief: 	Pop the common far nodes of children nodes from bottom to top
 */
		void clustertree::MergeFarNodes() {
			int l = levelsets.size() - 1;
			for (auto i = l; i>=0; --i) {
				auto &level = levelsets[i];
				for(auto &v:level)
				{
					auto kids = children[v];
					if(!kids.empty())
					{
						auto lkid = kids[0];
						std::vector<int> tmp;
						for(int j=1;j<kids.size();j++)
						{
							auto cid = kids[j];
							if(j==1){
								tmp=intersection(farnodes[lkid],farnodes[cid]);
							}
							else {
								tmp=intersection(farnodes[cid],tmp);
							}
						}

						if(!tmp.empty())
						{
							for(auto &cw:tmp)
							{
								for(auto &ck: kids)
								{

									auto search = std::find(farnodes[ck].begin(),farnodes[ck].end(),cw);
									farnodes[ck].erase(search);

									auto search1 = std::find(farnodes[cw].begin(),farnodes[cw].end(),ck);
									farnodes[cw].erase(search1);

								}

								farnodes[v].push_back(cw);
								farnodes[cw].push_back(v);
							}
						}
					}
				}
			}

		}


		void clustertree::MergeFarNodes(int idx) {
			if(idx==0) {
				return ;
			}
			if(children[idx].empty()){
				FindFarNodes(0, idx);
			}
			else {
				auto lc = 2*idx + 1;
				auto rc = 2*idx + 2;
				for (auto it = farnodes[lc].begin(); it != farnodes[lc].end() ; ++it) {

//            printf("node=%d, far=%d\n",idx,*it);
//					auto search = farnodes[rc].find(*it);

					auto search = std::find(farnodes[rc].begin(), farnodes[rc].end(),*it);


					if(search!=farnodes[rc].end()){
						farnodes[idx].push_back(*it);
//                printf("XXXXXtest,node=%d = %d\n", node, *it);
						auto search1= std::find(farnodes[lc].begin(), farnodes[lc].end(), *it);
						farnodes[lc].erase(search1);
						farnodes[rc].erase(search);
						--it;
					}

				}
			}

		}





		void clustertree::FindFarNodes(int root, int node) {
			assert(children[node].empty());
			auto &nnodes=nearnodes[node];

			int lchild = 2*root+1;
			int rchild = 2*root+2;

			bool ret = containany(root,nnodes);

			if(ret){
				if(!children[root].empty()){
					FindFarNodes(lchild,node);
					FindFarNodes(rchild,node);
				}
			}
			else{
				farnodes[node].push_back(root);
			}

		}

/**
 * this code only for binary perfect tree to do comparision with GOFMM
 * @todo: we can optimize this function in future if we found the performance is not good.
 * @param root
 * @param NearNodes
 * @return
 */
		bool clustertree::containany(int root, std::vector<int> &NearNodes)
		{
			for(auto it = NearNodes.begin(); it!=NearNodes.end();++it)
			{
				if(isancestor(*it,root))
				{
					return true;
				}
			}
			return false;
		}


		bool clustertree::isancestor(int node, int root)
		{
			int melevel = level(node);
			int rootlevel = level(root);

			int leftrange;
			int rightrange;
			leftrange=rightrange=root;
			if(rootlevel>melevel) return false;
			else{
				for (int i = rootlevel+1; i <= melevel ; ++i) {

					leftrange = 2 * leftrange + 1;
					rightrange = 2 * rightrange + 2;
				}

				return  (node<=rightrange) && (node>=leftrange);

			}
		}
/**
 * @brief partitioning tree levels
 * @param agg  Every agg levels are merged
 * @param llevel number of last llevel levels from bottom to top
 * @FIIXME There is a bug about level partitioning
 */
		void clustertree::parlevel(int agg, int llevel)
		{
			int tlevels = (sflevelsets.size()-1 -llevel + (agg-1))/agg + 1;

			coarlevelsets.resize(tlevels);

			for (int i = 0; i < llevel; ++i) {
				coarlevelsets[0].push_back(i);
			}

			int index = 1;
			for(int i = llevel; i < sflevelsets.size()-1;i++)
			{
				coarlevelsets[index].push_back(i);
				if(coarlevelsets[index].size()==agg) ++index;
			}
		}

/**
 * @brief generate w partitions in post order from bottom to top
 * @FIXME consider the conidtion that tree is pruned
 * This one is for binary complete tree. If setup.equal is false,
 * we need to use another function
 **/

		void clustertree::genpostw() {
			postw.resize(coarlevelsets.size());
			for(int i=0;i<coarlevelsets.size();i++)
			{
				auto tl=coarlevelsets[i].back();
				postw[i].resize(sflevelsets[tl].size());
//		auto levels = coarlevelsets[i].back()-coarlevelsets[i].front()+1;
				auto elevel = coarlevelsets[i].front();

				for(int j=0;j<sflevelsets[tl].size();j++)
				{
					auto idx = sflevelsets[tl][j];
					postorder(postw[i][j], idx, tl, elevel);
				}

//		int index = sflevelsets[tl].size();
				/**
				 * TODO
				 */
				if(npnodes!=0)
				{
					for(int l=tl-1;l>=elevel;l--)
					{
						std::vector<int> tmp;
						for(int j=0; j<sflevelsets[l].size();j++)
						{

							auto idx = sflevelsets[l][j];
							if(pflags[parents[idx]]){
								tmp.clear();
								postorder(tmp, idx, l, elevel);
								postw[i].push_back(tmp);
							}

						}
					}
				}
			}
		}


		void clustertree::imgenpostw() {
			levelmapping();
			postw.resize(coarlevelsets.size());
			for(int i=0;i<coarlevelsets.size();i++)
			{
				auto tl=coarlevelsets[i].back();
				postw[i].resize(sflevelsets[tl].size());
//		auto levels = coarlevelsets[i].back()-coarlevelsets[i].front()+1;
				auto elevel = coarlevelsets[i].front();

				for(int j=0;j<sflevelsets[tl].size();j++)
				{
					auto idx = sflevelsets[tl][j];
					postorder_v2(postw[i][j], idx, tl, elevel);
				}

				/**
				 * Here, need to consider two cases: 1. tree is pruned 2. imbalance partitioned
				 */
				if(npnodes==0)
				{
					for(int l=tl-1;l>=elevel;l--)
					{
						std::vector<int> tmp;
						for(int j=0; j<sflevelsets[l].size();j++)
						{
							auto idx = sflevelsets[l][j];
							auto plevel = levelmap.at(parents[idx]);
							if(plevel>tl){
								tmp.clear();
								postorder_v2(tmp, idx, l, elevel);
								postw[i].push_back(tmp);
							}
						}
					}
				}
				else {
					for(int l=tl-1;l>=elevel;l--)
					{
						std::vector<int> tmp;
						for(int j=0; j<sflevelsets[l].size();j++)
						{
							auto idx = sflevelsets[l][j];
							if(pflags[parents[idx]]){
								tmp.clear();
								postorder_v2(tmp, idx, l, elevel);
								postw[i].push_back(tmp);
							}
							else {
								auto plevel = levelmap.at(parents[idx]);
								if(plevel>tl){
									tmp.clear();
									postorder_v2(tmp, idx, l, elevel);
									postw[i].push_back(tmp);
								}
							}
						}
					}
				}
			}
		}


/**
 * @brief post order traversal of the tree
 * @param wpartition
 * @param index
 * @param clevel
 * @param elevel
 */

		void clustertree::postorder(std::vector<int>& wpartition, int index, int clevel, int elevel) {
			if(clevel==elevel){
				wpartition.push_back(index);
				return ;
			}

			for(int i=0; i<children[index].size();i++)
			{
				int klevel = clevel-1;
				postorder(wpartition,children[index][i], klevel, elevel);
			}
			wpartition.push_back(index);
		}


/**
 * @brief this is for general tree after generating level sets
 * @param wpartition
 * @param index
 * @param clevel
 * @param elevel
 */
		void clustertree::postorder_v2(std::vector<int>& wpartition, int index, int clevel, int elevel) {
			if(clevel==elevel){
				wpartition.push_back(index);
				return ;
			}

			for(int i=0; i<children[index].size();i++)
			{
				int klevel = clevel-1;

				auto cindex = children[index][i];
				auto clevel = levelmap.at(cindex);
				if(clevel>=elevel)postorder_v2(wpartition,children[index][i], klevel, elevel);
			}
			wpartition.push_back(index);
		}

/**
 * @brief build the mapping between index and level.
 */
		void clustertree::levelmapping() {

			for(int i=0;i<sflevelsets.size();i++)
			{
				for(int j=0;j<sflevelsets[i].size();j++)
				{
					levelmap.insert(std::make_pair(sflevelsets[i][j],i));
				}
			}
		}

		// write the leveset for generated code
		void clustertree::writelevelset() {
			/**Store in a CSR-like format**/
			int level=1;
			ofstream levelfile("level.txt");
			for(int i=sflevelsets.size()-2; i>=0; i--)
			{
				auto &levels=sflevelsets[i];
				for(int j=0; j<levels.size();j++)
				{
					auto idx = levels[j];
					levelfile<<level<<"\t"<<idx<<"\n";
				}
				++level;
			}
			levelfile.close();
		}



		void clustertree::getTreeDecl(std::vector<Expr> &el, std::vector<Argument> &al) {
			al.push_back(Argument(levelset,Argument::Kind::InputBuffer, halide_type_t(halide_type_int,32),1));
			al.push_back(Argument(idx, Argument::Kind::InputBuffer, halide_type_t(halide_type_int, 32),1));
		}

		/**
		 * Blocking on Near Interactions based on blockSize
		 * @param blockSize
		 */
		void clustertree::NearBlock(int blockSize) {
			int nBlockSet = (leafnodes.size() - 1 + blockSize) / blockSize;
			nearblocks.resize(nBlockSet);

			std::vector<std::vector<int>> xyblock;
//			std::map<int, int> buckmap;

			xyblock.resize(nBlockSet);


			unsigned *usedBlocks = new unsigned[nBlockSet];

			ncount = 0;

			for(int i = 0; i<leafnodes.size(); i+=blockSize)
			{
				int xbeg = i;
				int xstop = min(i+blockSize, (int)leafnodes.size());

				memset(usedBlocks, 0, sizeof(unsigned)*nBlockSet);
				auto idx = i/blockSize;

				for(int j=xbeg; j<xstop; ++j)
				{
					for(auto &w:nearnodes[leafnodes[j]])
					{
						auto y = leafoffset.at(w);
						auto idy = y / blockSize;
						++ncount;
						if(usedBlocks[idy]==0)
						{
							usedBlocks[idy]=1;
							xyblock[idx].push_back(idy);
						}

					}
				}


				nearblocks[idx].resize(xyblock[idx].size());

				sort(xyblock[idx].begin(), xyblock[idx].end());

				for(int j=0; j<xyblock[idx].size(); j++)
				{
					nbuckmap.insert(pair<int, int>(idx*nBlockSet + xyblock[idx][j],j));
				}

			}

			for(auto &v:leafnodes)
			{
				auto x = leafoffset.at(v);
				auto idx = x / blockSize;

				for(auto &w:nearnodes[v])
				{
					auto y = leafoffset.at(w);
					auto idy = y / blockSize;

					auto id = idx*nBlockSet+idy;

					int offset = nbuckmap.at(id);
					nearblocks[idx][offset].push_back(make_pair(x,y));
				}
			}
		}


		void clustertree::FarBlock(int blockSize, bool flag) {
			std::vector<int> nId;
			std::map<int, int> mapIdx;
			if(flag){
				nId = cnodeId;
				mapIdx = cskelmindex;
			}
			else {
				nId = nodeId;
				mapIdx = skelmidex;
			}
			int Nnodes = getnumnodes()-1;
			int fBlockSet = (Nnodes -1 + blockSize) / blockSize;
			farblocks.resize(fBlockSet);

			std::vector<std::vector<int>> xyblock;
			std::map<int, int> buckmap;

			xyblock.resize(fBlockSet);
			unsigned *usedBlocks = new unsigned[fBlockSet];

			fcount = 0;

			for(int i=0; i<nId.size();i+=blockSize)
			{
				int xbeg = i;
				int xstop = min(i+blockSize, (int)nId.size());

				memset(usedBlocks, 0, sizeof(unsigned)*fBlockSet);
				auto idx = i/blockSize;

				for(int j = xbeg; j<xstop; ++j)
				{
					for(auto &w:farnodes[nId[j]])
					{
						++fcount;

						auto y = mapIdx.at(w);
						auto idy = y/blockSize;
						if(usedBlocks[idy]==0)
						{
							usedBlocks[idy]=1;
							xyblock[idx].push_back(idy);
						}
					}
				}

				farblocks[idx].resize(xyblock[idx].size());
				sort(xyblock[idx].begin(), xyblock[idx].end());

				for(int j=0; j<xyblock[idx].size();j++)
				{
					fbuckmap.insert(pair<int,int>(idx*fBlockSet+xyblock[idx][j],j));
				}

			}

			for(auto &v:nId){
				auto x = mapIdx.at(v);
				auto idx = x / blockSize;
				for(auto &w:farnodes[v])
				{
					auto y = mapIdx.at(w);
					auto idy = y/blockSize;
					auto id = idx * fBlockSet + idy;
					int offset = fbuckmap.at(id);
					farblocks[idx][offset].push_back(make_pair(v,w));
				}
			}
		}

/**
 * @brief: mapping skelindex (mapping index)
 *
 */
		void clustertree::mapskelindex() {
			nodeId.resize(getnumnodes()-1);
			int index=0;
			for(int i = 0; i<omplevelsets.size()-1;++i){
				auto levels=omplevelsets[i];
				for(auto &v:levels)
				{
					nodeId[index] = v;
					skelmidex.insert(std::pair<int,int>(v,index));
					++index;
				}
			}
		}

		void clustertree::coarmapskelidx() {
			cnodeId.resize(getnumnodes()-1);

			int index = 0;

			auto &posw = this->opostw;

			for(int i=0; i<posw.size();i++)
			{
				auto &lpow=posw[i];
				for(int j=0; j<lpow.size(); j++)
				{
					auto wpart=lpow[j];
					for(auto &v:wpart)
					{
						cskelmindex.insert(std::pair<int, int>(v,index));
						cnodeId[index] = v;
						++index;
					}
				}
			}
		}


		void clustertree::BalanceCoarLevelSet() {
		   auto len = postw.size();
			opostw.resize(len);

//       		auto &pow=tree->postw;
//       		auto &opow=tree->opostw;
       		for(int i=0;i<postw.size();i++){
          		auto &lpow=postw[i];
          		int nw = lpow.size();

          		int nparts;

          		int nthreads=omp_get_max_threads();

          		if(nw>=nthreads){
             		nparts=nthreads;
          		}
          		else {
             		nparts=nw/2;
          		}
          		if(nparts==0)nparts=1;
          		opostw[i].resize(nparts);

				binpacking(lpow,opostw[i],nparts);
      		 }
		}

		void clustertree::binpacking(std::vector<std::vector<int>> &wpartitions,
									 std::vector<std::vector<int>> &owpartitions,
									 int numofbins) {
			auto &children = this->getchildren();
			auto &leafmap = this->getleafoffset();
//	auto &boxes = tree.getboxes();
			Dcost *ccost = new Dcost[wpartitions.size()];

			for(auto i = 0; i<wpartitions.size(); i++)
			{
				ccost[i].cost=0;
				ccost[i].index=i;
				for(auto j=0; j<wpartitions[i].size(); j++)
				{
					auto idx = wpartitions[i][j];
					unsigned long cost = 0;
					if(children[idx].empty())
					{
						cost+= 2* tmpresult[idx].skels_length * tmpresult[idx].proj_column * 1;
					}
					else {
						for(auto &v : children[idx])
						{
							cost += 2*tmpresult[idx].skels_length*tmpresult[v].skels_length * 1;
						}
					}
					ccost[i].cost+=cost;
				}
			}

			std::sort(ccost,ccost+wpartitions.size(),compare);

			uint64_t *ocost = new uint64_t[numofbins];

			memset(ocost, 0, sizeof(uint64_t)*numofbins);

			int partNo = wpartitions.size();

			int minBin = 0;
			for(int i = 0; i < partNo; i++)
			{
				minBin = findMin(ocost,numofbins);
				ocost[minBin] += ccost[i].cost;
				int index = ccost[i].index;
//owpartition

				owpartitions[minBin].insert(owpartitions[minBin].end(),
											wpartitions[index].begin(), wpartitions[index].end());
			}
		}

	}

}
