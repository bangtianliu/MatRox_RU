//
// Created by Bangtian Liu on 4/29/19.
//

#ifndef PROJECT_CLUSTERTREE_H
#define PROJECT_CLUSTERTREE_H
#include <vector>
#include <algorithm>
#include <set>
#include <map>
#include <unordered_set>
#include "boundingbox.h"
#include "config.h"
#include "Expr.h"
#include "Argument.h"
#include "nUtil.h"

using namespace std;

namespace Sympiler {
	namespace Internal {
		class clustertree{
			friend class CDS;
			friend class HGEMMInspector;
			friend class HMatrix;

		public:
//			Setup setup;
			std::vector<int> nodeId;
			std::vector<int> cnodeId;

			std::vector<int> leafnodes;
			std::map<int, int> leafoffset;
			std::vector<boundingbox> boxes;
			std::vector<std::vector<int>> levelsets;
			std::vector<std::set<int>> omplevelsets; // from bottom to up
			std::vector<std::vector<int>> sflevelsets; // output

			std::vector<std::vector<int>> coarlevelsets; // output

			std::vector<std::vector<std::vector<int>>> postw;

			std::vector<std::vector<std::vector<int>>> opostw;

			std::vector<int> parents;
			std::vector<std::vector<int>> children;


			std::vector<std::vector<int>> nearnodes;
			std::vector<std::vector<int>> farnodes;


			std::vector<std::vector<std::vector<pair<int,int>>>> nearblocks;
			std::vector<std::vector<std::vector<pair<int,int>>>> farblocks;

			std::vector<bool> pflags;
			std::map<int, int> skelmidex; // index mapping
			std::map<int, int> levelmap; // for finding the level of each node


			std::map<int, int> nbuckmap;
			std::map<int, int> fbuckmap;
			int ncount;
			int fcount;

			std::vector<std::unordered_set<int>> pnids;
			std::vector<std::map<int, double>> snids;

			std::vector<std::multimap<double, int>> ordered_snids;



			std::map<int, int> cskelmindex;

			int depth=0;
			int npnodes=0; /**number of pruned nodes**/
			int nplevels=0; /**number of pruned levels**/
			int nfplvels=0;

		public:

			std::vector<Ret> tmpresult; // for intemediate result from compression phase.

			std::string levelset="levelset";
			std::string idx="idx";

			clustertree(){};

			clustertree(std::string , double, int, int);
			clustertree(Setup &setup);

			~clustertree(){
			};

			void buildtree(boundingbox &box, Setup &_setup);

			bool isHSS();
			bool isPerfect();

			void getTreeDecl(std::vector<Expr> &el, std::vector<Argument> &al);

			void findleafmap();
			void findnearfarnodes(const int idx, const int idy, const double tau);

			void findnearfarnodes_buget(double budget);
			void findnearnodes_buget(double budget);
			void MergeFarNodes();
			void MergeFarNodes(int idx);
			void FindFarNodes(int root, int node);
			bool containany(int root, std::vector<int> &NearNodes);

			bool isancestor(int node, int root);

			void bfindnearfarnodes(const int idx, const int idy);
			bool bwell_sperated(const int idx, const int idy);

			void findleafnodes(int idx, int &beg, int &end);

			void NearBlock(int blockSize);
			void FarBlock(int blockSize, bool flag);

			void mapskelindex();
			void coarmapskelidx();

			int numNearnodes();
			int numFarnodes();

			void sortnodes();

			void parlevel(int agg, int llevel);

			void genpostw();
			void imgenpostw();

			void levelmapping();

			void postorder(std::vector<int>&wpartition, int index, int clevel, int elevel);
			void postorder_v2(std::vector<int>&wpartition, int index, int clevel, int elevel);

			void allocatenids();

			const std::vector<std::vector<int>>&  getchildren() const { return children; }

			const std::vector<int>& getparent(){ return parents; }


			const std::vector<std::vector<int>>& getlevelsets(){ return levelsets; }

			const std::vector<std::set<int>>& getomplevelsets(){ return omplevelsets; }

			const std::vector<std::vector<int>>& getsflevelsets(){ return sflevelsets; };

			int getnumnodes(){ return (int)parents.size(); };

			int getnumleafnodes(){ return (int)leafnodes.size(); }

			const std::vector<boundingbox>& getboxes(){ return boxes; }

			const std::vector<int>& getleaf(){ return leafnodes; }

			const std::vector<std::vector<int>>& getnearnodes(){ return nearnodes; }

			const std::vector<std::vector<int>>& getfarnodes(){ return farnodes; }

			const std::map<int, int>& getleafoffset(){ return leafoffset; };


			const std::map<int, int>& getskelmindex(){ return skelmidex; }


			const std::map<int, int>& getcskelmindex(){ return cskelmindex; };

			std::vector<std::unordered_set<int>> & getpnids(){ return pnids; }

			std::vector<std::map<int, double>> & getsnids(){ return snids; };

			std::vector<std::multimap<double, int>> & getordersnids(){ return ordered_snids;};



			std::vector<std::vector<std::vector<int>>>& getopostw(){ return opostw; }
			std::vector<std::vector<std::vector<int>>>& getpostw(){ return postw; }



			void compression();
			void skeletonize(int idx, boundingbox &box, Ret *rtmp);


			void binpacking(std::vector<std::vector<int>> &wpartitions, std::vector<std::vector<int>> &owpartitions, int numofbins);


			void BalanceCoarLevelSet();


			void writelevelset(); // write level set

			void prune();

			int getnplevels(){ return nplevels; }

			int getnfplevels(){ return nfplvels; }

			int getnpnodes(){ return npnodes; }

			void save();
		};


	}
}


#endif //PROJECT_CLUSTERTREE_H
