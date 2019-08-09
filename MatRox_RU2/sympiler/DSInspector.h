//
// Created by Bangtian Liu on 5/12/19.
//

#ifndef PROJECT_DSINSPECTOR_H
#define PROJECT_DSINSPECTOR_H

#include <vector>
#include "HMatrix.h"
#include "HTree.h"

using namespace std;

namespace Sympiler {
	namespace Internal {
		class StructureObject {
public:
			int *levelSet;
			int *wpart;
			int *idx;

			int *nblockset;
			int *nblocks;
			int *nxval;
			int *nyval;

			int *fblockset;
			int *fblocks;
			int *fxval;
			int *fyval;

			int cdepth;

			int nl;
			int fl;

public:
			std::string levelPtr;
			std::string wpartPtr;
			std::string idxPtr;

			std::string nblocksetPtr;
			std::string nblocksPtr;
			std::string nxvalPtr;
			std::string nyvalPtr;

			std::string fblocksetPtr;
			std::string fblocksPtr;
			std::string fxvalPtr;
			std::string fyvalPtr;

			bool isHSS;
			bool isPerfect;
		StructureObject();
		~StructureObject();


		};




		class DSInspector {
		protected:
			StructureObject *ds;
		public:
			bool isHSS(clustertree *tree);
			bool isPerfect(clustertree *tree);
			DSInspector();
			~DSInspector();
		};

		class HGEMMInspector: public DSInspector{
//				HMatrix *A;
//				Matrix *B;
			HTree *htree;
			void CoarsenLevelSetDetection();
			void BlockSetDetection();
			void StructureCoarsen(std::vector<std::vector<std::vector<int>>> &postw);
			void StructureNearBlock(std::vector<std::vector<std::vector<pair<int,int>>>> &blocks);
			void StructureFarBlock(std::vector<std::vector<std::vector<pair<int,int>>>> &blocks);
		public:
			HGEMMInspector(HTree *tree);
			StructureObject* Inspect();
			~HGEMMInspector();
		};
	}
}



#endif //PROJECT_DSINSPECTOR_H
