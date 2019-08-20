//
// Created by Bangtian Liu on 5/6/19.
//

#ifndef PROJECT_CDS_H
#define PROJECT_CDS_H

#include "ClusterTree.h"
#include "nUtil.h"

namespace Sympiler {
	namespace Internal {
		class CDS {
		public:
			double *CacheNear;
			double *utmp;
			int *ndimx;
			int *ndimy;
			unsigned long int *nearoffset;
			unsigned long int *utmpoffset;
			int *nidx; // node idx
			int *nidy; // node idy
			int ncount;

			double *CacheFar;
			double *ftmp;
			int *fdimx;
			int *fdimy;
			unsigned long int *faroffset;
			unsigned long int *ftmpoffset;
			int *fidx;
			int *fidy;
			int fcount;

			double *Proj;
			int *projoffset;


			int *skel_length;
			int *proj_column;

//    T *skel;
			double *w_skel;
			int *w_skeloffset;


			double *u_skel;
			int *u_skeloffset;

			unsigned long int totalsize;

			int nblocksize=2;
			int maxblockPerRow;

			std::vector<std::vector<pair<int,int>>> nmap;
			std::vector<std::vector<pair<int,int>>> nearlevel;
			std::vector<std::vector<unsigned long int>> noffset;
			std::map<int, int> nbuckmap;


			std::vector<std::vector<pair<int,int>>> fmap;
			std::vector<std::vector<pair<int,int>>> farlevel;
			std::vector<std::vector<unsigned long int>> foffset;
			std::map<int, int> fbuckmap;
			int fblocksize=4;
			int fmaxblockPerRow;

			double overhead=0.0;

//    int *skel_length;

		public:
			CDS(){};
			CDS(clustertree &tree, Setup &setup, std::vector<Ret> &tmp);
			void cacheNear(clustertree &tree, Setup &setup, std::vector<Ret> &tmp);
			// v1 version, each node may have different near nodes.
			void cacheFar(clustertree &tree, Setup &setup, std::vector<Ret> &tmp);

			void cacheNearBlock(clustertree &tree, Setup &setup, std::vector<Ret> &tmp);

			void cacheFarBlock(clustertree &tree, Setup &setup, std::vector<Ret> &tmp);

			void cacheProj(clustertree &tree, Setup &setup, std::vector<Ret> &tmp);
			void CoarcacheProj(clustertree &tree, Setup &setup, std::vector<Ret> &tmp);
			void cacheWUskel(clustertree &tree, Setup &setup, std::vector<Ret> &tmp);
			void CoarcacheWUskel(clustertree &tree, Setup &setup, std::vector<Ret> &tmp);

			void cacheSkeldim(clustertree & tree, Setup &setup, std::vector<Ret> &tmp);


			int getsize(){
				return totalsize;
			}

			double getoverhead(){
				return overhead;
			}
		};
	}
}
#endif //PROJECT_CDS_H
