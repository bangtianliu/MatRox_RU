//
// Created by Bangtian Liu on 4/30/19.
//

#ifndef PROJECT_BOUNDINGBOX_H
#define PROJECT_BOUNDINGBOX_H

#include <vector>
#include <cstdlib>
#include <cstring>
#include "config.h"
#include <mkl.h>

using namespace std;

namespace Sympiler {
	namespace Internal {
		class boundingbox {
		private:
			double *spoint; /**low and left corner point **/
			double *tpoint; /**up and right corner point**/
			double *mpoint; /**(spoint+tpoint)/2**/

			int num; /*number of points*/
			int dim;
			std::vector<int> lids;
			double _radius;

		public:

			boundingbox(){};
			boundingbox(double *_point, int _num, int _dim);
			boundingbox(double *_spoint, double *_tpoint, int _dim);

			//FIXME
			boundingbox(double *_spoint, double *_tpoint, std::vector<int> _lids, int _num, int _dim)
					: num(_num), dim(_dim){

				spoint=(double *)malloc(sizeof(double)*dim);
				tpoint=(double *)malloc(sizeof(double)*dim);
				memcpy(spoint,_spoint,sizeof(double)*dim);
				memcpy(tpoint,_tpoint,sizeof(double)*dim);

				lids.swap(_lids);

				mpoint=(double *)malloc(sizeof(double)*dim);
				for (int i = 0; i < dim; ++i) {
					mpoint[i] = (spoint[i]+tpoint[i])/2;
				}
				radius();
			};

			boundingbox(double *_point, std::vector<int> _lids, int _num, int _dim):num(_num),dim(_dim){
				spoint = (double *)malloc(sizeof(double)*dim);
				tpoint = (double *)malloc(sizeof(double)*dim);
				mpoint = (double *)malloc(sizeof(double)*dim);

				for(int i=0;i<dim;i++){
					spoint[i]=_point[_lids[0]*dim+i];
					tpoint[i]=_point[_lids[0]*dim+i];
				}

				for (int i = 0; i < num ; ++i) {
					for (int j = 0; j < dim; ++j){
						spoint[j] = min(spoint[j],_point[_lids[i]*dim+j]);
						tpoint[j] = max(tpoint[j],_point[_lids[i]*dim+j]);
					}
				}

				midpoint();
				radius();

				copy(_lids.begin(),_lids.end(), back_inserter(lids));
			}
			void binarysplit(boundingbox &left, boundingbox &right, bool equal, Setup &setup);
			void mlbinarysplit(boundingbox &left, boundingbox &right, bool equal, Setup &setup);


			void mlbinarysplit_v2(boundingbox &left, boundingbox &right, bool equal, Setup &setup);

			void split(Setup setup, std::vector<int>& lids, std::vector<int>& llids, std::vector<int> &rlids);
			void sbsplit(boundingbox &left, boundingbox &right, Setup &setup);

			int maxis();
			void midpoint();
			void radius();
			double *getmin(){
				return spoint;
			}

			double *getmax(){
				return tpoint;
			}

			double *getmid(){
				return mpoint;
			}

			double getradius(){
				return _radius;
			}
			std::vector<boundingbox> split(Setup &setup);

			void setlids(std::vector<int> _lids);

			void addlids(int idx){
				lids.push_back(idx);
				num=lids.size();
			}
			const std::vector<int>& getlids(){
				return lids;
			}

			const std::vector<int>& conlids() const{
				return lids;
			}

			const int &getnum(){
				return num;
			}

			const int &getdim(){
				return dim;
			}

			friend bool well_sperated(boundingbox &i, boundingbox &j, double tau);

		};
	}
}
#endif //PROJECT_BOUNDINGBOX_H
