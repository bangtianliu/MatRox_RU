//
// Created by Bangtian Liu on 4/30/19.
//

#include "boundingbox.h"
#include <assert.h>
#include <algorithm>
#include "nUtil.h"


namespace Sympiler {
	namespace Internal {
/**
 *
 * @tparam T
 * @return the dimension along which to perform binary partition
 */
		int boundingbox::maxis() {
			int max;
			double tmp=std::numeric_limits<double>::min();
			for(int i=0;i<dim;i++)
			{
				if((tpoint[i]-spoint[i])>tmp) {
					tmp=(tpoint[i]-spoint[i]);
					max = i;
				}
			}
			return max;
		}
/**
 * @tparam T
 */
		void boundingbox::midpoint() {
			for (int i = 0; i < dim; ++i) {
				mpoint[i] = (spoint[i]+tpoint[i])/2;
			}
		}

/**
 * @brief compute the raduis
 * @return
 */
		void boundingbox::radius() {
			_radius=distance(tpoint,mpoint,dim);
		}

/**
 *
 * @tparam T
 * @param point input points
 */
		boundingbox::boundingbox(double *_point, int _num, int _dim)
		{
			num=_num;
			dim=_dim;

			spoint = (double *)mkl_malloc(sizeof(double)*dim,64);
			tpoint = (double *)mkl_malloc(sizeof(double)*dim,64);
			mpoint = (double *)mkl_malloc(sizeof(double)*dim,64);

			lids.resize(num);
			for (int k = 0; k < num; ++k) {
				lids[k]=k;
			}

			memcpy(spoint,_point,sizeof(double)*dim);
			memcpy(tpoint,_point,sizeof(double)*dim);


			for(int i=1;i<num;i++)
			{
				for (int j = 0; j < dim; ++j) {
					spoint[j] = min(spoint[j],_point[i*dim+j]);
					tpoint[j] = max(tpoint[j],_point[i*dim+j]);
				}
			}

			midpoint();
			radius();
		}
/**
 *
 * @tparam T
 * @param _spoint
 * @param _tpoint
 * @param _dim
 */
		boundingbox::boundingbox(double *_spoint, double *_tpoint, int _dim) {
			dim=_dim;
			spoint=(double *)malloc(sizeof(double)*dim);
			tpoint=(double *)malloc(sizeof(double)*dim);
			mpoint=(double *)malloc(sizeof(double)*dim);
			memcpy(spoint,_spoint,sizeof(double)*dim);
			memcpy(tpoint, _tpoint,sizeof(double)*dim);

			num=0;
			midpoint();
			radius();
		}
/**
 * @param left  bounding box in left side along major axis
 * @param right bounding box in right side along major axis
 * @param flag about determining to perform balance partition or not
 **/
		void boundingbox::binarysplit(boundingbox &left, boundingbox &right, bool equal, Setup &setup) {
			auto axis=maxis();
			auto points=setup.X;
			double mid;
			std::vector<int> llids;
			std::vector<int> rlids;
			double lmax=std::numeric_limits<double>::min();
			double rmin=std::numeric_limits<double>::max();

			if(equal){

				double *tmp=(double *)malloc(sizeof(double)*num);
				double *tmp1=(double *)malloc(sizeof(double)*num);
				for (int i = 0; i < num; ++i) {
					tmp1[i] = points[lids[i]*dim+axis];
					tmp[i] =tmp1[i];
				}

				sort(tmp, tmp+num);

				mid = ((num%2)==0)?(tmp[num/2]+tmp[num/2-1])/2:tmp[num/2];

				std::vector<int> middle;
				int llength=num/2;

				if((num%2)!=0){
					++llength;
				}
				for(int i =0; i<num; ++i)
				{

					if(tmp1[i]<mid){
						llids.push_back(lids[i]);
						if(tmp1[i]>lmax)lmax=tmp1[i];
					}
					else if(tmp1[i]>mid){
						rlids.push_back(lids[i]);
						if(tmp1[i]<rmin)rmin=tmp1[i];
					}
					else{
						middle.push_back(lids[i]);
					}
				}
				for (int i = 0; i < middle.size(); ++i) {
					auto val=middle[i];
					if(llids.size()<llength+1)
					{
						llids.push_back(val);
						if(val>lmax)lmax=val;
					}
					else{
						rlids.push_back(val);
						if(val<rmin)rmin=val;
					}
				}
				//todo: split in two part equally

			}
			else {
				mid = (spoint[axis] + tpoint[axis]) / 2;
				//FIXME: maybe a bug
				for (int i = 0; i < num; i++) {
					auto val = points[lids[i] * dim + axis];
					if (val <= mid) {

						llids.push_back(lids[i]);

					} else {
						rlids.push_back(lids[i]);

					}
				}
			}
			double *lpoint = (double *) malloc(sizeof(double) * dim);

			memcpy(lpoint, tpoint, sizeof(double) * dim);
			lpoint[axis] = mid;

			double *rpoint = (double *) malloc(sizeof(double) * dim);
			memcpy(rpoint, spoint, sizeof(double) * dim);
			rpoint[axis] = mid;

			// todo: may have a bug here
			left  = boundingbox(spoint,lpoint,llids,(int)llids.size(),dim);
			right = boundingbox(rpoint,tpoint,rlids,(int)rlids.size(),dim);

		}



/**
 * @param left  bounding box in left side along major axis
 * @param right bounding box in right side along major axis
 * @param flag about determining to perform balance partition or not
 **/
		void boundingbox::mlbinarysplit(boundingbox &left, boundingbox &right, bool equal, Setup &setup) {
			auto axis=maxis();
			auto points=setup.X;
			double mid;
			std::vector<int> llids;
			std::vector<int> rlids;
			double lmax=std::numeric_limits<double>::min();
			double rmin=std::numeric_limits<double>::max();

			if(equal){
				double *tmp=(double *)malloc(sizeof(double)*num);
				double *tmp1=(double *)malloc(sizeof(double)*num);
				for (int i = 0; i < num; ++i) {
					tmp1[i] = points[lids[i]*dim+axis];
					tmp[i] =tmp1[i];
				}

				sort(tmp, tmp+num);

				mid = ((num%2)==0)?(tmp[num/2]+tmp[num/2-1])/2:tmp[num/2];

				std::vector<int> middle;
				int llength=num/2;
//        int rlength=num/2;
				if((num%2)!=0){
//            ++roffset;
					++llength;
				}
				for(int i =0; i<num; ++i)
				{
					if(tmp1[i]<mid){
						llids.push_back(lids[i]);
						if(tmp1[i]>lmax)lmax=tmp1[i];
					}
					else if(tmp1[i]>mid){
						rlids.push_back(lids[i]);
						if(tmp1[i]<rmin)rmin=tmp1[i];
					}
					else{
						middle.push_back(lids[i]);
					}
				}
				for (int i = 0; i < middle.size(); ++i) {
					auto val=middle[i];
					if(llids.size()<llength)
					{
						llids.push_back(val);
						//   if(val>lmax)lmax=val;
					}
					else{
						rlids.push_back(val);
						// if(val<rmin)rmin=val;
					}
				}
				//todo: split in two part equally
//        free(tmp);
			}
			else {
				mid = (spoint[axis] + tpoint[axis]) / 2;
				//FIXME: maybe a bug
				for (int i = 0; i < num; i++) {
					auto val = points[lids[i] * dim + axis];
					if (val <= mid) {
//                printf("TEST %d\n",lids[i]);
						llids.push_back(lids[i]);
//                if (val > lmax)lmax = val;
					} else {
						rlids.push_back(lids[i]);
//                if (val < rmin)rmin = val;
					}
				}
			}
			double *lpoint = (double *) malloc(sizeof(double) * dim);

			memcpy(lpoint, tpoint, sizeof(double) * dim);
			lpoint[axis] = mid;
//        lpoint[axis] = lmax;

			double *rpoint = (double *) malloc(sizeof(double) * dim);
			memcpy(rpoint, spoint, sizeof(double) * dim);
			rpoint[axis] = mid;


			double *lpoint1 = (double *)malloc(sizeof(double)*dim);
			double *lpoint2 = (double *)malloc(sizeof(double)*dim);

			double *rpoint1 = (double *)malloc(sizeof(double)*dim);
			double *rpoint2 = (double *)malloc(sizeof(double)*dim);


			for(int i=0;i<dim;i++){
				lpoint1[i]=points[llids[0]*dim+i];
				lpoint2[i]=points[llids[0]*dim+i];
				rpoint1[i]=points[rlids[0]*dim+i];
				rpoint2[i]=points[rlids[0]*dim+i];
			}

			for(int i=1;i<llids.size();i++){
				for(int j=0;j<dim;j++)
				{
					lpoint1[j]=std::min(lpoint[j],points[llids[i]*dim+j]);
					lpoint2[j]=std::max(lpoint[j],points[llids[i]*dim+j]);
				}
			}

			for(int i=1;i<rlids.size();i++){
				for(int j=0;j<dim;j++)
				{
					rpoint1[j]=std::min(rpoint1[j],points[rlids[i]*dim+j]);
					rpoint2[j]=std::max(rpoint2[j],points[rlids[i]*dim+j]);
				}
			}

			// todo: may have a bug here
			left  = boundingbox(lpoint1,lpoint2,llids,(int)llids.size(),dim);
			right = boundingbox(rpoint1,rpoint2,rlids,(int)rlids.size(),dim);

		}


		void boundingbox::mlbinarysplit_v2(boundingbox &left, boundingbox &right, bool equal, Setup &setup) {

			assert(equal==true);
			auto points=setup.X;

			std::vector<int> llids;
			std::vector<int> rlids;

			// split points

			split(setup,lids,llids, rlids);

			left=boundingbox(points, llids, (int)llids.size(),dim);
			right=boundingbox(points, rlids, (int)rlids.size(),dim);
		}


		void boundingbox::split(Setup setup, std::vector<int> &lids, std::vector<int> &llids, std::vector<int> &rlids) {
			double *center = Mean<double>(lids.data(), (int)lids.size(), setup);
			double *direction = (double *)mkl_malloc(sizeof(double)*dim, 64);

			double *projection = (double *)mkl_malloc(sizeof(double)*num, 64);
			memset(projection, 0, sizeof(double)*num);

			double *tmp_value = (double *)mkl_malloc(sizeof(double)*num, 64);
			auto X = setup.X;

			int d = setup.d;
			int length = (int)lids.size();
            #pragma omp parallel for
			for(int i=0; i<length; ++i)
			{
				double rcx = 0.0;
				for ( int p = 0; p < d; p ++ )
				{
					double tmp = X[ lids[ i ] * d + p ] - center[ p ];
					rcx += tmp * tmp;

				}

				tmp_value[i] = rcx;

			}

			double rcx = 0.0;
			int x0;

			double val = tmp_value[0];
			int index = 0;
//#pragma omp parallel for reduction(maximum:max)
			for (int i = 1;  i < length ; ++ i) {
				if(tmp_value[i]>val){
					val = tmp_value[i];
					index = i;
				}
			}  // here may have a bug, not sure.

			x0 = index;

//#pragma omp parallel for
			for ( int i = 0; i < length; i ++ )
			{
				double rxx = 0.0;
				for ( int p = 0; p < d; p ++ )
				{
					double tmp = X[ lids[ i ] * d + p ] - X[ lids[ x0 ] * d + p ];
					rxx += tmp * tmp;
				}
				tmp_value[i] = rxx;

			}

			val = tmp_value[0];
			index=0;
//#pragma omp parallel for reduction(maximum:max)
			for (int i = 1;  i < length ; ++ i) {
				if(tmp_value[i]>val){
					val = tmp_value[i];
					index = i;
				}
			}  // here may have a bug, not sure.

			int x1 = index;

#pragma omp parallel for
			for ( int p = 0; p < d; p ++ )
			{
				direction[ p ] = X[ lids[ x1 ] * d + p ] - X[ lids[ x0 ] * d + p ];
			}

#pragma omp parallel for
			for ( int i = 0; i < length; i ++ )
				for ( int p = 0; p < d; p ++ )
					projection[ i ] += X[ lids[ i ] * d + p ] * direction[ p ];

			double *projcpy = (double *)malloc(sizeof(double)*length);
			memcpy(projcpy,projection, sizeof(double)*length);
			sort(projcpy, projcpy+length);
			double median = (projcpy[length/2] + projcpy[length/2-1])/2;

			std::vector<int> middle;

			int llength=num/2;
			if((num%2)!=0){
				++llength;
			}

			for (int i = 0; i < length; ++i) {
				if(projection[ i ] < median) {
					llids.push_back(lids[i]);
				}
				else if(projection[ i ] > median){
					rlids.push_back(lids[i]);
				}
				else middle.push_back(lids[i]);

			}


			for (int i = 0; i < middle.size(); ++i) {
				if(llids.size()<llength)
				{
					llids.push_back(middle[i]);
				}
				else{
					rlids.push_back(middle[i]);
				}
			}
		}

		void boundingbox::sbsplit(boundingbox &left, boundingbox &right, Setup &setup) {
			auto points = setup.X;
			std::vector<int> llids;
			std::vector<int> rlids;
			int len =lids.size();

			llids.assign(lids.begin(),lids.begin()+len/2);
			rlids.assign(lids.begin()+len/2, lids.begin()+len);

			left=boundingbox(points, llids, (int)llids.size(),dim);
			right=boundingbox(points, rlids, (int)rlids.size(),dim);

		}

/**
 *
 *
 * @tparam T
 * @param _lids
 * FIXME: Maybe a bug.
 */
		void boundingbox::setlids(std::vector<int> _lids) {
			lids=_lids;
			num=_lids.size();
		}

/**
 * split the bounding pox into 2^{dim} bounding boxes
 * @return
 */
		std::vector<boundingbox> boundingbox::split(Setup &setup) {

			int nbox=pow(2,dim);

			auto points=setup.X;

			std::vector<boundingbox> ret(nbox);

			auto min=getmin();
			auto max=getmax();
			auto mid=getmid();

			double *tmin=(double *)malloc(sizeof(double)*dim);
			double *tmax=(double *)malloc(sizeof(double)*dim);

			for (int i = 0; i < nbox; ++i) {
				memcpy(tmin, min, sizeof(double)*dim);
				memcpy(tmax, max, sizeof(double)*dim);
				for(int j = 0; j<dim;j++)
				{
					if(0==((i>>j)&1)){
						tmax[j]=mid[j];
					}
					else{
						tmin[j]=mid[j];
					}
					boundingbox box(tmin,tmax,dim);
					ret[i]=box;
				}
			}

			for(int i=0;i<num;i++)
			{
				int index=0;
				for(int j=0;j<dim;j++)
				{
					int id;
					auto pd = points[lids[i]*dim+j];
					if(pd<=mid[j]){
						id=(0<<j);
					}
					else{
						id=(1<<j);
					}
					index+=id;
				}
				ret[index].addlids(lids[i]);

			}
			return ret;
		}

		bool well_sperated( boundingbox &i, boundingbox &j, double tau) {
			auto radius = i.getradius() + j.getradius();
			auto mpointi = i.getmid();
			auto mpointj = j.getmid();
			assert(i.getdim() == j.getdim());
			auto dist = distance(mpointi, mpointj, i.getdim());

			return radius < tau * dist;
		}
	}
}