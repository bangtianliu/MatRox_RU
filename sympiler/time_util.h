//
// Created by Bangtian Liu on 6/14/19.
//

#ifndef PROJECT_TIME_UTIL_H
#define PROJECT_TIME_UTIL_H


#include <assert.h>
#include <time.h>


// Get time elapsed between t0 and t1
static inline struct timespec difftimespec(struct timespec t1, struct timespec t0)
{
	assert(t1.tv_nsec < 1000000000);
	assert(t0.tv_nsec < 1000000000);

	return (t1.tv_nsec >= t0.tv_nsec)
		   ? (struct timespec){ t1.tv_sec - t0.tv_sec    , t1.tv_nsec - t0.tv_nsec             }
		   : (struct timespec){ t1.tv_sec - t0.tv_sec - 1, t1.tv_nsec - t0.tv_nsec + 1000000000};
}


// Convert struct timespec to seconds
static inline double timespec_to_sec(struct timespec t)
{
	return t.tv_sec * 1.0 + t.tv_nsec / 1000000000.0;
}




#endif //PROJECT_TIME_UTIL_H
