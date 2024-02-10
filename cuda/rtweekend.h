//
// Created by 17656 on 2024/2/3.
//
#include <curand_kernel.h>
#ifndef RAYTRACING_RTWEEKEND_H
#define RAYTRACING_RTWEEKEND_H
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <curand.h>


// Usings

using std::shared_ptr;
using std::make_shared;

// Constants

const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

// Utility Functions

inline double degrees_to_radians(double degrees) {
    return degrees * pi / 180;
}

inline double ffmin(double a, double b) { return a <= b ? a : b; }
inline double ffmax(double a, double b) { return a >= b ? a : b; }

#define RAND_DOUBLE (curand_uniform_double(&local_rand_state))
__host__ __device__ inline double random_double(curandState *rand_state) {
    curandState local_rand_state = *rand_state;
    // Returns a random real in [0,1).
    return RAND_DOUBLE;
}

__host__ __device__ inline double random_double(double min, double max,curandState *rand_state) {
    // Returns a random real in [min,max).
    return min + (max-min)*random_double(rand_state);
}

static double min_error=0.0001;
static double reflect_rate=0.9;

/*#include "vec3.h"
#include "interval.h"
#include "material.h"*/

#endif //RAYTRACING_RTWEEKEND_H
