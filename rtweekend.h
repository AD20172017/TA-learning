//
// Created by 17656 on 2024/2/3.
//

#ifndef RAYTRACING_RTWEEKEND_H
#define RAYTRACING_RTWEEKEND_H
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include "random"

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


inline double random_double() {
    // Returns a random real in [0,1).
    return rand() / (RAND_MAX + 1.0);
}
inline int random_int(int min,int max){
    //[min,max]
    //return min+rand()%(max-min+1);
    //or
    //return static_cast<int>(random_double(min, max+1));
    std::random_device rd;  // Obtain a random number from hardware
    std::mt19937 eng(rd()); // Seed the generator

    std::uniform_int_distribution<> distr(min, max); // Define the range

    int random_number = distr(eng);  // Generate a random number in the range
    return  random_number;
}
inline double random_double(double min, double max) {
    // Returns a random real in [min,max).
    return min + (max-min)*random_double();
}

static double min_error=0.0001;
static double reflect_rate=0.9;
#include "ray.h"
#include "vec3.h"
#include "interval.h"
#include "color.h"
#include "material.h"

#endif //RAYTRACING_RTWEEKEND_H
