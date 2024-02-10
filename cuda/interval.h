//
// Created by 17656 on 2024/2/3.
//

#ifndef RAYTRACING_INTERVAL_H
#define RAYTRACING_INTERVAL_H


#include "rtweekend.h"

class interval {
public:
    __host__ __device__ double min, max;

    __host__ __device__ interval() : min(+infinity), max(-infinity) {} // Default interval is empty

    __host__ __device__ interval(double _min, double _max) : min(_min), max(_max) {}

    __host__ __device__ bool contains(double x) const {
        return min <= x && x <= max;
    }

    __host__ __device__ bool surrounds(double x) const {
        return min < x && x < max;
    }
    __host__ __device__ double clamp(double x) const {
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }
    static const interval empty, universe;
};

const static interval empty   (+infinity, -infinity);
const static interval universe(-infinity, +infinity);

#endif //RAYTRACING_INTERVAL_H
