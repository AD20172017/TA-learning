//
// Created by 17656 on 2024/2/3.
//

#ifndef RAYTRACING_INTERVAL_H
#define RAYTRACING_INTERVAL_H


#include "rtweekend.h"

class interval {
public:
    double min, max;
    interval(const interval& a, const interval& b)
            : min(fmin(a.min, b.min)), max(fmax(a.max, b.max)) {}
    interval() : min(+infinity), max(-infinity) {} // Default interval is empty

    interval(double _min, double _max) : min(_min), max(_max) {}

    bool contains(double x) const {
        return min <= x && x <= max;
    }
    double size()const{
        return max-min;
    }
    interval expand(double delta)const{
        auto padding=delta/2;
        return interval(min-padding,max+padding);
    }

    bool surrounds(double x) const {
        return min < x && x < max;
    }
    double clamp(double x) const {
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }
    static const interval empty, universe;
};

const static interval empty   (+infinity, -infinity);
const static interval universe(-infinity, +infinity);

#endif //RAYTRACING_INTERVAL_H
