//
// Created by 17656 on 2024/2/2.
//

#ifndef RAYTRACING_RAY_H
#define RAYTRACING_RAY_H
#include "vec3.h"
class ray{
public:
    ray(){}
    ray(const vec3& origin, const vec3& direction)
    : ori(origin), dir(direction){}

    vec3 origin() const {return ori;}
    vec3 direction() const {return dir;}

    vec3 at(double t) const {
        return ori + t * dir;
    }

private:
    vec3 ori;
    vec3 dir;


};



#endif //RAYTRACING_RAY_H
