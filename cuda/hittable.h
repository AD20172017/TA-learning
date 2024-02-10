//
// Created by 17656 on 2024/2/3.
//

#ifndef RAYTRACING_HITTABLE_H
#define RAYTRACING_HITTABLE_H
#include "rtweekend.h"
#include "interval.h"
class material;
class hit_record {
public:
    vec3 p;
    vec3 normal;
    shared_ptr<material> mat;
    double  t;
    bool front_face;//面向

    __host__ __device__ inline void set_face_normal(const ray& r, const vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal :-outward_normal;
    }
};

class hittable {
public:
    __host__ __device__ virtual bool hit(const ray &r, interval ray_t, hit_record &rec) const=0;

};

#endif //RAYTRACING_HITTABLE_H
