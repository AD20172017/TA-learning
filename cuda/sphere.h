//
// Created by 17656 on 2024/2/3.
//

#ifndef RAYTRACING_SPHERE_H
#define RAYTRACING_SPHERE_H

#include "rtweekend.h"

class sphere : public hittable {
public:
    __host__ __device__ sphere(point3 _center, double _radius, shared_ptr<material> _material)
            : center(_center), radius(_radius), mat(_material) {}

    __host__ __device__ sphere(vec3 center, double r) : center(center), radius(r) {};

    __host__ __device__ virtual bool hit(const ray &r, interval ray_t, hit_record &rec) const;

public:
    vec3 center;
    double radius;
    shared_ptr<material> mat;
};


__host__ __device__ bool sphere::hit(const ray &r, interval ray_t, hit_record &rec) const {
    vec3 oc = r.origin() - center;
    auto a = r.direction().length_squared();
    auto half_b = dot(oc, r.direction());
    auto c = oc.length_squared() - radius * radius;
    auto discriminant = half_b * half_b - a * c;

    if (discriminant > 0) {
        auto root = sqrt(discriminant);
        auto temp = (-half_b - root) / a;//求得的t
        if (ray_t.surrounds(temp)) {
            rec.t = temp;
            rec.p = r.at(rec.t);
            rec.mat = mat;
            vec3 outward_normal = (rec.p - center) / radius;
            rec.set_face_normal(r, outward_normal);
            return true;
        }
        temp = (-half_b + root) / a;
        if (ray_t.surrounds(temp)) {
            rec.t = temp;
            rec.p = r.at(rec.t);
            rec.mat = mat;
            vec3 outward_normal = (rec.p - center) / radius;
            rec.set_face_normal(r, outward_normal);
            return true;
        }
    }
    return false;
}

#endif //RAYTRACING_SPHERE_H
