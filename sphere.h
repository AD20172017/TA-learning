//
// Created by 17656 on 2024/2/3.
//

#ifndef RAYTRACING_SPHERE_H
#define RAYTRACING_SPHERE_H

#include "aabb.h"
#include "rtweekend.h"

class sphere : public hittable {
public:
    sphere(point3 _center1, double _radius, shared_ptr<material> _material)
            : center1(_center1), radius(_radius), mat(_material), is_moving(false) {
        auto rvec = vec3(radius, radius, radius);
        bbox = aabb(center1 - rvec, center1 + rvec);
    }

    sphere(point3 _center1, point3 _center2, double _radius, shared_ptr<material> _material)
            : center1(_center1), center_vec(_center2 - _center1), radius(_radius), mat(_material), is_moving(true) {
        auto rvec = vec3(radius, radius, radius);
        aabb box1(_center1 - rvec, _center1 + rvec);
        aabb box2(_center2 - rvec, _center2 + rvec);
        bbox = aabb(box1, box2);
    }

    sphere(vec3 center, double r) : center1(center), radius(r), is_moving(false) {
        auto rvec = vec3(radius, radius, radius);
        bbox = aabb(center1 - rvec, center1 + rvec);
    };

    virtual bool hit(const ray &r, interval ray_t, hit_record &rec) const;

    aabb bounding_box() const override { return bbox; }

private:
    point3 center1;
    double radius;
    shared_ptr<material> mat;
    bool is_moving;
    vec3 center_vec;
    aabb bbox;

    point3 move_center(double time) const {
        // Linearly interpolate from center1 to center2 according to time, where t=0 yields
        // center1, and t=1 yields center2.
        return center1 + time * center_vec;
    }
};


bool sphere::hit(const ray &r, interval ray_t, hit_record &rec) const {
    point3 center = is_moving ? move_center(r.t()) : center1;
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
            rec.u = rec.p.x();
            rec.v = rec.p.y();
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
