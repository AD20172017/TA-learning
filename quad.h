//
// Created by 17656 on 2024/2/18.
//

#ifndef RAYTRACING_QUAD_H
#define RAYTRACING_QUAD_H

#include "rtweekend.h"
#include "hittable.h"

class quad : public hittable {
public:
    quad(const point3 &_Q, const vec3 &_u, const vec3 &_v, shared_ptr<material> m)
            : Q(_Q), u(_u), v(_v), mat(m) {
        auto n = cross(u, v);
        normal = unit_vector(n);
        D = dot(normal, Q);
        w = n / dot(n, n);
        set_bbox();
    }

    virtual void set_bbox() {
        bbox = aabb(Q, Q + u + v).pad();
    }

    aabb bounding_box() const override { return bbox; }

    bool hit(const ray &r, interval ray_t, hit_record &rec) const override {
        //点积判断是否和法向量垂直
        auto cos = dot(normal, r.direction());
        if (fabs(cos) < 1e-8)return false;
        //点到平面的距离
        auto t = (D - dot(normal, r.origin())) / cos;
        if (!ray_t.contains(t))return false;

        auto intersection = r.at(t);
        vec3 planar_hitpt_vector = intersection - Q;
        auto alpha = dot(w, cross(planar_hitpt_vector, v));
        auto beta = dot(w, cross(u, planar_hitpt_vector));

        if (!is_interior(alpha, beta, rec))
            return false;

        rec.t = t;
        rec.p = intersection;
        rec.mat = mat;
        rec.set_face_normal(r, normal);
        return true;
    }

    virtual bool is_interior(double a, double b, hit_record &rec) const {
        interval v(0, 1);
        if (!(v.contains(a) && v.contains(b)))return false;
        rec.u = a;
        rec.v = b;
        return true;
    }


private:
    point3 Q;
    vec3 u, v;
    shared_ptr<material> mat;
    aabb bbox;

    //从没这么想过空间几何

    //法向量
    vec3 normal;
    //常数项
    double D;
    //uv空间来表示hit点,先叉乘在点乘求出系数
    vec3 w;

};


#endif //RAYTRACING_QUAD_H
