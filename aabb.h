//
// Created by 17656 on 2024/2/17.
//

#ifndef RAYTRACING_AABB_H
#define RAYTRACING_AABB_H

#include "rtweekend.h"

class aabb {
public:
    aabb() {

    }

    aabb(const aabb &box0, const aabb &box1) {
        x = interval(box0.x, box1.x);
        y = interval(box0.y, box1.y);
        z = interval(box0.z, box1.z);
    }

    aabb(const interval &a, const interval &b, const interval &c) : x(a), y(b), z(c) {}

    aabb(const point3 &a, const point3 &b) {
        x = interval(fmin(a.x(), b.x()), fmax(a.x(), b.x()));
        y = interval(fmin(a.y(), b.y()), fmax(a.y(), b.y()));
        z = interval(fmin(a.z(), b.z()), fmax(a.z(), b.z()));

    }

    aabb pad() {
        double delta = 0.0001;
        //当平面平行与xy,xz,yz面,数值过小带来的问题??溢出还是啥?
        interval new_x = (x.size() >= delta) ? x : x.expand(delta);
        interval new_y = (y.size() >= delta) ? y : y.expand(delta);
        interval new_z = (z.size() >= delta) ? z : z.expand(delta);

        return aabb(new_x, new_y, new_z);

    }

    const interval &axis(int n) const {
        if (n == 1)return y;
        if (n == 2)return z;
        return x;
    }

    bool hit(const ray &r, interval ray_t) const {
        for (int i = 0; i < 3; ++i) {
            auto invD = 1 / r.direction()[i];
            auto orig = r.origin()[i];

            auto t0 = (axis(i).min - orig) * invD;
            auto t1 = (axis(i).max - orig) * invD;
            //如果是反方向射入
            if (invD < 0)
                std::swap(t0, t1);

            if (t0 > ray_t.min) ray_t.min = t0;
            if (t1 < ray_t.max) ray_t.max = t1;
            if (ray_t.size() <= 0)return false;
        }
        return true;


    }

private:
    interval x, y, z;
};

#endif //RAYTRACING_AABB_H
