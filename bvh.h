//
// Created by 17656 on 2024/2/17.
//

#ifndef RAYTRACING_BVH_H
#define RAYTRACING_BVH_H

#include "rtweekend.h"
#include "hittable.h"
#include "hittable_list.h"
#include <algorithm>

class bvh_node : public hittable {
public:

    bvh_node(const hittable_list &list)
            : bvh_node(list.objs(), 0, list.objs().size()) {}

    bvh_node(const std::vector<shared_ptr<hittable>> &src_objects,
             size_t start, size_t end) {
        // To be implemented later.int axis = random_int(0,2);
        auto obj = src_objects;
        int axis = random_int(0, 2);
        //函数指针??
        auto comparator = (axis == 0) ? box_x_compare
                                      : (axis == 1) ? box_y_compare
                                                    : box_z_compare;
        size_t obj_scan = end - start;

        if (obj_scan == 1) {
            left = right = obj[start];
        } else if (obj_scan == 2) {
            if (comparator(obj[start], obj[start + 1])) {
                left = obj[start];
                right = obj[start + 1];
            } else {
                left = obj[start + 1];
                right = obj[start];
            }
        } else {
            std::sort(obj.begin() + start, obj.begin() + end, comparator);
            auto mid = start + obj_scan / 2;
            left = make_shared<bvh_node>(obj, start, mid);
            right = make_shared<bvh_node>(obj, mid, end);

        }
        bbox = aabb(left->bounding_box(), right->bounding_box());
    }

    bool hit(const ray &r, interval ray_t, hit_record &rec) const override {
        if (!bbox.hit(r, ray_t))return false;
        bool hit_left = left->hit(r, ray_t, rec);
        bool hit_right = right->hit(r, ray_t, rec);
        return hit_left || hit_right;
    }

    aabb bounding_box() const override { return bbox; }


private:
    shared_ptr<hittable> left;
    shared_ptr<hittable> right;
    aabb bbox;


    static bool box_compare(
            const shared_ptr<hittable> a, const shared_ptr<hittable> b, int axis_index
    ) {
        return a->bounding_box().axis(axis_index).min < b->bounding_box().axis(axis_index).min;
    }

    static bool box_x_compare(const shared_ptr<hittable> a, const shared_ptr<hittable> b) {
        return box_compare(a, b, 0);
    }

    static bool box_y_compare(const shared_ptr<hittable> a, const shared_ptr<hittable> b) {
        return box_compare(a, b, 1);
    }

    static bool box_z_compare(const shared_ptr<hittable> a, const shared_ptr<hittable> b) {
        return box_compare(a, b, 2);
    }
};

#endif //RAYTRACING_BVH_H
