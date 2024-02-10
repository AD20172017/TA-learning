//
// Created by 17656 on 2024/2/3.
//

#ifndef RAYTRACING_HITTABLE_LIST_H
#define RAYTRACING_HITTABLE_LIST_H
#include "hittable.h"
#include <memory>
#include <vector>

using std::shared_ptr;
using std::make_shared;

class hittable_list: public hittable{
public:
    __device__ hittable_list(){}
    __device__ hittable_list(shared_ptr<hittable> obj){add(obj);}

    __device__ void clear(){objects.clear();}
    __device__ void add(shared_ptr<hittable> object) { objects.push_back(object); }
    __device__ virtual bool hit(const ray &r, interval ray_t, hit_record &rec) const;
public:
    std::vector<shared_ptr<hittable>> objects;
};

__device__ bool hittable_list::hit(const ray &r, interval ray_t, hit_record &rec) const{
    hit_record temp_rec;
    bool hit_anything = false;
    auto closest_so_far = ray_t.max;

    for (const auto& object : objects) {
        if (object->hit(r, interval(ray_t.min,closest_so_far), temp_rec)) {//找到光线第一个射入的地方
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}

#endif //RAYTRACING_HITTABLE_LIST_H
