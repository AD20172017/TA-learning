//
// Created by 17656 on 2024/2/3.
//

#ifndef RAYTRACING_HITTABLE_LIST_H
#define RAYTRACING_HITTABLE_LIST_H
#include "hittable.h"
#include <memory>
#include <vector>
#include "omp.h"

using std::shared_ptr;
using std::make_shared;

class hittable_list: public hittable{
public:
    hittable_list(){}
    hittable_list(shared_ptr<hittable> obj){add(obj);}

    void clear(){objects.clear();}
    void add(shared_ptr<hittable> object) { objects.push_back(object); }
    virtual bool hit(const ray &r, interval ray_t, hit_record &rec) const;
public:
    std::vector<shared_ptr<hittable>> objects;
};

bool hittable_list::hit(const ray &r, interval ray_t, hit_record &rec) const{
    hit_record temp_rec;
    bool hit_anything = false;
    auto closest_so_far = ray_t.max;
    //奇了怪.可能还是物体太少,再次使用openmp速度居然下降了,也可能是嵌套的原因
    //#pragma omp parallel for
    /*//事实证明太过高级的语法反而会降低速度,使用下标减了10s
        for (const auto& object : objects) {
        if (object->hit(r, interval(ray_t.min,closest_so_far), temp_rec)) {//找到光线第一个射入的地方
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }*/
    for(int i=0;i<objects.size();++i){
        if (objects[i]->hit(r, interval(ray_t.min,closest_so_far), temp_rec)) {//找到光线第一个射入的地方
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}

#endif //RAYTRACING_HITTABLE_LIST_H
