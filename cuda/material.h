//
// Created by 17656 on 2024/2/3.
//

#ifndef RAYTRACING_MATERIAL_H
#define RAYTRACING_MATERIAL_H
#include "rtweekend.h"
#include "hittable.h"
class material {
public:
    __host__ __device__ virtual ~material() = default;

    __host__ __device__ virtual bool scatter(
            const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const = 0;
};

class lambertian : public material {
public:
    __host__ __device__ lambertian(const color& a) : albedo(a) {}

    __host__ __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered)const override {
        auto scatter_direction = rec.normal + random_unit_vector();

        // Catch degenerate scatter direction
        if (scatter_direction.near_zero())
            scatter_direction = rec.normal;
        scattered = ray(rec.p, scatter_direction);
        attenuation = albedo;
        return true;
    }

private:
    color albedo;
};


class metal : public material {
public:
    vec3 albedo;
    double fuzz;
public:
    __host__ __device__ metal(const color& a, double f) : albedo(a), fuzz(f < 1 ? f : 1) {}

    __host__ __device__ virtual bool  scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered)const override{
        vec3 reflected =reflect(unit_vector(r_in.direction()), rec.normal);



        scattered = ray(rec.p, reflected + fuzz*random_unit_vector());
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal)>0);
    }
};

class dielectric : public material {
public:
    __host__ __device__ dielectric(double index_of_refraction) : ir(index_of_refraction) {}

    __host__ __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered)
    const override {
        attenuation = color(1.0, 1.0, 1.0);

        //!!
        double refraction_ratio = rec.front_face ? (1.0/ir) : ir;

        /*vec3 unit_direction = unit_vector(r_in.direction());
        vec3 refracted = refract(unit_direction, rec.normal, refraction_ratio);

        scattered = ray(rec.p, refracted);*/
        vec3 unit_direction = unit_vector(r_in.direction());
        double cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
        double sin_theta = sqrt(1.0 - cos_theta*cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0;
        vec3 direction;
        //????
        if (cannot_refract|| reflectance(cos_theta, refraction_ratio) > random_double())
            direction = reflect(unit_direction, rec.normal);
        else
            direction = refract(unit_direction, rec.normal, refraction_ratio);

        scattered = ray(rec.p, direction);
        return true;
    }

private:
    double ir; // Index of Refraction
    __host__ __device__ static double reflectance(double cosine, double ref_idx) {
        // Use Schlick's approximation for reflectance.
        auto r0 = (1-ref_idx) / (1+ref_idx);
        r0 = r0*r0;
        return r0 + (1-r0)*pow((1 - cosine),5);
    }
};
#endif //RAYTRACING_MATERIAL_H
