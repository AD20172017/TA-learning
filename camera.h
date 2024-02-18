//
// Created by 17656 on 2024/2/3.
//

#ifndef RAYTRACING_CAMERA_H
#define RAYTRACING_CAMERA_H

#include "rtweekend.h"

#include "color.h"
#include "hittable.h"
#include <iostream>

class camera {
public:
    /* Public Camera Parameters Here */
    double aspect_ratio = 1.0;
    int image_width = 100;
    int samples_per_pixel = 10;
    int max_depth = 10;
    color background = color(0, 0, 0);

    double vfov = 90;  // Vertical view angle (field of view)
    point3 lookfrom = point3(0, 0, -1);  // Point camera is looking from
    point3 lookat = point3(0, 0, 0);   // Point camera is looking at
    vec3 vup = vec3(0, 1, 0);     // Camera-relative "up" direction

    //???
    double defocus_angle = 0;
    double focus_dist = 10;

    void render(const hittable &world) {
        initialize();

        std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";


        for (int j = 0; j < image_height; ++j) {
            std::clog << "\rScanlines remaining: ------------------" << image_height - j << ' ' << std::flush;
            for (int i = 0; i < image_width; ++i) {
                color pixel_color(0.0, 0.0, 0.0);
                #pragma omp parallel for
                for (int sample = 0; sample < samples_per_pixel; ++sample) {
                    ray r = get_ray(i, j);
                    pixel_color += ray_color(r, world, max_depth);
                }
                write_color(std::cout, pixel_color, samples_per_pixel);
            }
        }
        std::clog << "\nDone-------------------------------------\n";
    }

private:
    /* Private Camera Variables Here */
    int image_height;
    point3 center;
    point3 pixel00_loc;
    vec3 pixel_delta_u;
    vec3 pixel_delta_v;
    vec3 u, v, w;        // Camera frame basis vectors

    vec3 defocus_disk_u;  // Defocus disk horizontal radius
    vec3 defocus_disk_v;  // Defocus disk vertical radius


    void initialize() {
        image_height = static_cast<int>(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;

        center = lookfrom;


        //auto focal_length = (lookfrom - lookat).length();
        auto theta = degrees_to_radians(vfov);
        auto h = tan(theta / 2);
        auto viewport_height = 2 * h * focus_dist;
        auto viewport_width = viewport_height * ((static_cast<double>(image_width) / image_height));//数值转换的时候要注意!!

        // Calculate the u,v,w unit basis vectors for the camera coordinate frame.左手来判断叉乘结果
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        //viewport,image_height取整必须大于0,而view必须和image的比列相同
        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        vec3 viewport_u = viewport_width * u;    // Vector across viewport horizontal edge
        vec3 viewport_v = viewport_height * -v; // Vector down viewport vertical edge

        //离散化
        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;
        //计算像素中心值??
        auto viewport_upper_left = center - (focus_dist * w)
                                   - viewport_u / 2 - viewport_v / 2;
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

        auto defocus_radius = focus_dist * tan(degrees_to_radians(defocus_angle / 2));
        defocus_disk_u = u * defocus_radius;
        defocus_disk_v = v * defocus_radius;

    }

    ray get_ray(int i, int j) const {
        // Get a randomly sampled camera ray for the pixel at location i,j.

        auto pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
        auto pixel_sample = pixel_center + pixel_sample_square();
        //?
        auto ray_origin = (defocus_angle <= 0) ? center : defocus_disk_sample();
        auto ray_direction = pixel_sample - ray_origin;
        auto ray_time = random_double();

        return ray(ray_origin, ray_direction, ray_time);

    }

    point3 defocus_disk_sample() const {
        // Returns a random point in the camera defocus disk.
        auto p = random_in_unit_disk();
        return center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
    }

    vec3 pixel_sample_square() const {
        // Returns a random point in the square surrounding a pixel at the origin.
        auto px = -0.5 + random_double();
        auto py = -0.5 + random_double();
        return (px * pixel_delta_u) + (py * pixel_delta_v);
    }

    color ray_color(const ray &r, const hittable &world, int depth) const;
};

color camera::ray_color(const ray &r, const hittable &world, int depth) const {
    hit_record rec;
    if (depth <= 0)return color(0, 0, 0);
    /*if(world.hit(r, interval(min_error,infinity), rec)){
        ray scattered;
        color atten;
        if (rec.mat->scatter(r, rec, atten, scattered))
            return atten * ray_color(scattered, world, depth-1);
        return color(0,0,0);
    }
    vec3 unit_direction = unit_vector(r.direction());
    auto a = 0.5*(unit_direction.y() + 1.0);
    return (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
*/
    if (!world.hit(r, interval(min_error, infinity), rec))
        return background;

    ray scattered;
    color attenuation;
    color color_from_emission = rec.mat->emitted(rec.u, rec.v, rec.p);

    if (!rec.mat->scatter(r, rec, attenuation, scattered))
        return color_from_emission;

    color color_from_scatter = attenuation * ray_color(scattered, world, depth - 1);

    return color_from_emission + color_from_scatter;
}

#endif //RAYTRACING_CAMERA_H
