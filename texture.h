//
// Created by 17656 on 2024/2/17.
//

#ifndef RAYTRACING_TEXTURE_H
#define RAYTRACING_TEXTURE_H

#include "rtw_stb_image.h"
#include "rtweekend.h"

class texure {
public:
    virtual  ~texure() = default;

    virtual color value(double u, double v, const point3 &p) const = 0;

};

class solid_color : public texure {
public:
    solid_color(color c) : color_val(c) {}

    solid_color(double r, double g, double b) : color_val(color(r, g, b)) {}

    color value(double u, double v, const point3 &p) const override {
        return color_val;
    }

private:
    color color_val;
};

class checker_tex : public texure {
public:
    checker_tex(double _scale, shared_ptr<texure> _even, shared_ptr<texure> _odd)
            : inv_scale(_scale), even(_even), odd(_odd) {}


    checker_tex(double _scale, color _c1, color _c2)
            : inv_scale(_scale), even(make_shared<solid_color>(_c1)), odd(make_shared<solid_color>(_c2)) {}

    color value(double u, double v, const point3 &p) const override {
        auto xInteger = static_cast<int>(std::floor(inv_scale * p.x()));
        auto yInteger = static_cast<int>(std::floor(inv_scale * p.y()));
        auto zInteger = static_cast<int>(std::floor(inv_scale * p.z()));

        bool isEven = (xInteger + yInteger + zInteger) % 2 == 0;

        return isEven ? even->value(u, v, p) : odd->value(u, v, p);
    }

private:
    double inv_scale;
    shared_ptr<texure> even;
    shared_ptr<texure> odd;
};

class image_texture : public texure {
public:
    image_texture(const char *filename) : image(filename) {}

    color value(double u, double v, const point3 &p) const override {
        // If we have no texture data, then return solid cyan as a debugging aid.
        if (image.height() <= 0) return color(0, 1, 1);

        // Clamp input texture coordinates to [0,1] x [1,0]
        u = interval(0, 1).clamp(u);
        v = 1.0 - interval(0, 1).clamp(v);  // Flip V to image coordinates

        auto i = static_cast<int>(u * image.width());
        auto j = static_cast<int>(v * image.height());
        auto pixel = image.pixel_data(i, j);

        auto color_scale = 1.0 / 255.0;
        return color(color_scale * pixel[0], color_scale * pixel[1], color_scale * pixel[2]);
    }

private:
    rtw_image image;
};

#endif //RAYTRACING_TEXTURE_H
