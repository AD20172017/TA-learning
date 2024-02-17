//
// Created by 17656 on 2024/2/17.
//

#ifndef RAYTRACING_TEXTURE_H
#define RAYTRACING_TEXTURE_H

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

#endif //RAYTRACING_TEXTURE_H
