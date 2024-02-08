cmake_minimum_required(VERSION 3.23)
project(RayTracing)
set(CMAKE_EXE_LINKER_FLAGS "-static")
set(CMAKE_CXX_STANDARD 20)

add_executable(RayTracing main.cpp vec3.h ray.h color.h hittable.h sphere.h hittable_list.h rtweekend.h interval.h camera.h material.h)
