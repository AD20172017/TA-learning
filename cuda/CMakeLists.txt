cmake_minimum_required(VERSION 3.23)
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/bin/nvcc.exe")
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_EXE_LINKER_FLAGS "-static")
project(cuda LANGUAGES CXX CUDA)
find_package(cuda)
set(CMAKE_CUDA_STANDARD 17 CMAKE_C_STANDARD CMAKE_CXX_STANDARD)

add_executable(cuda add.cu)
add_executable(cpp demo.cpp)
add_executable(raytracing main.cu ray.h vec3.h color.h hittable.h sphere.h hittable_list.h rtweekend.h interval.h camera.h material.h)

set_target_properties(cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
