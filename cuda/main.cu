// -*- coding: gbk -*-
#include<iostream>
#include "time.h"
#include "vec3.h"
#include "ray.h"
#include "hittable_list.h"
#include "sphere.h"
#include "camera.h"
#include "material.h"
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
                  file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

void get_gpu(){
    int dev=0;
    cudaDeviceProp devProp;
    checkCudaErrors(cudaGetDeviceProperties(&devProp, dev));
    std::cerr << "GPU device "<< dev<<':'<<devProp.name<<std::endl;
    std::cerr << "SM 个数: "<< devProp.multiProcessorCount<<std::endl;
    std::cerr << "每个线程块的共享内存大小: "<< devProp.sharedMemPerBlock/1024.0 <<"KB"<<std::endl;
    std::cerr << "每个线程块最大线程数: "<< devProp.maxThreadsPerBlock<<std::endl;
    std::cerr << "每个EM的最大线程数: "<< devProp.maxThreadsPerMultiProcessor<<std::endl;
    std::cerr << "每个SM的最大线程数: "<< devProp.maxThreadsPerMultiProcessor/32<<std::endl;
}
//这个圆貌似会随着图片尺寸形变,之后还要添加视口
__device__ bool hit_sphere(const vec3& center, double radius, const ray& r) {
    vec3 oc = r.origin() - center;
    double a = dot(r.direction(), r.direction());
    double b = 2.0f * dot(oc, r.direction());
    double c = dot(oc, oc) - radius*radius;
    double discriminant = b*b - 4.0f*a*c;
    return (discriminant > 0.0f);
}

__device__ vec3 color(const ray& r) {
    if (hit_sphere(vec3(0,0,-1), 0.5, r))
        return vec3(0.5,0.7,0.2);
    vec3 unit_direction = unit_vector(r.direction());
    double t = 0.5f*(unit_direction.y() + 1.0f);
    return (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
}

__global__
void render(vec3 *fb, int max_x, int max_y ,
            vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin){
    int i = threadIdx.x+blockIdx.x*blockDim.x;
    int j = threadIdx.y+blockIdx.y*blockDim.y;
    if((i>max_x)||(j>=max_y)) return;
    int pixel_index = j*max_x + i;
    double u = double(i) / double(max_x);
    double v = double(j) / double(max_y);
    ray r(origin, lower_left_corner + u*horizontal + v*vertical);
    fb[pixel_index] = color(r);
}

int main() {

    get_gpu();

    // Image
    bool in= false;
    std::cerr<<"need write size of image?"<<std::endl;
    std::cin>>in;

    int image_width = 256;
    int image_height = 184;

    int block_x=8,block_y=8;

    if(in){
        std::cerr<<"image_width: ";
        std::cin>>image_width;
        std::cerr<<'\n';

        std::cerr<<"image_height: ";
        std::cin>>image_height;
        std::cerr<<'\n';
    }
    std::cerr << "Rendering a " << image_width << "x" << image_height << " image ";
    std::cerr << "in " << block_x << "x" << block_y << " blocks.\n";

    int num_pixels = image_width*image_height;
    size_t fb_size = num_pixels*sizeof(vec3);

    // allocate FB
    vec3 *fb;
    //统一内存
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    clock_t start, stop;
    start = clock();

    // Render
    dim3 blocks(image_width/block_x+1,image_height/block_y+1);
    dim3 threads(block_x,block_y);
    render<<<blocks, threads>>>(fb, image_width, image_height,
                                vec3(-2.0, -1.0, -1.0),
                                vec3(4.0, 0.0, 0.0),
                                vec3(0.0, 2.0, 0.0),
                                vec3(0.0, 0.0, 0.0));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int j = image_height-1; j >= 0; j--) {
        std::clog << "\rScanlines remaining: ------------------" << j << ' ' << std::flush;
        for (int i = 0; i < image_width; i++) {
            size_t pixel_index = j*image_width + i;
            int ir = int(255.99*fb[pixel_index].x());
            int ig = int(255.99*fb[pixel_index].y());
            int ib = int(255.99*fb[pixel_index].z());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }
    std::clog << "\nDone-------------------------------------\n";

    checkCudaErrors(cudaFree(fb));
}