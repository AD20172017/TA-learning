#include<iostream>
#include "time.h"
#include "vec3.h"
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

__global__
void render(vec3 *fb, int max_x, int max_y){
    int i = threadIdx.x+blockIdx.x*blockDim.x;
    int j = threadIdx.y+blockIdx.y*blockDim.y;
    if((i>max_x)||(j>=max_y)) return;
    int pixel_index = j*max_x + i;
    fb[pixel_index] = vec3( double (i) / max_x, double (j) / max_y, 0.2f);
}

int main() {

    // Image
    bool in= false;
    std::cerr<<"need write size of image?"<<std::endl;
    std::cin>>in;

    int image_width = 1600;
    int image_height = 1000;

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
    render<<<blocks, threads>>>(fb, image_width, image_height);
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