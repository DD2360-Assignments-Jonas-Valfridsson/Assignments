#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <fstream>
#include <iostream>
#include <chrono>

#define CHECK_CUDA_ERR(cudaerr)                        \
{                                                      \
auto err = cudaerr;                                    \
if (err != cudaSuccess) {                              \
    printf("kernel launch failed with error \"%s\".\n",\
           cudaGetErrorString(err));                   \
    exit(1);                                           \
}                                                      \
}

__device__ void color_pixel(
    char *colors, char *pixels, 
    float c_re, float c_im, 
    int global_index, int max_iter) {
    
    float i = 0, j = 0, ii = 0, jj = 0;
    int iteration = 0;
    while ( ii + jj < 4.0 && iteration < max_iter) {
      j = 2 * i * j + c_im;
      i = ii - jj + c_re;

      ii = i * i;
      jj = j * j;

      iteration++;
    }	

    int color_index = global_index * 3;

    if (iteration < max_iter) {
      int it_offset = 3 + iteration * 3;

      pixels[color_index] = colors[it_offset];
      pixels[color_index + 1] = colors[it_offset + 1];
      pixels[color_index + 2] = colors[it_offset + 2];
    } else {
      pixels[color_index] = colors[0];
      pixels[color_index + 1] = colors[1];
      pixels[color_index + 2] = colors[2];
    }
}

__global__ void mandelbrot(char *colors, char* pixels, int height, int width, int max_iter) {
    int global_index = threadIdx.x + blockDim.x * blockIdx.x;

    float x = (float)(global_index % width);
    float y = (float)(global_index / width);

    float f_width = (float)width, f_height = (float)height;


    float c_re = (x - f_height / 2.0) * 4.0 / f_height;
    float c_im = (y - f_height / 2.0) * 4.0 / f_height;

    if (global_index < height * width)
      color_pixel(colors, pixels, c_re, c_im, global_index, max_iter);
}


void fill_colors(char *colors, int max_iter) {
  colors[0] = 200;
  colors[1] = 200;
  colors[2] = 200;
  int shade = 1, speed1 = 0, speed2 = 10, speed3 = 0, j = 1;
  for (int i = 0; i < max_iter; i+=3) {
    if (j % 50 == 0)
      shade <<= 1;

    int red = colors[0] + i * speed1 - j;
    int green = colors[1] + i * speed2;
    int blue = colors[2] + i * speed3 - j;

    if (red < 0) red = 0;
    if (green < 0) green = 0;
    if (blue < 0) blue = 0;


    colors[3 + i]     = (red) % (256 / shade);
    colors[3 + i + 1] = (green) % (256 / shade);
    colors[3 + i + 2] = (blue) % (256 / shade);

    j += 1;
  }

}

int main(int argc, char **argv) {
  int write_to_file_flag = std::atoi(argv[1]);

    
  int x_pixels = 19968, y_pixels = 13730, max_iter = 150;
  int n_pixels = x_pixels * y_pixels;

  char *host_pixels, *device_pixels, *host_colors, *device_colors;

  size_t pixel_size = sizeof(char) * n_pixels * 3; // * 3 for RGB
  // This allocates pinned memory to speed-up memory transfers

  CHECK_CUDA_ERR(cudaMallocHost(&host_pixels, pixel_size));
  CHECK_CUDA_ERR(cudaMalloc(&device_pixels, pixel_size));

  size_t color_size = sizeof(char) * (max_iter * 3 + 3);
  CHECK_CUDA_ERR(cudaMallocHost(&host_colors, color_size));
  CHECK_CUDA_ERR(cudaMalloc(&device_colors, color_size));

  fill_colors(host_colors, max_iter);

  CHECK_CUDA_ERR(cudaMemcpy(device_colors, host_colors, color_size, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERR(cudaDeviceSynchronize());

  auto start = std::chrono::steady_clock::now();

  mandelbrot<<<(32 + n_pixels) / 32, 32>>>(
      /*colors=*/device_colors, 
      /*pixels=*/device_pixels, 
      /*height=*/y_pixels, 
      /*width=*/x_pixels, 
      /*max_iter*/max_iter);


  CHECK_CUDA_ERR(cudaDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  std::cout << "RUN " 
    << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() 
    << std::endl;

  start = std::chrono::steady_clock::now();
  CHECK_CUDA_ERR(cudaMemcpy(host_pixels, device_pixels, pixel_size, cudaMemcpyDeviceToHost));
  CHECK_CUDA_ERR(cudaDeviceSynchronize());
  end = std::chrono::steady_clock::now();
  std::cout << "READ " 
    << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() 
    << std::endl;


  if (write_to_file_flag) {
    long long current_time = time(nullptr);
    std::ofstream image (std::to_string(current_time).append("-gpu.bmp"), std::ofstream::binary);
    image << 
      (uint8_t)0x42 << 
      (uint8_t)0x4D << 
      (uint8_t)0x7C << 
      (uint8_t)0x00 << 
      (uint8_t)0x00 << 
      (uint8_t)0x00 << 
      (uint8_t)0x00 << 
      (uint8_t)0x00 << 
      (uint8_t)0x00 << 
      (uint8_t)0x00 << 
      (uint8_t)0x1A << 
      (uint8_t)0x00 << 
      (uint8_t)0x00 << 
      (uint8_t)0x00 << 
      (uint8_t)0x0C << 
      (uint8_t)0x00 << 
      (uint8_t)0x00 << 
      (uint8_t)0x00 << 
      (uint8_t)0x00 << // Image Width
      (uint8_t)0x4E << // Image Width
      (uint8_t)0xA2 << // Image Height
      (uint8_t)0x45 << // Image height
      (uint8_t)0x01 << 
      (uint8_t)0x00 << 
      (uint8_t)0x18 << 
      (uint8_t)0x00;

    for (int i = 0; i < n_pixels * 3; i++) 
      image << host_pixels[i];

    image << 0x00 << 0x00;	
  }

  CHECK_CUDA_ERR(cudaFreeHost(host_pixels));
  CHECK_CUDA_ERR(cudaFreeHost(host_colors));
  CHECK_CUDA_ERR(cudaFree(device_pixels));
  CHECK_CUDA_ERR(cudaFree(device_colors));

  return 0;
}


