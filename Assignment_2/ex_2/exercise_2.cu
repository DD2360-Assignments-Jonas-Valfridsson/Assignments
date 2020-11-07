#include "stdio.h"
#include <limits>
#include <iostream>
#include <chrono>


__global__ void GPU_SAXPY(int n, float *x,  float a, float* y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) y[index] = a * x[index] + y[index];
}

void CPU_SAXPY(int n, float *x, float a, float * y) {
  for (int i = 0; i < n; i++) {
    y[i] = a* x[i] + y[i];
  }

}

void initialize(int N, float *x, float *a, float * y) {
  // Initialize SAXPY computation with some values
  for (int i = 0; i < N; i++) {
    y[i] = 2.0;
    x[i] = (float)(rand()) / ((float)(RAND_MAX));
  }
  a[0] = 2.0;
}

int main(int argc, char **argv) {


  int N, threads, seed = 12;
  N = std::atoi(argv[1]);
  threads = std::atoi(argv[2]);

  float *x_host, *y_host, *x_device, *y_device, a, *y_cpu_result, *y_gpu_result;

  // Cuda errs
  cudaError_t err;

  // Allocate in cpu memory

  x_host = (float*)malloc(sizeof(float) * N);
  y_host = (float*)malloc(sizeof(float) * N);

  srand(seed);
  initialize(N, x_host, &a, y_host);

  auto start = std::chrono::steady_clock::now();
  CPU_SAXPY(N, x_host, a, y_host);
  auto end = std::chrono::steady_clock::now();

  std::cout << "CPU " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;

  // store result cpu
  y_cpu_result = (float*)malloc(sizeof(float) * N);
  memcpy(y_cpu_result, y_host, sizeof(float) * N);

  // reset seed and reinit
  srand(seed);
  initialize(N, x_host, &a, y_host);

  // Allocate memory on GPU
  cudaMalloc(&x_device, sizeof(float) * N);
  cudaMalloc(&y_device, sizeof(float) * N);

  // Copy to device
  cudaMemcpy(x_device, x_host, sizeof(float) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(y_device, y_host, sizeof(float) * N, cudaMemcpyHostToDevice);

  // Synchronize Copy
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("Error %s", cudaGetErrorString(err));
  }

  start = std::chrono::steady_clock::now();
  // +1 to make sure we get the full array
  GPU_SAXPY<<<(N / threads) + 1, threads>>>(N, x_device, a, y_device);

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("Error %s", cudaGetErrorString(err));
  }
  end = std::chrono::steady_clock::now();

  std::cout << "GPU " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;

  // store gpu result
  y_gpu_result = (float*)malloc(sizeof(float) * N);
  cudaMemcpy(y_gpu_result , y_device, sizeof(float) * N, cudaMemcpyDeviceToHost);


  float mse = 0;
  for (int i = 0; i < N; i ++) 
    mse += sqrt((y_cpu_result[i] - y_gpu_result[i]) * (y_cpu_result[i] - y_gpu_result[i]));

  std::cout << "mean squared error between GPU and CPU is: " << mse << std::endl;


  // Free all memory
  free(y_gpu_result);
  free(y_cpu_result);

  free(x_host);
  free(y_host);

  cudaFree(x_device);
  cudaFree(y_device);
  return 0;
}
