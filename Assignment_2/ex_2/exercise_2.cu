#include "stdio.h"
#include <limits>
#include <iostream>
#include <chrono>


__global__ void SAXPY(float *x,  float * c, float* y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  y[index] = c[index] + x[index];
}

int main(int argc, char **argv) {


  int N, threads;
  N = std::atoi(argv[1]);
  threads = std::atoi(argv[2]);

  float *x, *y, *c;

  // Allocate in shared memory
  cudaMallocManaged(&x, N * sizeof(float));
  cudaMallocManaged(&y, N * sizeof(float));
  cudaMallocManaged(&c, N * sizeof(float));

  // Initialize with some values
  for (int i = 0; i < N; i++) {

    x[i] = (float)(rand()) / ((float)(RAND_MAX));
    c[i] = 1;
  }

  auto start = std::chrono::steady_clock::now();
  // Make sure threads divides N evenly
  SAXPY<<<N / threads, threads>>>(x, c, y);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("Error %s", cudaGetErrorString(err));
  }
  auto end = std::chrono::steady_clock::now();

  std::cout
    << "SAXPY took: "
    << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "µs ≈ "
    << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms.\n";

  return 0;
}
