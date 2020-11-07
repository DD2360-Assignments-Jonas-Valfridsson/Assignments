#include <curand_kernel.h>
#include <curand.h>
#include <chrono>
#include <iostream>
#include <string.h>
#include <math.h>
#include <time.h>

__global__ void setup_gpu_rng(int n, curandState *rng_states, int seed) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) curand_init(seed, i, 0, &rng_states[i]);
}

__global__ void gpu_estimate_pi(int n, curandState *rng_states, int samples, float *pi_estimates) {
  int i = blockIdx.x * blockDim.x + threadIdx.x, count = 0;

  float x, y, z;
  if (i < n) {
    for (int j = 0; j < samples; j++) {
      x = curand_uniform(&rng_states[i]);
      y = curand_uniform(&rng_states[i]);

      z = sqrt((x*x) + (y*y));

      if (z <= 1.0) count++;
    }

    pi_estimates[i] = 4.0 * (float)count / (float)samples;
  }
}

float cpu_estimate_pi(int n) {
  // Calculate PI following a Monte Carlo method
  float x, y, z;
  int count;
  for (int iter = 0; iter < n; iter++) {
    // Generate random (X,Y) points
    x = (float)random() / (float)RAND_MAX;
    y = (float)random() / (float)RAND_MAX;
    z = sqrt((x*x) + (y*y));

    // Check if point is in unit circle
    if (z <= 1.0)
    {
      count++;
    }
  }

  return ((float)count / (float)n) * 4.0;

}

int main(int argc, char* argv[]) {
  int samples = std::atoi(argv[1]), samples_per_thread = std::atoi(argv[2]), thread_per_block = std::atoi(argv[3]);

  auto start = std::chrono::steady_clock::now();
  float pi = cpu_estimate_pi(samples);
  auto end = std::chrono::steady_clock::now();

  std::cout << "CPU-PI: " << pi << std::endl;
  std::cout << "CPU " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;

  curandState *rng;
  cudaMalloc(&rng, sizeof(curandState) * samples);

  int blocks = (samples + thread_per_block - 1) / (thread_per_block * samples_per_thread);
  int threads = blocks * thread_per_block;

  setup_gpu_rng<<<blocks, thread_per_block>>>(samples, rng, 12);

  float *device_pi_estimates, *host_pi_estimates;
  cudaMalloc(&device_pi_estimates, sizeof(float) * threads);
  host_pi_estimates = (float*)malloc(sizeof(float) * threads);

  start = std::chrono::steady_clock::now();
  gpu_estimate_pi<<<blocks, thread_per_block>>>(samples, rng, samples_per_thread, device_pi_estimates);

  cudaMemcpy(host_pi_estimates, device_pi_estimates, sizeof(float) * threads, cudaMemcpyDeviceToHost);
  float gpu_pi = 0;
  for (int i = 0; i < threads; i++) {
    gpu_pi += host_pi_estimates[i];
  }

  gpu_pi /= (float)threads;
  end = std::chrono::steady_clock::now();

  std::cout << "GPU-PI: " << gpu_pi << std::endl;
  std::cout << "GPU " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;

  return 0;
}

