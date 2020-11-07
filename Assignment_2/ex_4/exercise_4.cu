#include <curand_kernel.h>
#include <curand.h>
#include <chrono>
#include <iostream>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdio.h>

__global__ void setup_gpu_rng(long long n, curandState *rng_states, long long seed) {
  long long i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) curand_init(seed, i, 0, &rng_states[i]);
}

__global__ void gpu_estimate_pi(long long n, curandState *rng_states, long long samples, double *pi_estimates) {
  long long i = blockIdx.x * blockDim.x + threadIdx.x, count = 0;


  double x, y, z;
  if (i < n) {
    for (long long j = 0; j < samples; j++) {
      x = curand_uniform(&rng_states[i]);
      y = curand_uniform(&rng_states[i]);

      z = sqrt((x*x) + (y*y));

      if (z <= 1.0) count++;
    }

    pi_estimates[i] = 4.0 * (double)count / (double)samples;

    //printf("i: %lld est: %.2f\n", i, pi_estimates[i]);
  }

}

double cpu_estimate_pi(long long n) {
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

  return ((double)count / (double)n) * 4.0;

}

int main(int argc, char* argv[]) {
  long long samples = std::atoi(argv[1]), samples_per_thread = std::atoi(argv[2]), thread_per_block = std::atoi(argv[3]);

  auto start = std::chrono::steady_clock::now();
  double pi = cpu_estimate_pi(samples);
  auto end = std::chrono::steady_clock::now();

  std::cout << "CPU-PI: " << pi << std::endl;
  std::cout << "CPU " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;


  long long blocks = (samples + thread_per_block - 1) / (thread_per_block * samples_per_thread) + 1;
  long long total_threads = blocks * thread_per_block;

  curandState *rng;
  cudaMalloc(&rng, sizeof(curandState) * total_threads);
  setup_gpu_rng<<<blocks, thread_per_block>>>(total_threads, rng, time(NULL));


  double *device_pi_estimates, *host_pi_estimates;
  cudaMalloc(&device_pi_estimates, sizeof(double) * total_threads);
  host_pi_estimates = (double*)malloc(sizeof(double) * total_threads);

  start = std::chrono::steady_clock::now();
  gpu_estimate_pi<<<blocks, thread_per_block>>>(total_threads, rng, samples_per_thread, device_pi_estimates);

  cudaMemcpy(host_pi_estimates, device_pi_estimates, sizeof(double) * total_threads, cudaMemcpyDeviceToHost);
  double gpu_pi = 0;
  for (long long i = 0; i < total_threads; i++) {
    gpu_pi += host_pi_estimates[i];
  }

  gpu_pi /= (double)total_threads;
  end = std::chrono::steady_clock::now();

  std::cout << "GPU-PI: " << gpu_pi << std::endl;
  std::cout << "GPU " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;

  return 0;
}

