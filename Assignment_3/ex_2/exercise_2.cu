#include "stdio.h"
#include <iostream>
#include <chrono>


struct Particle {
  float3 position;
  float3 velocity;
};

__host__ __device__ float3 velocity_update(float3 velocity, float time) {
  float3 u_vel;

  u_vel.x = velocity.x + sin(time);
  u_vel.y = velocity.y + sin(time);
  u_vel.z = velocity.z + sin(time);

  return u_vel;
}

__global__ void gpu_step(int n, Particle *particles, float time) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < n) {
    particles[i].velocity = velocity_update(particles[i].velocity, time);

    particles[i].position.x += particles[i].velocity.x;
    particles[i].position.y += particles[i].velocity.y;
    particles[i].position.z += particles[i].velocity.z;
  }
}

void cpu_step(int n, Particle *particles, float time) {
  for (int i = 0; i < n; i++) {
    particles[i].velocity = velocity_update(particles[i].velocity, time);

    particles[i].position.x += particles[i].velocity.x;
    particles[i].position.y += particles[i].velocity.y;
    particles[i].position.z += particles[i].velocity.z;
  }
}

float rand_float() {
  return (float)(rand()) / ((float)(RAND_MAX));
}

void init_particles(int n, Particle *particles) {
  for (int i = 0; i < n; i++) {
    float3 pos, vel;

    pos.x = rand_float();
    pos.y = rand_float();
    pos.z = rand_float();

    vel.x = rand_float();
    vel.y = rand_float();
    vel.z = rand_float();

    particles[i].position = pos;
    particles[i].velocity = vel;
  }
}

double mse_difference(int n, Particle *xp, Particle *yp) {
  double mse = 0;
  for (int i = 0; i < n; i++) {
    mse += (xp[i].position.x - yp[i].position.x) * (xp[i].position.x - yp[i].position.x);
    mse += (xp[i].position.y - yp[i].position.y) * (xp[i].position.y - yp[i].position.y);
    mse += (xp[i].position.z - yp[i].position.z) * (xp[i].position.z - yp[i].position.z);

    mse += (xp[i].velocity.x - yp[i].velocity.x) * (xp[i].velocity.x - yp[i].velocity.x);
    mse += (xp[i].velocity.y - yp[i].velocity.y) * (xp[i].velocity.y - yp[i].velocity.y);
    mse += (xp[i].velocity.z - yp[i].velocity.z) * (xp[i].velocity.z - yp[i].velocity.z);

  }

  return mse / (double)n;
}

void print_particles(int n, Particle *par) {
  for (int i = 0; i < n; i++) {
    std::cout << "n: " << i << " px: " << par[i].position.x << " py: " << par[i].position.y
      << " pz: " << par[i].position.z << " vx: " << par[i].velocity.x << " vy: " << par[i].velocity.y
      << " vz: " << par[i].velocity.z << std::endl;
  }
}

int main(int argc, char **argv) {
  int n_par = std::atoi(argv[1]), n_it =std::atoi(argv[2]), block_size = std::atoi(argv[3]);

  cudaError_t err;

  Particle *par_host, *par_device, *par_device_result_on_host;


  par_host = (Particle*)malloc(sizeof(Particle) * n_par);
  init_particles(n_par, par_host);

  // Initialize memory that will contain the GPU particles on host
  // Copy the CPU particles to the GPU particles
  err = cudaMallocHost(&par_device_result_on_host, sizeof(Particle) * n_par);
  if (err != cudaSuccess) {
    printf("Error %s", cudaGetErrorString(err));
  }

  memcpy(par_device_result_on_host, par_host, sizeof(Particle) * n_par);


  float t = 0;

  // Simulate on CPU
  auto start = std::chrono::steady_clock::now();
  //for (int i = 0; i < n_it; i++) {
    //cpu_step(n_par, par_host, t);

    //t += 1.0;
  //}

  auto end = std::chrono::steady_clock::now();

  //std::cout << "CPU " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;


  // reset time
  t = 0;

  // Allocate device memory 
  cudaMalloc(&par_device, sizeof(Particle)*n_par);

  // Simulate on GPU
  start = std::chrono::steady_clock::now();
  for (int i = 0; i < n_it; i++) {
    // At the beginning of timestep copy from host to device
    cudaMemcpy(par_device, par_device_result_on_host, sizeof(Particle) * n_par, cudaMemcpyHostToDevice);

    // Perform one update
    gpu_step<<<(n_par / block_size) + 1, block_size>>>(n_par, par_device, t);

    // Copy from devince to host
    cudaMemcpy(par_device_result_on_host, par_device, sizeof(Particle) * n_par, cudaMemcpyDeviceToHost);
    //err = cudaDeviceSynchronize();
    //if (err != cudaSuccess) {
    //printf("Error %s", cudaGetErrorString(err));
    //}

    t += 1.0;
  }

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("Error %s", cudaGetErrorString(err));
  }

  end = std::chrono::steady_clock::now();

  std::cout << "GPU " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;


  //std::cout << "Particles from GPU:\n";
  //print_particles(n_par, par_device_result_on_host);
  //std::cout << "\n";

  //std::cout << "Particles from CPU:\n";
  //print_particles(n_par, par_host);
  //std::cout << "\n";

  //double mse = mse_difference(n_par, par_device_result_on_host, par_host);
  //std::cout << "GPU - CPU mean squared error: " << mse << std::endl;


  // Free memory
  free(par_host);
  cudaFree(par_device);

  return 0;
}
