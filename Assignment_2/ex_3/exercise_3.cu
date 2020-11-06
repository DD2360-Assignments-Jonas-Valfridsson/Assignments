const float dx = 0.1;
const float dy = 0.1;
const float dz = 0.1;

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
  int i = threadIdx.x * blockIdx.x * blockDim.x;

  if (i < n) {
    particles[i].velocity = velocity_update(particles[i].velocity, time);

    particles[i].position.x += particles[i].velocity.x;
    particles[i].position.y += particles[i].velocity.y;
    particles[i].position.z += particles[i].velocity.z;
  }
}

void cpu_step(int n, float3 *dv, Particle *particles, float time) {
  for (int i = 0; i < n; i++) {
    particles[i].velocity = velocity_update(particles[i].velocity, time);

    particles[i].position.x += dv[i].x;
    particles[i].position.y += dv[i].y;
    particles[i].position.z += dv[i].z;
  }
}


int main(int argc, char **argv) {
  int n_par = std::atoi(argv[1]), n_it =std::atoi(argv[2]);

  float3 *dv;
  Particle *par;

  // Allocate shared memory
  cudaMallocManaged(&dv, sizeof(float3) * n_par);
  cudaMallocManaged(&par, sizeof(Particle) * n_par);

  return 0;
}
