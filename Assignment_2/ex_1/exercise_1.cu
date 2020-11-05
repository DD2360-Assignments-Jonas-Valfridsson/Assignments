#include "stdio.h"

__global__ void cuda_hello(){
    printf("Hello World! My thread ID is %d\n\n", threadIdx.x);
}

int main() {
    
    cuda_hello<<<1,256>>>(); 
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));
    return 0;
}

