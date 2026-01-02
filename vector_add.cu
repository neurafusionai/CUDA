#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 10000000

// CUDA kernel for vector addition
__global__ void vector_add(float *out, float *a, float *b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

int main(){
    float *a, *b, *out; 
    float *d_a, *d_b, *d_out;
    
    size_t size = N * sizeof(float);
    
    // Allocate host memory
    a   = (float*)malloc(size);
    b   = (float*)malloc(size);
    out = (float*)malloc(size);
    
    // Initialize host arrays
    for(int i = 0; i < N; i++){
        a[i] = 1.0f; 
        b[i] = 2.0f;
    }
    
    // Allocate device memory
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_out, size);
    
    // Transfer data from host to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    
    // Configure kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch kernel
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_a, d_b, N);
    
    // Wait for kernel to complete
    cudaDeviceSynchronize();
    
    // Transfer result back to host
    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);
    
    // Display results
    printf("CUDA Vector Addition Results:\n");
    printf("First 5 elements: ");
    for(int i = 0; i < 5; i++) {
        printf("%.1f ", out[i]);
    }
    printf("\nLast 5 elements: ");
    for(int i = N-5; i < N; i++) {
        printf("%.1f ", out[i]);
    }
    
    // Cleanup
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    free(a); free(b); free(out);
    
    return 0;
}
