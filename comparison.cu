#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 10000000

// CPU version
void vector_add_cpu(float *out, float *a, float *b, int n) {
    for(int i = 0; i < n; i++){
        out[i] = a[i] + b[i];
    }
}

// CUDA kernel
__global__ void vector_add_gpu(float *out, float *a, float *b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

int main(){
    float *a, *b, *out_cpu, *out_gpu;
    float *d_a, *d_b, *d_out;
    
    size_t size = N * sizeof(float);
    
    // Allocate memory
    a = (float*)malloc(size);
    b = (float*)malloc(size);
    out_cpu = (float*)malloc(size);
    out_gpu = (float*)malloc(size);
    
    // Initialize arrays
    for(int i = 0; i < N; i++){
        a[i] = 1.0f; 
        b[i] = 2.0f;
    }
    
    // CPU version timing
    clock_t start_cpu = clock();
    vector_add_cpu(out_cpu, a, b, N);
    clock_t end_cpu = clock();
    double cpu_time = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC;
    
    // GPU version setup
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_out, size);
    
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    
    // GPU version timing
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    
    cudaEventRecord(start_gpu);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vector_add_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_a, d_b, N);
    
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);
    
    cudaMemcpy(out_gpu, d_out, size, cudaMemcpyDeviceToHost);
    
    // Verify results
    int errors = 0;
    for(int i = 0; i < N; i++) {
        if (abs(out_cpu[i] - out_gpu[i]) > 1e-5) {
            errors++;
            if (errors < 10) printf("Error at index %d: CPU=%.6f, GPU=%.6f\n", 
                                   i, out_cpu[i], out_gpu[i]);
        }
    }
    
    // Results
    printf("CUDA Vector Addition Performance Comparison\n");
    printf("==========================================\n");
    printf("Array size: %d elements\n", N);
    printf("CPU Execution time: %.6f seconds\n", cpu_time);
    printf("GPU Execution time: %.6f seconds\n", gpu_time / 1000.0); // Convert ms to seconds
    printf("Speedup: %.2fx\n", cpu_time / (gpu_time / 1000.0));
    printf("Verification errors: %d\n", errors);
    
    // Cleanup
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    free(a); free(b); free(out_cpu); free(out_gpu);
    
    return 0;
}
