#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 10000000

void vector_add(float *out, float *a, float *b, int n) {
    for(int i = 0; i < n; i++){
        out[i] = a[i] + b[i];
    }
}

int main(){
    float *a, *b, *out; 
    clock_t start, end;
    double cpu_time_used;

    // Allocate memory
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    // Initialize array
    for(int i = 0; i < N; i++){
        a[i] = 1.0f; 
        b[i] = 2.0f;
    }

    // Measure execution time
    start = clock();
    vector_add(out, a, b, N);
    end = clock();
    
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    
    printf("CPU Vector Addition Results:\n");
    printf("First 5 elements: ");
    for(int i = 0; i < 5; i++) {
        printf("%.1f ", out[i]);
    }
    printf("\nLast 5 elements: ");
    for(int i = N-5; i < N; i++) {
        printf("%.1f ", out[i]);
    }
    printf("\nCPU Execution time: %f seconds\n", cpu_time_used);
    
    // Cleanup
    free(a); free(b); free(out);
    return 0;
}
