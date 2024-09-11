// 原子操作(串行): histogram latency = 1.269152 ms

# include <cuda.h>
# include "cuda_runtime.h"
# include <bits/stdc++.h>
__global__ void histgram(int *hist_data, int *bin_data) {
    
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    // error: 资源竞争
    // bin_data[hist_data[gtid]]++;
    // 原子加法, 并行 -> 串行
    atomicAdd(&bin_data[hist_data[gtid]], 1);
}

bool CheckResult(int *out, int* groudtruth, int N){
    for (int i = 0; i < N; i++){
        if (out[i] != groudtruth[i]) {
            return false;
        }
    }
    return true;
}


int main () {
    // init
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    float milliseconds = 0;
    const int N = 25600000;
    const int blockSize = 256;
    int GridSize = std::min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]);
    dim3 Grid(GridSize);
    dim3 Block(blockSize);

    int* hist = (int* )malloc(N * sizeof(int));
    int* bin  = (int* )malloc(N * sizeof(int));
    int* bin_data;
    int* hist_data;
    cudaMalloc((void **)&bin_data, 256 * sizeof(int));
    cudaMalloc((void **)&hist_data, N * sizeof(int));

    for(int i = 0; i < N; i++){
        hist[i] = i % 256;
    }

    int *groudtruth = (int *)malloc(256 * sizeof(int));;
    for(int j = 0; j < 256; j++){
        groudtruth[j] = 100000;
    }
    cudaMemcpy(hist_data, hist, N * sizeof(int), cudaMemcpyHostToDevice);

    // operation
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    histgram<<<Grid, Block>>>(hist_data, bin_data);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(bin, bin_data, 256 * sizeof(int), cudaMemcpyDeviceToHost);
    bool is_right = CheckResult(bin, groudtruth, 256);
    if(is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        for(int i = 0; i < 256; i++){
            printf("%lf ", bin[i]);
        }
        printf("\n");
    }
    printf("histogram latency = %f ms\n", milliseconds);    

    // free
    cudaFree(bin_data);
    cudaFree(hist_data);
    free(bin);
    free(hist);
}