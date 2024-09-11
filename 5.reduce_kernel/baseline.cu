// reduce 类算子 -> 累加
// baseline: CPU 的处理逻辑
// ms: 519.084167 ms
#include <cuda.h>
#include "cuda_runtime.h"

#include <bits/stdc++.h>


__global__ void reduce_baseline(int* input, int* output, size_t n) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    printf("tid: %d\n", idx);

    int sum = 0;
    for (int i = 0; i < n; ++i) {
        sum += input[i];
    }

    *output = sum;
}

bool CheckResult(int *out, int groudtruth) {
    if (*out != groudtruth) {
        return false;
    }
    return true;
}

int main() {

    // 初始化变量
    float ms = 0;
    int groudtruth = 0;
    const int N = 25600000; // 数据量

    // 获取设备属性
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    // thread
    int gridSize  = 1;
    int blockSize = 1;
    dim3 Grid(gridSize);
    dim3 Block(blockSize);

    // allocate memory
    int* device_in;
    int* device_out;
    int* host_in  = (int* )malloc(N * sizeof(int));
    int* host_out = (int* )malloc(sizeof(int));
    for (int i = 0; i < N; ++i) {
        host_in[i] = 1;
        groudtruth += host_in[i];
    }
    cudaMalloc((void** )&device_in, N * sizeof(int));
    cudaMalloc((void** )&device_out, sizeof(int));
    cudaMemcpy(device_in, host_in, N * sizeof(int), cudaMemcpyHostToDevice);

    // 开始处理
    // CPU 的方式处理, 分配 1 个 thread
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_baseline<<<Grid, Block>>>(device_in, device_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(host_out, device_out, 1 * sizeof(int), cudaMemcpyDeviceToHost);

    // 验证
    if (CheckResult(host_out, groudtruth)) {
        printf("the ans is right\n");
    }
    else {
        printf("groudtruth: %d\n", groudtruth);
        printf("result: %d\n", host_out);
        printf("the ans is wrong\n");
    }
    printf("reduce_baseline latency = %f ms\n", ms);

    // free
    cudaFree(device_in);
    cudaFree(device_out);
    free(host_in);
    free(host_out);


    return 0;
}