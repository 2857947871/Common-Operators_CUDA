// reduce 类算子 -> 累加
// v3: 空闲 thread(每个 stage 过后仅有前一个 stage 的一半在工作)也干活
// baseline:  519.084167 ms
// reduce_v0: 0.696928 ms      bank conflict: 834
// reduce_v1: 0.510880 ms(old) bank conflict: 7001056
// reduce_v1: 0.478528 ms(new) bank conflict: 689
// reduce_v2: 0.454912 ms      bank conflict: 891
// reduce_v3: 0.237376 ms      bank conflict: 987


#include <cuda.h>
#include "cuda_runtime.h"

#include <bits/stdc++.h>


void CheckResult(int *out, int groudtruth, int n) {
    float res = 0;
    for (int i = 0; i < n; i++){
        res += out[i];
    }
    if (res != groudtruth) {
        printf("groudtruth: %d\n", groudtruth);
        printf("result: %d\n", res);
        printf("the ans is wrong\n");
    }

    else
        printf("the ans is right\n");
}


template<int blockSize>
__global__ void reduce_v3(int* input, int* output, size_t n) {

    __shared__ float smem[blockSize];
    unsigned int tid = threadIdx.x;
    unsigned int gtid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // load
    smem[tid] = input[gtid] + input[gtid + blockSize];
    __syncthreads();

    // operation
    for (unsigned int idx = blockDim.x / 2; idx > 0; idx >>= 1) {
        if (tid < idx) {
            smem[tid] += smem[tid + idx];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
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
    const int blockSize = 256;
    const int blockSize_v3 = blockSize / 2;
    int gridSize  = std::min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]); // 向上加 1 -> 防止 N = 255 -> gridSize = 0

    dim3 Grid(gridSize);
    dim3 Block(blockSize_v3);

    // allocate memory
    int* device_in;
    int* device_out;
    int* host_in  = (int* )malloc(N * sizeof(int));
    int* host_out = (int* )malloc(gridSize * sizeof(int));
    for (int i = 0; i < N; ++i) {
        host_in[i] = 1;
        groudtruth += host_in[i];
    }
    cudaMalloc((void** )&device_in, N * sizeof(int));
    cudaMalloc((void** )&device_out, gridSize * sizeof(int));
    cudaMemcpy(device_in, host_in, N * sizeof(int), cudaMemcpyHostToDevice);

    // 开始处理
    // grid: (100000, 1, 1)   block: (256, 1, 1)   thread = 25600000
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_v3<blockSize_v3><<<Grid, Block>>>(device_in, device_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(host_out, device_out, gridSize * sizeof(int), cudaMemcpyDeviceToHost);

    // 验证
    CheckResult(host_out, groudtruth, gridSize);

    printf("reducev2 latency = %f ms\n", ms);

    // free
    cudaFree(device_in);
    cudaFree(device_out);
    free(host_in);
    free(host_out);


    return 0;
}