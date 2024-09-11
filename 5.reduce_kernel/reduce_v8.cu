// reduce 类算子 -> 累加
// v8: warp_level
// baseline:  519.084167 ms
// reduce_v0: 0.696928 ms      bank conflict: 834
// reduce_v1: 0.510880 ms(old) bank conflict: 7001056
// reduce_v1: 0.478528 ms(new) bank conflict: 689
// reduce_v2: 0.454912 ms      bank conflict: 891
// reduce_v3: 0.237376 ms      bank conflict: 987
// reduce_v4: 0.197024 ms      bank conflict: 2343
// reduce_v5: 0.236256 ms
// reduce_v6: 0.202784 ms(blockSize = 128)
// reduce_v7: 0.422016 ms(两次 reduce)
// reduce_v8: 0.297760 ms(大数据量不适合 warp_level)

#include <cuda.h>
#include "cuda_runtime.h"
#include <bits/stdc++.h>


# define WarpSize 32

template <int blockSize>
__device__ float WarpShuffle(float sum) {
    // down: 前面的线程向后面的线程要数据
    // up: 后面的线程向前面的线程要数据
    sum += __shfl_down_sync(0xffffffff, sum, 16);   // 0-15 = 0-15 + 16-31
    sum += __shfl_down_sync(0xffffffff, sum, 8);    // 0-7  = 0-7  + 8-15
    sum += __shfl_down_sync(0xffffffff, sum, 4);    // 0-3  = 0-3  + 4-7 
    sum += __shfl_down_sync(0xffffffff, sum, 2);    // 0-1  = 0-1  + 2-3
    sum += __shfl_down_sync(0xffffffff, sum, 1);    // 0    = 0    + 1 

    return sum;
}

template <int blockSize>
__global__ void reduce_warp_level(float *input,float *output, unsigned int N){
    // 当前线程的私有寄存器 -> 每个线程都会有一个 sum 寄存器
    float sum = 0;

    unsigned int tid  = threadIdx.x;
    unsigned int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_tid = blockSize * gridDim.x;

    // 一个线程处理 m 个元素
    // eg: total_ti = 10  N = 50
    // i = 0 -> 10 -> 20 -> 30 -> 40
    // i = 9 -> 19 -> 29 -> 39 -> 49
    for (int i = gtid; i < N; i += total_tid)
        sum += input[i];

    // 存储 partial sums for each warp of a block
    // WarpSums: 位于 shared_memory
    // blockSize / WarpSize: 32 个线程为一个 warp
    __shared__ float WarpSums[blockSize / WarpSize];
    
    // 当前线程所在的 warp 内的 ID(线程 ID)
    // tid.max() = blockSize, 因此 WarpSum 也可以用 laneId 索引
    const int laneId = tid % WarpSize;
    
    // 当前线程所在 warp 在所有 warp 内的 ID(warp ID)
    const int warpId = tid / WarpSize;
    
    // 交换 warp 内线程间的寄存器数据
    sum = WarpShuffle<blockSize>(sum);
    
    // 将每个 warp 中的第 0 个线程中的 sum 取出放入 WarpSums, 准备求和(block 尺度)
    if(laneId == 0) {
        WarpSums[warpId] = sum;
    }
    __syncthreads();

    // 得到了每个 block 的每个 warp 层次的累和
    // 求 block 层次的累和
    // 从 WarpSums 取数据, 超出索引部分置 0
    sum = (tid < blockSize / WarpSize) ? WarpSums[laneId] : 0;

    // 最后一次累和 -> warp
    if (warpId == 0) 
        sum = WarpShuffle<blockSize/WarpSize>(sum);

    // 求出每个 block 的结果并返回
    if (tid == 0)
        output[blockIdx.x] = sum;
}

bool CheckResult(float *out, float groudtruth, int n){
    float res = 0;
    for (int i = 0; i < n; i++){
        res += out[i];
    }
    if (res != groudtruth) {
        return false;
    }
    return true;
}

int main() {
    // properites
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    // init
    float milliseconds = 0;
    const int N = 25600000;
    float groudtruth = N * 1.0f;
    const int blockSize = 256;                                                      // blocksize(256, 1, 1)
    int GridSize = std::min((N + 256 - 1) / blockSize, deviceProp.maxGridSize[0]);   // gridsize(10000, 1, 1)
    dim3 Grid(GridSize);
    dim3 Block(blockSize);


    // allocate memory
    float* device_in;
    cudaMalloc((void** )&device_in, N * sizeof(float));
    float* host_in = (float* )malloc(N * sizeof(float));
    float* device_out;
    cudaMalloc((void** )&device_out, GridSize * sizeof(float));
    float* host_out = (float* )malloc(GridSize * sizeof(float));
    for(int i = 0; i < N; i++)
        host_in[i] = 1.0f;
    cudaMemcpy(device_in, host_in, N * sizeof(float), cudaMemcpyHostToDevice);

    // operation
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_warp_level<blockSize><<<Grid, Block>>>(device_in, device_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(host_out, device_out, GridSize * sizeof(float), cudaMemcpyDeviceToHost);
    bool is_right = CheckResult(host_out, groudtruth, GridSize);
    if(is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        printf("groudtruth is: %f \n", groudtruth);
    }
    printf("reduce_warp_level latency = %f ms\n", milliseconds);

    // free
    cudaFree(device_in);
    cudaFree(device_out);
    free(host_in);
    free(host_out);


    return 0;
}