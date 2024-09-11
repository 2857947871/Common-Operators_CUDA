// reduce 类算子 -> 累加
// v0: 利用 shared memory 进行并行化处理(局部)
// ms: 0.696928 ms
#include <cuda.h>
#include "cuda_runtime.h"

#include <bits/stdc++.h>

// blockSize 作为模板参数的效果主要是用于静态 shared memory 的申请, 需要传入编译期(间)常量指定大小
template<int blockSize>
__global__ void reduce_v0(int* input, int* output, size_t n) {

    int tid  = threadIdx.x;
    int gtid = blockIdx.x * blockSize + threadIdx.x;
    
    // load
    // 申请静态 shared memory, 需要传入编译期(间)常量(blockDim.x: 变量)
    // blockSize 是模板参数, 在核函数被调用时就已经知道它的值(const int)
    // 申请 blocksize * float 个 shared memory 空间
    // shared memoty <--> block   每个 block 各自拥有 shared memory
    __shared__ float smem[blockSize];
    smem[tid] = input[gtid];
    __syncthreads(); // 确保所有 thread 完成 shared memory 的写入

    // operation
    // idx: shared memory 的偏移量, 最大偏移 blockDim.x
    for (int idx = 1; idx < blockDim.x; idx *= 2) {
        // stage1: input:  0 1 2 3 4 5 6...
        //                 |/  |/  |/
        // stage1: output: 0 1 2 3 4 5 6... -> 0 2 4 6 8 10 12...
        // stage2: input:  0 2 4 6 8 10 12 14...
        //                 |/  |/  |/    |/
        // stage2: output: 0 2 4 6 8 10 12 14... -> 0 4 8 12...
        if (tid % (2 * idx) == 0) {
            // 左值: 当前 stage 的 output -> smem[0] 是当前 block 的累和
            smem[tid] += smem[tid + idx];
        } 
        // 问题: 发生 warp divergence ?   解答: if-else 而不是 if
        // warp divergence: 0 2 4 6 在工作, 1 3 5 7 也在工作
        // 因为条件判断 -> 1 3 5 7 没和 0 2 4 6 一起工作, 而是有了一个时间差, 这个时间差导致性能下降
        // 0 2 4 6 在工作, 1 3 5 7 在待着, 并没有时间差产生
        __syncthreads(); // 确保所有的 thread 均完成 block 的累和
    }

    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}

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

    printf("the ans is right\n");
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
    // 向上加 1 -> 防止 N = 255
    int gridSize  = std::min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]); 

    dim3 Grid(gridSize);
    dim3 Block(blockSize);

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
    reduce_v0<blockSize><<<Grid, Block>>>(device_in, device_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(host_out, device_out, gridSize * sizeof(int), cudaMemcpyDeviceToHost);

    // 验证
    CheckResult(host_out, groudtruth, gridSize);

    printf("reducev0 latency = %f ms\n", ms);

    // free
    cudaFree(device_in);
    cudaFree(device_out);
    free(host_in);
    free(host_out);


    return 0;
}