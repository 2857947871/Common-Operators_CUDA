// 原子操作(串行): histogram latency = 1.269152 ms
// 原子操作(并行): histogram latency = 1.275328 ms


# include <cuda.h>
# include "cuda_runtime.h"
# include <bits/stdc++.h>
__global__ void histgram_atomic(int *hist_data, int *bin_data) {
    
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    // error: 资源竞争
    // bin_data[hist_data[gtid]]++;
    // 原子加法, 并行 -> 串行
    atomicAdd(&bin_data[hist_data[gtid]], 1);
}

template <int blockSize>
__global__ void histgram(int *hist_data, int *bin_data, int N) {
    
    // 问题: 使用 shared_memory -> bank conflict
    __shared__ int cache[blockSize];
    int gtid = blockIdx.x * blockSize + threadIdx.x;
    int tid = threadIdx.x;

    // 每个 thread 初始化 shared mem
    // 一个 block 公用一个 shared mem, 用 tid 作为索引
    cache[tid] = 0;
    __syncthreads();

    // total_tid: 25600000
    // eg: gtid = 100   step: 100000 * 256 
    for (int i = gtid; i < N; i += gridDim.x * blockSize) {
        // 每个单线程计算全局内存中的若干个值
        int val = hist_data[i];
        
        // bank conflict -> 其实比完全的 atomic 要慢
        // tid1 的 val 为 2   bank[0][2]
        // tid2 的 val 为 2   bank[0][2]
        // tid1 和 tid2 同时要访问 cache(smem) 的同一个 bank
        // tid1 的 val 为 0   bank[0][0]
        // tid2 的 val 为 32  bank[1][0]
        // tid1 和 tid2 同时要访问 cache(smem) 的同一个 bank
        atomicAdd(&cache[val], 1);
    }

    //此刻每个 block 的 bin 都已统计在 cache 这个 smem 中
    __syncthreads();

    // 汇总所有 block 的 cache(smem)
    atomicAdd(&bin_data[tid], cache[tid]);
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
    // histgram_atomic<<<Grid, Block>>>(hist_data, bin_data);
    histgram<blockSize><<<Grid, Block>>>(hist_data, bin_data, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(bin, bin_data, 256 * sizeof(int), cudaMemcpyDeviceToHost);
    bool is_right = CheckResult(bin, groudtruth, 256);
    if(is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
    }
    printf("histogram latency = %f ms\n", milliseconds);    

    // free
    cudaFree(bin_data);
    cudaFree(hist_data);
    free(bin);
    free(hist);
}