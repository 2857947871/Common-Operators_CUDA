// 优化手段:
//  优化共享内存的使用
//  向量化的读写
//  warp层面求max_val和sum
// CPU: 16.8773 ms
// v0: 0.291008 ms
// v1: 0.287360 ms
// v2: 0.115872 ms
// v3: 0.065824 ms

# include <cuda.h>
# include <chrono>
# include <iostream>
# include <bits/stdc++.h>
# include "cuda_runtime.h"
# include <device_launch_parameters.h>


// softmax公式
// e^(xi - max(xi)) / sigma(e^(xi - max(xi)))
void softmaxCPU(float* input, float* result, int rows, int cols){
    for (int j = 0; j < rows; j++) {
        float total = 0;
        float MAX = 0;

        // 找到最大值
        for(int i = 0; i < cols; i++) {
            MAX = max(input[j * cols + i], MAX);
        }

        // 计算total
        for(int i = 0; i < cols; i++) {
            total += exp(input[j * cols + i] - MAX);
        }

        // 计算softmax
        for(int i = 0; i < cols; i++) {
            result[j * cols + i] = exp(input[j * cols + i] - MAX) / total;
        }
    }
}

int CheckResult(float *out, float* groudtruth, int N){
    for (int i = 0; i < N; i++){
      if (abs(out[i] - groudtruth[i]) > 1e-6) {
          printf("i: %d\n", i);
          return i;
      }
    }

    return -1;
}

// warp层面求max_val
template<typename T>
__device__ T warp_reduce_max(T val) {

    // val: 当前thread的值(smem[tid])
    // 完成归约操作的线程掩码(32个thread)
    const unsigned int mask = 0xFFFFFFFF;

    // 依次归约
    // offset = 16
    //  前16个thread和后16个thread比较, 取最大值
    // offset = 8
    //  前8个thread和后8个thread比较, 取最大值
    //                  .
    //                  .
    //                  .
    // offset = 1 -> 当前warp的最大值
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        
        // __shfl_down_sync: 同一warp中传递数据(warp级别的同步操作)
        //  mask: 线程掩码, 表示参与的线程
        //  val: 当前线程的值
        //  offset: 偏移量
        val = max(val, __shfl_down_sync(mask, val, offset));
    }

    return val;
}

// warp层面求sum
template<typename T>
__device__ T warp_reduce_sum(T val, T max) {

    // val: 当前thread的值(smem[tid])
    // 完成归约操作的线程掩码(32个thread)
    const unsigned int mask = 0xFFFFFFFF;

    // 依次归约
    // offset = 16
    //  前16个thread和后16个thread相加
    // offset = 8
    //  前8个thread和后8个thread相加
    //                  .
    //                  .
    //                  .
    // offset = 1 -> 当前warp的和
    val = exp(val - max);
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        
        // __shfl_down_sync: 同一warp中传递数据(warp级别的同步操作)
        //  mask: 线程掩码, 表示参与的线程
        //  val: 当前线程的值
        //  offset: 偏移量
        val += __shfl_down_sync(mask, val, offset);
    }

    return val;
}

// 策略:
// 1000 * 1024
//  1000个block, 每个block有1024个thread
//  每个block也互不干扰(因为是不同的batch) -> 每个block都有自己的max_val和sum -> shared memory(每个block公用一个)
//  -> 选择每个block的第0个thread来计算当前blokc的max_val(shared memory)和sum(shared memory) -> 最后求解
// warp层面求max_val和sum
// softmax:
//  y = e^(x - max(x)) / sigma(e^(x - max(x)))
template<int classSize,  typename T>
__global__ void softmax_gpu_v2(T* x, T* y, int N) {

    int tid = threadIdx.x; // 0 ~ 255
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid * 4 >= N) {
        return;
    }

    // 向量化加载数据到shared memory
    __shared__ T smem[classSize]; // 一个block有1024个thread

    smem[tid * 4 + 0] = x[gid * 4 + 0];
    smem[tid * 4 + 1] = x[gid * 4 + 1];
    smem[tid * 4 + 2] = x[gid * 4 + 2];
    smem[tid * 4 + 3] = x[gid * 4 + 3];
    __syncthreads();

    // 计算每个warp的max_val
    //  注: 向量化计算, 每个warp的4个值
    T warp_max_val = -FLT_MAX; // 普通变量, 每个线程私有, 不共享
    warp_max_val = warp_reduce_max(smem[tid * 4 + 0]);
    warp_max_val = max(warp_max_val, warp_reduce_max(smem[tid * 4 + 1]));
    warp_max_val = max(warp_max_val, warp_reduce_max(smem[tid * 4 + 2]));
    warp_max_val = max(warp_max_val, warp_reduce_max(smem[tid * 4 + 3]));

    // 计算每个block的max_val
    __shared__ T max_val;
    max_val = -FLT_MAX;
    if (tid % warpSize == 0) {
        max_val = max(max_val, warp_max_val);
    }
    __syncthreads();

    // 计算每个block(每个smem)的sum
    //  注: 向量化计算, 每个warp的4个值
    T warp_sum = 0;
    warp_sum = warp_reduce_sum(smem[tid * 4 + 0], max_val);
    warp_sum += warp_reduce_sum(smem[tid * 4 + 1], max_val);
    warp_sum += warp_reduce_sum(smem[tid * 4 + 2], max_val);
    warp_sum += warp_reduce_sum(smem[tid * 4 + 3], max_val);

    // 计算每个block的sum
    __shared__ T sum;
    if (tid == 0) sum = 0;
    __syncthreads();

    if (tid % warpSize == 0) {

        // 使用原子操作, 累加每个warp的sum(线程安全)
        atomicAdd(&sum, warp_sum);
    }
    __syncthreads();

    // 向量化计算softmax
    y[gid * 4 + 0] = exp(smem[tid * 4 + 0] - max_val) / sum;
    y[gid * 4 + 1] = exp(smem[tid * 4 + 1] - max_val) / sum;
    y[gid * 4 + 2] = exp(smem[tid * 4 + 2] - max_val) / sum;
    y[gid * 4 + 3] = exp(smem[tid * 4 + 3] - max_val) / sum;

    __syncthreads();
}


int main() {

    // 初始化
    int N = 1000 * 1024; // 1000行(batch_xize) 1024列(种类)
    float* x_host = (float* )malloc(N * sizeof(float));
    float* y_host = (float* )malloc(N * sizeof(float));
    float* groundtruth = (float* )malloc(N * sizeof(float));

    for (int i = 0; i < N; ++i) {
        x_host[i] = i % 10;
        y_host[i] = 0;
        groundtruth[i] = 0;
    }

    float* x_device;
    float* y_device;
    cudaMalloc((void** )&x_device, N * sizeof(float));
    cudaMalloc((void** )&y_device, N * sizeof(float));
    cudaMemcpy(x_device, x_host, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_device, y_host, N * sizeof(float), cudaMemcpyHostToDevice);

    constexpr int classSize = 1024;                  // 1024个thread(种类)
    int gridSize = (N + classSize - 1) / classSize;  // 1000个block(batch_size)

    float ms = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    softmax_gpu_v2<classSize><<<gridSize, classSize / 4>>>(x_device, y_device, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(y_host, y_device, N * sizeof(float), cudaMemcpyDeviceToHost);

    // 验证
    auto start_cpu = std::chrono::high_resolution_clock::now();
    softmaxCPU(x_host, groundtruth, 1000, 1024);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    int is_right = CheckResult(y_host, groundtruth, N);
    if(is_right == -1) {
        printf("the ans is right\n");
    } else {
        for(int i = is_right; i < is_right + 10; i++){ printf("%lf ", y_host[i]); }
        printf("\n");

        for(int i = is_right; i < is_right + 10; i++){ printf("%lf ", groundtruth[i]); }
        printf("\n");

        printf("the ans is wrong\n");
    }

    std::chrono::duration<double, std::milli> duration = end_cpu - start_cpu;
    printf("Softmax_CPU: %f ms\n", duration);
    printf("Softmax_GPU: %f ms\n", ms);

    // free
    free(x_host);
    free(y_host);
    free(groundtruth);
    cudaFree(x_device);
    cudaFree(y_device);

    return 0;
}