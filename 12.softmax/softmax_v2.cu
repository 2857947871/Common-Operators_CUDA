// 优化手段:
//  优化共享内存的使用
//  向量化的读写
// v0: 0.291008 ms
// v1: 0.287360 ms
// v2: 0.115872 ms

# include <cuda.h>
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
      if (abs(out[i] - groudtruth[i]) > 1e-5) {
          printf("i: %d\n", i);
          return i;
      }
    }

    return -1;
}

// 策略:
// 1000 * 1024
//  1000个block, 每个block有1024个thread
//  每个block也互不干扰(因为是不同的batch) -> 每个block都有自己的max_val和sum -> shared memory(每个block公用一个)
//  -> 选择每个block的第0个thread来计算当前blokc的max_val(shared memory)和sum(shared memory) -> 最后求解
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

    // 计算每个block(每个smem)的max_val
    __shared__ T max_val;
    max_val = 0;
    if (tid == 0) {
        for (int i = 0; i < classSize; ++i) {
            max_val = max(max_val, smem[i]);
        }
    }
    __syncthreads();

    // 计算每个block(每个smem)的sum
    __shared__ T sum;
    sum = 0;
    if (tid == 0) {
        for (int i = 0; i < classSize; ++i) {
            sum += exp(smem[i] - max_val);
        }
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
    softmaxCPU(x_host, groundtruth, 1000, 1024);
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
    printf("Softmax_GPU: %f ms\n", ms);

    // free
    free(x_host);
    free(y_host);
    free(groundtruth);
    cudaFree(x_device);
    cudaFree(y_device);

    return 0;
}