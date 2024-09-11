# include <cuda.h>
# include <iostream>
# include <bits/stdc++.h>
# include "cuda_runtime.h"

// 实现FP32的biasadd mask scale and add的算子融合
//  (x + bias) * mask * scale + add
template <typename T>
struct MaskScaleAndElemwiseAddFunctor {

    // 有参构造函数
    MaskScaleAndElemwiseAddFunctor(const uint8_t* mask, const T* add_val, float scale) {
        _mask = mask;
        _add_val = add_val;
        _scale = scale;
    }

    // 结构体内运算符重载, 
    __device__ T operator () (T x, int i) const {
        return x * static_cast<T>(static_cast<bool>(_mask[i]) * _scale) + _add_val[i];
    }

    const uint8_t* _mask;
    const T* _add_val;
    float _scale;
};

// 朴素写法: costs 0.002484 s 
// 并行性: 每个x互不干扰, 可以并行
template <int biasSize, typename FUNCTOR, typename T>
__global__ void FusedBaisAdd(FUNCTOR func, T* x, T* y, T* bias, 
                                const int n, const int bias_size) {
    int git = blockIdx.x * blockDim.x + threadIdx.x;

    // 最多分配blockSize * gridSize个线程
    // 如果n大于最大线程数, 则每个线程处理多个元素
    // eg: n = 1050   blockSize = 512   gridSize = 2
    // gid=1 -> i=1 -> i+1024=1025 -> gid=1的线程继续处理n=1025
    for (int i = git; i < n; i += blockDim.x * gridDim.x) {
        T tmp = x[i] + bias[i % bias_size];
        y[i] = func(tmp, i);
    }
}

// 向量化: costs 0.003008 s
template <int biasSize, typename FUNCTOR, typename T>
__global__ void FusedBaisAddVecSmem(FUNCTOR func, T* x, T* y, T* bias, 
                                        const int n, const int bias_size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // 使用shared_memory
    //  每个block共享smem
    __shared__ T smem[biasSize];
    if (tid < bias_size) {
        smem[tid] = bias[tid];
    }
    __syncthreads();

    // 向量化读取, 每个thread读取4个数
    for (int i = gid; i < n / 4; i += blockDim.x * gridDim.x) {
        float4 a = reinterpret_cast<float4*>(x)[i];
        float4 b;

        b.x = func(a.x + smem[(i * 4) % bias_size], i * 4);
        b.y = func(a.y + smem[(i * 4 + 1) % bias_size], i * 4 + 1);
        b.z = func(a.z + smem[(i * 4 + 2) % bias_size], i * 4 + 2);
        b.w = func(a.w + smem[(i * 4 + 3) % bias_size], i * 4 + 3);

        reinterpret_cast<float4*>(y)[i] = b;
    }
}

bool CheckRight(float * y, float * groudTruth, const int n)
{
    for (int i = 0; i < n; ++i)
    {
        if (y[i] != groudTruth[i])
        {
            printf("y[%d] : %f \n", i, y[i]);
            printf("groundTruth[%d] : %f\n", i, groudTruth[i]);
            return false;
        }
    }
    return true;
}

int main() {

    // 参数初始化
    float scale             = 0.5;
    constexpr int n         = 100000;
    constexpr int bias_size = 10;
    uint8_t* mask_tensor    = new uint8_t[n];
    float* add_value        = new float[n];
    for (int i = 0; i < n; ++i) {

        mask_tensor[i] = (uint8_t)(i);
        add_value[i]   = (float)(i);
    }

    float* x_host    = (float* )malloc(sizeof(float) * n);
    float* y_host    = (float* )malloc(sizeof(float) * n);
    float* bias_host = (float* )malloc(sizeof(float) * bias_size);
    for (int i = 0; i < n; ++i)
    {
        x_host[i] = (float)(i);
        y_host[i] = 0.0f;
    }
    for (int i = 0; i < bias_size; ++i) {
        bias_host[i] = i;
    }

    // groundTruth
    //  (x + bias) * mask * scale + add
    //  scale: ??? 不会影响bool -> 不会影响结果
    float* groundTruth = (float* )malloc(sizeof(float) * n);
    for (int i = 0; i< n; ++i) {
        groundTruth[i] = (x_host[i] + bias_host[i % bias_size]) *
            static_cast<float>(static_cast<bool>(mask_tensor[i]) * scale) + add_value[i];
    }

    uint8_t* mask_device;
    float* add_device;
    cudaMalloc((void **)&mask_device, sizeof(uint8_t) * n);
    cudaMalloc((void **)&add_device, sizeof(float) * n);
    cudaMemcpy(mask_device, mask_tensor, sizeof(uint8_t) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(add_device, add_value, sizeof(float) * n, cudaMemcpyHostToDevice);

    float* x_device;
    float* y_device;
    float* bias_device;
    cudaMalloc((void** )&x_device, sizeof(float) * n);
    cudaMalloc((void** )&y_device, sizeof(float) * n);
    cudaMalloc((void** )&bias_device, sizeof(float) * bias_size);
    cudaMemcpy(x_device, x_host, sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(y_device, y_host, sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(bias_device, bias_host, sizeof(float) * bias_size, cudaMemcpyHostToDevice);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int blockSize = 512;
    int gridSize = std::min((n + blockSize - 1) / blockSize, deviceProp.maxGridSize[0]);
    dim3 Block(blockSize);
    dim3 Grid(gridSize);

    // operation
    MaskScaleAndElemwiseAddFunctor<float> functor(mask_device, add_device, scale);

    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 朴素版
    cudaEventRecord(start);
    for (int i = 0; i < 1000; ++i)
        FusedBaisAdd<bias_size><<<Grid, Block>>>(functor, x_device, y_device, bias_device, n, bias_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(y_host, y_device, sizeof(float) * n, cudaMemcpyDeviceToHost);
    
    bool isRight = CheckRight(y_host, groundTruth, n);
    if (isRight) {
        printf("朴素版   costs %f s \n", milliseconds/1000);
    } else {
        printf("结果错误\n"); 
    }

    // 向量化版
    milliseconds = 0;
    cudaEventRecord(start);
    for (int i = 0; i < 1000; ++i)
        FusedBaisAddVecSmem<bias_size><<<Grid, Block>>>(functor, x_device, y_device, bias_device, n, bias_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(y_host, y_device, sizeof(float) * n, cudaMemcpyDeviceToHost);
    
    isRight = CheckRight(y_host, groundTruth, n);
    if (isRight) {
        printf("向量化版 costs %f s \n", milliseconds/1000);
    } else {
        printf("结果错误\n"); 
    }

    // free
    free(x_host);
    free(y_host);
    free(bias_host);
    free(groundTruth);
    cudaFree(mask_device);
    cudaFree(add_device);
    cudaFree(x_device);
    cudaFree(y_device);
    cudaFree(bias_device);
    delete[] mask_tensor;
    mask_tensor = nullptr;
    delete[] add_value;
    add_value = nullptr;
    
    
    return 0;


}