#include <stdio.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "cuda_runtime.h"
#include <device_launch_parameters.h>

# define LOOP_TIMES 1000

__global__ void FP32FLOPS(int* start, int* stop, float* x, float* y, float* result) {

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float d1  = x[gid];
    float d2  = y[gid];
    float res = 0.0f;
    int start_time = 0;

    // asm CUDA内联汇编   volatile 防止编译器优化
    // 将当前GPU的时钟计数器存储到寄存器%0中
    // %%clock: GPU内部的时候总计数器寄存器, 记录GPU执行的时钟周期数
    // "=r"(start_time): 输出结果, 寄存器的值存储到start_time中
    asm volatile("mov.u32 %0, %%clock;" : "=r"(start_time));
    for (int i = 0; i < LOOP_TIMES; ++i) {
        res = d1 * d2 + res;
        res = d1 * d2 + res;
        res = d1 * d2 + res;
        res = d1 * d2 + res;
    }

    // 同一block中所有线程同步
    asm volatile("bar.sync 0;");
    
    int stop_time = 0;
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(stop_time) :: "memory");
    start[gid]  = start_time;
    stop[gid]   = stop_time;
    result[gid] = res;
}


int main() {

    int N = 1024;
    float* x_host = (float* )malloc(N * sizeof(float));
    float* y_host = (float* )malloc(N * sizeof(float));
    for(int i = 0; i < 1024; i++) {
        x_host[i] = static_cast<float>(i);
        y_host[i] = static_cast<float>(i);
    }
    
    float* x_device;
    float* y_device;
    cudaMalloc((void**)&x_device, N * sizeof(float));
    cudaMalloc((void**)&y_device, N * sizeof(float));
    cudaMemcpy(x_device, x_host, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_device, y_host, N * sizeof(float), cudaMemcpyHostToDevice);

    float* result_device;
    int* startClock = (int* )malloc(sizeof(int));
    int* stopClock  = (int* )malloc(sizeof(int));
    int* startClock_device;
    int* stopClock_device;
    cudaMalloc((void**)&result_device,  N * sizeof(float));
    cudaMalloc((void **)&startClock_device, N * sizeof(int)); 
    cudaMalloc((void **)&stopClock_device, N * sizeof(int)); 
    FP32FLOPS<<<1, N>>>(startClock_device, stopClock_device, x_device, y_device, result_device);
    cudaMemcpy(startClock, startClock_device, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(stopClock, stopClock_device, sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    float FLOPS = (LOOP_TIMES * 4 * 2 * 1024) / (static_cast<float>(stopClock[0] - startClock[0]));
    printf( "GPU Max Clock rate: %0.2f GHz\n" , props.clockRate * 1e-6f);
    printf("SM counts is %d\n", props.multiProcessorCount);
    printf("actual NVIDIA 3080 GPU peak FLOPS is %f (TFLOPS) \n", FLOPS * props.clockRate * 1e-9 * props.multiProcessorCount);

    // free
    free(x_host);
    free(y_host);
    free(startClock);
    free(stopClock);
    cudaFree(x_device);
    cudaFree(y_device);
    cudaFree(result_device);
    cudaFree(startClock_device);
    cudaFree(stopClock_device);
}