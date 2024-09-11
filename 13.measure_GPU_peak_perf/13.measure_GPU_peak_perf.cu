# include <stdio.h>
# include <cuda.h>
# include <cuda_fp16.h>
# include "cuda_runtime.h"

# define LOOP_TIMES 1000


__global__ void FP32FLOPS(int* start, int* stop, float* x, float* y, float* result) {

    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    float d1 = x[gid];
    float d2 = y[gid];
    float res = 0;
    int start_time = 0;

    // 仅测量运算时间, 忽略访存时间 -> 不能使用cuda自带的计时, 使用asm指令
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(start_time) :: "memory");

    for(int i = 0; i < LOOP_TIMES; ++i) {
        
        // 为什么连续四次有依赖的FMA?
        //  减少判断指令对时间的影响(i < LOOP_TIMES; ++i)
        res = d1 * d2 + res;
        res = d1 * d2 + res;
        res = d1 * d2 + res;
        res = d1 * d2 + res;
    }

    // sync all threads而不是仅仅同步该 block 内的 thread
    // 仅有 asm volatile 能做到
    asm volatile("bar.sync 0;");

    int stop_time = 0;
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(stop_time) :: "memory");
    start[gid]  = start_time;
    stop[gid]   = stop_time;
    result[gid] = res;
}

int main() {

    // 初始化
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    int ThreadsPerSM = props.maxThreadsPerMultiProcessor;

    int N = 1024;
    float* x_host = (float* )malloc(N * sizeof(float));
    float* y_host = (float* )malloc(N * sizeof(float));
    float* x_device;
    float* y_device;
    float* res_device;
    int* startClock_host = (int*)malloc(N * sizeof(int));
    int* stopClock_host  = (int*)malloc(N * sizeof(int));
    int* startClock_device;
    int* stopClock_device;
    cudaMalloc((void**)&x_device, N * sizeof(float));
    cudaMalloc((void**)&y_device, N * sizeof(float));
    cudaMalloc((void**)&res_device, N * sizeof(float));
    cudaMalloc((void**)&startClock_device, N * sizeof(int));
    cudaMalloc((void**)&stopClock_device, N * sizeof(int));

    for (int i = 0; i < 1024; ++i) {
        x_host[i] = static_cast<float>(i);
        x_host[i] = static_cast<float>(i);
    }
    cudaMemcpy(x_device, x_host, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_device, y_host, N * sizeof(float), cudaMemcpyHostToDevice);

    // operation
    FP32FLOPS<<<1, 1024>>>(startClock_device, stopClock_device, x_device, y_device, res_device);
    cudaMemcpy(startClock_host, startClock_device, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(stopClock_host, stopClock_device, N * sizeof(int), cudaMemcpyDeviceToHost);

    // 计算量: LOOP_TIMES * n次res * FMA(1次add 1次mul) * 1024个thread / 时间 -> FP32/S
    float FLOPS = (LOOP_TIMES * 4 * 2 * 1024) / (static_cast<float>(stopClock_host[0] - startClock_host[0]));
    printf("GPU Max Clock rate: %0.2f GHz\n" , props.clockRate * 1e-6f);
    printf("SM counts is %d\n", props.multiProcessorCount);
    printf("actual GPU peak FLOPS is %f (TFLOPS) \n", FLOPS * props.clockRate * 1e-9 * props.multiProcessorCount);

    // free
    free(x_host);
    free(y_host);
    free(startClock_host);
    free(stopClock_host);
    cudaFree(x_device);
    cudaFree(y_device);
    cudaFree(res_device);
    cudaFree(startClock_device);
    cudaFree(stopClock_device);


    return 0;
}