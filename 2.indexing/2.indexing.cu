#include <cuda.h>
#include <stdio.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>


__global__ void sum(float* x) {

    // block 和 thread 都是 一维
    int block_tid = blockIdx.x; // block index
    int global_tid = block_tid * blockDim.x + threadIdx.x; // global thread index
    int local_tid = threadIdx.x; // local thread index

    printf("global_tid: %d, block_tid: %d, local_tid: %d\n", global_tid, block_tid, local_tid);

    x[global_tid] += 1; // 因为 block = 1, 所以 global_tid = local_tid
}

int main() {

    int N = 32;
    int nbytes = N * sizeof(float);
    float* dx = nullptr;
    float* hx = nullptr;

    // Allocate memory on the device
    cudaMalloc((float** ) &dx, nbytes);

    // Allocate memory on the host
    hx = (float* )malloc(nbytes);

    // Initialize the host memory
    printf("hx original:\n");
    for (int i = 0; i < N; ++i) {
        
        hx[i] = i;
        printf("%f ", hx[i]);
    }
    printf("\n");

    // host -> device
    cudaMemcpy(dx, hx, nbytes, cudaMemcpyHostToDevice);

    // 处理数据
    sum<<<1, N>>>(dx); // 1 个 gird 有 1 个block, 每个 block 有 N 个线程 -> thread = 1 * N

    // 同步等待所有 thread 结束
    cudaDeviceSynchronize();

    // device -> host
    cudaMemcpy(hx, dx, nbytes, cudaMemcpyDeviceToHost);

    printf("hx after:\n");

    for (int i = 0; i < N; ++i)
        printf("%f ", hx[i]);

    printf("\n");

    // free
    cudaFree(dx);
    free(hx);


    return 0;
}