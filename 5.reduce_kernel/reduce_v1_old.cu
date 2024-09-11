#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"

// v1新版本: 用位运算替换除余操作
// latency: 2.825ms
// blockSize作为模板参数的效果主要用于静态shared memory的申请需要传入编译期常量指定大小（L120)
template<int blockSize>
__global__ void reduce_v1(float *d_in,float *d_out) {
     __shared__ float smem[blockSize];

     unsigned int tid = threadIdx.x;

     unsigned int gtid = blockIdx.x * blockSize + threadIdx.x;

     smem[tid] = d_in[gtid];

     __syncthreads();

     for(unsigned int s = 1; s < blockDim.x; s *= 2) {
         int index = 2 * s * tid;
         if (index < blockDim.x) {
             smem[index] += smem[index + s];
         }
         __syncthreads();
     }

     if (tid == 0) {
        d_out[blockIdx.x] = smem[0];
    }
}
bool CheckResult(float *out, float groudtruth, int n){
    float res = 0;
    for (int i = 0; i < n; i++){
        res += out[i];
    }
    if (res != groudtruth) {

        printf("res is: %f \n", res);
        return false;
    }
    return true;
}

int main(){
    float milliseconds = 0;
    const int N = 25600000;
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    const int blockSize = 256;
    int GridSize = std::min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]);
    //int GridSize = 100000;
    float *a = (float *)malloc(N * sizeof(float));
    float *d_a;
    cudaMalloc((void **)&d_a, N * sizeof(float));

    float *out = (float*)malloc((GridSize) * sizeof(float));
    float *d_out;
    cudaMalloc((void **)&d_out, (GridSize) * sizeof(float));

    for(int i = 0; i < N; i++){
        a[i] = 1.0f;
    }

    float groudtruth = N * 1.0f;

    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(GridSize);
    dim3 Block(blockSize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_v1<blockSize><<<Grid,Block>>>(d_a, d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(out, d_out, GridSize * sizeof(float), cudaMemcpyDeviceToHost);
    printf("allcated %d blocks, data counts are %d", GridSize, N);
    bool is_right = CheckResult(out, groudtruth, GridSize);
    if(is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");

        printf("groudtruth is: %f \n", groudtruth);
    }
    printf("reduce_v1 latency = %f ms\n", milliseconds);

    cudaFree(d_a);
    cudaFree(d_out);
    free(a);
    free(out);
}

