// x + y = z   x: 100个FP32
// 2次读(读取x和y) + 1次写(写入z) -> 3次显存操作
// 100 * 3 * 4 = 1200 bytes
// 测量时间, 用时t秒
// 峰值性能 显存带宽 = 1200 / t
// CUDA中一次最多读写128bit(4字节), 向量化读写, 一次读写的单位是4字节

// 注: 测量时, 关闭缓存, 保证每次读写都是从显存读取, 保证测量的是显存带宽

# include <iostream>
# include <stdio.h>
# include <cuda.h>
# include "cuda_runtime.h"
# include <device_launch_parameters.h>

#define ARRAY_SIZE	  100000000 // 400M   Array size has to exceed L2 size to avoid L2 cache residence
#define MEMORY_OFFSET 10000000	// “关闭”缓存
#define BENCH_ITER    10
#define THREADS_NUM   256


__global__ void mem_bw(float* A, float* B, float* C) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // 向量化读写, 每次读写4字节(128bit)
    for (int i = gid; i < MEMORY_OFFSET / 4; i += blockDim.x * gridDim.x) {
        
        // A是flaot类型的指针 -> 强制类型转换 -> A是float4类型的指针, 每次读取4个float
        // 指针[i] -> 相当于解引用 
        // 向量化的读取
        float4 a1 = reinterpret_cast<float4*>(A)[i];
        float4 b1 = reinterpret_cast<float4*>(B)[i];
        float4 c1;

        // 不直接相加 -> cuda支持向量化的读和写, 不支持向量化的加法
        c1.x = a1.x + b1.x;
        c1.y = a1.y + b1.y;
        c1.z = a1.z + b1.z;
        c1.w = a1.w + b1.w;

        // 向量化的写入
        reinterpret_cast<float4*>(C)[i] = c1;
    }

}

void vec_add_cpu(float *x, float *y, float *z) {
    for (int i = 0; i < 20; i++) z[i] = y[i] + x[i];
}

int main() {

    float ms = 0;

	float* A_host = (float* )malloc(ARRAY_SIZE*sizeof(float));
	float* B_host = (float* )malloc(ARRAY_SIZE*sizeof(float));
	float* C_host = (float* )malloc(ARRAY_SIZE*sizeof(float));
	for (uint32_t i=0; i<ARRAY_SIZE; i++) {
		A_host[i] = (float)i;
		B_host[i] = (float)i;
	}

    float* A_dev;
    float* B_dev;
    float* C_dev;
    cudaMalloc((void**)&A_dev, ARRAY_SIZE*sizeof(float));
    cudaMalloc((void**)&B_dev, ARRAY_SIZE*sizeof(float));
    cudaMalloc((void**)&C_dev, ARRAY_SIZE*sizeof(float));
    cudaMemcpy(A_dev, A_host, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_dev, B_host, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);

    int BlockNums = MEMORY_OFFSET / THREADS_NUM;

    // “关闭”L2 cache
	// L2 cache: 40M, 因此warm up时, 塞入40M, 将L2 cache占满 -> “关闭”L2 cache
	printf("warm up start\n");
	mem_bw<<<BlockNums / 4, THREADS_NUM>>>(A_dev, B_dev, C_dev);
    printf("warm up end\n");

    // 计时
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	for (int i = BENCH_ITER - 1; i >= 0; --i) {
		mem_bw<<<BlockNums / 4, THREADS_NUM>>>(A_dev + i * MEMORY_OFFSET, B_dev + i * MEMORY_OFFSET, C_dev + i * MEMORY_OFFSET);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);

	cudaMemcpy(C_host, C_dev, ARRAY_SIZE*sizeof(float), cudaMemcpyDeviceToHost);

    // check
	float* C_cpu_res = (float *) malloc(20*sizeof(float));
	vec_add_cpu(A_host, B_host, C_cpu_res);
	for (int i = 0; i < 20; ++i) {
		/* 测量显存带宽时, 修改C_cpu_res[i]为0 */
		if (fabs(C_cpu_res[i] - C_host[i]) > 1e-6) {
			printf("Result verification failed at element index %d!\n", i);
		}
	}
	printf("Result right\n");

    // 计算显存带宽
	unsigned N = ARRAY_SIZE * 4; // 4 bytes per float
    
	printf("Mem BW= %f (GB/sec)\n", 3 * (float)N / ms / 1e6);

    // free
  	cudaFree(A_dev);
  	cudaFree(B_dev);
  	cudaFree(C_dev);

  	free(A_host);
  	free(B_host);
  	free(C_host);
  	free(C_cpu_res);


    return 0;
}