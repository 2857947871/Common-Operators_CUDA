# include <string>
# include <stdio.h>
# include <iostream>
# include <stdexcept>
# include <cuda.h>
# include "cuda_fp16.h"
# include "cuda_runtime.h"
# include "cuda_runtime_api.h"
# include "device_functions.h"

static const char* _cudaGetErrorEnum(cudaError_t error) {
    return cudaGetErrorString(error);
}
#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)


// 为了下面的模板特化作准备
template <typename T>
struct Vec;

template <>
struct Vec<float>
{
    static constexpr int size = 4;
};

template <>
struct Vec<half>
{
    static constexpr int size = 8;
};

template <typename T>
struct SumOp;

template <>
struct SumOp<float>
{
    __device__ __forceinline__ float operator() (const float& a, const float& b) const
    {
        return a + b;
    }
};

template <>
struct SumOp<half>
{
    __device__ __forceinline__ half operator() (const half& a, const half& b) const 
    {
        return __hadd(a, b);
    }
};

template <template <typename> class ReductionOp, typename T>
__device__ __forceinline__ T warpReduce(T val) {
	for (int mask = 16; mask > 0; mask >>= 1) {
		val = ReductionOp<T>() (val, __shfl_xor_sync(0xffffffff, val, mask));
	}
	return val;
}

template <template <typename> class ReductionOp, typename T>
__device__ __forceinline__ T blockReduce(T val) {

	int tid = threadIdx.x;
	int warp_id = tid / 32;
	int lane_id = tid % 32;

	// 向上进一, 防止最后几个元素分配thread小于32, warp_nums 为0
	int warp_nums = (blockDim.x + 31) / 32;
	static __shared__ float warpres[64];

	// block内每个warp reduce的结果, 该结果保存在每个warp的0号thread
	// warp: 规约求和
	val = warpReduce<ReductionOp, T>(val);
	if (lane_id == 0) {
		warpres[warp_id] = val;
	}
	__syncthreads();

	// 最后指定 warp_nums 个thread将所有中间结果相加
	float warp_val = tid < warp_nums ? warpres[tid] : 0;
	return warpReduce<ReductionOp, T>(warp_val);
}


template <int VECS_PER_THREAD, int VEC_SIZE>
__global__ void gemv(float* matrix, float* vector, float* res, const int cols)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x; // 行号 -> 一个blokc掌管一行

	float thread_local_sum = 0;

	// 并不是用for循环取match一行中的每一个元素, 
	// 而是固定每个thread向量化读取的次数
	for (int i = 0; i < VECS_PER_THREAD; ++i) {
		
		// 类型转换: float -> float4
		// matrix[]: 解引用 -> 取地址
		float4* mat4 = reinterpret_cast<float4*>(&matrix[bid * cols + tid * VEC_SIZE]);
        float4* vec4 = reinterpret_cast<float4*>(&vector[tid * VEC_SIZE]);
		thread_local_sum += mat4[i].x * vec4[i].x;
		thread_local_sum += mat4[i].y * vec4[i].y;
		thread_local_sum += mat4[i].z * vec4[i].z;
		thread_local_sum += mat4[i].w * vec4[i].w;
	}

	float reduce_res = blockReduce<SumOp, float>(thread_local_sum);
	if (tid == 0) {
		res[bid] = reduce_res;
	}
	__syncthreads();
}

template <int VECS_PER_THREAD, int VEC_SIZE, int THREAD_NUMS>
struct DispatchLauncher
{
	template <typename T>
	static void launcher(T* d_mat, T* d_vec, T* d_dst, int M, int N) {
		
		dim3 Grid(M);
		dim3 Block(THREAD_NUMS);
		
		float time = 0.0f;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
		gemv<VECS_PER_THREAD, VEC_SIZE><<<Grid, Block>>>(d_mat, d_vec, d_dst, N);
        cudaError_t error = cudaGetLastError(); /// 当前行所在位置的上一个kernel调用是否出错
        if (error) {
            throw std::runtime_error(std::string("[ERROR] CUDA runtime error: ") +
                (_cudaGetErrorEnum(error)) + " " + __FILE__ + ":" + std::to_string(__LINE__) + " \n");
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        printf("gemv latency: %f ms\n", time);
    }
};







