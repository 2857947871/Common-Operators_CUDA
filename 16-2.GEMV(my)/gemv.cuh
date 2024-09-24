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


namespace MatVec
{
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
} // namespace MatVec

namespace VecMat
{
template <int M, typename T>
struct get_thread_per_mat_row
{
	// 每一行, 需要多少threads
	// M / 4 or 8
	// float: 一次读4个   half: 一次读8个
	static const int value = M * sizeof(T) / 16;
};

// 一个block处理一个[1, M], 循环N次处理[N, M]
// THREADS_PER_VALUE: 每一行需要多少threads
template<int THREADS_PER_BLOCK, int THREADS_PER_VALUE, int VEC_SIZE>
__global__ void gemv(float* matrix, float* vector, float* res, int N, int M) {

	int tid = threadIdx.x;
	
	// eg: 
	// [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
	//  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
	//  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
	// 	THREADS_PER_VALUE = 3   VEC_SIZE = 4   M = 12
	//	tid = 5   mat_o: 1   mat_i: 8 -> [1, 8]
	//	tid = 6   mat_o: 0   mat_i: 0
	//	tid = 4   mat_o: 1   mat_i: 4
	int mat_o = tid / THREADS_PER_VALUE;			// 当前线程处理的行号
	int mat_i = tid % THREADS_PER_VALUE * VEC_SIZE;	// 当前线程处理的列号

	// ROW_PER_ITER: 每个iter处理的行数: 
	//	THREADS_PER_BLOCK(一个block中分配的threads总数) / THREADS_PER_VALUE(一行需要多少thread)
	// 假设每个iter处理x行
	constexpr int ROW_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_VALUE;

	// eg: [1, 2, 3] * [[1, 3, 5]
	//			        [2, 4, 6]
	//					[3, 6, 9]]
	// [1, 3, 5] + [2, 4, 6] + [9, 18, 27]
	float4 out; // 每个线程独有
	for (int i = mat_o; i < N; i += ROW_PER_ITER) {
		float4 mat = *reinterpret_cast<float4*>(&matrix[i * M + mat_i]);
		float logits = vector[i];

		out.x += logits * mat.x;
		out.y += logits * mat.y;
		out.z += logits * mat.z;
		out.w += logits * mat.w;
	}

	// intra block 累加(规约)
	// 将[1, 3, 5] + [2, 4, 6] + [9, 18, 27]加在一起
	// ROWS_PER_BLOCK: 每个block处理多少行
	// mindpoint: 规约的中点
	__shared__ float out_smem[512];
	for (int ROWS_PER_BLOCK = ROW_PER_ITER; ROWS_PER_BLOCK >= 2; ROWS_PER_BLOCK /= 2) {
		int mindpoint = ROWS_PER_BLOCK / 2;
		
		// 开始赋值, 下半行的out赋给smem
		if (mat_o >= mindpoint && mat_o < ROWS_PER_BLOCK) {

			// 赋值
			*reinterpret_cast<float4*>(&out_smem[(mat_o - mindpoint) * M + mat_i]) = out;
		}
		// 赋值结束, 开始规约
		__syncthreads();

		// 上半行与smem相加求和
		if (mat_o < mindpoint) {
			out.x = (*reinterpret_cast<float4*>(&out_smem[mat_o * M + mat_i])).x + out.x;
			out.y = (*reinterpret_cast<float4*>(&out_smem[mat_o * M + mat_i])).y + out.y;
			out.z = (*reinterpret_cast<float4*>(&out_smem[mat_o * M + mat_i])).z + out.z;
			out.w = (*reinterpret_cast<float4*>(&out_smem[mat_o * M + mat_i])).w + out.w;
		}
		__syncthreads();

		// 所有结果相加到第0行
		if (mat_o == 0) {
			*reinterpret_cast<float4*>(&res[mat_i]) = out;
		}
	}
}

template<int THREADS_PER_BLOCK, int THREADS_PER_VALUE, int VEC_SIZE>
struct DispatchLauncher {
    template<typename T>
    static void launcher(T* d_mat, T* d_vec, T* d_dst, int M, int N){
        
        // 仅分配一个block
        dim3 Grid(1);
        dim3 Block(THREADS_PER_BLOCK);
        float milliseconds = 0;
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        printf("calling\n");
        
        // 启动cuda kernel
        gemv<THREADS_PER_BLOCK, THREADS_PER_VALUE, VEC_SIZE><<<Grid, Block>>>(d_mat, d_vec, d_dst, N, M);
        cudaError_t result = cudaGetLastError();
        if (result) {
            throw std::runtime_error(std::string("[ERROR] CUDA runtime error: ") +  (_cudaGetErrorEnum(result)) + " " + __FILE__ + ":" + std::to_string(__LINE__) + " \n");
        }
        printf("called\n");
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("gemv latency = %f ms\n", milliseconds);
    }
};
}; // namespace VecMat
