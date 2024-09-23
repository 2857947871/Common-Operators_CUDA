# include <string>
# include <stdio.h>
# include <iostream>
# include <stdexcept>
# include <cuda.h>
# include <cuda_fp16.h>
# include "cuda_runtime.h"


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

template<typename T>
struct Vec;

template<>
struct Vec<float>
{ static constexpr int size = 4; };

template<>
struct Vec<half> { static constexpr int size = 8; };

template<typename T>
struct SumOp;

template<>
struct SumOp<float> {
  __device__ __forceinline__ float operator()(const float& a, const float& b) const { return a + b; }
};

template<>
struct SumOp<half> {
  __device__ __forceinline__ half operator()(const half& a, const half& b) const { return __hadd(a, b); }
};

template<template<typename> class ReductionOp, typename T>
__device__ __forceinline__ T warpReduce(T val){
    for(int mask = 16; mask > 0; mask >>= 1){
        val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

// 把block reduce拆分为多个warp reduce来计算
template<template<typename> class ReductionOp, typename T>
__device__ __forceinline__ T blockReduce(T val){
    
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    // 向上进1，以防分配的线程数量小于32导致warp nums为0
    int warp_nums = (blockDim.x + 31) / 32;
    static __shared__ float warpres[64];
    
    // block内每个warp reduce的结果，该结果保存在每个warp内的0号线程，所以L65用0号线程写入warp res
    val = warpReduce<ReductionOp, T>(val);
    if (lane_id == 0){
        warpres[warp_id] = val;
    }
    __syncthreads();

    // 最后把每个warp的结果再作一个reduce得到最终一个block的结果
    float warp_val = tid < warp_nums ? warpres[tid] : 0;
    return warpReduce<ReductionOp, T>(warp_val);
}


template<int VECS_PER_THREAD, int VEC_SIZE>
__global__ void gemv(float* matrix, float* vector, float* res, int cols) {

    int tid = threadIdx.x;
    int bid = blockIdx.x; // 行号, 每个block对应一行
    
    float thread_local_sum = 0;
    for (int i = 0; i < VECS_PER_THREAD; ++i) {
        
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

struct half8 {
    half2 x;
    half2 y;
    half2 z;
    half2 w;
};
template<int VECS_PER_THREAD, int VEC_SIZE>
__global__ void gemv(half* matrix, half* vector, half* res, int cols) {

    int tid = threadIdx.x;
    int bid = blockIdx.x; // 行号, 每个block对应一行
    
    half thread_local_sum = 0;
    for (int i = 0; i < VECS_PER_THREAD; ++i) {

        float4 mat4 = reinterpret_cast<float4*>(matrix)[bid * (cols / VEC_SIZE) + i * blockDim.x + tid]; // 4 * half2
        float4 vec4 = reinterpret_cast<float4*>(vector)[i * blockDim.x + tid];

        half2* vec_h1 = (half2*)&vec4.x;
        half2* vec_h2 = (half2*)&vec4.y;
        half2* vec_h3 = (half2*)&vec4.z;
        half2* vec_h4 = (half2*)&vec4.w;

        half2* mat_h1 = (half2*)&mat4.x;
        half2* mat_h2 = (half2*)&mat4.y;
        half2* mat_h3 = (half2*)&mat4.z;
        half2* mat_h4 = (half2*)&mat4.w;
           
        half2 res1 = __hmul2(*mat_h1, *vec_h1);
        half2 res2 = __hmul2(*mat_h2, *vec_h2);
        half2 res3 = __hmul2(*mat_h3, *vec_h3);
        half2 res4 = __hmul2(*mat_h4, *vec_h4); 
        half2 res = __hadd2(__hadd2(__hadd2(res1, res2), res3), res4);
        thread_local_sum = __hadd(res.x, res.y);
    }

    float reduce_res = blockReduce<SumOp, half>(thread_local_sum);
    if (tid == 0) {
        res[bid] = reduce_res;
    }
    __syncthreads();
}


template<int VECS_PER_THREAD, int VEC_SIZE, int THREAD_NUMS>
struct DispatchLauncher {
    template<typename T>
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


// vec * mat, mat is row major
// [1, N] * [N, M]
// logits * v
// 有关fp32/fp16 fma和add的各种重载操作
namespace gemv2 {
    struct half8 {
        half2 h1;
        half2 h2;
        half2 h3;
        half2 h4;

        __device__ half8& operator = (half8 h8) {
            h1 = h8.h1;
            h2 = h8.h2;
            h3 = h8.h3;
            h4 = h8.h4;
            return *this;
        }
    };

    template<int M, typename T>
    struct get_threads_per_mat_row {

        // M / 4 or 8
        // float: 一次读4个   half: 一次读8个
        // 一共M个数据, 需要多少thread
        static const int value = M * sizeof(T) / 16;
    };

    inline __device__ float add(float a, float b)
    {
        return a + b;
    }

    inline __device__ float4 add(float4 a, float4 b)
    {
        float4 c;
        c.x = gemv2::add(a.x, b.x);
        c.y = gemv2::add(a.y, b.y);
        c.z = gemv2::add(a.z, b.z);
        c.w = gemv2::add(a.w, b.w);
        return c;
    }
    inline __device__ half add(half a, half b)
    {
        //return __hadd(a, b);
        //if use L216, half+half is not really adding, its so weird, which  cause our result is 32, not 256
        return (half)((float)a+(float)b);
    }

    inline __device__ half2 add(half2 a, half2 b)
    {
        half2 res;
        res.x = gemv2::add(a.x, b.x);
        res.y = gemv2::add(a.y, b.y);
        return res;
    }

    inline __device__ half8 add(half8 a, half8 b)
    {
        half8 c;
        c.h1 = gemv2::add(a.h1, b.h1);
        c.h2 = gemv2::add(a.h2, b.h2);
        c.h3 = gemv2::add(a.h3, b.h3);
        c.h4 = gemv2::add(a.h4, b.h4);
        return c;
    }

    inline __device__ half fma(half a, half b, half c)
    {
        // 有的编译器会不认识half intrinsic 例如__hmul或者__hadd，这很奇怪
        // 所以粗暴转成fp32计算再转回fp16
        return __float2half((float)a * (float)b + (float)c);
    }


    inline __device__ half2 fma(half a, half2 b, half2 c)
    {
        half2 res;
        res.x = gemv2::fma(a, b.x, c.x);
        res.y = gemv2::fma(a, b.y, c.y);
        return res;
    }

    inline __device__ half8 fma(half a, half8 b, half8 c)
    {
        half8 d;
        d.h1 = gemv2::fma(a, b.h1, c.h1);
        d.h2 = gemv2::fma(a, b.h2, c.h2);
        d.h3 = gemv2::fma(a, b.h3, c.h3);
        d.h4 = gemv2::fma(a, b.h4, c.h4);
        return d;
    }

    inline __device__ float fma(float a, float b, float c)
    {
        return a * b + c;
    }

    inline __device__ float4 fma(float a, float4 b, float4 c)
    {
        float4 d;
        d.x = gemv2::fma(a, b.x, c.x);
        d.y = gemv2::fma(a, b.y, c.y);
        d.z = gemv2::fma(a, b.z, c.z);
        d.w = gemv2::fma(a, b.w, c.w);
        return d;
    }
} // namespace gemv2

// 1个block处理一个[1, M], 循环处理完[N, M]
// for fp32: <64, M * sizeof(T) / 16 = M / 4, 4>
template<int THREADS_PER_BLOCK, int THREADS_PER_VALUE, int VEC_SIZE>
__global__ void gemv2_kernel(float* matrix, float* vector, float* res, int N, int M) {

    // THREADS_PER_VALUE: 一行需要多少thread
    // 根据编译期常量获取每个线程处理的行列号
    int tid = threadIdx.x;

    // 获得当前线程处理的行号
    int mat_o = tid / THREADS_PER_VALUE;

    // 获得当前线程处理的向量号
    int mat_i = tid % THREADS_PER_VALUE * VEC_SIZE;

    // ROW_PER_ITER: 每个iter处理的行数: THREADS_PER_BLOCK(一个block中thread的总数) / THREADS_PER_VALUE(一行需要多少thread)
    // 假设每个iter处理x行
    constexpr int ROW_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_VALUE;
    
    // 线程内点乘 + inter block 累加
    // N: 矩阵的行数   M: 矩阵的列数
    float4 out;
    for (int i = mat_o; i < N; i += ROW_PER_ITER) {
        float4 mat = *reinterpret_cast<float4*>(&matrix[i * M + mat_i]);
        float logits = vector[i];

        // FMA: Fused Mul Add: y = a * b + y
        out = gemv2::fma(logits, mat, out);
    }

    // intra block累加
    // ROWS_PER_BLOCK: 每个blokc处理多少行
    __shared__ float out_smem[512];
    for (int ROWS_PER_BLOCK = ROW_PER_ITER; ROWS_PER_BLOCK >= 2; ROWS_PER_BLOCK /= 2) {
        int midpoint = ROWS_PER_BLOCK / 2;
        if (mat_o >= midpoint && mat_o < ROWS_PER_BLOCK) {
            
            // M: 列数, mat_o: 行号 mat_i: 向量号
            *reinterpret_cast<float4*>(&out_smem[(mat_o - midpoint) * M + mat_i]) = out;
        }
        __syncthreads();
        if (mat_o < midpoint) {
            out = gemv2::add(*reinterpret_cast<float4*>(&out_smem[mat_o * M + mat_i]), out);
        }
        __syncthreads();
    }
    if (mat_o == 0) {
        *reinterpret_cast<float4*>(&res[mat_i]) = out;
    }
}

// for fp16: <64, M * sizeof(T) / 16 = M / 8, 8>
template<int THREADS_PER_BLOCK, int THREADS_PER_VALUE, int VEC_SIZE>
__global__ void gemv2_kernel(half* matrix, half* vector, half* res, int N, int M) {
    










}
// TODO: 修改float4部分为可以泛化表示float4和half8类型的代码, 而后此模板函数可以取代以上fp32和fp16的gemv2
template<int THREADS_PER_BLOCK, int THREADS_PER_VALUE, int VEC_SIZE, typename T>
__global__ void gemv2_kernel_template(T* matrix, T* vector, T* res, int N, int M) {
    int tid = threadIdx.x;
    int mat_o = tid / THREADS_PER_VALUE;
    int mat_i = tid % THREADS_PER_VALUE * VEC_SIZE;
    constexpr int ROW_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_VALUE;
    __shared__ T out_smem[512];
    float4 out; //TODO
    for (int ti = mat_o; ti < N; ti += ROW_PER_ITER) {
        float4 mat = *reinterpret_cast<float4*>(&matrix[ti * M + mat_i]);//TODO
        T logits = vector[ti];
        out = gemv2::fma(logits, mat, out);
    }
    for (int ROWS_PER_BLOCK = ROW_PER_ITER; ROWS_PER_BLOCK >= 2; ROWS_PER_BLOCK /= 2) {
        int midpoint = ROWS_PER_BLOCK / 2;
        if (mat_o >= midpoint && mat_o < ROWS_PER_BLOCK) {
            *reinterpret_cast<float4*>(&out_smem[(mat_o - midpoint) * M + mat_i]) = out;//TODO
        }
        __syncthreads();
        if (mat_o < midpoint) {
            // ROW_PER_ITER中上半部分out和下半部分out相加
            out = gemv2::add(*reinterpret_cast<float4*>(&out_smem[mat_o * M + mat_i]), out);//TODO
        }
        __syncthreads();
    }
    if (mat_o == 0) {
        *reinterpret_cast<float4*>(&res[mat_i]) = out;//TODO
    }
}

template<int THREADS_PER_BLOCK, int THREADS_PER_VALUE, int VEC_SIZE>
struct DispatchLauncher2 {
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
        gemv2_kernel<THREADS_PER_BLOCK, THREADS_PER_VALUE, VEC_SIZE><<<Grid, Block>>>(d_mat, d_vec, d_dst, N, M);
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