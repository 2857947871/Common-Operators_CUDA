# include <iostream>
# include <bits/stdc++.h>
# include "cooperative_groups.h"
# include <cuda.h>
# include <cuda_fp16.h>
# include "cuda_runtime.h"


// 见笔记CUDA_lesson
// 确保数据按内存对齐的方式存储
template <typename T, int Size>
struct alignas(sizeof(T) * Size) AlignedVector {
    // 向量由size个类型为T的元素组成
    T val[Size];
    
    // 向量支持[]访问
    __host__ __device__ inline const T& operator[](int i) const {
        
        // 运算符重载的实现
        return val[i];
    }
    __host__ __device__ inline T& operator[](int i) {
        
        // 运算符重载的实现
        return val[i];
    }
};

__device__ float TanhApprox(float x) {
    // ptx指令，是CUDA的更底层的语言，类似于汇编对于C/C++
    //float r;
    //asm("tanh.approx.f32 %0,%1; \n\t" : "=f"(r) : "f"(x));
    //return r;
    return tanhf(x); // CUDA内置的math API
}

// gelu公式：x / 2 * (1 + tan(0.7978845608028654 * (x + 0.044714998453855515 * x^3)))
template<typename T>
struct GeluFunctor {

    static constexpr T alpha = static_cast<T>(0.7978845608028654);
    static constexpr T beta = static_cast<T>(0.044714998453855515);

    // 构造函数, 使得该构造体可以在device上使用
    __device__ GeluFunctor() {};

    // operator: 运算符重载
    // 重载()运算符
    __device__ T operator()(T x) const {
        const T half    = static_cast<T>(0.5);
        const T one     = static_cast<T>(1);
        const T tanh_in = alpha * (x + beta * x * x * x);
        return half * x * (one + tanh(tanh_in));
    }
};

// 模板特化
// 专门处理half类型的GeluFunctor
template<>
struct GeluFunctor<half> {

    static constexpr float alpha = GeluFunctor<float>::alpha;
    static constexpr float beta  = GeluFunctor<float>::beta;
    
    // float_functor: 用于处理float类型的GeluFunctor
    GeluFunctor<float> float_functor;
    __device__ GeluFunctor() {};

    __device__ half operator()(const half x) const {
        
        // float_functor是一个GeluFunctor<float>类型的对象,
        // GeluFunctor<float>中进行了()的重载
        // 所以(x)中的x会进行gelu操作
        return static_cast<half>(float_functor(static_cast<float>(x)));
    }
};

template <int VecSize>
__global__ void FP16GeluCUDAKernel(const __half* x, __half* y, int n) {
    // 向量化load & store
    // 读取向量的offset
    int offset = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x) * VecSize;
    // 循环读取向量的stride
    int stride = static_cast<int>(blockDim.x * gridDim.x) * VecSize;
    GeluFunctor<half> gelu_fwd;
    __half y_reg[VecSize];

    // using: 类型别名
    // ArrT: AlignedVector<__half, VecSize>类型的别名
    using ArrT = AlignedVector<__half, VecSize>; // 声明向量类型

    for (; offset < n; offset += stride) {
        // 先求出每个线程所读向量的起始offset
        const __half* in = x + offset;

        if (VecSize == 1){
            y_reg[0] = gelu_fwd(in[0]);
        } else {
            //标量计算
            for (int i = 0; i < VecSize; i++) {
                y_reg[i] = gelu_fwd(in[i]);
            }
        }
        // 将计算结果写回显存
        *reinterpret_cast<ArrT*>(y + offset) = *reinterpret_cast<ArrT*>(y_reg);
    }
}


int main() {

    // 设备信息
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    // 创建并初始化 FP16 的数组
    int n = 128;
    __half* device_x;
    __half* device_y;
    __half* host_x = new __half[n];
    __half* host_y = new __half[n];
    for (int i = 0; i < n; ++i) {
        // 类型转换 int -> FP16
        host_x[i] = (__half)(i - 50);
    }

    // malloc
    cudaMalloc((void** )&device_x, n * sizeof(__half));
    cudaMalloc((void** )&device_y, n * sizeof(__half));
    cudaMemcpy(device_x, host_x, sizeof(__half) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(device_y, host_y, sizeof(__half) * n, cudaMemcpyHostToDevice);

    // lambda表达式
    // 检查是否内存对齐
    // 返回一个bool值
    auto is_aligned = [](const void* p, int alignment) {
        
        // p转换为uintptr_t类型 
        // typedef unsigned long int	uintptr_t;
        // p: 指针(地址) 与 alignment 取余, 计算是否对齐
        return reinterpret_cast<uintptr_t>(p) % alignment == 0;
    };

    // 见笔记
    // kAlignment: int
    // constexpr: 常量表达式, 在编译时就能计算出结果
    // alignof：返回类型的对齐要求(对齐的字节数)
    constexpr auto kAlignment = alignof(AlignedVector<__half, 8>);

    // is_aligned(host_x, kAlignment)
    // 计算host_x是否对齐(host_x是一个指针(地址), 与kAlignment取余, 计算是否对齐)
    if (n % 8 == 0 && is_aligned(host_x, kAlignment) && is_aligned(host_y, kAlignment))
    {
        int thread = std::min<int>(512, deviceProp.maxThreadsPerBlock);
        int block = (n + thread - 1) / thread;
        block = std::min<int>(block, deviceProp.maxGridSize[0]);

        // operation
        FP16GeluCUDAKernel<1><<<block, thread>>>(device_x, device_y, n);
        cudaMemcpy(host_y, device_y, sizeof(__half) * n, cudaMemcpyDeviceToHost);
    }

    // free
    printf("pass\n");

    for (int i = 0; i < n; i++) {
        std::cout << static_cast<float>(host_y[i]) << " ";
    }

    delete host_x;
    delete host_y;
    host_x = nullptr;
    host_y = nullptr;
    cudaFree(device_x);
    cudaFree(device_y);


    return 0;
}