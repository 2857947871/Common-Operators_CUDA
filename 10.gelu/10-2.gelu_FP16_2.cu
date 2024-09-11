# include <stdio.h>
# include <iostream>
# include <cuda.h>
# include <cuda_fp16.h>
# include "cuda_runtime.h"


__device__ float TanhApprox(float x) {
  // ptx指令，是CUDA的更底层的语言，类似于汇编对于C/C++
    float r;
    asm("tanh.approx.f32 %0,%1; \n\t" : "=f"(r) : "f"(x));
    return r;
  // return tanhf(x); // CUDA内置的math API
}

// alignas(sizeof(T) * Size): 强制变量内存对齐
// 使得 AlignedVector 对象可以像标准数组一样, 通过下标访问其元素
template<typename T, int Size>
struct alignas(sizeof(T) * Size) AlignedVector {

    // 向量由Size个类型为T的元素组成
    T val[Size];

    // 运算符重载
    // T&: 返回值类型(引用)   operator []: 重载[]运算符   int i: 参数列表
    __host__ __device__ inline const T& operator [] (int i) const{
        
        return val[i];
    }

    __host__ __device__ inline T& operator[](int i) {
        
        return val[i];
    }
};

template<typename T>
struct GeluFunctor {

    static constexpr T alpha = static_cast<T>(0.7978845608028654); 
    static constexpr T beta = static_cast<T>(0.044714998453855515);

    // 与默认构造函数等价
    __device__ GeluFunctor() {};

    // 运算符重载
    // __device__: 表示该函数在设备上执行
    // T: 返回值类型   operator (): 重载()运算符   T x: 参数
    __device__ T operator () (T x) const {
        const T half = static_cast<T>(0.5);
        const T one  = static_cast<T>(1);
        const T tanh_in = alpha * (x + beta * x * x * x);
        return half * x * (one + tanh(tanh_in));
    }
};

// 模板的特化
template<>
struct GeluFunctor<half> {

    static constexpr float alpha = static_cast<float>(0.7978845608028654); 
    static constexpr float beta  = static_cast<float>(0.044714998453855515);
    GeluFunctor<float> float_functor;

    __device__ GeluFunctor() {};

    // half: 返回值类型   operator (): 重载()运算符   const half x: 参数 
    __device__ half operator () (const half x) const {

        const float tanh_in = 
            __half2float(__float2half_rn(alpha) * (x + __float2half_rn(beta) * x * x * x));    
        const float tanh_out = TanhApprox(tanh_in);

        return __float2half_rn(0.5f) * x * (__float2half_rn(1.0f) + __float2half_rn(tanh_out));
    }

    __device__ void apply2(half* y, const half* x) const {

        // 输入: float -> half2
        // half2: 2个half类型的元素组成的结构体 -> FP16 * 2 -> FP32
        const half2 x2 = *(reinterpret_cast<const half2*>(x));

        // tanh_in = alpha * (x + beta * x * x * x)
        // float2: 2个float类型的元素组成的结构体
        // __half22float2: half2 -> float2
        // __float2half2_rn: float -> half2
        // __hmul2(const __half2 a, const __half2 b): a * b
        // __hadd2(const __half2 a, const __half2 b): a + b
        const float2 tanh_in = __half22float2(
                __hmul2(
                    __float2half2_rn(alpha),
                    __hadd2(x2, __hmul2(__hmul2(__hmul2(__float2half2_rn(beta), x2), x2), x2))));

        // tanh_out = tanh(tanh_in)
        float2 tanh_out;
        tanh_out.x = TanhApprox(tanh_in.x);
        tanh_out.y = TanhApprox(tanh_in.y);
        
        // y = 0.5 * x * (1.0 + tanh_out)
        const half2 y2 = __hmul2(__hmul2(__float2half2_rn(0.5F), x2),
                                __hadd2(__float2half2_rn(1.0F), __float22half2_rn(tanh_out)));
        
        // 向量化将结果写回显存
        // 此时已经在device中, 无需再次转换
        *(reinterpret_cast<half2*>(y)) = y2;
    }
};

// 非类型模板参数声明, 它允许你为模板类或模板函数定义一个常量参数
template <int VecSize>
__global__ void FP16GeluCUDAKernel(const __half* x, __half* y, int n) {

    // 向量化load&save, 一个thread处理VecSize个元素
    __half y_reg[VecSize];
    int offset = (threadIdx.x + blockIdx.x * blockDim.x) * VecSize;
    int stride = (blockDim.x * gridDim.x) * VecSize;

    GeluFunctor<half> gelu_fwd;

    using ArrT = AlignedVector<__half, VecSize>; // 声明向量类型
    for (; offset < n; offset += stride) {

        // 每个线程所读向量的起始地址
        const __half* in = x + offset;

        // 每个线程处理VecSize个元素
        if (VecSize == 1) {
            y_reg[0] = gelu_fwd(in[0]);
            *reinterpret_cast<ArrT*>(y + offset) = *reinterpret_cast<ArrT*>(y_reg);
        } else {
            for (int i = 0; i < VecSize; i += 2) {
                gelu_fwd.apply2(y+offset+i, in+i);
            }
        }
    }
}


int main() {

    int n = 1280;

    // 初始化参数并分配内存
    __half *x_host = new __half[n];
    __half *y_host = new __half[n];
    __half *x_device = nullptr;
    __half *y_device = nullptr;

    for (int i = 0; i < n; ++i) {
        x_host[i] = (__half)(i - 50);
    }

    cudaMalloc((void** )&x_device, n * sizeof(__half));
    cudaMalloc((void** )&y_device, n * sizeof(__half));
    cudaMemcpy(x_device, x_host, sizeof(__half) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(y_device, y_host, sizeof(__half) * n, cudaMemcpyHostToDevice);
    
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    
    // 内存对齐
    auto is_aligned = [](const void* p, int alignment) {
        return reinterpret_cast<uintptr_t>(p) % alignment == 0;
    };

    // constexpr: 常量表达式, 在编译时就能计算出结果
    // alignof: 关键字, 返回类型的对齐要求(对齐的字节数)
    //  在使用 alignof 运算符时, 传入的参数是一个类型, alignof 的主要作用是返回这个类型的对齐要求
    //  eg: size_t intAlignment = alignof(int);  // 获取 int 类型的对齐要求
    //  eg: struct MyStruct {
    //           char a;
    //           int  b;
    //       };
    //  size_t myStructAlignment = alignof(MyStruct);  // 获取 MyStruct 类型的对齐要求
    //  此时传入AlignedVector<__half, 8>, 我们想知道, 8个__half类型的元素(这些元素也是内存对齐存储的), 对齐要求是多少
    constexpr auto kAlignment = alignof(AlignedVector<__half, 8>);

    if (n % 8 == 0 && is_aligned(x_host, kAlignment) && is_aligned(y_host, kAlignment)) 
    {
        int thread = std::min<int>(512, deviceProp.maxThreadsPerBlock);
        int block = (n / 8 + thread - 1) / thread;  
        block = std::min<int>(block, deviceProp.maxGridSize[0]);
        
        // VecSize = 2, 其他会报错, 暂不明原因
        FP16GeluCUDAKernel<8><<<block, thread>>>(x_device, y_device, n); 
        cudaMemcpy(y_host, y_device, sizeof(__half) * n, cudaMemcpyDeviceToHost);
    }
    printf("pass\n");

    for (int i = 0; i < n; i++) {
        std::cout << static_cast<float>(y_host[i]) << " ";
    }
    printf("\n");
  
    delete x_host;
    x_host = nullptr;
    delete y_host;
    y_host = nullptr;
    cudaFree(x_device);
    cudaFree(y_device);
}