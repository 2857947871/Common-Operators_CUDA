# include <cmath>
# include <cfenv>
# include <random>
# include <cuda.h>
# include <float.h>
# include <bits/stdc++.h>
# include "cuda_runtime.h"


int CheckResult(float *input, float *output, float* groudtruth, int nums){
    int sign = 1;
    for(int i = 0; i < 20; i++){
      if(groudtruth[i] == output[i]) {
        printf("the index is %d, the input is %f, the groudtruth is %f, the res is %f\n", i, input[i], groudtruth[i], output[i]);
      }
      if (groudtruth[i] != output[i]) {
        sign = 0;
        printf("the index is %d, the input is %f, the groudtruth is %f, the res is %f\n", i, input[i], groudtruth[i], output[i]);
      }
    }

    return sign;
}

// PerTensor + Sym: scale = max(abs(weight)) / 127, zeropoint = 0, 
//                          input_int8 = clamp(input_fp32/scale , -128, 127)
// PerTensor + Asym: scale = (max(weight) - min(weight)) / 255, zeropoint = -round(min(weight))/scale

/*
=======================================================================================
========================================= CPU =========================================
=======================================================================================
*/
/*
=======================================================================================
========================================= Sym =========================================
=======================================================================================
*/

// 使用Sym -> 映射到[-128, 127]
// scale = max(abs(val)) / (2 ^ bit - 1)
// zero_point = 0
template <typename T>
void ScalePerTensorSymCPU(const T* in_ptr, const int quantization_bit,
        const int num_elements, T* scale, T* zeropoint)
{
    T in_max = *std::max_element(in_ptr, in_ptr + num_elements);
    T in_min = *std::min_element(in_ptr, in_ptr + num_elements);

    // 对称量化 -> 找绝对值的最大值
    T out_max = std::max(std::abs(in_max), std::abs(in_min));

    // 获取量化位数
    T denominator = static_cast<T>(pow(2.0, quantization_bit - 1) - 1);
    *scale = out_max / denominator;
    *zeropoint = 0;
}

// INT8 = (clip(input / scale).round(), -128, 127)
template <typename T>
void QuantizationPerTensorSymCPU(const T* in_ptr, T* out_ptr, 
        const T scale, const int quantization_bit, const int num_elements)
{
    T upper_bound = static_cast<T>(pow(2.0, quantization_bit - 1) - 1);
    T lower_bound = -upper_bound;
    for (int i = 0; i < num_elements; ++i) {

        // 量化
        T out = std::nearbyint(in_ptr[i] / scale);
        
        // clip
        out = out > upper_bound ? upper_bound : out;
        out = out < lower_bound ? lower_bound : out;
        out_ptr[i] = out;
    }
}

/*
=======================================================================================
========================================= ASym ========================================
=======================================================================================
*/
template <typename T>
void ScalePerTensorASymCPU(const T* in_ptr, const int quantization_bit,
        const int num_elements, T* scale, T* zeropoint)
{
    T in_max = *std::max_element(in_ptr, in_ptr + num_elements);
    T in_min = *std::min_element(in_ptr, in_ptr + num_elements);

    T upper_bound = static_cast<T>(pow(2.0, quantization_bit) - 1);
    T lower_bound = 0;

    *scale = (in_max - in_min) / (upper_bound - lower_bound);

    *zeropoint = -std::round(in_min / *scale);
}

template <typename T>
void QuantizationPerTensorASymCPU(const T* in_ptr, T* out_ptr, 
        const T scale, const T ZP, const int quantization_bit, const int num_elements)
{
    T upper_bound = static_cast<T>(pow(2.0, quantization_bit) - 1);
    T lower_bound = 0;

    for (int i = 0; i < num_elements; ++i) {
        T out = std::nearbyint(in_ptr[i] / scale + ZP);
        out = out > upper_bound ? upper_bound : out;
        out = out < lower_bound ? lower_bound : out;
        
        out = out > upper_bound ? upper_bound : out;
        out = out < lower_bound ? lower_bound : out;
        
        out_ptr[i] = out;
    }
}


/*
=======================================================================================
========================================= GPU =========================================
=======================================================================================
*/
__device__ float gpunearbyint(float val) {
    return std::nearbyint(val);
}

inline __device__ float atomicMax(float* address, float val) {
    
    // CUDA不支持对float类型的原子操作, 所以需要转换为int类型
    int* address_as_i = (int*)address;
    int old = *address_as_i;
    int assumed = 0;

    do {
        // __int_as_float: 将int转换为float
        // __float_as_int: 将float转换为int
        // atomicCAS(address_as_i, assumed, new_value): 比较并交换
        //  作用: 将address_as_i指向的内存位置的值从assumed修改为new_value, 
        //  仅当当前值(*address_as_i)等于assumed时才会更新, 
        //  返回原来的值, 如果更新失败, 函数返回当前值, 循环
        assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (old != assumed);

    return __int_as_float(old);    
}

inline __device__ float atomicMin(float* address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i;
    int assumed = 0;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (old != assumed);

    return __int_as_float(old);    
}

template <typename T>
__global__ void GetMinMaxPerTensor(const T* input_ptr, 
                    const int nums, const int channel, const int HW,
                    T* max_ptr, T* min_ptr)
{
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float smem[];
    
    // smem被分为两个部分, 一个存放最大值, 一个存放最小值
    //       tid1, .., tidN, tid1, .., tidN
    // smem: [max, ..., max, min, ..., min]
    // shared_max: 指向共享内存的开头
    T* shared_max = (T* )smem;
    T* shared_min = (T* )(smem + blockDim.x);

    // 便于数据更新, max和min的初始值设置为最小值和最大值
    shared_max[tid] = FLT_MIN;
    shared_min[tid] = FLT_MAX;

    // 1. 赋值, 数据传入smem
    // 并不是真的在比大小, 而是将数据传入shared_max和shared_min
    // 所以不需要进行原子操作
    // 由于是smem, 每个blokc独有, 所以不会出现gid不同但是tid相同导致数据冲突
    //  eg: gid = 0,  tid = 0
    //      gid = 32, tid = 0
    //      虽然是同一个tid, 但是由于gid不同, 所以对应不同的block -> 对应不同的smem
    // shared_min和shared_max都是第一次赋值
    for (int i = gid; i < nums; i += blockDim.x * gridDim.x) {
        T val = input_ptr[i];
        
        // 由于119和120的操作, 几乎一定传入的是val
        shared_max[tid] = fmaxf(val, shared_max[tid]);
        shared_min[tid] = fminf(val, shared_min[tid]);
    }
    __syncthreads();

    // 2. 每个block内部进行比较
    // 归约
    //  eg: blockDim.x = 8, s = 4 tid < 4, gid < 8
    //      tid = 0, shared_max[0] = max(shared_max[0], shared_max[4]) -> 
    //      ...
    //      tid = 3, shared_max[3] = max(shared_max[3], shared_max[7])
    //      tid = 0, shared_max[0] = max(shared_max[0], shared_max[2]) ->
    //      tid = 1, shared_max[1] = max(shared_max[1], shared_max[3])
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && gid < nums) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + s]);
            shared_min[tid] = min(shared_min[tid], shared_min[tid + s]);
        }
        __syncthreads();
    }

    // 3. 每个block的smem[0]和smem[blockDim.x]保存了block内的max和min
    if (tid == 0) {
        atomicMax(max_ptr, shared_max[0]);
        atomicMin(min_ptr, shared_min[0]);
    }
}

/*
=======================================================================================
========================================= Sym =========================================
=======================================================================================
*/
// 使用Sym -> 映射到[-128, 127]
// scale = max(abs(val)) / (2 ^ bit - 1)
// zero_point = 0
template <typename T>
__global__ void GetScaleAndZPSym(const T* max_ptr, const T* min_ptr, 
                    const int nums, const int quantization_bit, T* scale)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    while (gid < nums) {
        T weight_max = max(fabs(max_ptr[gid]), fabs(min_ptr[gid]));
        T denominator = static_cast<T>(pow(2.0, quantization_bit - 1) - 1);
        scale[gid] = weight_max / denominator;
        gid += blockDim.x * gridDim.x;
    }
}

template <typename T>
__global__ void QuantizePerTensorSym(const T* input_ptr, T* out_ptr, 
                    const T* scale_ptr, const int nums, const int quantization_bit, const int channel, const int HW)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    T upper_bound = static_cast<T>(pow(2.0, quantization_bit - 1) - 1);
    T lower_bound = -upper_bound - 1;
    T scale = scale_ptr[0];
    
    while (gid < nums) {
        T out = gpunearbyint(input_ptr[gid] / scale);
        out = out > upper_bound ? upper_bound : out;
        out = out < lower_bound ? lower_bound : out;
        out_ptr[gid] = out;
        gid += blockDim.x * gridDim.x;
    }
}

/*
=======================================================================================
========================================= ASym ========================================
=======================================================================================
*/
template <typename T>
__global__ void GetScaleAndZPASym(const T* max_ptr, const T* min_ptr, const int nums,
                    const int quantization_bit, T* scale, T* zero_point)
{
# if 0
    int tid = threadIdx.x;
    int gid = (blockDim.x * blockIdx.x) + tid;
    while (gid < nums) {
        T denominator = static_cast<T>(pow(2.0, quantization_bit)) - 1;
        T min = -min_ptr[gid];
        T s = (max_ptr[gid] - min) / denominator;
        scale[gid] = s;
        zero_point[gid] = -1 * std::nearbyint(min / s);
        gid += gridDim.x * blockDim.x;
    }
# endif


# if 1
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    while (gid < nums) {
        T denominator = static_cast<T>(pow(2.0, quantization_bit) - 1);
        T max_val = max_ptr[gid];
        T min_val = min_ptr[gid];
        scale[gid] = (max_val - min_val) / denominator;
        zero_point[gid] = -1 * gpunearbyint(min_val / scale[gid]);
        gid += blockDim.x * gridDim.x;
    }
# endif

}

template <typename T>
__global__ void QuantizePerTensorAsym(const T* in_ptr, T* out_ptr, 
                    const T* scale_ptr, const T* zero_point_ptr,
                    const int nums, const int quantization_bit, const int channel)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    T upper_bound = static_cast<T>(pow(2.0, quantization_bit) - 1);
    T lower_bound = 0;
    T scale = *scale_ptr;
    T zero_point = *zero_point_ptr;

    while (gid < nums) {
        T out = gpunearbyint(in_ptr[gid] / scale + zero_point);
        out = out > upper_bound ? upper_bound : out;
        out = out < lower_bound ? lower_bound : out;
        out_ptr[gid] = out;
        gid += blockDim.x * gridDim.x;
    }
}


# define LAUNCH_GPU_KERNEL_Sym(GetMinMaxFunc, QuantFunc, scale_size, channel, HW) \
    cudaMalloc((void **)&d_scale, scale_size * sizeof(float)); \
    cudaMalloc((void **)&d_zeropoint, scale_size * sizeof(float)); \
    cudaMalloc((void **)&d_max, scale_size * sizeof(float)); \
    cudaMalloc((void **)&d_min, scale_size * sizeof(float)); \
    cudaEvent_t start, stop; \
    cudaEventCreate(&start); \
    cudaEventCreate(&stop); \
    cudaEventRecord(start); \
    GetMinMaxFunc<float><<<gridSize, blockSize, blockSize * 2 * sizeof(float), 0>>>(d_input, nums, channel, HW, d_max, d_min);  \
    GetScaleAndZPSym<float><<<1, blockSize>>>(d_max, d_min, channel, quantization_bit, d_scale); \
    QuantFunc<float><<<gridSize, blockSize>>>(d_input, d_output, d_scale, nums, quantization_bit, channel, HW); \
    cudaEventRecord(stop); \
    cudaEventSynchronize(stop); \
    cudaEventElapsedTime(&ms, start, stop); \
    cudaMemcpy(output, d_output, sizeof(float) * nums, cudaMemcpyDeviceToHost); \
    ScalePerTensorSymCPU<float>(input, quantization_bit, nums, scale, zeropoint); \
    QuantizationPerTensorSymCPU<float>(input, CPUOutput, *scale, quantization_bit, nums);

# define LAUNCH_GPU_KERNEL_ASym(GetMinMaxFunc, QuantFunc, scale_size, channel, HW) \
    cudaMalloc((void **)&d_scale, scale_size * sizeof(float)); \
    cudaMalloc((void **)&d_zeropoint, scale_size * sizeof(float)); \
    cudaMalloc((void **)&d_max, scale_size * sizeof(float)); \
    cudaMalloc((void **)&d_min, scale_size * sizeof(float)); \
    cudaEvent_t start, stop; \
    cudaEventCreate(&start); \
    cudaEventCreate(&stop); \
    cudaEventRecord(start); \
    GetMinMaxFunc<float><<<gridSize, blockSize, blockSize * 2 * sizeof(float), 0>>>(d_input, nums, channel, HW, d_max, d_min);  \
    GetScaleAndZPASym<float><<<1, blockSize>>>(d_max, d_min, channel, quantization_bit, d_scale, d_zeropoint); \
    QuantFunc<float><<<gridSize, blockSize>>>(d_input, d_output, d_scale, d_zeropoint, nums, quantization_bit, channel); \
    cudaEventRecord(stop); \
    cudaEventSynchronize(stop); \
    cudaEventElapsedTime(&ms, start, stop); \
    cudaMemcpy(output, d_output, sizeof(float) * nums, cudaMemcpyDeviceToHost); \
    ScalePerTensorASymCPU<float>(input, quantization_bit, nums, scale, zeropoint); \
    QuantizationPerTensorASymCPU<float>(input, CPUOutput, *scale, *zeropoint, quantization_bit, nums);


int main() {

    // init
    float ms             = 0;
    int nums             = 400 * 20 * 10;
    int HW               = 20 * 10;
    int channel          = 400;
    int quantization_bit = 8;
    float cpu_min        = FLT_MAX;
    float cpu_max        = FLT_MIN;
    float* input         = new float[nums];
    float* output        = new float[nums];
    float* CPUOutput     = (float*) malloc(sizeof(float) * nums);
    float* scale         = (float*) malloc(sizeof(float) * 1);
    float* zeropoint     = (float*) malloc(sizeof(float) * 1);
    for(int i = 0; i < nums; i++) {
        // input[i] = -5 + static_cast<float> (rand()) / ( static_cast <float> (RAND_MAX / 10));
        input[i] = -5 + i % 20;
        cpu_min = std::min(input[i], cpu_min);
        cpu_max = std::max(input[i], cpu_max);
    }

    float* d_input, *d_output;
    cudaMalloc((void**)&d_input, sizeof(float) * nums);
    cudaMalloc((void**)&d_output, sizeof(float) * nums);
    cudaMemcpy(d_input, input, sizeof(float) * nums, cudaMemcpyHostToDevice);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int maxblocks = deviceProp.maxGridSize[0];
    int blockSize = 256;
    int gridSize = std::min<int>((nums + blockSize - 1) / blockSize,  std::min<int>(maxblocks, channel));
    float *d_scale, *d_zeropoint, *d_max, *d_min;

    // quantize
    // LAUNCH_GPU_KERNEL_Sym(GetMinMaxPerTensor, QuantizePerTensorSym, 1, nums, HW);
    LAUNCH_GPU_KERNEL_ASym(GetMinMaxPerTensor, QuantizePerTensorAsym, 1, nums, HW);


    // check
    if (CheckResult(input, output, CPUOutput, nums)) {
        printf("the ans is right\n");
        printf("Quantize kernel latency = %f ms\n", ms);
    } else {
        printf("the ans is wrong\n");
    }
    
    // free
    delete[] input;
    delete[] output;
    free(scale);
    free(zeropoint);
    free(CPUOutput);
    cudaFree(d_max);
    cudaFree(d_min);
    cudaFree(d_input);
    cudaFree(d_scale);
    cudaFree(d_output);
    cudaFree(d_zeropoint);
    
    return 0;
}