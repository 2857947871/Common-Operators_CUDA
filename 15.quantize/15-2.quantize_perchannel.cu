# include <cmath>
# include <cfenv>
# include <random>
# include <cuda.h>
# include <float.h>
# include <bits/stdc++.h>
# include "cuda_runtime.h"


int CheckResult(float *input, float *output, float* groudtruth, int nums){
    int sign = 1;
    for(int i = 0; i < 30; i++){
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

// PerChannel + Sym: scale[channel_id] = max(abs(weight[channel_id])) / 127 , zeropoint = 0, 
//                          input_int8[channel_id * HW + (channel_id + 1) * HW] = clamp(input_fp32[channel_id * HW + (channel_id + 1) * HW]/scale[channel_id], -128, 127)
// PerChannel + Asym: scale[channel_id] = (max(weight[channel_id]) - min(weight[channel_id])) / 255, 
//                          zeropoint = -round(min(weight[channel_id])) / scale[channel_id]

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
void ScalePerChannelSymCPU(const T* in_ptr, const int quantization_bit, const int HW, const int channel,
        const int num_elements, T* scale, T* zero_point)
{
    for (int cid = 0; cid < channel; ++cid) {
        int start = cid * HW;
        int end = (cid + 1) * HW;
        T channel_max = *std::max_element(in_ptr + start, in_ptr + end);
        T channel_min = *std::min_element(in_ptr + start, in_ptr + end);
        T out_max = std::max(std::fabs(channel_max), std::fabs(channel_min));
        T denominator = static_cast<T>(std::pow(2.0, quantization_bit - 1) - 1);
        scale[cid] = out_max / denominator;
        zero_point[cid] = 0;
    }
}

template <typename T>
void QuantizationPerChannelSymCPU(const T* in_ptr, T* out_ptr, const T* scale, const int quantization_bit, const int HW,
        const int num_elements)
{
    T upper_bound = static_cast<T>(pow(2.0, quantization_bit - 1)) - 1;
    T lower_bound = -upper_bound - 1;
    
    for (int i = 0; i < num_elements; ++i) {
        T out = std::nearbyint(in_ptr[i] / scale[i / HW]);
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
void ScalePerChannelAsymCPU(const T* in_ptr, const int quantization_bit, const int HW, const int channel,
        const int num_elements, T* scale, T* zero_point)
{
    for (int cid = 0; cid < channel; ++cid) {
        int start = cid * HW;
        int end = (cid + 1) * HW;
        T channel_max = *std::max_element(in_ptr + start, in_ptr + end);
        T channel_min = *std::min_element(in_ptr + start, in_ptr + end);

        // 获取量化位数，非对称量化计算scale和zero_point
        T denominator = static_cast<T>(std::pow(2.0, quantization_bit) - 1); // 使用[0, 2^bit - 1]的范围
        scale[cid] = (channel_max - channel_min) / denominator;

        // zero_point 应该使得 input 0 映射到量化值
        zero_point[cid] = std::nearbyint(-channel_min / scale[cid]);
    }
}

template <typename T>
void QuantizationPerChannelAsymCPU(const T* in_ptr, T* out_ptr, const T* scale, const T* zero_point, const int quantization_bit, const int HW,
        const int num_elements)
{
    T upper_bound = static_cast<T>(std::pow(2.0, quantization_bit) - 1);
    T lower_bound = 0;

    for (int i = 0; i < num_elements; ++i) {
        int channel_index = i / HW;

        // 使用每个通道的 scale 和 zero_point 进行量化
        T out = std::nearbyint(in_ptr[i] / scale[channel_index] + zero_point[channel_index]);

        // clip 到合法范围内 [0, 2^bit - 1]
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
__global__ void GetMinMaxPerChannel(const T* input_ptr, const int nums,
                    T* max_ptr, T* min_ptr, const int channel, const int HW)
{
    extern __shared__ float smem[];
    T* smem_max = (T* )smem;
    T* smem_min = (T* )(smem + blockDim.x);

    // 一个blokc处理一个channel
    int cur_channel = blockIdx.x;
    int tid = threadIdx.x;

    while (cur_channel < channel) {
        smem_max[tid] = FLT_MIN;
        smem_min[tid] = FLT_MAX;

        int index = (HW * cur_channel) + tid;
        int end = HW * (cur_channel + 1);

        while (index < end && index < nums) {
            smem_max[tid] = fmaxf(smem_max[tid], input_ptr[index]);
            smem_min[tid] = fminf(smem_min[tid], input_ptr[index]);
            index += blockDim.x;
        }
        __syncthreads();

        // 归约求min和max
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                smem_max[tid] = fmaxf(smem_max[tid], smem_max[tid + s]);
                smem_min[tid] = fminf(smem_min[tid], smem_min[tid + s]);
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            atomicMax(max_ptr + cur_channel, smem_max[0]);
            atomicMin(min_ptr + cur_channel, smem_min[0]);
        }

        cur_channel += gridDim.x;
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
__global__ void QuantizationPerChannelSym(const T* in_ptr, T* out_ptr, const T* scale_ptr, const int nums,
                    const double quantization_bit, const int scale_size, const int HW)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
        
    T upper_bound = static_cast<T>(pow(2.0, quantization_bit - 1) - 1);
    T lower_bound = -upper_bound - 1;

    while (gid < nums) {
        int channel_index = gid / HW;
        int scale_index = min(scale_size - 1, channel_index);
        
        T scale = scale_ptr[scale_index];
        T out = gpunearbyint(in_ptr[gid] / scale);
        
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
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    while (gid < nums) {
        T denominator = static_cast<T>(pow(2.0, quantization_bit) - 1);
        T max_val = max_ptr[gid];
        T min_val = min_ptr[gid];
        scale[gid] = (max_val - min_val) / denominator;
        zero_point[gid] = -1 * gpunearbyint(min_val / scale[gid]);

        gid += blockDim.x * gridDim.x;
    }
}

template <typename T>
__global__ void QuantizationPerChannelAsym(const T* in_ptr, T* out_ptr, 
                    const T* scale_ptr, const T* zero_point_ptr,
                    const int nums, const int quantization_bit, const int HW)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    T upper_bound = static_cast<T>(pow(2.0, quantization_bit) - 1);
    T lower_bound = 0;

    while (gid < nums) {
        int channel_index = gid / HW;                   // 获取当前元素对应的通道
        T scale = scale_ptr[channel_index];             // 使用每个通道的scale
        T zero_point = zero_point_ptr[channel_index];   // 使用每个通道的zero point

        T out = gpunearbyint(in_ptr[gid] / scale + zero_point);

        out = out > upper_bound ? upper_bound : out;
        out = out < lower_bound ? lower_bound : out;

        out_ptr[gid] = out;

        gid += blockDim.x * gridDim.x;
    }
}


# define LAUNCH_GPU_KERNEL_ASym(GetMinMaxFunc, QuantFunc, scale_size, channel, HW) \
    cudaMalloc((void **)&d_scale, scale_size * sizeof(float)); \
    cudaMalloc((void **)&d_zeropoint, scale_size * sizeof(float)); \
    cudaMalloc((void **)&d_max, scale_size * sizeof(float)); \
    cudaMalloc((void **)&d_min, scale_size * sizeof(float)); \
    cudaEvent_t start, stop; \
    cudaEventCreate(&start); \
    cudaEventCreate(&stop); \
    cudaEventRecord(start); \
    GetMinMaxFunc<float><<<gridSize, blockSize, blockSize * 2 * sizeof(float), 0>>>(d_input, nums, d_max, d_min, channel, HW);  \
    GetScaleAndZPASym<float><<<1, blockSize>>>(d_max, d_min, channel, quantization_bit, d_scale, d_zeropoint); \
    QuantFunc<float><<<gridSize, blockSize>>>(d_input, d_output, d_scale, d_zeropoint, nums, quantization_bit, HW); \
    cudaEventRecord(stop); \
    cudaEventSynchronize(stop); \
    cudaEventElapsedTime(&ms, start, stop); \
    cudaMemcpy(output, d_output, sizeof(float) * nums, cudaMemcpyDeviceToHost); \
    ScalePerChannelAsymCPU<float>(input, quantization_bit, HW, channel, nums, scale, zeropoint); \
    QuantizationPerChannelAsymCPU<float>(input, CPUOutput, scale, zeropoint, quantization_bit, HW, nums);


# define LAUNCH_GPU_KERNEL_Sym(GetMinMaxFunc, QuantFunc, scale_size, channel, HW) \
    cudaMalloc((void **)&d_scale, scale_size * sizeof(float)); \
    cudaMalloc((void **)&d_zeropoint, scale_size * sizeof(float)); \
    cudaMalloc((void **)&d_max, scale_size * sizeof(float)); \
    cudaMalloc((void **)&d_min, scale_size * sizeof(float)); \
    cudaEvent_t start, stop; \
    cudaEventCreate(&start); \
    cudaEventCreate(&stop); \
    cudaEventRecord(start); \
    GetMinMaxFunc<float><<<gridSize, blockSize, blockSize * 2 * sizeof(float), 0>>>(d_input, nums, d_max, d_min, channel, HW);  \
    GetScaleAndZPSym<float><<<1, blockSize>>>(d_max, d_min, channel, quantization_bit, d_scale); \
    QuantFunc<float><<<gridSize, blockSize>>>(d_input, d_output, d_scale, nums, quantization_bit, channel, HW); \
    cudaEventRecord(stop); \
    cudaEventSynchronize(stop); \
    cudaEventElapsedTime(&ms, start, stop); \
    cudaMemcpy(output, d_output, sizeof(float) * nums, cudaMemcpyDeviceToHost); \
    ScalePerChannelSymCPU<float>(input, quantization_bit, HW, channel, nums, scale, zeropoint); \
    QuantizationPerChannelSymCPU<float>(input, CPUOutput, scale, quantization_bit, HW, nums);

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
    float* scale         = (float*) malloc(sizeof(float) * channel);
    float* zeropoint     = (float*) malloc(sizeof(float) * channel);
    for(int i = 0; i < nums; i++) {
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
    // LAUNCH_GPU_KERNEL_Sym(GetMinMaxPerChannel, QuantizationPerChannelSym, channel, channel, HW);
    LAUNCH_GPU_KERNEL_ASym(GetMinMaxPerChannel, QuantizationPerChannelAsym, channel, channel, HW);

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