# include <cmath>
# include <cfenv>
# include <random>
# include <cuda.h>
# include <float.h>
# include <bits/stdc++.h>
# include "cuda_runtime.h"


bool CheckResult(float *out, float* groudtruth, int nums){
    for(int i = 0; i < nums; i++){
      if(groudtruth[i] != out[i]) {
        printf("the wrong index is %d, the groudtruth is %f, the res is %f\n", i, groudtruth[i], out[i]);
        return false;
      }
    }
    return true;
}

// PerTensor + Sym: scale = max(abs(weight)) / 127, zeropoint = 0, 
//                          input_int8 = clamp(input_fp32/scale , -128, 127)
// PerTensor + Asym: scale = (max(weight) - min(weight)) / 255, zeropoint = -round(min(weight))/scale
//
// PerChannel + Sym: scale[channel_id] = max(abs(weight[channel_id])) / 127 , zeropoint = 0, 
//                          input_int8[channel_id * HW + (channel_id + 1) * HW] = clamp(input_fp32[channel_id * HW + (channel_id + 1) * HW]/scale[channel_id], -128, 127)
// PerChannel + Asym: scale[channel_id] = (max(weight[channel_id]) - min(weight[channel_id])) / 255, 
//                          zeropoint = -round(min(weight[channel_id])) / scale[channel_id]


// ========================================= CPU =========================================
// scale = max(abs(val)) / (2 ^ bit - 1)
// zero_point = 0
template <typename T>
void GenScalePerTensorSymCPU(const T* in_ptr, const int quantization_bit,
        const int num_elements, T* scale, T* zeropoint)
{
    T in_max = *std::max_element(in_ptr, in_ptr + num_elements);
    T in_min = *std::min_element(in_ptr, in_ptr + num_elements);
    T out_max = std::max(std::abs(in_max), std::abs(in_min));

    // 获取量化位数
    T denominator = static_cast<T>(pow(2.0, quantization_bit - 1) - 1);
    *scale = out_max / denominator;
    *zeropoint = 0;
}

// INT8 = (clip(input / scale).round(), -128, 127)
template <typename T>
void QuantizationPerTensorSymCPU(const T* in_ptr, const T scale, const int quantization_bit,
        const int num_elements, T* out_ptr)
{
    T upper_bound = static_cast<T>(pow(2.0, quantization_bit - 1) - 1);
    T lower_bound = -upper_bound;
    for (int i = 0; i < num_elements; ++i) {
        T out = std::nearbyint(in_ptr[i] / scale);
        out = out > upper_bound ? upper_bound : out;
        out = out < lower_bound ? lower_bound : out;
        out_ptr[i] = out;
    }
}

// scale[cid] = max(abs(weight[cid])) / 127
// zero_point = 0
template <typename T>
void GenScalePerChannelSymCPU(const T* in_ptr, const int quantization_bit, const int HW, const int channel,
        const int num_elements, T* scale, T* zero_point)
{
    for (int cid = 0; cid < channel; ++cid) {
        int start = cid * HW;
        int end   = (cid + 1) * HW;
        T channel_max = *std::max_element(in_ptr + start, in_ptr + end);
        T channel_min = *std::min_element(in_ptr + start, in_ptr + end);
        T out_max = std::max(std::abs(channel_max), std::abs(channel_min));
        T denominator = static_cast<T>(pow(2.0, quantization_bit - 1) - 1);
        scale[cid] = out_max / denominator;
        zero_point[cid] = 0;   
    }
}

// INT8 = (clip(input[cid] / scale[cid]).round(), -128, 127)
template <typename T>
void QuantizationPerChannelSymCPU(const T* in_ptr, const T* scale, const int quantization_bit, const int HW,
        const int num_elements, T* out_ptr)
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


// ========================================= GPU =========================================
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
__global__ void GetScaleAndZPSym(const T* max_ptr, const T* min_ptr, 
                    const int nums, const double quantization_bit, 
                    T* scale, T* zeropoint)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    while (gid < nums) {
        T weight_max = max(fabs(max_ptr[gid]), fabs(min_ptr[gid])); // fabs: 取绝对值(float)
        T denominator = static_cast<T>(pow(2.0, quantization_bit - 1) - 1);
        scale[gid] = weight_max / denominator;
        zeropoint[gid] = 0;
        gid += blockDim.x * gridDim.x;
    }
}

// PerTensor
template <typename T>
__global__ void GetMaxMinPerTensor(const T* input_ptr, const int nums, T* max_ptr,
                    T* min_ptr, const int channel, const int HW)
{
    // extern: 共享内存大小在编译时不确定, 在运行时确定
    extern __shared__ unsigned char shared_max_min_memory[];
    
    // 共享内存被分为两个部分, 一个存储最大值, 一个存储最小值
    //       tid1, .., tidN, tid1, .., tidN
    // smem: [max, ..., max, min, ..., min]
    // shared_max: 指向共享内存的开头
    T* shared_max = reinterpret_cast<T*>(shared_max_min_memory);
    
    // shared_min: 指向共享内存+offset, offset: block中threads的数量
    T* shared_min = shared_max + blockDim.x;
    int total_thread_num = blockDim.x * gridDim.x;
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 便于数据更新, max和min的初始值设置为最小值和最大值
    shared_max[tid] = FLT_MIN;
    shared_min[tid] = FLT_MAX;

    // 1. 赋值, 数据传入共享内存
    // block数量可能无法覆盖总数据量 -> 循环赋值
    // 这个for循环, 主要是为了赋值, 而不是真正的maxmin, 所以不需要同步, 仅需要保证数据不丢失(124: __syncthreads();)
    for (int i = gid; i < nums; i += total_thread_num) {
        T val = input_ptr[i];
        shared_max[tid] = max(val, shared_max[tid]);
        shared_min[tid] = min(val, shared_min[tid]);
    }
    __syncthreads();

    // 2. 每个block内部进行比较(inter-block)
    // 归约(类似my_softmax中的归约): 
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

    // 3. 每个block的smem[0]保存了block内的max和min
    if (tid == 0) {
        atomicMax(max_ptr, shared_max[0]);
        atomicMin(min_ptr, shared_min[0]);
    }
}

template <typename T>
__global__ void GetScaleAndZPAsymPerTensor(const T* max_ptr, const T* min_ptr, const int nums,
                    const double quantization_bit, T* scale, T* zero_point)
{
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
 
    while(gid < nums) {
        T denominator = static_cast<T>(pow(2.0, quantization_bit) - 1);
        T max = max_ptr[gid];

        // 为什么是负的?
        T min = -min_ptr[gid];



        T s = (max - min) / denominator;
        scale[gid] = s;
        zero_point[gid] = -1 * gpunearbyint(min / s);
        gid += blockDim.x * gridDim.x;
    }
}

template <typename T>
__global__ void QuantizePerTensorSym(const T* in_ptr, const T* scale_ptr,
                    const int nums, const double quantization_bit, T* out_ptr, const int channel, const int HW)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    T upper_bound = static_cast<T>(pow(2.0, quantization_bit - 1) - 1);
    T lower_bound = -upper_bound - 1;
    T scale = *scale_ptr;
    if (gid == 0)
        printf("scale: %f\n", scale);

    while (gid < nums) {
        T out = gpunearbyint(in_ptr[gid] / scale);
        out = out > upper_bound ? upper_bound : out;
        out = out < lower_bound ? lower_bound : out;
        out_ptr[gid] = out;

        gid += blockDim.x * gridDim.x;
    }
}

template <typename T>
__global__ void QuantizePerTensorAsym(const T* in_ptr, const T* scale_ptr, const T* zero_point_ptr,
                    const int nums, const double quantization_bit, T* out_ptr, const int channel, const int HW)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // 由于是Asym, 所以上下界不同, 量化后范围为[0, 255]
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


// PerChannel
template <typename T>
__global__ void GetMaxMinPerChannel(const T* input_ptr, const int nums,
                    T* max_ptr, T* min_ptr, const int channel, const int HW)
{
    extern __shared__ unsigned char shared_max_min_memory[];
    T* shared_max = reinterpret_cast<T*>(shared_max_min_memory);
    T* shared_min = shared_max + blockDim.x;
    
    // 一个block处理一个channel
    int cur_channel = blockIdx.x;
    int tid = threadIdx.x;

    while (cur_channel < channel) {
        shared_max[tid] = FLT_MIN;
        shared_min[tid] = FLT_MAX;

        int index = (HW * cur_channel) + tid;
        int end = HW * (cur_channel + 1);

        while (index < end && index < nums) {
            shared_max[tid] = max(input_ptr[index], shared_max[tid]);
            shared_min[tid] = min(input_ptr[index], shared_min[tid]);
            index += blockDim.x;
        }
        __syncthreads();

        // 归约求min和max
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_max[tid] = max(shared_max[tid], shared_max[tid + s]);
                shared_min[tid] = min(shared_min[tid], shared_min[tid + s]);
            }
            __syncthreads();
        }

        if (tid == 0) {
            atomicMax(max_ptr + cur_channel, shared_max[0]);
            atomicMin(min_ptr + cur_channel, shared_min[0]);
        }

        cur_channel += gridDim.x;
    }
}

template <typename T>
__global__ void QuantizePerChannelSym(const T* in_ptr, const T* scale_ptr, const int nums,
                    const double quantization_bit, T* out_ptr, const int scale_size, const int HW)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    T upper_bound = static_cast<T>(pow(2.0, quantization_bit - 1) - 1);
    T lower_bound = -upper_bound - 1;

    if (gid == 0)
    for (int i = 0; i < scale_size; i += 50)
        printf("scale: %f\n", scale_ptr[i]);

    while (gid < nums) {
        int channel_index = gid / HW;
        int scale_idx = min(scale_size - 1, channel_index);
        T scale = scale_ptr[scale_idx];

        T out = gpunearbyint(in_ptr[gid] / scale);
        out = out > upper_bound ? upper_bound : out;
        out = out < lower_bound ? lower_bound : out;
        out_ptr[gid] = out;

        gid += blockDim.x * gridDim.x;
    }
}


#define LAUNCH_GPU_KERNEL(GetMinMaxFunc, QuantFunc, scale_size, channel, HW) \
    cudaMalloc((void **)&d_scale, scale_size * sizeof(float)); \
    cudaMalloc((void **)&d_zeropoint, scale_size * sizeof(float)); \
    cudaMalloc((void **)&d_max, scale_size * sizeof(float)); \
    cudaMalloc((void **)&d_min, scale_size * sizeof(float)); \
    cudaEvent_t start, stop; \
    cudaEventCreate(&start); \
    cudaEventCreate(&stop); \
    cudaEventRecord(start); \
    GetMinMaxFunc<float><<<gridSize, blockSize, blockSize * 2 * sizeof(float), 0>>>(d_input, nums, d_max, d_min, channel, HW);  \
    GetScaleAndZPSym<float><<<1, blockSize>>>(d_max, d_min, channel, quantization_bit, d_scale, d_zeropoint); \
    QuantFunc<float><<<gridSize, blockSize>>>(d_input, d_scale, nums, quantization_bit, d_output, channel, HW); \
    cudaEventRecord(stop); \
    cudaEventSynchronize(stop); \
    cudaEventElapsedTime(&milliseconds, start, stop);

int main() {

    float milliseconds = 0;
    constexpr int nums              = 400 * 20 * 10;
    constexpr int HW                = 20 * 10;
    constexpr int channel           = 400;
    constexpr int quantization_bit  = 8;
    float* input = (float*) malloc(sizeof(float) * nums);
    float cpu_min = FLT_MAX;
    float cpu_max = FLT_MIN;
    for(int i = 0; i < nums; i++) {
        
        // 生成随机数, 范围为[-5, 5]
        input[i] = -5 + static_cast <float> (rand()) / ( static_cast <float> (RAND_MAX / 10));
        cpu_min = std::min(input[i], cpu_min);
        cpu_max = std::max(input[i], cpu_max);
    }

    float* output = (float*) malloc(sizeof(float) * nums);
    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, nums * sizeof(float));
    cudaMalloc((void **)&d_output, nums * sizeof(float));
    cudaMemcpy(d_input, input, sizeof(float) * nums, cudaMemcpyHostToDevice);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int maxblocks = deviceProp.maxGridSize[0];
    int blockSize = 256;
    int gridSize = std::min<int>((nums + blockSize - 1) / blockSize,  std::min<int>(maxblocks, channel));
    float *d_scale, *d_zeropoint, *d_max, *d_min;

    bool per_tensor_quantize = false;
    if(per_tensor_quantize) {

        // switch to per tensor
        LAUNCH_GPU_KERNEL(GetMaxMinPerTensor, QuantizePerTensorSym, 1, nums, HW);
    } else {

        // switch to per channel
        LAUNCH_GPU_KERNEL(GetMaxMinPerChannel, QuantizePerChannelSym, channel, channel, HW);
    }

    cudaMemcpy(output, d_output, sizeof(float) * nums, cudaMemcpyDeviceToHost);

    float* CPUOutput= (float*) malloc(sizeof(float) * nums);

    if (per_tensor_quantize) {
        float* scale = (float*) malloc(sizeof(float) * 1);
        float* zeropoint = (float*) malloc(sizeof(float) * 1);
        GenScalePerTensorSymCPU<float>(input, quantization_bit, nums, scale, zeropoint);
        QuantizationPerTensorSymCPU<float>(input, *scale, quantization_bit, nums, CPUOutput);
        free(scale);
        free(zeropoint);
    } else {
        float* scale = (float*) malloc(sizeof(float) * channel);
        float* zeropoint = (float*) malloc(sizeof(float) * channel);
        GenScalePerChannelSymCPU<float>(input, quantization_bit, HW, channel, nums, scale, zeropoint);
        QuantizationPerChannelSymCPU<float>(input, scale, quantization_bit, HW, nums, CPUOutput);
        free(scale);
        free(zeropoint);
    }
    if (CheckResult(output, CPUOutput, nums)) {
        printf("the ans is right\n");
        printf("Quantize kernel latency = %f ms\n", milliseconds);
    } else {
        printf("the ans is wrong\n");
        printf("first two CPUoutput are %f, %f\n", CPUOutput[0], CPUOutput[1]);
        printf("first two output are %f, %f\n", output[0], output[1]);
    }

    free(input);
    free(output);
    free(CPUOutput);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_scale);
    cudaFree(d_zeropoint);
    cudaFree(d_max);
    cudaFree(d_min);
}