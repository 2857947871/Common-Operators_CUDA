# include <math.h>
# include <vector>
# include <stdio.h>
# include <iostream>

# include <cuda.h>
# include <cuda_runtime.h>
# include <device_launch_parameters.h>

using namespace std;


__global__ void add_gpu(float* c, float* a, float* b, int N) {
    
    int global_tid = blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x) + threadIdx.x;

    if (global_tid < N)
        c[global_tid] = a[global_tid] + b[global_tid];

}


void add_cpu(float* c, float* a, float* b, int N) {
    
    for (int i = 0; i < N; i++)
        c[i] = a[i] + b[i];
}

int main() {

    int N = 10000;
    int bytes = N * sizeof(int);
    float milliseconds = 0;
    
    // thread: 1-D
    int thread = 128;

    // block: 1-D
    // threads: 40 * 256 = 10240
    // int block = (N + thread - 1) / thread;

    // block: 2-D
    // threads: 7 * 7 * 256 = 12544
    int block = ceil(sqrt((N + thread - 1.) / thread)); // ceil: 向上取整 -> 7
    dim3 grid(block, block);

    float* host_x = nullptr;
    float* host_y = nullptr;
    float* host_z = nullptr;
    float* device_x = nullptr;
    float* device_y = nullptr;
    float* device_z = nullptr;

    // allocate memory on the host
    host_x = (float* )malloc(bytes);
    host_y = (float* )malloc(bytes);
    host_z = (float* )malloc(bytes);

    // allocate memory on the device
    cudaMalloc((float** )&device_x, bytes);
    cudaMalloc((float** )&device_y, bytes);
    cudaMalloc((float** )&device_z, bytes);

    // initialize the host memory
    for (int i = 0; i < N; ++i) {

        host_x[i] = i;
        host_y[i] = i;
    }

    // host -> device
    cudaMemcpy(device_x, host_x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_y, host_y, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_z, host_z, bytes, cudaMemcpyHostToDevice);

    // 处理数据
    // grid: 7 * 7, block: 256
    // block_num = 7 * 7 -> 49   thread_num = 256
    cudaEvent_t start, stop; // 记时
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    add_gpu<<<grid, thread>>>(device_z, device_x, device_y, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU time: %d\n", milliseconds);
    printf("Mem BW: %f GB/sec\n", 3 * N * sizeof(float) / milliseconds / 1e6);

    // 同步等待所有 thread 结束 -> 因为上述计时操作 -> 不再需要显式调用同步函数
    // cudaDeviceSynchronize();

    // device -> host
    cudaMemcpy(host_z, device_z, bytes, cudaMemcpyDeviceToHost);

    // check the result
    vector<float> host_z_cpu(N);
    vector<float> host_z_gpu(N);

    // 此时 host_z_gpu 中存储的是 device 计算的结果
    for (int i = 0; i < N; ++i) 
        host_z_gpu.push_back(host_z[i]);

    // 此时 host_z_cpu 中存储的是 host 计算的结果
    add_cpu(host_z, host_x, host_y, N);    
    for (int i = 0; i < N; ++i) 
        host_z_cpu.push_back(host_z[i]);

    for (int i = 0; i < N; ++i) 
        if (abs(host_z_cpu[i] - host_z_gpu[i]) > 1e-5) {
            cout << "Error: " << i << " " << host_z_cpu[i] << " " << host_z_gpu[i] << endl;
            break;
        }


    // free
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_z);
    free(host_x);
    free(host_y);
    free(host_z);

    return 0;
}