# include <cuda.h>
# include "cuda_runtime.h"
# include "cooperative_groups.h"
# include <bits/stdc++.h>

// naive: 0.208416 ms
__global__ void filter_k(int *dst, int *nres, const int *src, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n && src[tid] > 0)
        dst[atomicAdd(nres, 1)] = src[tid];
}

// blcok level: 0.171200 ms
// 先在 shared memory 中计算每个 block 内的大于 0 的数量, 然后再将这个数量累加到全局计数器中
__global__ void filter_block_k(int *dst, int *nres, const int *src, int n) {

    // l_n: shared_memory, 每个 block 所有线程共享, 不同 block 之间不共享
    __shared__ int l_n; // 计数器
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_thread_num = blockDim.x * gridDim.x;

    // block_level: step: total_thread_num
    for (int i = gtid; i < n; i += total_thread_num) {
        
        // 对每个 block 的 l_n 进行初始化
        if (threadIdx.x == 0)
            l_n = 0;
        __syncthreads();

        // l_n: 每个 block 中大于 0 的元素的数量
        // pos: 每个线程私有的寄存器, 作为 atomicAdd 的返回值, 表示当前线程对 l_n 原子加 1 之前的 l_n
        int pos;
        if (i < n && src[i] > 0)
            // block1: 大于 0: 1 2 4 号线程, l_n = 3
            //  pos: 0 1 2
            // block2: 大于 0: 1 3 号线程, l_n = 2
            //  pos: 0 1
            // block3: 大于 0: 0 2 4 5 号线程, l_n = 4
            //  pos: 0 1 2 3
            pos = atomicAdd(&l_n, 1);
        __syncthreads();

        // block 中的 tid = 0 作为 leader
        // l_n: 每个 block 中大于 0 的元素的数量
        // 此时, l_n 为全局计数器, 各个 block 的 l_n 的累和
        if (threadIdx.x == 0)

            // block1: l_n: 0
            // block2: l_n: 3
            // block3: l_n: 5
            l_n = atomicAdd(nres, l_n);
        __syncthreads();

        // write && store
        // pos: 每个线程私有的, src[thread] > 0 的 thread 在当前 block 的 index
        if (i < n && src[i] > 0) {
            
            // block1:
            // l_n(同一 block 共享): 0
            //  i: 1 2 4
            //  pos(每个 thread 独享): 0 1 2 -> 0 1 2
            // block2:
            // l_n(同一 block 共享): 2
            //  i: 1 3
            //  pos(每个 thread 独享): 0 1 -> 2 3
            // block3:
            // l_n(同一 block 共享): 4
            //  i: 0 2 4 5
            //  pos(每个 thread 独享): 0 1 2 3 -> 4 5 6 7
            pos += l_n;
            dst[pos] = src[i];
        }
        __syncthreads();
    }
}

// warp level: 0.200640 ms
__device__ int atomicAggInc(int *ctr) {
    unsigned int active = __activemask();
    int leader = __ffs(active) - 1; // 视频所示代码这里有误，leader应该表示warp里面第一个src[threadIdx.x]>0的threadIdx.x
    int change = __popc(active);
    int lane_mask_lt;
    asm("mov.u32 %0, %%lanemask_lt;" : "=r"(lane_mask_lt));
    unsigned int rank = __popc(active & lane_mask_lt); // 比当前线程id小且值为1的mask之和
    int warp_res;
    if(rank == 0)
    warp_res = atomicAdd(ctr, change);
    warp_res = __shfl_sync(active, warp_res, leader);
    return warp_res + rank;
}

__global__ void filter_warp_k(int *dst, int *nres, const int *src, int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if(i >= n)
    return;
  if(src[i] > 0)
    dst[atomicAggInc(nres)] = src[i];
}


bool CheckResult(int *out, int groudtruth, int n);
int main() {

    // init
    float ms = 0;
    int N = 2560000;

    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    const int blockSize = 256;
    int GridSize = std::min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]);
    dim3 Grid(GridSize);
    dim3 Block(blockSize);

    int* src_host = (int* )malloc(N * sizeof(int));
    int* dst_host = (int* )malloc(N * sizeof(int));
    int* nres_host = (int* )malloc(1 * sizeof(int)); // 计数器
    int* dst_device;
    int* src_device;
    int* nres_device;
    cudaMalloc((void** )&src_device, N * sizeof(int));
    cudaMalloc((void** )&dst_device, N * sizeof(int));
    cudaMalloc((void** )&nres_device, 1 * sizeof(int));

    for (int i = 0; i < N; i++) {
        if (i % 2 == 0) {
            src_host[i] = -1;
        } else {
            src_host[i] = 1;
        }
    }

    int groudtruth = 0;
    for (int i = 0; i < N; i++) {
        if (src_host[i] > 0) {
            groudtruth += 1;
        }
    }   

    cudaMemcpy(src_device, src_host, N * sizeof(int), cudaMemcpyHostToDevice);

    // operation
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    filter_warp_k<<<Grid, Block>>>(dst_device, nres_device, src_device, N);
    // filter_k<<<Grid, Block>>>(dst_device, nres_device, src_device, N);
    // filter_block_k<<<Grid, Block>>>(dst_device, nres_device, src_device, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(nres_host, nres_device, 1 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(dst_host, dst_device, N * sizeof(int), cudaMemcpyDeviceToHost);

    // val
    // for (int i = 0; i < N; ++i)
    //     if (dst_host[i] == 1)
    //         printf("%d ", i);
    // printf("\n");

    bool is_right = CheckResult(nres_host, groudtruth, N);
    if(is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        printf("%lf ",*nres_host);
        printf("\n");
    }
    printf("filter_k latency = %f ms\n", ms);

    // free
    cudaFree(src_device);
    cudaFree(dst_device);
    cudaFree(nres_device);
    free(src_host);
    free(dst_host);
    free(nres_host);  
}


bool CheckResult(int *out, int groudtruth, int n) {
    if (*out != groudtruth) {
        return false;
    }
    return true;
}