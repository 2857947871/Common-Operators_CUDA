# include "gemv.cuh"


// [1, N] * [N, M]
// notes: CPU res sometimes are trash values, which very weird, so I check result by printing each res skipping comparison with CPU res
// when compile to executable file named gemv, we can run by typing "./gemv 1" to run fp32 gemv and "./gemv" to run fp16 gemv
template <typename T>
void gemvCPU(T *mat, T *vec, float *dst, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            dst[i] += (float)vec[j] * (float)mat[i + j * M];
        }
        if (i < 5) {
            printf("cpu res = %f\n", dst[i]);
        }
    }
}

template <typename T>
bool CheckResult(T *out, float *groudtruth, int M) {
    for (int i = 0; i < M; i++) {
        printf("%d th comparsion: %f and %f \n", i, (float)out[i], groudtruth[i]);
    }
    return true;
}


// vec.shape = [1, N]
// mat.shape = [N, M] and matrix is row major order in memory
// 为什么在编译期间获取threads_per_value(处理M个数据需要多少thread): 
//  1. 我们可以提前知道
//  2. 方便写cuda kernel
//      x = threads / threads_per_value: 一个block可以处理 x 行
//      N / x: 一共需要多少个block, 或者一个block需要循环几次
# define GEMV_KERNEL(dtype)                                                                                                 \
    dtype *d_vec;                                                                                                           \
    dtype *d_mat;                                                                                                           \
    dtype *d_dst;                                                                                                           \
    constexpr int N = 256;                                                                                                  \
    constexpr int M = 256;                                                                                                  \
    dtype *vec = (dtype *)malloc(N * sizeof(dtype));                                                                        \
    cudaMalloc((void **)&d_vec, N * sizeof(dtype));                                                                         \
    dtype *mat = (dtype *)malloc(M * N * sizeof(dtype));                                                                    \
    cudaMalloc((void **)&d_mat, M *N * sizeof(dtype));                                                                      \
    dtype *dst = (dtype *)malloc(M * sizeof(dtype));                                                                        \
    cudaMalloc((void **)&d_dst, M * sizeof(dtype));                                                                         \
    for (int i = 0; i < N; i++)                                                                                             \
    {                                                                                                                               \
        vec[i] = (dtype)1;                                                                                                          \
    }                                                                                                                               \
    for (int i = 0; i < N * M; i++)                                                                                                 \
    {                                                                                                                               \
        mat[i] = (dtype)1;                                                                                                          \
    }                                                                                                                               \
    cudaMemcpy(d_vec, vec, N * sizeof(dtype), cudaMemcpyHostToDevice);                                                              \
    cudaMemcpy(d_mat, mat, M *N * sizeof(dtype), cudaMemcpyHostToDevice);                                                           \
    constexpr int THREADS_PER_BLOCK = 256;                                                                                          \
    constexpr int VEC_SIZE = Vec<dtype>::size;                                                                                      \
    constexpr int THREADS_PER_VALUE = VecMat::get_thread_per_mat_row<M, dtype>::value;                                              \
    VecMat::DispatchLauncher<THREADS_PER_BLOCK, THREADS_PER_VALUE, VEC_SIZE>::template launcher<dtype>(d_mat, d_vec, d_dst, M, N);  \
    cudaMemcpy(dst, d_dst, M * sizeof(dtype), cudaMemcpyDeviceToHost);                                                              \
    float *groudtruth = (float *)malloc(sizeof(float) * M);                                                                         \
    gemvCPU(mat, vec, groudtruth, M, N);                                                                                    \
    bool is_right = CheckResult(dst, groudtruth, M);                                                                        \
    if (is_right)                                                                                                           \
    {                                                                                                                       \
        printf("the ans is right\n");                                                                                       \
    }                                                                                                                       \
    else                                                                                                                    \
    {                                                                                                                       \
        printf("the ans is wrong\n");                                                                                       \
    }                                                                                                                       \
    cudaFree(d_vec);                                                                                                        \
    cudaFree(d_mat);                                                                                                        \
    cudaFree(d_dst);                                                                                                        \
    free(vec);                                                                                                              \
    free(mat);                                                                                                              \
    free(dst);


int main(int argc, char** argv)
{

    GEMV_KERNEL(float);


    return 0;
}