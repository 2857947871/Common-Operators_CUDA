# include "16_gemv.cuh"


bool CheckResult(float *out, float* groudtruth, int M){
    for (int i = 0; i < M; i++){
      if(i == 0){
        printf("1st comparsion: %f and %f \n" , out[i], groudtruth[i] );
      }
      if (out[i] != groudtruth[i]) {
        printf("%dth res is wrong: %f and %f \n" , i, out[i], groudtruth[i] );
        return false;
      }
    }
    return true;
}

void gemvCPU(float* mat, float* vec, float* dst, int M, int N){
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            dst[i] +=  mat[i * N + j] * vec[j];
        }
        if (i < 5){
            printf("cpu res = %f\n", dst[i]);
        }
    }
}

template <typename T>
void gemv_kernel(T* vec, T* d_vec, T* mat, T* d_mat, T* dst, T* d_dst) {

    // 数据初始化
    // [256, 2048] * [2048, 1] -> [256, 1]
    constexpr int M = 256;  // 行
    constexpr int N = 2048; // 列

    // 定义向量
    vec = (T* )malloc(N * sizeof(T));
    cudaMalloc((void** )&d_vec, N * sizeof(T));
    for (int i = 0; i < N; i++) {
        vec[i] = (T)1.0;
    }
    cudaMemcpy(d_vec, vec, N * sizeof(T), cudaMemcpyHostToDevice);

    // 定义矩阵
    mat = (T* )malloc(M * N * sizeof(T));
    cudaMalloc((void** )&d_mat, M * N * sizeof(T));
    for (int i = 0; i < N * M; i++) {
        mat[i] = (T)1.0;
    }
    cudaMemcpy(d_mat, mat, M * N * sizeof(T), cudaMemcpyHostToDevice);

    // 定义结果
    dst = (T* )malloc(M * sizeof(T));
    cudaMalloc((void** )&d_dst, M * sizeof(T));

    // block nums = M, thread nums = 256
    // constexpr: 必须在编译期间进行计算(const没有这个限制), 模板参数需要编译时常量
    // template launcher: 调用类模板里面的函数模板, 要进行声明
    constexpr int VEC_SIZE = Vec<T>::size;
    constexpr int THREAD_NUMS = 256;
    constexpr int VECS_PER_THREAD = (N / VEC_SIZE) / THREAD_NUMS; // N / VEC_SIZE: 处理多少次向量, / THREAD_NUMS: 每个线程处理多少次向量
    DispatchLauncher<VECS_PER_THREAD, VEC_SIZE, THREAD_NUMS>::template launcher<T>(d_mat, d_vec, d_dst, M, N);
    CHECK(cudaMemcpy(dst, d_dst, M * sizeof(T), cudaMemcpyDeviceToHost));
    
    // check
    float* vec_cpu = (float* )malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        vec_cpu[i] = 1.0;
    }
    float* mat_cpu = (float* )malloc(M * N * sizeof(float));
    for (int i = 0; i < N * M; i++) {
        mat_cpu[i] = 1.0;
    }
    float* cpu_res = (float* )malloc(M * sizeof(float));
    gemvCPU(mat_cpu, vec_cpu, cpu_res, M, N);

    if (CheckResult((float* )dst, cpu_res, M)) {
        printf("right\n");
    } else {
        printf("wrong\n");
        for (int i = 0; i < 5; ++i) {
            printf("dst[%d]: %f ", i, (float)dst[i]);
        }
        printf("\n");
    }

    cudaFree(d_vec);
    cudaFree(d_mat);
    cudaFree(d_dst);
    free(vec);
    free(mat);
    free(dst);
    free(cpu_res);
}


int main() {

    if(false) {
        float* vec   = nullptr;
        float* d_vec = nullptr;
        float* mat   = nullptr;
        float* d_mat = nullptr;
        float* dst   = nullptr;
        float* d_dst = nullptr;
        gemv_kernel<float>(vec, d_vec, mat, d_mat, dst, d_dst);
    } else {
        half *vec   = nullptr;
        half *d_vec = nullptr;
        half *mat   = nullptr;
        half *d_mat = nullptr;
        half *dst   = nullptr;
        half *d_dst = nullptr;
        gemv_kernel<half>(vec, d_vec, mat, d_mat, dst, d_dst);
    }

    return 0;
}