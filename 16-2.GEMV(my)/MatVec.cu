# include "gemv.cuh"


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
	// 全1矩阵和全1向量
    // [256, 2048] * [2048, 1] -> [256, 1]
	constexpr int M = 256;  // 行
    constexpr int N = 2048; // 列

    // 定义向量
	vec = (T* )malloc(N * sizeof(T));
	cudaMalloc((void** )&d_vec, N * sizeof(T));
	for (int i = 0; i < N; ++i) {
		vec[i] = (T)1.0;
	}
	cudaMemcpy(d_vec, vec, N * sizeof(T), cudaMemcpyHostToDevice);

	// 定义矩阵
	mat = (T* )malloc(N * M * sizeof(T));
	cudaMalloc((void** )&d_mat, M * N * sizeof(T));
	for (int i = 0; i < N * M; ++i) {
		mat[i] = (T)1.0;
	}
	cudaMemcpy(d_mat, mat, M * N * sizeof(T), cudaMemcpyHostToDevice);
	
	// 定义结果
	dst = (T* )malloc(M * sizeof(T));
	cudaMalloc((void** )&d_dst, M * sizeof(T));

	// dispatch
	// blocks num: M   threads num: 256
	// 行 * 列, 矩阵为 row major 在内存中存储, 
	// 每一行分配一个 block, 256个 threads 处理这一行 -> M个block, 256个thread
	constexpr int VEC_SIZE = Vec<T>::size; // float: 4   half: 8
	constexpr int THREAD_NUMS = 256;

	// 每个 thread 分配多少向量(向量化读取的向量, 不是与mat相乘的向量)
	// 一个 block 处理一行, 一行 N 个元素, 每次读取 VEC_SIZE 个元素, THREAD_NUMS 个thread进行读取
	//	eg: float N = 2048   VECS_PER_THREAD = 2048 / 4 / 256 = 2
	//		每个 thread 向量化读取 2 次
	//		最后并不是用for循环来match, 而是固定每个thread向量化读取的次数
	constexpr int VECS_PER_THREAD = (N / VEC_SIZE) / THREAD_NUMS;
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

// 仅有FP32的实现, FP16还在整理
int main() 
{
    float* vec      = nullptr;
    float* d_vec    = nullptr;
    float* mat      = nullptr;
    float* d_mat    = nullptr;
    float* dst      = nullptr;
    float* d_dst    = nullptr;

    gemv_kernel<float>(vec, d_vec, mat, d_mat, dst, d_dst);


    return 0;
}