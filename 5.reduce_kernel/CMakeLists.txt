cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(CUDATutorial LANGUAGES CXX CUDA)

find_package(CUDA 11.8 REQUIRED)

# 需下载安装cuda toolkit，直接搜即可
set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})

# set compiler flags
set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -Xcompiler -Wall")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  \
                        -gencode=arch=compute_70,code=\\\"sm_70,compute_70\\\" \
                        -gencode=arch=compute_75,code=\\\"sm_75,compute_75\\\" \
                        -gencode=arch=compute_80,code=\\\"sm_80,compute_80\\\" \
                        -gencode=arch=compute_86,code=\\\"sm_86,compute_86\\\" \
                        ")
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)

#输出的可执行文件保存地址
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/app)
    
set(COMMON_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/15_gemv
    ${CUDA_PATH}/include
)

set(COMMON_LIB_DIRS
    ${CUDA_PATH}/lib64
)
#指定头文件路径
include_directories(
    ${COMMON_HEADER_DIRS}
)
#指定待链接的lib路径
link_directories(
    ${COMMON_LIB_DIRS}
)


add_executable(baseline
    baseline.cu
)

# 0
add_executable(reduce_v0
    reduce_v0.cu
)

# 1
add_executable(reduce_v1
    reduce_v1.cu
)

# 1-old
add_executable(reduce_v1_old
    reduce_v1_old.cu
)

# 2
add_executable(reduce_v2
    reduce_v2.cu
)

# 3
add_executable(reduce_v3
    reduce_v3.cu
)

# 4
add_executable(reduce_v4
    reduce_v4.cu
)

# 5
add_executable(reduce_v5
    reduce_v5.cu
)

# 6
add_executable(reduce_v6
    reduce_v6.cu
)

# 7
add_executable(reduce_v7
    reduce_v7.cu
)

# 8
add_executable(reduce_v8
    reduce_v8.cu
)