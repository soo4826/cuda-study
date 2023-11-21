// Reference: https://www.youtube.com/watch?v=8sDg-lD1fZQ
// Tutorial for GPGPU Programming with NVIDIA CUDA

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <chrono>

// __global__: GPU에서 사용하는 함수임을 명시
__global__ void vectorAdd(int* a, int* b, int* c)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];

    return;
}

int main(){
    
    // CPU Calculation
    int a[] = { 1,2,3 };
    int b[] = { 4,5,6 };
    int c[sizeof(a) / sizeof(int)] = {0};


    auto start_cpu = std::chrono::high_resolution_clock::now();
    for (int i = 0 ; i < sizeof(c) / sizeof(int); i++){
        c[i] = a[i] + b[i];
        // std::cout<<c[i]<<std::endl;
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_cpu = end_cpu - start_cpu;

    std::cout << "CPU Execution Time: " << duration_cpu.count()/1000 << " ms" << std::endl;

    

    // GPU Calculation

    // create pointers into the gpu
    int *cudaA = 0;
    int *cudaB = 0;
    int *cudaC = 0;

    // allocate memory in the gpu
    cudaMalloc(&cudaA, sizeof(a));
    cudaMalloc(&cudaB, sizeof(b));
    cudaMalloc(&cudaC, sizeof(c));

    // copy the vectors into the gpu
    cudaMemcpy(cudaA, a, sizeof(a), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaB, b, sizeof(b), cudaMemcpyHostToDevice);
    
    auto start_gpu = std::chrono::high_resolution_clock::now();
    // Syntax: vectorAdd<<<GRID_SIZE, BLOCK_SIZE>>> 
    vectorAdd <<< 1, sizeof(a) / sizeof(int) >>> (cudaA, cudaB, cudaC);
    // <<<>>> Special syntex
    // Grid size: # of blocks which i have --> 현재는 연산수준이 작기때문에 1로 설정
    //            # of threads exists per block??
    // Block size: # of vectors --> 여기서는 a 배열의 크기
    
    cudaMemcpy(c, cudaC, sizeof(c), cudaMemcpyDeviceToHost);

    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_gpu = end_gpu - start_gpu;

    std::cout << "GPU Execution Time: " << duration_gpu.count()/1000 << " ms" << std::endl;

    // for (int i = 0 ; i < sizeof(c) / sizeof(int); i++){
    //     std::cout<<c[i]<<std::endl;
    // }
    
    
    return 1;
}

/*
* Questions
1) What is cuda block grid thread framework?
  - thread 인덱스를 기반하여 요소에 접근
  - 즉, vectorAdd 함수가 여러 개의 vector 에 대해 동시에 호출된다는 의미임!
  - 이를 고유하게 구분하기 위해서 thread idx를 사용함!!


*/