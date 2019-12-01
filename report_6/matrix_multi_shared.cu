/*
 * matrix_mult.cu
 *
 *  Created on: Nov 14, 2019
 *      Author: cuda-s18
 */

#include <stdio.h>

#include <assert.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <stdlib.h>
#include <helper_cuda.h>
#include <sys/time.h>
#include <time.h>


double cpuTimer()
{
	struct timeval clock;
	gettimeofday(&clock, NULL);
	return ((double)clock.tv_sec + (double)clock.tv_usec * 1e-6);
}

void initWith(float *a, int N)
{
  for(int i = 0; i < N; ++i)
  {
          for( int j = 0; j < N; ++j){
                a[i*N+j] = (i+j);
          }
  }
}
__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width) {

    __shared__ int tile_M[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int tile_N[BLOCK_SIZE][BLOCK_SIZE];

    int Row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int Col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int tmp = 0;
    int idx;

    for (int i = 0; i < gridDim.x; ++i) 
    {
        idx = Row * Width + i * BLOCK_SIZE + threadIdx.x;
        if(idx >= Width*Width)
        {
            // Width may not divisible by BLOCK_SIZE
            tile_M[threadIdx.y][threadIdx.x] = 0;
        }
        else 
        {
            tile_M[threadIdx.y][threadIdx.x] = M[idx];
        }

        idx = (i * BLOCK_SIZE + threadIdx.y) * Width + Col;
        if(idx >= Width*Width)
        {
            tile_N[threadIdx.y][threadIdx.x] = 0;
        }  
        else
        {
            tile_N[threadIdx.y][threadIdx.x] = N[idx];
        }
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) 
        {
            tmp += tile_M[threadIdx.y][k] * tile_N[k][threadIdx.x];
        }
        __syncthreads();
    }
    if(Row < Width && Col < Width)
    {
        P[Row * Width + Col] = tmp;
    }
}


int main()
{
  const int N = 2<<8;
  size_t size = N * N * sizeof(float);

  float *a;
  float *b;
  float *c;

  cudaMallocManaged(&a, size);
  cudaMallocManaged(&b, size);
  cudaMallocManaged(&c, size);

  int deviceId;
  cudaGetDevice(&deviceId);

  cudaMemPrefetchAsync(a, size, deviceId);
  cudaMemPrefetchAsync(a, size, cudaCpuDeviceId);
  cudaMemPrefetchAsync(b, size, deviceId);
  cudaMemPrefetchAsync(b, size, cudaCpuDeviceId);
  cudaMemPrefetchAsync(c, size, deviceId);
  cudaMemPrefetchAsync(c, size, cudaCpuDeviceId);

  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, deviceId);

  int multiProcessorCount = props.multiProcessorCount;

  size_t threadsPerBlock;
  size_t numberOfBlocks;

  threadsPerBlock = 1024;
  numberOfBlocks = multiProcessorCount/10;

  initWith(a, N);
  initWith(b, N);

  cudaError_t addVectorsErr;
  cudaError_t asyncErr;

  double ti = cpuTimer();
  MatrixMulKernel<<<numberOfBlocks, threadsPerBlock>>>(a,b, c, N);
  double elapsed = cpuTimer - ti;
  printf("The time of matrix multiplication is equal to: %.6f", elapsed);
  
  addVectorsErr = cudaGetLastError();
  if(addVectorsErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(addVectorsErr));

  asyncErr = cudaDeviceSynchronize();
  if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}


