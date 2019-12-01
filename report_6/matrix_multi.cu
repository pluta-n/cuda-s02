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

#include <helper_cuda.h>


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

        // Calculate the row index of the P element and M
                int Row = blockIdx.y*blockDim.y+threadIdx.y;

        // Calculate the column index of P and N
                int Col = blockIdx.x*blockDim.x+threadIdx.x;
                        if ((Row < Width) && (Col < Width)) {
                                float Pvalue = 0;

        // each thread computes one element of the block sub-matrix
                                for (int k = 0; k < Width; ++k) {
                                        Pvalue += M[Row*Width+k]*N[k*Width+Col];
                                }
                        P[Row*Width+Col] = Pvalue;
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

  //stuff to register elapsed time
  float el_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);

  cudaError_t addVectorsErr;
  cudaError_t asyncErr;

  MatrixMulKernel<<<1, 1>>>(a,b, c, N);

  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);


  cudaEventElapsedTime(&el_time, start, stop);
  printf("Time elapsed on single-threaded matrix multiplication: %f", el_time);
 
 //multi
  float el_time = 0.0f;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);

  cudaError_t addVectorsErr;
  cudaError_t asyncErr;

  MatrixMulKernel<<<numberOfBlocks, threadsPerBlock>>>(a,b, c, N);

  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);


  cudaEventElapsedTime(&el_time, start, stop);
  printf("Time elapsed on multi-threaded matrix multiplication: %f", el_time);
  
  
  addVectorsErr = cudaGetLastError();
  if(addVectorsErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(addVectorsErr));

  asyncErr = cudaDeviceSynchronize();
  if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}


