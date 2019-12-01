// cuBLAS matrix multiplication - lab 6 
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

// Allocates a matrix with random float entries.
void randomInit(float *data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

double cpuTimer()
{
	struct timeval clock;
	gettimeofday(&clock, NULL);
	return ((double)clock.tv_sec + (double)clock.tv_usec * 1e-6);
}

int main() {

	int rows_A, cols_A, rows_B, cols_B;
	rows_A = cols_A = rows_B = cols_B = 1 << 10;

	// (rows_A x cols_A) * (rows_B x cols_B) = (rows_A x cols_B) && cols_A == rows_B
	if( cols_A != rows_B )
    {
       printf("ERROR: Matrix sizes do not match!\n");
       exit(-1);
    }

	int rows_C = rows_A;
	int cols_C = cols_B;


	// Problem size
	int size_A = rows_A * cols_A;
	int size_B = rows_B * cols_B;
	int size_C = rows_C * cols_C;
	size_t bytes_A = size_A * sizeof(float);
	size_t bytes_B = size_B * sizeof(float);
	size_t bytes_C = size_C * sizeof(float);

	// Pointers to host and device
	float *h_a, *h_b, *h_c;
	float *d_a, *d_b, *d_c;

	// Allocate memory for hosts
	h_a = (float*)malloc(bytes_A);
	h_b = (float*)malloc(bytes_B);
	h_c = (float*)malloc(bytes_C);

	// Alocate memory for devices
	cudaMalloc(&d_a, bytes_A);
	cudaMalloc(&d_b, bytes_B);
	cudaMalloc(&d_c, bytes_C);

	// set seed for rand()
    srand(2006);

    // initialize host memory
    randomInit(h_a, size_A);
    randomInit(h_b, size_B);

	// cudaMemcpy from host to device
	cudaMemcpy(h_a, d_a, bytes_A, cudaMemcpyHostToDevice);
	cudaMemcpy(h_b, d_b, bytes_B, cudaMemcpyHostToDevice);
	cudaMemcpy(h_c, d_c, bytes_C, cudaMemcpyHostToDevice);

	// cuBLAS handle
	cublasHandle_t handle;
	cublasCreate(&handle);

	// Scalaing factors
	float alpha = 1.0f;
	float beta = 0.0f;

	double ti = cpuTimer();
	// c = (alpha*a) * b + (beta*c)
	// (m X n) * (n X k) = (m X k)    || (rows_A x cols_A) * (rows_B x cols_B) = (rows_A x cols_B)
	// cublasSgemm(handle, operation, operation, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
	// lda = rows_A = ldc 		ldb = cols_B
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, rows_A, cols_A, cols_B, &alpha, d_a, rows_A, d_b, cols_B, &beta, d_c, rows_A);
	double elapsed = cpuTimer - ti;

	printf("The time of matrix multiplication using cuBLAS library is equal to: %.6f", elapsed);
	
	// Destroying the handle
	cublasDestroy(handle)
	
	// Copying back the three matrices is optionall. For example if we want to print the result.
	cudaMemcpy(h_a, d_a, bytes_A, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_b, d_b, bytes_B, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_c, d_c, bytes_C, cudaMemcpyDeviceToHost);


	//Free GPU memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	// Free CPU memory
	free(h_a);
	free(h_b);
	free(h_c);


	return 0;
}

