#include <iostream>
#include <time.h>
#include <stdio.h>

// For the CUDA runtime routines
#include <cuda_runtime.h>

//initializing vectors with random numbers
void initVec(float *a, int N)
{
    for(int i = 0; i < N; ++i)
    {
        a[i] = rand()%100;
    }
}

//initializing vector with appointed number(probably useless)
void initWith(float num, float *a, int N)
{
    for(int i = 0; i < N; ++i)
    {
        a[i] = num;
    }
}

//dotting your vectors on cpu
void dotVectorsCPU(float result, float *a, float *b, int N)
{
    for(int i = 0; i < N; i++)
    {
        result = result + a[i] * b[i];
    }
}

//dotting your vectors on gpu
__global__
void dotVectorsGpu(float result, float *a, float *b, int N)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < N; i += stride)
    {
        result = result + a[i] * b[i];
    }
}

//adding your vectors on cpu
void addVectorsCPU(float *result, float *a, float *b, int N)
{
    for(int i = 0; i < N; i++)
    {
        result[i] = a[i] + b[i];
    }
}

//adding your vectors on gpu
__global__
void addVectorsGpu(float *result, float *a, float *b, int N)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < N; i += stride)
    {
        result[i] = a[i] + b[i];
    }
}

//subtracting your vectors on cpu
void subVectorsCPU(float *result, float *a, float *b, int N)
{
    for(int i = 0; i < N; i++)
    {
        result[i] = a[i] - b[i];
    }
}

//subtracting your vectors on gpu
__global__
void subVectorsGpu(float *result, float *a, float *b, int N)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < N; i += stride)
    {
        result[i] = a[i] - b[i];
    }
}

bool ask_repeat()
{
    char decision;
    while(1)
    {
        printf("\nDo You want to improve your score? (y/n)");
        printf("\n");
        scanf(" %c", &decision);
        if(decision == 'y' || decision == 'Y')
            return 1;
        else if(decision == 'n' || decision == 'N')
            return 0;
        else
            printf("\nWrong answer! Give y or n!");
    }
}
int main()
{

    //device's variables
    int deviceId;
    //int numberOfSMs;

    cudaGetDevice(&deviceId);
    //cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    //timing variables
    clock_t start, end;
    double cpu_time_used;
    float el_time;   // 1x1 kernel
    cudaEvent_t start_gpu, stop_gpu;

    //take value N of vectors' length
    int N;
    printf("Give the length of vectors you want to work with:");
    scanf("%d", &N);
    size_t size = N * sizeof(float);

    float *a;
    float *b;
    float c = 0.0f;
    float *d;

    //allocate data
    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    //cudaMallocManaged(&c, size);

    cudaMemPrefetchAsync(a, size, cudaCpuDeviceId);
    cudaMemPrefetchAsync(b, size, cudaCpuDeviceId);

    //initWith(0, c, N);

    //size_t threadsPerBlock;
    //size_t numberOfBlocks;

    //generate 2 vectors of length N
    initVec(a, N);
    initVec(b, N);

    cudaMemPrefetchAsync(a, size, deviceId);
    cudaMemPrefetchAsync(b, size, deviceId);

    //switch statement for different operations
    char operat;
    printf("Choose mathematical operations:\n");
    printf("Type '*' for dotting vectors\n");
    printf("Type '+' for adding vectors\n");
    printf("Type '-' for subtracting vectors\n");
    scanf(" %c", &operat);
    switch(operat)
    {
    case '*':
        //dot vec on cpu
        start = clock();
        dotVectorsCPU(c, a, b, N);

        //give time of execution
        end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("\nTime elapsed dotting vectors on CPU is: %f", cpu_time_used);

        //dot vecs on 1x1 kernel
        cudaEventCreate(&start_gpu);
        cudaEventCreate(&stop_gpu);
        cudaEventRecord(start_gpu, 0);

        c = 0;
        dotVectorsGpu<<<1,1>>>(c, a, b, N);

        //give time of execution
        cudaDeviceSynchronize();

        cudaEventRecord(stop_gpu,0);
        cudaEventSynchronize(stop_gpu);

        cudaEventElapsedTime(&el_time, start_gpu, stop_gpu);
        printf("\nTime elapsed on single-threaded vector dotting: %f \n", el_time);
        break;
    /////////
    case '+':
        cudaMallocManaged(&d, size);
        cudaMemPrefetchAsync(d, size, cudaCpuDeviceId);
        initWith(0,d,N);
        cudaMemPrefetchAsync(d, size, deviceId);
        //add vec on cpu
        start = clock();
        addVectorsCPU(d, a, b, N);

        //give time of execution
        end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("\nTime elapsed adding vectors on CPU is: %f", cpu_time_used);

        //add vecs on 1x1 kernel
        cudaEventCreate(&start_gpu);
        cudaEventCreate(&stop_gpu);
        cudaEventRecord(start_gpu, 0);

        initWith(0,d,N);
        addVectorsGpu<<<1,1>>>(d, a, b, N);

        //give time of execution
        cudaDeviceSynchronize();
        cudaEventRecord(stop_gpu,0);
        cudaEventSynchronize(stop_gpu);

        cudaEventElapsedTime(&el_time, start_gpu, stop_gpu);
        printf("\nTime elapsed on single-threaded vector addition: %f \n", el_time);
        break;
    /////////
    case '-':
        cudaMallocManaged(&d, size);
        cudaMemPrefetchAsync(d, size, cudaCpuDeviceId);
        initWith(0,d,N);
        cudaMemPrefetchAsync(d, size, deviceId);
        //sub vec on cpu
        start = clock();
        subVectorsCPU(d, a, b, N);

        //give time of execution
        end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("\nTime elapsed subtracting vectors on CPU is: %f", cpu_time_used);

        //sub vecs on 1x1 kernel
        cudaEventCreate(&start_gpu);
        cudaEventCreate(&stop_gpu);
        cudaEventRecord(start_gpu, 0);

        initWith(0,d,N);
        subVectorsGpu<<<1,1>>>(d, a, b, N);

        //give time of execution
        cudaDeviceSynchronize();
        cudaEventRecord(stop_gpu,0);
        cudaEventSynchronize(stop_gpu);

        cudaEventElapsedTime(&el_time, start_gpu, stop_gpu);
        printf("\nTime elapsed on single-threaded vector subtracting: %f \n", el_time);
        break;

    // operator doesn't match any case constant +, -, *,
    default:
        printf("Error! operator is not correct\n");
        //printf("Operat = %c",operat);
        printf("Quiting...");
        return 0;
    }
    bool repeat = 1;
    while (repeat)
    {

        //take size of grid and etc
        int user_block;
        printf("\nGive the size of block you want to use:");
        scanf("%d", &user_block);
        int user_thread;
        printf("Give the number of threads per block you want to use:");
        scanf("%d", &user_thread);

        //timing variables
        float el_time2;
        cudaEvent_t start_gpu2, stop_gpu2;

        switch(operat)
        {
        case '*':
            cudaEventCreate(&start_gpu2);
            cudaEventCreate(&stop_gpu2);
            cudaEventRecord(start_gpu2, 0);

            //dot vecs on got sizes kernel
            c=0;
            dotVectorsGpu<<<user_block,user_thread>>>(c, a, b, N);

            //give time of execution
            cudaEventRecord(stop_gpu2,0);
            cudaEventSynchronize(stop_gpu2);

            cudaEventElapsedTime(&el_time2, start_gpu2, stop_gpu2);
            printf("\nTime elapsed on your size vectors dot product: %f\n", el_time2);

            //print all execution times in table
            cudaDeviceSynchronize();

            //ask if user want to boost the score
            repeat = ask_repeat();
            break;
        /////////
        case '+':
            cudaEventCreate(&start_gpu2);
            cudaEventCreate(&stop_gpu2);
            cudaEventRecord(start_gpu2, 0);

            //add vecs on got sizes kernel
            initWith(0,d,N);
            addVectorsGpu<<<user_block,user_thread>>>(d, a, b, N);

            //give time of execution
            cudaEventRecord(stop_gpu2,0);
            cudaEventSynchronize(stop_gpu2);

            cudaEventElapsedTime(&el_time2, start_gpu2, stop_gpu2);
            printf("\nTime elapsed on your size vectors addition: %f\n", el_time2);

            //print all execution times in table
            cudaDeviceSynchronize();

            //ask if user want to boost the score
            repeat = ask_repeat();
            break;
        /////////
        case '-':
            cudaEventCreate(&start_gpu2);
            cudaEventCreate(&stop_gpu2);
            cudaEventRecord(start_gpu2, 0);

            //sub vecs on got sizes kernel
            initWith(0,d,N);
            subVectorsGpu<<<user_block,user_thread>>>(d, a, b, N);

            //give time of execution
            cudaEventRecord(stop_gpu2,0);
            cudaEventSynchronize(stop_gpu2);

            cudaEventElapsedTime(&el_time2, start_gpu2, stop_gpu2);
            printf("\nTime elapsed on your size vectors subtraction: %f\n", el_time2);

            //print all execution times in table
            cudaDeviceSynchronize();

            //ask if user want to boost the score
            repeat = ask_repeat();
            break;
        ////////
        default:
            printf("Error!\n");
            printf("Quiting...");
            return 0;
        }
    }
    printf("Quiting...");
    return 0;
}
