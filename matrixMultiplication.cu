#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#include<math.h>

#define BLOCK_SIZE 8
#define WIDTH 512
#define HEIGHT 512

__global__ void matrixMulCuda(float *A, float *B, float *C, int N, int d)
{ // Each thread calculates one entry of C by accumulating results into tmpSum
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if (row >= N && col >= N) return;
    double tmpSum = 0.0;
    for (int k=0; k<d; k++)
    {
        tmpSum += A[row*d + k] * B[col + k*N];
    }
    C[row*N + col] = tmpSum;
}

__global__ void matrixMulSharedMemory(float *A, float *B, float *C, int N, int d)
{   // Shared memory used to store Asub and Bsub, respectively.
    // shared memory is shared within one thread block.
    __shared__ float Asub[BLOCK_SIZE][BLOCK_SIZE];  
    __shared__ float Bsub[BLOCK_SIZE][BLOCK_SIZE];
    // each thread has one tmpSum to accumulates the results
    double tmpSum = 0; // tmpSum is stored in registers
    // row and column in C
    unsigned int col = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    unsigned int row = BLOCK_SIZE * blockIdx.y + threadIdx.y;
    if (row >= N && col >= N) return; // check boundary condition 
    unsigned int col_sub = threadIdx.x; // Thread column within Csub
    unsigned int row_sub = threadIdx.y; // Thread row within Csub
    // Loop over all the sub-matrices of A and B required to compute Csub
    for (int m=0; m<d/BLOCK_SIZE; m++)
    {   // load Asub and Bsub from device memory to shared memory 
        // each thread load one entry of each sub-matrix        
        Asub[row_sub][col_sub] = A[row*d + m*BLOCK_SIZE + col_sub];
        Bsub[row_sub][col_sub] = B[col + (row_sub + m*BLOCK_SIZE) * N];
        // Synchronize to ensure the sub-matrices are loaded before computation.
        __syncthreads();
        // Multiply Asub and Bsub together
        for (int k=0; k<BLOCK_SIZE; k++)
        {
            tmpSum += Asub[row_sub][k] * Bsub[k][col_sub];
        }
        // Before loading the next two sub-matrices, synchronize to ensure 
        // the computation involving these two sub-matrices is done.
        __syncthreads();
    }
    C[row*N + col] =tmpSum; // write Csub to device memory; each thread writes one
}

int main()
{
    // const int WIDTH = 512;
    long size = HEIGHT * WIDTH * sizeof(float);
    long size_result = HEIGHT * HEIGHT * sizeof(float);
    printf("size: %ld, size_result: %ld\n", size, size_result);
    float array1_h[HEIGHT][WIDTH], array2_h[WIDTH][HEIGHT], result_array_h[HEIGHT][HEIGHT];
    memset(result_array_h, 0.0, size_result);
    float *array1_d, *array2_d, *result_array_d;
    int i, j;
    cudaEvent_t start, stop;
    float elapsed_time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // initialization
    for (i=0; i<HEIGHT; i++)
    {
        for (j=0; j<WIDTH; j++)
        {
            array1_h[i][j] = sin(i+j);
            array2_h[i][j] = cos(i+j);
        }
    }
    printf("array1_h[0][0]: %f\n", array1_h[5][5]);
    cudaMalloc((void**) &array1_d, size);
    cudaMalloc((void**) &array2_d, size);
    cudaMalloc((void**) &result_array_d, size_result);
    cudaMemcpy(array1_d, array1_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(array2_d, array2_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(result_array_d, result_array_h, size_result, cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(HEIGHT/BLOCK_SIZE, HEIGHT/BLOCK_SIZE);
    cudaEventRecord(start, 0);
    matrixMulSharedMemory<<<dimGrid, dimBlock>>>(array1_d, array2_d, result_array_d, HEIGHT, WIDTH);
    // matrixMulCuda<<<dimGrid, dimBlock>>>(array1_d, array2_d, result_array_d, HEIGHT, WIDTH);
    cudaMemcpy(result_array_h, result_array_d, size_result, cudaMemcpyDeviceToHost);
    cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
  		fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
	}
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("elapsed time: %.3f\n", elapsed_time);
    
    float err = 0.0;
    float cpu_C[HEIGHT][HEIGHT];
    double tmpSum;
    for (i=0; i<HEIGHT; i++)
    {
        for (j=0; j<HEIGHT; j++)
        {
            tmpSum = 0.0;
            for (int k=0; k<WIDTH; k++)
            {
                tmpSum += array1_h[i][k] * array2_h[k][j];
            }
            cpu_C[i][j] = tmpSum;
            err += fabs(cpu_C[i][j] - result_array_h[i][j]);
            if (i<3 && j<3)
            {
                printf("cpu_C[%d][%d]: %f, host_C[%d][%d]: %f\n", i, j, cpu_C[i][j], i, j, result_array_h[i][j]);
            }
        }
    }
    printf("err: %f\n", err);
    cudaFree(array1_d);
    cudaFree(array2_d);
    cudaFree(result_array_d);

    return 0;
}