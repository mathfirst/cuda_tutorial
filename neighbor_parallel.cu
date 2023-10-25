#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#include<math.h>

#define TILE_WIDTH 4

__global__ void write(float *M, float *N, int WIDTH)
{   // write sth to global memory
    int col = TILE_WIDTH * blockIdx.x + threadIdx.x;
    int row = TILE_WIDTH * blockIdx.y + threadIdx.y;
    for (int k = 0; k < WIDTH; k++)
    {
        M[row*WIDTH + k] = k;
        N[col+k*WIDTH] = k;
    }
}

__global__ void read(float *M, float *N, int WIDTH)
{   // write sth from global memory
    int col = TILE_WIDTH * blockIdx.x + threadIdx.x;
    int row = TILE_WIDTH * blockIdx.y + threadIdx.y;
    double tmpSum = 0.0, tmpSum1 = 0.0;
    for (int k = 0; k < WIDTH; k++)
    {
        tmpSum = M[row*WIDTH + k];
        tmpSum1 =  N[col+k*WIDTH];
    }
    // tmpSum += tmpSum1;
}

__global__ void neighbor_parallel(float *M, int WIDTH)
{   // write sth to global memory
    int col = TILE_WIDTH * blockIdx.x + threadIdx.x;
    int row = TILE_WIDTH * blockIdx.y + threadIdx.y;
    long idx = row * (gridDim.x * blockDim.x) + col;
    if (idx >= WIDTH) return;
    for (long k = 1; k<=WIDTH; k*=2)
    {
        if (idx % (2*k) == 0)
        {   
            if(idx + k < WIDTH)
            {
                printf("row: %d, col: %d, k: %ld, idx: %ld\n", row, col, k, idx);
                M[idx] += M[idx + k];}
            // else if (/* condition */)
            // {
            //     /* code */
            // }
            
        }
        __syncthreads();
    }
}

int main()
{
    const int WIDTH = 64;
    long numElem = WIDTH * WIDTH;
    long size = numElem * sizeof(float);
    float *array1_d;
    float array1_h[numElem];
    memset(array1_h, 0.0, size);
    for (long i=0; i<numElem; ++i)
    {
        array1_h[i] = 1;
    }
    cudaMalloc((void**) &array1_d, size);
    cudaMemcpy(array1_d, array1_h, size, cudaMemcpyHostToDevice);
    float result = 0;
    for (long i=0; i<numElem; ++i)
    {
        result += array1_h[i];
    }
    // printf("cpu result: %f\n", result);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(WIDTH / TILE_WIDTH, WIDTH / TILE_WIDTH);
    neighbor_parallel<<<dimGrid, dimBlock>>>(array1_d, numElem);
    cudaDeviceSynchronize();
    cudaMemcpy(array1_h, array1_d, size, cudaMemcpyDeviceToHost);
    printf("gpu result: %f, cpu result: %f\n", array1_h[0], result);
    cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
  		fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
	}
    
    cudaFree(array1_d);

    return 0;
}