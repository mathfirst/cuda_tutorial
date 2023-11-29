#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#include<math.h>

#define TILE_WIDTH 16

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

#define numThreadsPerBlock 1024

template<typename scalar_t>
__global__ void neighbor_parallel(float *M, float *outArr)
{   __shared__ scalar_t input_arr[numThreadsPerBlock]; 
    // int col = blockDim.x * blockIdx.x + threadIdx.x;
    // int row = TILE_WIDTH * blockIdx.y + threadIdx.y;
    long thread_idx = threadIdx.x; // within a block
    long idx = blockDim.x * blockIdx.x + threadIdx.x; // global index
    input_arr[thread_idx] = M[idx];
    __syncthreads();
    long N = numThreadsPerBlock * gridDim.x;
    if (idx >= N) return;
    for (long k=1; k<=numThreadsPerBlock; k*=2)
    {
        if (thread_idx % (2*k) == 0)
        {   
            if(thread_idx + k < numThreadsPerBlock)
            {
                // printf("k: %ld, thread_idx: %ld\n", k, thread_idx);
                input_arr[thread_idx] += input_arr[thread_idx + k];
            }         
        }
        __syncthreads();
    }
    if (thread_idx == 0)
    {
        outArr[blockIdx.x] = input_arr[0];
    }
}

template<typename scalar_t>
__global__ void interleaved_parallel(float *M, float *outArr)
{   __shared__ scalar_t input_arr[numThreadsPerBlock]; 
    // int col = blockDim.x * blockIdx.x + threadIdx.x;
    // int row = TILE_WIDTH * blockIdx.y + threadIdx.y;
    long thread_idx = threadIdx.x; // within a block, i.e. local thread index
    long tid = blockDim.x * blockIdx.x + threadIdx.x; // global thread index
    input_arr[thread_idx] = M[tid];
    __syncthreads();
    long N = blockDim.x * gridDim.x;
    if (tid >= N) return;
    for (long k=1; k<blockDim.x; k*=2)
    {
        long idx = 2 * k * thread_idx;
        if (idx < blockDim.x)
        {   //printf("blockIdx.x: %d, k: %ld, idx: %ld, idx+k: %ld\n", blockIdx.x, k, idx, idx +k);
            input_arr[idx] += input_arr[idx + k];      
        }
        __syncthreads();
    }
    if (thread_idx == 0)
    {
        outArr[blockIdx.x] = input_arr[0];
    }
}

template<typename scalar_t>
__global__ void sequential_addressing(float *M, float *outArr)
{   __shared__ scalar_t input_arr[numThreadsPerBlock]; 
    // int col = blockDim.x * blockIdx.x + threadIdx.x;
    // int row = TILE_WIDTH * blockIdx.y + threadIdx.y;
    long thread_idx = threadIdx.x; // within a block, i.e. local thread index
    long tid = blockDim.x * blockIdx.x + threadIdx.x; // global thread index
    input_arr[thread_idx] = M[tid];
    __syncthreads();
    long N = numThreadsPerBlock * gridDim.x;
    if (tid >= N) return;
    for (long k=blockDim.x/2; k>0; k >>= 1)
    {        
        if (thread_idx < k)
        {   
            // long next = tid + k;
            input_arr[thread_idx] += input_arr[thread_idx + k];      
        }
        __syncthreads();
    }
    if (thread_idx == 0)
    {
        outArr[blockIdx.x] = input_arr[0];
    }
}

template<typename scalar_t>
__global__ void load_add(float *M, float *outArr)
{   __shared__ scalar_t input_arr[numThreadsPerBlock]; 
    long thread_idx = threadIdx.x; // within a block, i.e. local thread index
    long tid = (2*blockDim.x) * blockIdx.x + threadIdx.x; // global thread index
    input_arr[thread_idx] = M[tid] + M[tid + blockDim.x];
    __syncthreads();
    // long N = numThreadsPerBlock * gridDim.x;
    // if (tid >= N) return;
    for (long k=blockDim.x/2; k>0; k>>=1)
    {        
        if (thread_idx < k)
        {   
            // long next = tid + k;
            input_arr[thread_idx] += input_arr[thread_idx + k];      
        }
        __syncthreads();
    }
    if (thread_idx == 0)
    {
        outArr[blockIdx.x] = input_arr[0];
    }
}

__device__ void warpReduce(volatile float* sdata, long tid)
{
    sdata[tid] += sdata[tid + 32]; 
    __syncthreads();
    sdata[tid] += sdata[tid + 16]; 
    __syncthreads();
    sdata[tid] += sdata[tid + 8]; 
    __syncthreads();
    sdata[tid] += sdata[tid + 4]; 
    __syncthreads();
    sdata[tid] += sdata[tid + 2]; 
    __syncthreads();
    sdata[tid] += sdata[tid + 1];
}

template<typename scalar_t>
__global__ void unroll(float *M, float *outArr)
{   
    __shared__ scalar_t input_arr[numThreadsPerBlock]; 
    long thread_idx = threadIdx.x; // within a block, i.e. local thread index
    long tid = (2*blockDim.x) * blockIdx.x + threadIdx.x; // global thread index
    input_arr[thread_idx] = M[tid] + M[tid + blockDim.x];
    __syncthreads();
    // long N = numThreadsPerBlock * gridDim.x;
    // if (tid >= N) return;
    for (long k=blockDim.x/2; k>32; k >>= 1)
    {        
        if (thread_idx < k)
        {   
            // long next = tid + k;
            input_arr[thread_idx] += input_arr[thread_idx + k]; 
        }
        __syncthreads();        
    }
    if (thread_idx < 32)
    {
        warpReduce(input_arr, thread_idx);
    }
    __syncthreads();        
    if (thread_idx == 0)
    {
        outArr[blockIdx.x] = input_arr[0];
    }
}

int main()
{
    int iDev = 1;
	cudaError_t err = cudaSetDevice(iDev);
	if(err != cudaSuccess)
	{
		printf("failed to set GPU %d for computing.\n", iDev);
		exit(-1);
	}
	else
	{
		printf("set GPU %d for computing.\n", iDev);
	}

    const int WIDTH = 256;
    long numElem = WIDTH * WIDTH;
    long size = numElem * sizeof(float);
    long numOutElements = numElem / numThreadsPerBlock;
    long size_out = numOutElements * sizeof(float);
    long size_out2 = numOutElements / 2 * sizeof(float);
    float *array1_d, *outArr_d, *array2_d, *array3_d, *array4_d, *outArr2_d;
    float array1_h[numElem], outArr_h[numOutElements], outArr2_h[numOutElements/2];
    memset(array1_h, 0.0, size);
    for (long i=0; i<numElem; ++i)
    {
        array1_h[i] = sin(i);
    }
    cudaMalloc((void**) &array1_d, size);
    cudaMalloc((void**) &array2_d, size);
    cudaMalloc((void**) &array3_d, size);
    cudaMalloc((void**) &array4_d, size);
    cudaMalloc((void**) &outArr_d, size_out);
    cudaMalloc((void**) &outArr2_d, size_out);
    cudaMemcpy(array1_d, array1_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(array2_d, array1_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(array3_d, array1_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(array4_d, array1_h, size, cudaMemcpyHostToDevice);
    float result = 0;
    for (long i=0; i<numElem; ++i)
    {
        result += array1_h[i];
    }
    // printf("cpu result: %f\n", result);
    dim3 dimBlock(numThreadsPerBlock);
    dim3 dimGrid(numOutElements);
    // neighbor_parallel<float><<<dimGrid, dimBlock>>>(array1_d, outArr_d);
    
    float elapsed_time, t_neighbor_parallel=0.0, t_interleaved_parallel=0.0, t_seq_addressing=0.0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // double iStart, iStop;
    
    for (int i=0; i<100; ++i)        
    {   
        cudaEventRecord(start, 0);
        interleaved_parallel<float><<<dimGrid, dimBlock>>>(array2_d, outArr_d);
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        if (i >= 50)
            t_interleaved_parallel += elapsed_time;
    }
    cudaMemcpy(outArr_h, outArr_d, size_out, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    float tmp2 = 0;
    for (int i=0; i<numOutElements; ++i)
    {
        tmp2 += outArr_h[i];
    }
    printf("interleaved parallel, gpu result: %.3f, cpu result: %.3f, elapsed time: %.6f\n", tmp2, result, t_interleaved_parallel);
    for (int i=0; i<100; ++i)        
    {   
        cudaEventRecord(start, 0);
        sequential_addressing<float><<<dimGrid, dimBlock>>>(array3_d, outArr_d);
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        if (i >= 50)
            t_seq_addressing += elapsed_time;
    }
    cudaMemcpy(outArr_h, outArr_d, size_out, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    float tmp3 = 0;
    for (int i=0; i<numOutElements; ++i)
    {
        tmp3 += outArr_h[i];
    }
    printf("sequential parallel, gpu result: %.3f, cpu result: %.3f, elapsed time: %.6f\n", tmp3, result, t_seq_addressing);
    t_seq_addressing = 0.0;
    dim3 dimGrid2(numOutElements / 2);
    for (int i=0; i<100; ++i)        
    {   
        cudaEventRecord(start, 0);
        load_add<float><<<dimGrid2, dimBlock>>>(array3_d, outArr2_d);
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        if (i >= 50)
            t_seq_addressing += elapsed_time;
    }
    cudaMemcpy(outArr2_h, outArr2_d, size_out/2, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    tmp3 = 0;
    for (int i=0; i<numOutElements; ++i)
    {
        tmp3 += outArr2_h[i];
    }
    printf("load and add, gpu result: %.3f, cpu result: %.3f, elapsed time: %.6f\n", tmp3, result, t_seq_addressing);
    t_seq_addressing = 0.0;
    for (int i=0; i<100; ++i)        
    {   
        cudaEventRecord(start, 0);
        unroll<float><<<dimGrid2, dimBlock>>>(array1_d, outArr2_d);
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        if (i >= 50)
            t_seq_addressing += elapsed_time;
    }
    cudaMemcpy(outArr2_h, outArr2_d, size_out/2, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    tmp3 = 0;
    for (int i=0; i<numOutElements; ++i)
    {
        tmp3 += outArr2_h[i];
    }
    printf("unroll, gpu result: %.3f, cpu result: %.3f, elapsed time: %.6f\n", tmp3, result, t_seq_addressing);
    for (int i=0; i<100; ++i)        
    {   
        cudaEventRecord(start, 0);
        neighbor_parallel<float><<<dimGrid, dimBlock>>>(array1_d, outArr_d);
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        if (i >= 50)
            t_neighbor_parallel += elapsed_time;
            // elapsed_time_accum += iStop - iStart;
    }
    cudaMemcpy(outArr_h, outArr_d, size_out, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    float tmp1 = 0;
    for (int i=0; i<numOutElements; ++i)
    {
        tmp1 += outArr_h[i];
    }
    printf("neighbor parallel, gpu result: %.3f, cpu result: %.3f, elapsed time: %.6f\n", tmp1, result, t_neighbor_parallel);
    cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
  		fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
	}
    
    cudaFree(array1_d);
    return 0;
}