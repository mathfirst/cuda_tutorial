#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#include<math.h>

#define N 512
#define d 64
#define Br 16
#define Bc 16

typedef float arrtype[d];
typedef float arrtype1[N];

__global__ void attention_kernel(arrtype *Q, arrtype *K, arrtype *V, arrtype1 *O){
// __global__ void attention_kernel(float Q[N][d], float K[N][d], float V[N][d], float O[N][N]){
    __shared__ float Qi[Br][d];
    __shared__ float Oi[Br][d];
    __shared__ float Kj[Bc][d];
    __shared__ float Vj[Bc][d];
    __shared__ float l[Br];
    __shared__ float m[Br];
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int col_sub = threadIdx.x; // Thread column within Csub
    unsigned int row_sub = threadIdx.y; // Thread row within Csub
    int Tr = N / Br;
    int Tc = N / Bc;
    const int Tk = d / Bc;
    const int Tk_inner = d / Br;
    int i, j, k, Q_col_idx, K_row_idx, K_col_idx, Q_row_idx, O_col_idx;
    for(int i=0; i<Tr; ++i) // outer loop
    {
        Q_row_idx = i*Br + row_sub;        
        if (Q_row_idx >= N) return;
        l[col_sub] = 0.0;
        m[col_sub] = -1.0e-9;
        for(k=0; k<Tk; ++k) // load Q from global memory into shared memory
        {
            Q_col_idx = col_sub + k*Bc;   
            if (Q_col_idx >= d) return;
            Qi[row_sub][Q_col_idx] = Q[Q_row_idx][Q_col_idx];
            // printf("row_sub: %d, col_sub: %d, Q_row_idx: %d, Q_col_idx: %d, Qi[row_sub][Q_col_idx]: %f\n", row_sub, col_sub, Q_row_idx, Q_col_idx, Qi[row_sub][Q_col_idx]);
            Oi[row_sub][Q_col_idx] = 0.0;            
        }
        __syncthreads();    
        // printf("Qi[0][0]: %f", Qi[0][0]);    
        for(j=0; j<Tc; ++j) // inner loop
        {
            // int Tk = d / Br;
            K_row_idx = j*Bc + row_sub; // K is a tall matrix and K^T is a wide matrix            
            if (K_row_idx >= N) return;
            for(k=0; k<Tk_inner; ++k) // load K, V from global memory into shared memory
            {
                K_col_idx = col_sub + k*Br;   
                if (K_col_idx >= d) return;
                Kj[row_sub][K_col_idx] = K[K_row_idx][K_col_idx];
                Vj[row_sub][K_col_idx] = V[K_row_idx][K_col_idx];
                // printf("row_sub: %d, col_sub: %d, K_row_idx: %d, K_col_idx: %d, Kj[row_sub][K_col_idx]: %f\n", row_sub, col_sub, K_row_idx, K_col_idx, Kj[row_sub][K_col_idx]);
            }
            __syncthreads();
            double tmpSum = 0.0;
            for(k=0; k<d; ++k)
            {
                tmpSum += Qi[row_sub][k] * Kj[col_sub][k];
            } 
            __syncthreads();
            // printf("row_sub: %d, col_sub: %d, i: %d, Q_row_idx: %d, j: %d, K_row_idx, tmpSum: %f\n", row_sub, col_sub, i, Q_row_idx, j, K_row_idx, tmpSum);
            O_col_idx = j*Bc + col_sub;
            O[Q_row_idx][O_col_idx] = tmpSum;
            // printf("row: %d, col: %d, tmpsum: %f\n", Q_row_idx, j*Bc + col_sub, tmpSum);
        }
    }
}


int main()
{
    // const int WIDTH = 64;
    long numElem = N * d;
    long size = numElem * sizeof(float);
    long size_out = N*N * sizeof(float);
    // float array1_d[N][d], O_d[N][N];
    // float *array1_d, *O_d;
    arrtype *array1_d;
    arrtype1 *O_d;// *O_h;
    float A[N][d], B[N][N], O_h[N][N];
    memset(A, 0.0, size);
    // memset(O_h, 0.0, size_out);
    for (long i=0; i<N; ++i)
    {
        for (long j=0; j<d; ++j)
            A[i][j] = sin(i+j);
    }
    cudaMalloc((void**) &array1_d, size);
    cudaMalloc((void**) &O_d, size_out);
    cudaMemcpy(array1_d, A, size, cudaMemcpyHostToDevice);
    float result = 0;
    for (long i=0; i<N; ++i)
    {        
        for (long j=0; j<N; ++j)
        {
            double tmp = 0.0;
            for (long k=0; k<d; ++k)
            {
                tmp += A[i][k] * A[j][k];
            }
            B[i][j] = tmp;
        }            
    }
    printf("cpu result: %f, %f\n", B[0][0], B[1][1]);
    dim3 dimBlock(Br, Bc);
    dim3 dimGrid(1, 1);
    attention_kernel<<<dimGrid, dimBlock>>>(array1_d, array1_d, array1_d, O_d);
    cudaDeviceSynchronize();
    cudaMemcpy(O_h, O_d, N*N*sizeof(float), cudaMemcpyDeviceToHost);
    printf("gpu result: %f, %f\n", O_h[0][0], O_h[1][1]);
    double err = 0.0;
    for (int i=0; i<N; i++)
    {
        for (int j=0; j<N; j++)
        {            
            err += fabs(B[i][j] - O_h[i][j]);
            // if (i<3 && j<3)
            // {
            //     printf("cpu_C[%d][%d]: %f, host_C[%d][%d]: %f\n", i, j, cpu_C[i][j], i, j, result_array_h[i][j]);
            // }
        }
    }
    printf("err: %f\n", err);
    cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
  		fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
	}
    
    cudaFree(array1_d);

    return 0;
}