#include<torch/extension.h>
#include "cuda_fp16.h"
#include <mma.h>
#include<cuda_runtime.h>
using namespace nvcuda;

/* If you need to write c++/cuda with PyTorch, https://witnessj.com/archives/cuda 
** is a good reference. */


using namespace at;

/* When we write some custom CUDA kernel functions for torch.cuda.HalfTensor, 
** we use at::Half instead of half. Also, we should use AT_DISPATCH_FLOATING_TYPES_AND_HALF
** instead of AT_DISPATCH_FLOATING_TYPES_AND_HALF. */

/* static allocation size is limited to 48KB in one block, but A100 has 164KB shared memory on one SM.

*/


template<typename scalar_t>
__global__ void matmul_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> A,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> B,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> C
){
    // Each thread calculates one entry of C by accumulating results into tmpSum
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int height_A = A.size(0), width_A = A.size(1), width_B = B.size(1);
    if (row >= height_A || col >= width_B) return;
    double tmpSum = 0.0;
    for (int k=0; k<width_A; k++)
    {
        tmpSum += A[row][k] * B[k][col];
    }
    C[row][col] = tmpSum;
}


#define BLOCK_SIZE 32
template<typename scalar_t>
__global__ void matmul_shared_memory_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> A,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> B,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> C
){
    // Shared memory used to store Asub and Bsub, respectively.
    // shared memory is shared within one thread block.
    __shared__ float Asub[BLOCK_SIZE][BLOCK_SIZE];  
    __shared__ float Bsub[BLOCK_SIZE][BLOCK_SIZE];
    // Each thread calculates one entry of C by accumulating results into tmpSum
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int col_sub = threadIdx.x; // Thread column within Csub
    unsigned int row_sub = threadIdx.y; // Thread row within Csub
    int height_A = A.size(0), width_A = A.size(1), width_B = B.size(1);
    if (row >= height_A || col >= width_B) return;
    double tmpSum = 0.0;
    for (int m=0; m<width_A/BLOCK_SIZE; m++)
    {
        Asub[row_sub][col_sub] = A[row][m*BLOCK_SIZE + col_sub];
        Bsub[row_sub][col_sub] = B[m*BLOCK_SIZE + row_sub][col];
        // Synchronize to ensure the sub-matrices are loaded before computation.
        __syncthreads();
        for (int k=0; k<BLOCK_SIZE; k++)
        {
            tmpSum += Asub[row_sub][k] * Bsub[k][col_sub];
        }
        // Before loading the next two sub-matrices, synchronize to ensure 
        // the computation involving these two sub-matrices is done.
        __syncthreads();
    }
    C[row][col] = tmpSum;
}


#define Br 16
#define Bc 16
#define d 32

template<typename scalar_t>
__global__ void attention_bwd_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> Q,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> K,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> V,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> O,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dO,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> L,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> D,
    const float tau,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dQ,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dK,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dV
){
    __shared__ scalar_t Qi[Br][d], dQi[Br][d];
    __shared__ scalar_t Oi[Br][d], dOi[Br][d];
    __shared__ scalar_t Kj[Bc][d], dKj[Bc][d];
    __shared__ scalar_t Vj[Bc][d], dVj[Bc][d];
    __shared__ scalar_t Si[Br][Bc], dSi[Br][Bc], Pi[Br][Bc], dPi[Br][Bc];
    __shared__ scalar_t Si_tmp[Br][Bc/2];
    __shared__ scalar_t Li[Br];
    __shared__ scalar_t Di[Br];
    const int offset = Bc / 2;
    __shared__ scalar_t l[Br];
    __shared__ scalar_t m[Br];
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int col_sub = threadIdx.x; // Thread column within Csub
    const int row_sub = threadIdx.y; // Thread row within Csub
    const int N = Q.size(0);
    const int Tr = N / (Br * gridDim.x);
    // const int Tr = N / Br;
    const int Tc = N / Bc;
    const int Td_c = d / Bc;
    const int Td_r = d / Br;
    #pragma unroll
    for(int j = 0; j < Tc; ++j) // outer loop    
    {   
        int K_row_idx = j * Bc + col_sub; // Here we are considering K, not K^T
        if (K_row_idx >= N) // check the boundary condition
        {
            return;
        }
        #pragma unroll
        for(int kk = 0; kk < Td_r; ++kk) // load K, V from global memory into shared memory
        {
            int K_col_idx = row_sub + kk * Br;   
            if (K_col_idx >= d) // check the boundary condition
            {
                return;
            }
            Kj[col_sub][K_col_idx] = K[K_row_idx][K_col_idx];
            Vj[col_sub][K_col_idx] = V[K_row_idx][K_col_idx];
            dKj[col_sub][K_col_idx] = 0.0;
            dVj[col_sub][K_col_idx] = 0.0;
        }
        __syncthreads();
        for(int i = 0; i < Tr; ++i) // inner loop    
        {   
            int Q_row_idx = i * Br + row_sub + Tr * Br * blockIdx.x;   
            Li[row_sub] = L[row_sub + i * Br];     
            Di[row_sub] = D[row_sub + i * Br];     
            if(Q_row_idx >= Tr * Br * (blockIdx.x + 1))
            { 
                return;
            }
            for(int kq = 0; kq < Td_c; ++kq) // load Q from global memory into shared memory
            {
                int Q_col_idx = col_sub + kq * Bc;   
                if (Q_col_idx >= d) 
                {
                    return;
                }
                Qi[row_sub][Q_col_idx] = Q[Q_row_idx][Q_col_idx];
                Oi[row_sub][Q_col_idx] = O[Q_row_idx][Q_col_idx];  
                dOi[row_sub][Q_col_idx] = dO[Q_row_idx][Q_col_idx];
                dQi[row_sub][Q_col_idx] = dQ[Q_row_idx][Q_col_idx];
            }
            __syncthreads();
            // compute Si
            scalar_t tmpSum = 0.0;
            #pragma unroll
            for(int ks=0; ks<d; ++ks) // Q @ K^T / sqrt(d)
            {
                tmpSum += tau * Qi[row_sub][ks] * Kj[col_sub][ks];
            }
            Si[row_sub][col_sub] = tmpSum;
            __syncthreads();
            // compute Pi
            Pi[row_sub][col_sub] = exp(Si[row_sub][col_sub] - Li[row_sub]);
            
            // compute dPi
            tmpSum = 0.0;
            #pragma unroll
            for(int kp = 0; kp < d; ++kp) // Q @ K^T / sqrt(d)
            {
                tmpSum += tau * dOi[row_sub][kp] * Vj[col_sub][kp];
            }
            dPi[row_sub][col_sub] = tmpSum;
            __syncthreads();
            // compute dSi
            dSi[row_sub][col_sub] = Pi[row_sub][col_sub] * (dPi[row_sub][col_sub] - Di[row_sub]);
            __syncthreads();
            // update dQi
            for(int kq=0; kq < Td_c; ++kq)
            {
                int K_col_idx = col_sub + kq * Bc;
                scalar_t tmp = 0.0;
                for(int k = 0; k < Bc; ++k)
                {
                    tmp += dSi[row_sub][k] * Kj[k][K_col_idx];
                }
                dQi[row_sub][K_col_idx] += tmp;
                // write back to global memory
                dQ[Q_row_idx][K_col_idx] = dQi[row_sub][K_col_idx]; 
            }
            // compute dVj, dKj
            for(int kv=0; kv < Td_r; ++kv)
            {
                int V_col_idx = row_sub + kv * Br;
                scalar_t tmp_k = 0.0, tmp_v = 0.0;
                for(int k = 0; k < Br; ++k)
                {
                    tmp_v += Pi[k][col_sub] * dOi[k][V_col_idx];
                    tmp_k += dSi[k][col_sub] * Qi[k][V_col_idx];
                }
                dVj[col_sub][V_col_idx] += tmp_v;
                dKj[col_sub][V_col_idx] += tmp_k;
            }
        }
        __syncthreads();
        // write dKj, dVj to global memory
        #pragma unroll
        for(int kk = 0; kk < Td_r; ++kk) // load K, V from global memory into shared memory
        {
            int K_col_idx = row_sub + kk * Br;   
            if (K_col_idx >= d) // check the boundary condition
            {
                return;
            }
            dK[K_row_idx][K_col_idx] = dKj[col_sub][K_col_idx];
            dV[K_row_idx][K_col_idx] = dVj[col_sub][K_col_idx];
        }
    }    
}

std::vector<torch::Tensor> attention_bwd(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    torch::Tensor dO,
    torch::Tensor L,
    torch::Tensor D,
    const float tau,
    const int iDev
){
    // int iDev = 2;
	cudaError_t error = cudaSetDevice(iDev);
    const int N = Q.size(0);//, N = K.size(0);
    torch::Tensor dQ = torch::empty({N, d}, Q.options());
    torch::Tensor dK = torch::empty({N, d}, Q.options());
    torch::Tensor dV = torch::empty({N, d}, Q.options());
    // custom dtype
    // torch::zeros({N, F}, torch::dtype(kInt32).device(feats.device)); 
    const dim3 threads(Bc, Br);
    const dim3 blocks(N / Br, 1);
    // printf("Br: %d, Bc: %d\n", Br, Bc);
    // launch a kernel function
    AT_DISPATCH_FLOATING_TYPES(Q.type(), "attention_bwd",
    ([&]
        {
            attention_bwd_kernel<scalar_t><<<blocks, threads>>>(
                Q.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(), // .packed_accessor is only used for tensor
                K.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                V.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                O.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                dO.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                L.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
                D.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
                tau,
                dQ.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                dK.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                dV.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
            );
        }
    ));
    cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
  		printf("ERROR: %s \n", cudaGetErrorString(err));
	}
    return {dQ, dK, dV};
}

template<typename scalar_t>
__global__ void attention_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> Q,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> K,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> V,
    const float tau,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> O,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> M
){
    __shared__ scalar_t Qi[Br][d];
    __shared__ scalar_t Oi[Br][d];
    __shared__ scalar_t Kj[Bc][d];
    __shared__ scalar_t Vj[Bc][d];
    __shared__ scalar_t Si[Br][Bc];
    __shared__ scalar_t Si_tmp[Br][Bc/2];
    const int offset = Bc / 2;
    __shared__ scalar_t l[Br];
    __shared__ scalar_t m[Br];
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int col_sub = threadIdx.x; // Thread column within Csub
    const int row_sub = threadIdx.y; // Thread row within Csub
    const int N = Q.size(0);
    const int Tr = N / (Br * gridDim.x);
    // const int Tr = N / Br;
    const int Tc = N / Bc;
    const int Tk = d / Bc;
    const int Tk_inner = d / Br;
    #pragma unroll
    for(int i=0; i<Tr; ++i) // outer loop    
    {     
        int Q_row_idx = i*Br + row_sub + Tr*Br*blockIdx.x;        
        if (Q_row_idx >= Tr*Br*(blockIdx.x+1))
        { 
            return;
        }
        l[row_sub] = 0.0;
        m[row_sub] = -1.0e-9;  
        #pragma unroll
        for(int kq=0; kq<Tk; ++kq) // load Q from global memory into shared memory
        {
            int Q_col_idx = col_sub + kq*Bc;   
            if (Q_col_idx >= d) return;
            Qi[row_sub][Q_col_idx] = Q[Q_row_idx][Q_col_idx] * tau;
            Oi[row_sub][Q_col_idx] = 0.0;            
        }
        __syncthreads();
        #pragma unroll
        for(int j=0; j<Tc; ++j) // inner loop
        {            
            int K_row_idx = j*Bc + col_sub; // K is a tall matrix and K^T is a wide matrix
            if (K_row_idx >= N) 
            {
                return;
            }
            #pragma unroll
            for(int kk=0; kk<Tk_inner; ++kk) // load K, V from global memory into shared memory
            {
                int K_col_idx = row_sub + kk*Br;   
                if (K_col_idx >= d) return;
                Kj[col_sub][K_col_idx] = K[K_row_idx][K_col_idx];
                Vj[col_sub][K_col_idx] = V[K_row_idx][K_col_idx];
            }
            __syncthreads();
            scalar_t tmpSum = 0.0;
            #pragma unroll
            for(int ks=0; ks<d; ++ks) // Q @ K^T / sqrt(d)
            {
                tmpSum += Qi[row_sub][ks] * Kj[col_sub][ks]; //tau * 
            }
            Si[row_sub][col_sub] = tmpSum;
            __syncthreads();
            scalar_t m_prev = m[row_sub]; // m_{j-1}
            // if (col_sub == 0)
            // {
            //     #pragma unroll
            //     for(kmax=0; kmax<Bc; ++kmax)
            //     {
            //         m[row_sub] = max(m[row_sub], Si[row_sub][kmax]);
            //     }                
            // }
            // __syncthreads(); // necessary
            if (col_sub < offset)
                Si_tmp[row_sub][col_sub] = max(Si[row_sub][col_sub], Si[row_sub][col_sub + offset]);  
            __syncthreads();       
            for(int kmax=1; kmax<offset; kmax*=2)
            {
                if (col_sub % (2*kmax) == 0)
                {
                    if (col_sub + kmax < offset)
                        Si_tmp[row_sub][col_sub] = max(Si_tmp[row_sub][col_sub], Si_tmp[row_sub][col_sub + kmax]);
                }
                __syncthreads();
            }
            if (col_sub == 0)
                m[row_sub] = max(m[row_sub], Si_tmp[row_sub][0]);
            __syncthreads();
            Si[row_sub][col_sub] = exp(Si[row_sub][col_sub] - m[row_sub]); // compute P_i^j, we reuse Si to save memory          
            __syncthreads();
            // if (col_sub == 0)
            // {
            //     scalar_t tmpSum = 0.0;
            //     #pragma unroll
            //     for(kl=0; kl<Bc; ++kl)
            //     {
            //         tmpSum += Si[row_sub][kl];                    
            //     }
            //     l[row_sub] = exp(m_prev - m[row_sub]) * l[row_sub] + tmpSum;
            // }            
            // __syncthreads();  
            if (col_sub < offset)
                Si_tmp[row_sub][col_sub] = Si[row_sub][col_sub] + Si[row_sub][col_sub + offset];  
            __syncthreads();       
            for(int kmax=1; kmax<offset; kmax*=2)
            {
                if (col_sub % (2*kmax) == 0)
                {
                    if (col_sub + kmax < offset)
                        Si_tmp[row_sub][col_sub] = Si_tmp[row_sub][col_sub] + Si_tmp[row_sub][col_sub + kmax];
                }
                __syncthreads();
            }
            if (col_sub == 0)
                l[row_sub] = exp(m_prev - m[row_sub]) * l[row_sub] + Si_tmp[row_sub][0];
            __syncthreads();
            #pragma unroll
            for(int kd=0; kd<Tk; ++kd) // d is a multiple of Bc
            {
                int V_col_idx = col_sub + kd*Bc;
                if(V_col_idx >= d) return;
                scalar_t tmpSum = 0.0;
                #pragma unroll
                for(int ko=0; ko<Bc; ++ko) // To calculate PV, load Vj first
                {
                    tmpSum += Si[row_sub][ko] * Vj[ko][V_col_idx]; // PV,                     
                }                      
                Oi[row_sub][V_col_idx] = exp(m_prev - m[row_sub]) * Oi[row_sub][V_col_idx] + tmpSum;                          
            }   
            __syncthreads();
            // O_col_idx = j*Bc + col_sub;
            // O[Q_row_idx][O_col_idx] = tmpSum;
        }
        // __syncthreads();
        // int M_idx = i*Br + row_sub;
        if(col_sub==0) // && M_idx < N
        {
            M[Q_row_idx] = m[row_sub] + logf(l[row_sub]); // compute L
        }
        #pragma unroll
        for(int kq=0; kq<Tk; ++kq) // update O block by block
        {
            int O_col_idx = col_sub + kq*Bc;   
            if (O_col_idx >= d) return;
            O[Q_row_idx][O_col_idx] = (1.0 / l[row_sub]) * Oi[row_sub][O_col_idx];         
        }
    }
}

// #include "cublas_v2.h"

// __global__ void invokeDeviceCublasSgemm(cublasStatus_t *returnValue,
//                                         int n,
//                                         const float *d_alpha,
//                                         const float *d_A,
//                                         const float *d_B,
//                                         const float *d_beta,
//                                         float *d_C)
// {
//     cublasHandle_t cnpHandle;
//     cublasStatus_t status = cublasCreate(&cnpHandle);

//     if (status != CUBLAS_STATUS_SUCCESS)
//     {
//         *returnValue = status;
//         return;
//     }

//     /* Perform operation using cublas */
//     status =
//         cublasSgemm(cnpHandle,
//                     CUBLAS_OP_T, CUBLAS_OP_N,
//                     n, n, n,
//                     d_alpha,
//                     d_A, n,
//                     d_B, n,
//                     d_beta,
//                     d_C, n);

//     cublasDestroy(cnpHandle);

//     *returnValue = status;
// }

template<typename scalar_t>
__global__ void attention_kernel_test(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> Q,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> K,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> V,
    const float tau,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> O,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> M
){
    __shared__ scalar_t Qi[Br][d];
    __shared__ scalar_t Oi[Br][d];
    __shared__ scalar_t Kj[Bc][d];
    __shared__ scalar_t Vj[Bc][d];
    __shared__ scalar_t Si[Br][Bc];
    __shared__ scalar_t Si_tmp[Br][Bc/2];
    __shared__ scalar_t l[Br];
    __shared__ scalar_t m[Br];
    const int offset = Bc / 2;
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int col_sub = threadIdx.x; // Thread column within Csub
    const int row_sub = threadIdx.y; // Thread row within Csub
    const int N = Q.size(0);
    const int Tr = N / (Br * gridDim.x);
    const int Tc = N / Bc;
    const int Tk = d / Bc;
    const int Tk_inner = d / Br;
    // cublasStatus_t *d_status;
    const float alpha = 1.0, beta = 0.0;
    #pragma unroll
    for(int i=0; i<Tr; ++i) // outer loop    
    {     
        int Q_row_idx = i*Br + row_sub + Tr*Br*blockIdx.x;        
        if (Q_row_idx >= Tr*Br*(blockIdx.x+1)) return;
        l[row_sub] = 0.0;
        m[row_sub] = -1.0e-9;  
        #pragma unroll
        for(int kq=0; kq<Tk; ++kq) // load Q from global memory into shared memory
        {
            int Q_col_idx = col_sub + kq*Bc;   
            if (Q_col_idx >= d) return;
            Qi[row_sub][Q_col_idx] = Q[Q_row_idx][Q_col_idx] * tau;
            Oi[row_sub][Q_col_idx] = 0.0;            
        }
        __syncthreads();
        #pragma unroll
        for(int j=0; j<Tc; ++j) // inner loop
        {            
            int K_row_idx = j*Bc + col_sub; // K is a tall matrix and K^T is a wide matrix
            if (K_row_idx >= N) return;
            #pragma unroll
            for(int kk=0; kk<Tk_inner; ++kk) // load K, V from global memory into shared memory
            {
                int K_col_idx = row_sub + kk*Br;   
                if (K_col_idx >= d) return;
                Kj[col_sub][K_col_idx] = K[K_row_idx][K_col_idx];
                Vj[col_sub][K_col_idx] = V[K_row_idx][K_col_idx];
            }
            __syncthreads();
            scalar_t tmpSum = 0.0;
            #pragma unroll
            for(int ks=0; ks<d; ++ks) // matrix multiplication
            {
                tmpSum += Qi[row_sub][ks] * Kj[col_sub][ks];
            }
            Si[row_sub][col_sub] = tmpSum;
            __syncthreads();
            // invokeDeviceCublasSgemm<<<1, 1>>>(d_status, Br, &alpha, Kj, Qi, &beta, Si);
            scalar_t m_prev = m[row_sub]; // m_{j-1}
            if (col_sub < offset)
                Si_tmp[row_sub][col_sub] = max(Si[row_sub][col_sub], Si[row_sub][col_sub + offset]);  
            __syncthreads();       
            for(int kmax=1; kmax<offset; kmax*=2)
            {
                if (col_sub % (2*kmax) == 0)
                {
                    if (col_sub + kmax < offset)
                        Si_tmp[row_sub][col_sub] = max(Si_tmp[row_sub][col_sub], Si_tmp[row_sub][col_sub + kmax]);
                }
                __syncthreads();
            }
            if (col_sub == 0)
                m[row_sub] = max(m[row_sub], Si_tmp[row_sub][0]);
            __syncthreads();
            Si[row_sub][col_sub] = exp(Si[row_sub][col_sub] - m[row_sub]); // compute P_i^j, we reuse Si to save memory          
            __syncthreads();
            if (col_sub < offset) // load and add before for-loop
                Si_tmp[row_sub][col_sub] = Si[row_sub][col_sub] + Si[row_sub][col_sub + offset];  
            __syncthreads();       
            for(int kmax=1; kmax<offset; kmax*=2) // reduction for l
            {
                if (col_sub % (2*kmax) == 0)
                {
                    if (col_sub + kmax < offset)
                        Si_tmp[row_sub][col_sub] = Si_tmp[row_sub][col_sub] + Si_tmp[row_sub][col_sub + kmax];
                }
                __syncthreads();
            }
            if (col_sub == 0)
                l[row_sub] = exp(m_prev - m[row_sub]) * l[row_sub] + Si_tmp[row_sub][0];
            __syncthreads();
            #pragma unroll
            for(int kd=0; kd<Tk; ++kd)
            {
                int V_col_idx = col_sub + kd*Bc;
                if(V_col_idx >= d) return;
                scalar_t tmpSum = 0.0;
                #pragma unroll
                for(int ko=0; ko<Bc; ++ko) // To calculate PV, load Vj first
                {
                    tmpSum += Si[row_sub][ko] * Vj[ko][V_col_idx]; // PV,                     
                }                      
                Oi[row_sub][V_col_idx] = exp(m_prev - m[row_sub]) * Oi[row_sub][V_col_idx] + tmpSum;                          
            }   
            __syncthreads();
        }   
        if(col_sub==0) // && M_idx < N
        {
            M[Q_row_idx] = m[row_sub] + logf(l[row_sub]); // compute L
            // M[Q_row_idx] = l[row_sub];
        } 
        #pragma unroll
        for(int kq=0; kq<Tk; ++kq) // update O block by block
        {
            int O_col_idx = col_sub + kq*Bc;   
            if (O_col_idx >= d) return;
            O[Q_row_idx][O_col_idx] = (1.0 / l[row_sub]) * Oi[row_sub][O_col_idx];           
        }        
    }
}

#define d_ 32

__device__ void rowMax_warpReduce(volatile float* Si2, int row_sub, int col_sub)
{
    Si2[row_sub * Bc / 2 + col_sub] = max(Si2[row_sub * Bc / 2 + col_sub], Si2[row_sub * Bc / 2 + col_sub + 4]);
    Si2[row_sub * Bc / 2 + col_sub] = max(Si2[row_sub * Bc / 2 + col_sub], Si2[row_sub * Bc / 2 + col_sub + 2]);
    Si2[row_sub * Bc / 2 + col_sub] = max(Si2[row_sub * Bc / 2 + col_sub], Si2[row_sub * Bc / 2 + col_sub + 1]);
}

__device__ void rowSum_warpReduce(volatile float* Si2, int row_sub, int col_sub)
{
    Si2[row_sub * Bc / 2 + col_sub] += Si2[row_sub * Bc / 2 + col_sub + 4];
    Si2[row_sub * Bc / 2 + col_sub] += Si2[row_sub * Bc / 2 + col_sub + 2];
    Si2[row_sub * Bc / 2 + col_sub] += Si2[row_sub * Bc / 2 + col_sub + 1];
}

template<typename scalar_t>
__global__ void attention_kernel_test_half2(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> Q,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> K,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> V,
    const __half tau,
    torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> O,
    torch::PackedTensorAccessor<float, 1, torch::RestrictPtrTraits, size_t> M,
    const int headDim
){
    extern __shared__ float l[];
    float *normalizer = l;  // 16 * 4Byte = 64Byte
    float *m = (float*)&normalizer[Br];  // 16 * 4Byte = 64Byte
    __half *Qi = (__half*)&m[Br];  // 16 * 128 * 2Byte = 4KB
    __half *Kj = (__half*)&Qi[Br * headDim];  // 16 * 128 * 2Byte = 4KB
    __half *Vj = (__half*)&Kj[Bc * headDim];  // 16 * 128 * 2Byte = 4KB
    float *Oi = (float*)&Vj[Bc * headDim];  // 16 * 128 * 4Byte = 8KB
    float *Si = (float*)&Oi[Br * headDim];  // 16 * 16 * 4Byte = 1KB
    // for reduction
    float *Si2 = (float*)&Si[Br * Bc];  // 16 * 8 * 4Byte = 0.5KB
    __half *Pi = (__half*)&Si2[Br * Bc / 2];  // 16 * 16 * 2Byte = 0.5KB

    const int N = Q.size(0);
    // Here blockIdx.x is used to parallelize sequence length.
    const int Tr = N / (Br * gridDim.x); 
    const int Tc = N / Bc;

    // Declare the fragments for Q @ K^T
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    // P@V
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag1;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag1;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag2;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag1;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag2;

    for (int i = 0; i < Tr; ++i)
    {
        int row_idx_global = blockIdx.x * Br * Tr + i * Br + threadIdx.y;
        // load Qi, initialize Oi and l with 0, m with -inf
        if (threadIdx.x < headDim)
        {
            Qi[threadIdx.y * headDim + threadIdx.x] = __hmul(tau, Q[row_idx_global][threadIdx.x]);
            Oi[threadIdx.y * headDim + threadIdx.x] = 0.0;
            l[threadIdx.y] = 0.0;
            m[threadIdx.y] = -1e-9;
        }

        for (int j = 0; j < Tc; ++j)
        {
            // load Kj, Vj
            if (threadIdx.x < headDim)
            {
                Kj[threadIdx.y * headDim + threadIdx.x] = K[j * Bc + threadIdx.y][threadIdx.x];
                Vj[threadIdx.y * headDim + threadIdx.x] = V[j * Bc + threadIdx.y][threadIdx.x];
            }
            __syncthreads();
            wmma::fill_fragment(c_frag, 0.0);
            for(int k = 0; k < headDim; k += 16)
            {
                // Load the inputs
                wmma::load_matrix_sync(a_frag, &Qi[k], headDim);
                wmma::load_matrix_sync(b_frag, &Kj[k], headDim);
                // Perform the matrix multiplication
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }            
            // Store the output
            wmma::store_matrix_sync(&Si[0], c_frag, 16, wmma::mem_row_major);
            __syncthreads();
            float m_prev = m[threadIdx.y];  // m^{j-1}  
            // max reduction 
            if (threadIdx.x < Bc/2)
            {
                Si2[threadIdx.y * Bc / 2 + threadIdx.x] = max(Si[threadIdx.y * Bc + threadIdx.x], Si[threadIdx.y * Bc + threadIdx.x + Bc / 2]);
            }
            __syncthreads();
            if (threadIdx.x < Bc/4) // suppose Bc = 16
            {
                rowMax_warpReduce(Si2, threadIdx.y, threadIdx.x);
            }   
            if (threadIdx.x == 0)
            {
                m[threadIdx.y] = max(m[threadIdx.y], Si2[threadIdx.y * Bc / 2]);
            }
            __syncthreads();  
            if (threadIdx.x < Bc)
            {
                Si[threadIdx.y * Bc + threadIdx.x] = exp(Si[threadIdx.y * Bc + threadIdx.x] - m[threadIdx.y]);
            }            
            __syncthreads();
            // sum reduction
            if (threadIdx.x < Bc/2)
            {
                Si2[threadIdx.y * Bc / 2 + threadIdx.x] = Si[threadIdx.y * Bc + threadIdx.x] + Si[threadIdx.y * Bc + threadIdx.x + Bc / 2];
            }
            __syncthreads();
            if (threadIdx.x < Bc/4) // suppose Bc = 16
            {
                rowSum_warpReduce(Si2, threadIdx.y, threadIdx.x);
            } 
            if (threadIdx.x == 0)
            {
                l[threadIdx.y] = exp(m_prev - m[threadIdx.y]) * l[threadIdx.y] + Si2[threadIdx.y * Bc / 2];
            }
            __syncthreads(); 
            if (threadIdx.x < Bc)
            {
                Pi[threadIdx.y * Bc + threadIdx.x] = __float2half(Si[threadIdx.y * Bc + threadIdx.x]);  // convert float into __half         
            }   
            __syncthreads();       
            // store Oi with registers
            float ele_batch = 1.0;
            if (threadIdx.x < headDim)
            {
                ele_batch = Oi[threadIdx.y * headDim + threadIdx.x];   
                // float ele_2_batch = Oi[threadIdx.y * headDim + threadIdx.x];            
            }
            __syncthreads();   
            wmma::fill_fragment(c_frag1, 0.0);
            wmma::fill_fragment(c_frag2, 0.0);
            wmma::load_matrix_sync(a_frag1, &Pi[0], Bc);            
            // Initialize the output to zero           
            wmma::load_matrix_sync(b_frag1, &Vj[0], headDim);
            wmma::load_matrix_sync(b_frag2, &Vj[Bc], headDim);
            wmma::mma_sync(c_frag1, a_frag1, b_frag1, c_frag1);
            wmma::mma_sync(c_frag2, a_frag1, b_frag2, c_frag2);
            wmma::store_matrix_sync(&Oi[0], c_frag1, headDim, wmma::mem_row_major);
            wmma::store_matrix_sync(&Oi[Bc], c_frag2, headDim, wmma::mem_row_major);
            __syncthreads();
            if (threadIdx.x < headDim)
            {
                Oi[threadIdx.y * headDim + threadIdx.x] += exp(m_prev - m[threadIdx.y]) * ele_batch;
            }
            
        }
        __syncthreads();
        M[blockIdx.x * Br * Tr + i * Br + threadIdx.y] = m[threadIdx.y] + logf(l[threadIdx.y]);
        if (threadIdx.x < headDim && threadIdx.y < Br)
        {
            O[row_idx_global][threadIdx.x] = Oi[threadIdx.y * headDim + threadIdx.x] / l[threadIdx.y];             
        }
    }
    
    // if (blockIdx.x == 0 && threadIdx.x == 0)
    // {
    //     printf("l[%d]= %f\n", threadIdx.y, M[threadIdx.y + blockIdx.x * Br * Tr]);
    // }
}

template<typename scalar_t>
__global__ void attention_kernel_test_half(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> Q,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> K,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> V,
    const float tau,
    torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> O,
    torch::PackedTensorAccessor<float, 1, torch::RestrictPtrTraits, size_t> M
){
    __shared__ __half Qi[Br][d_];    
    // __shared__ half Kj[Bc][d];
    __shared__ __half Kj[Bc][d_];
    extern __shared__ float l[];
    float *normalizer = l;
    float *m = (float*)&normalizer[Br];
    float *Oi_ = (float*)&m[Br];
    __half *Vj_ = (__half*)&Oi_[Br * d_];
    float *Si_ = (float*)&Vj_[Bc * d_];
    float *Si_tmp_ = (float*)&Si_[Br * Bc]; // for 2d indexing
    __half *Pi_ = (__half*)&Si_tmp_[Br * Bc / 2];
    float (*Oi)[d_] = reinterpret_cast<float (*)[d_]>(Oi_);
    __half (*Vj)[d_] = reinterpret_cast<__half (*)[d_]>(Vj_);
    float (*Si)[Bc] = reinterpret_cast<float (*)[Bc]>(Si_);
    float (*Si_tmp)[Bc/2] = reinterpret_cast<float (*)[Bc/2]>(Si_tmp_);
    __half (*Pi)[Bc] = reinterpret_cast<__half (*)[Bc]>(Pi_);
    // __shared__ float Oi[Br][d_];    
    // __shared__ float Vj[Bc][d_];
    // __shared__ float Si[Br][Bc];
    // __shared__ float Si_tmp[Br][Bc/2];
    // __shared__ float l[Br];
    // __shared__ float m[Br];
    const int offset = Bc / 2;
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int col_sub = threadIdx.x; // Thread column within Csub
    const int row_sub = threadIdx.y; // Thread row within Csub
    const int N = Q.size(0);
    const int Tr = N / (Br * gridDim.x);
    const int Tc = N / Bc;
    const int Tk = d_ / Bc;
    const int Tk_inner = d_ / Br;
    // const half tau_half =  __float2half(tau);
    const at::Half tau_half =  __float2half(tau);
    // Declare the fragments for Q @ K^T
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    // P@V
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag1;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag1;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag2;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag1;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag2;
    #pragma unroll
    for(int i = 0; i < Tr; ++i) // outer loop    
    {     
        int Q_row_idx = i * Br + row_sub + Tr * Br * blockIdx.x;        
        if (Q_row_idx >= Tr * Br * (blockIdx.x + 1)) 
        {
            return;
        }
        if (col_sub == 0)
        {
            l[row_sub] = 0.0;
            m[row_sub] = -1.0e-9; 
        } 
        #pragma unroll
        for(int kq = 0; kq < Tk; ++kq) // load Q from global memory into shared memory
        {
            int Q_col_idx = col_sub + kq * Bc;   
            if (Q_col_idx >= d_) 
            {
                return;
            }
            Qi[row_sub][Q_col_idx] = __hmul(tau_half, Q[Q_row_idx][Q_col_idx]);
            Oi[row_sub][Q_col_idx] = 0.0;
        }
        __syncthreads();
        #pragma unroll
        for(int j = 0; j < Tc; ++j) // inner loop
        {            
            int K_row_idx = j * Bc + col_sub; // K is a tall matrix and K^T is a wide matrix
            if (K_row_idx >= N) 
            {
                return;
            }
            #pragma unroll
            for(int kk = 0; kk < Tk_inner; ++kk) // load K, V from global memory into shared memory
            {
                int K_col_idx = row_sub + kk * Br;   
                if (K_col_idx >= d_) 
                {
                    return;
                }
                Kj[col_sub][K_col_idx] = K[K_row_idx][K_col_idx];
                Vj[col_sub][K_col_idx] = V[K_row_idx][K_col_idx];
            }
            __syncthreads();
            // __half tmpSum = __float2half(0);
            // float tmpSum = 0.0f;
            // #pragma unroll // Q @ K^T
            // for(int ks = 0; ks < d; ++ks) // matrix multiplication
            // {   
            //     tmpSum += __half2float(__hmul(Qi[row_sub][ks], Kj[col_sub][ks]));
            //     // tmpSum += Qi[row_sub][ks] * Kj[col_sub][ks];
            // }
            // // Si[row_sub][col_sub] = __half2float(tmpSum);
            // Si[row_sub][col_sub] = tmpSum;
            // __syncthreads();            
            // Initialize the output to zero
            wmma::fill_fragment(c_frag, 0.0);
            for(int k = 0; k < d_; k += 16)
            {
                // Load the inputs
                wmma::load_matrix_sync(a_frag, &Qi[0][k], d_);
                wmma::load_matrix_sync(b_frag, &Kj[0][k], d_);
                // Perform the matrix multiplication
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }            
            // Store the output
            // wmma::store_matrix_sync(C[0], c_frag, 16, wmma::mem_row_major);
            wmma::store_matrix_sync(&Si[0][0], c_frag, 16, wmma::mem_row_major);
            float m_prev = m[row_sub]; // m_{j-1}
            if (col_sub < offset) // reduction 
                Si_tmp[row_sub][col_sub] = max(Si[row_sub][col_sub], Si[row_sub][col_sub + offset]);  
            __syncthreads(); 
            for(int kmax = 1; kmax < offset; kmax *= 2)
            {   // reduction
                if (col_sub % (2 * kmax) == 0)
                {
                    if (col_sub + kmax < offset)
                        Si_tmp[row_sub][col_sub] = max(Si_tmp[row_sub][col_sub], Si_tmp[row_sub][col_sub + kmax]);
                }
                __syncthreads();
            }
            if (col_sub == 0) // update m
                m[row_sub] = max(m[row_sub], Si_tmp[row_sub][0]);
            __syncthreads();
            // compute P_i^j, we reuse Si to save memory  
            Si[row_sub][col_sub] = exp(Si[row_sub][col_sub] - m[row_sub]);         
            __syncthreads();
            // first reduction for l
            if (col_sub < offset) // load and add before for-loop
                Si_tmp[row_sub][col_sub] = Si[row_sub][col_sub] + Si[row_sub][col_sub + offset];  
            __syncthreads();    
            Pi[row_sub][col_sub] = __float2half(Si[row_sub][col_sub]);  // convert float Pi into half Pi
            // further reduction for l 
            for(int kmax = 1; kmax < offset; kmax *= 2) 
            {
                if (col_sub % (2 * kmax) == 0)
                {
                    if (col_sub + kmax < offset)
                        Si_tmp[row_sub][col_sub] += Si_tmp[row_sub][col_sub + kmax];
                }
                __syncthreads();
            }
            // update l
            if (col_sub == 0)
                l[row_sub] = exp(m_prev - m[row_sub]) * l[row_sub] + Si_tmp[row_sub][0];
            
            __syncthreads();
            // #pragma unroll
            // for(int kd = 0; kd < Tk; ++kd)
            // {
            //     int V_col_idx = col_sub + kd * Bc;
            //     // check boundary conditions
            //     if(V_col_idx >= d_) 
            //     {
            //         return;
            //     }
            //     float tmpSum = 0.0;
            //     #pragma unroll // To calculate PV (matrix multiplication)
            //     for(int ko=0; ko<Bc; ++ko) 
            //     {
            //         tmpSum += __half2float(__hmul(Pi[row_sub][ko], Vj[ko][V_col_idx])); // PV                    
            //     }
            //     // update Oi           
            //     Oi[row_sub][V_col_idx] = exp(m_prev - m[row_sub]) * Oi[row_sub][V_col_idx] + tmpSum;                          
            // }   
            // __syncthreads();
            // store Oi with registers
            float ele_1_batch = Oi[row_sub][col_sub];
            float ele_2_batch = Oi[row_sub][col_sub + Bc];
            // Initialize the output to zero
            wmma::fill_fragment(c_frag1, 0.0);
            wmma::fill_fragment(c_frag2, 0.0);
            wmma::load_matrix_sync(a_frag1, &Pi[0][0], Bc);
            wmma::load_matrix_sync(b_frag1, &Vj[0][0], d_);
            wmma::load_matrix_sync(b_frag2, &Vj[0][Bc], d_);
            wmma::mma_sync(c_frag1, a_frag1, b_frag1, c_frag1);
            wmma::mma_sync(c_frag2, a_frag1, b_frag2, c_frag2);
            // #pragma unroll
            // for(int kd = 0; kd < Tk; ++kd)
            // {
            // int V_col_idx = col_sub + kd * Bc;
            // c_frag1.x[row_sub * d_ + col_sub] += exp(m_prev - m[row_sub]) * Oi[row_sub][col_sub];
            // c_frag2.x[row_sub * d_ + col_sub + Bc] += exp(m_prev - m[row_sub]) * Oi[row_sub][col_sub + Bc];
            // }
            // __syncthreads();
            // wmma::store_matrix_sync(&Oi[0][0], c_frag1, d_, wmma::mem_row_major);
            // wmma::store_matrix_sync(&Oi[0][Bc], c_frag2, d_, wmma::mem_row_major);
            wmma::store_matrix_sync(&Oi[0][0], c_frag1, d_, wmma::mem_row_major);
            wmma::store_matrix_sync(&Oi[0][Bc], c_frag2, d_, wmma::mem_row_major);
            Oi[row_sub][col_sub] = exp(m_prev - m[row_sub]) * ele_1_batch + Oi[row_sub][col_sub];
            Oi[row_sub][col_sub + Bc] = exp(m_prev - m[row_sub]) * ele_2_batch + Oi[row_sub][col_sub + Bc];            
        }  
        __syncthreads(); 
        // compute logsumexp L
        if(col_sub == 0) // && M_idx < N
        {
            M[Q_row_idx] = m[row_sub] + logf(l[row_sub]); // compute L
            // M[Q_row_idx] = l[row_sub];
        } 
        #pragma unroll // update O block by block
        for(int kq = 0; kq < Tk; ++kq) 
        {
            int O_col_idx = col_sub + kq * Bc;   
            if (O_col_idx >= d_) return;
            O[Q_row_idx][O_col_idx] = (1.0 / l[row_sub]) * Oi[row_sub][O_col_idx];           
        }        
    }
}

std::vector<torch::Tensor> attention_half_test2(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    const float tau,
    const int headDim,
    const int iDev
){
    // int iDev = 2;
	cudaError_t error = cudaSetDevice(iDev);
    int maxbytes = 16*4*2 + 16*headDim*2*3 + 16*headDim*4 + 16*16*4 + 16*8*4 + 16*16*2;
    cudaFuncSetAttribute(attention_kernel_test_half<at::Half>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    const int N = Q.size(0);//, N = K.size(0);
    const __half tau_half = __float2half(tau);
    torch::Tensor O = torch::empty({N, headDim}, torch::dtype(kFloat).device(Q.device()));
    torch::Tensor M = torch::empty({N}, torch::dtype(kFloat).device(Q.device()));
    // custom dtype
    // torch::zeros({N, F}, torch::dtype(kInt32).device(feats.device)); 
    const dim3 threads(64, Br);
    const dim3 blocks(N / Br, 1);
    // printf("Br: %d, Bc: %d\n", Br, Bc);
    // launch a kernel function
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(V.type(), "attention_h",
    ([&]
        {
            attention_kernel_test_half2<at::Half><<<blocks, threads, maxbytes>>>(
                Q.packed_accessor<at::Half, 2, torch::RestrictPtrTraits, size_t>(), 
                K.packed_accessor<at::Half, 2, torch::RestrictPtrTraits, size_t>(),
                V.packed_accessor<at::Half, 2, torch::RestrictPtrTraits, size_t>(),
                tau_half,
                O.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
                M.packed_accessor<float, 1, torch::RestrictPtrTraits, size_t>(),
                headDim
            );
        }
    ));
    cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
  		printf("ERROR: %s \n", cudaGetErrorString(err));
	}
    return {O, M};
}

std::vector<torch::Tensor> attention_half_test(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    const float tau,
    const int iDev
){
    // int iDev = 2;
	cudaError_t error = cudaSetDevice(iDev);
	// if(error != cudaSuccess)
	// {
	// 	printf("failed to set GPU %d for computing.\n", iDev);
	// 	exit(-1);
	// }
	// else
	// {
	// 	printf("set GPU %d for computing.\n", iDev);
	// }
    // cudaFuncCache cacheConfig = cudaFuncCachePreferShared;
    // cudaDeviceSetCacheConfig(cacheConfig);
    // cudaFuncSetCacheConfig(attention_kernel_test_half, cacheConfig);
    int maxbytes = 98304;
    cudaFuncSetAttribute(attention_kernel_test_half<at::Half>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    const int N = Q.size(0);//, N = K.size(0);
    torch::Tensor O = torch::empty({N, d_}, torch::dtype(kFloat).device(Q.device()));
    torch::Tensor M = torch::empty({N}, torch::dtype(kFloat).device(Q.device()));
    // custom dtype
    // torch::zeros({N, F}, torch::dtype(kInt32).device(feats.device)); 
    const dim3 threads(Bc, Br);
    const dim3 blocks(N / Br, 1);
    // printf("Br: %d, Bc: %d\n", Br, Bc);
    // launch a kernel function
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(V.type(), "attention_h",
    ([&]
        {
            attention_kernel_test_half<at::Half><<<blocks, threads, maxbytes>>>(
                // .packed_accessor is only used for tensor
                // Q.packed_accessor<half, 2, torch::RestrictPtrTraits, size_t>(), 
                // K.packed_accessor<half, 2, torch::RestrictPtrTraits, size_t>(),
                Q.packed_accessor<at::Half, 2, torch::RestrictPtrTraits, size_t>(), 
                K.packed_accessor<at::Half, 2, torch::RestrictPtrTraits, size_t>(),
                V.packed_accessor<at::Half, 2, torch::RestrictPtrTraits, size_t>(),
                tau,
                O.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
                M.packed_accessor<float, 1, torch::RestrictPtrTraits, size_t>()
            );
        }
    ));
    cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
  		printf("ERROR: %s \n", cudaGetErrorString(err));
	}
    return {O, M};
}

std::vector<torch::Tensor> attention_Br_Bc(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    const float tau,
    const int iDev
){
    // int iDev = 2;
	cudaError_t error = cudaSetDevice(iDev);
	// if(error != cudaSuccess)
	// {
	// 	printf("failed to set GPU %d for computing.\n", iDev);
	// 	exit(-1);
	// }
	// else
	// {
	// 	printf("set GPU %d for computing.\n", iDev);
	// }
    const int N = Q.size(0);//, N = K.size(0);
    torch::Tensor O = torch::empty({N, d}, Q.options());
    torch::Tensor M = torch::empty({N}, Q.options());
    // custom dtype
    // torch::zeros({N, F}, torch::dtype(kInt32).device(feats.device)); 
    const dim3 threads(Bc, Br);
    const dim3 blocks(N / Br, 1);
    // printf("Br: %d, Bc: %d\n", Br, Bc);
    // launch a kernel function
    AT_DISPATCH_FLOATING_TYPES(Q.type(), "attention_Br_Bc",
    ([&]
        {
            attention_kernel_test<scalar_t><<<blocks, threads>>>(
                Q.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(), // .packed_accessor is only used for tensor
                K.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                V.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                tau,
                O.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                M.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>()
            );
        }
    ));
    cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
  		printf("ERROR: %s \n", cudaGetErrorString(err));
	}

    return {O, M};
}

std::vector<torch::Tensor> attention_shared_memory(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    const float tau,
    const int iDev
){
    // int iDev = 2;
	cudaError_t error = cudaSetDevice(iDev);
	// if(error != cudaSuccess)
	// {
	// 	printf("failed to set GPU %d for computing.\n", iDev);
	// 	exit(-1);
	// }
	// else
	// {
	// 	printf("set GPU %d for computing.\n", iDev);
	// }
    const int N = Q.size(0);//, N = K.size(0);
    torch::Tensor O = torch::empty({N, d}, Q.options());
    torch::Tensor M = torch::empty({N}, Q.options());
    // custom dtype
    // torch::zeros({N, F}, torch::dtype(kInt32).device(feats.device)); 
    const dim3 threads(Bc, Br);
    const dim3 blocks(N / Br, 1);
    // printf("Br: %d, Bc: %d\n", Br, Bc);
    // launch a kernel function
    AT_DISPATCH_FLOATING_TYPES(Q.type(), "attention_shared_memory",
    ([&]
        {
            attention_kernel<scalar_t><<<blocks, threads>>>(
                Q.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(), // .packed_accessor is only used for tensor
                K.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                V.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                tau,
                O.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                M.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>()
            );
        }
    ));
    cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
  		printf("ERROR: %s \n", cudaGetErrorString(err));
	}

    return {O, M};
}
    

torch::Tensor matmul_cuda(
    torch::Tensor A,
    torch::Tensor B
){
    const int N = A.size(0), M = B.size(1);
    torch::Tensor C = torch::zeros({N, M}, A.options());
    // custom dtype
    // torch::zeros({N, F}, torch::dtype(kInt32).device(feats.device)); 
    const dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 blocks((N + threads.x - 1)/threads.x, (M + threads.y -1)/threads.y);
    // launch a kernel function
    AT_DISPATCH_FLOATING_TYPES(A.type(), "matmul_cuda",
    ([&]
        {
            matmul_kernel<scalar_t><<<blocks, threads>>>(
                A.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(), // .packed_accessor is only used for tensor
                B.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                C.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
            );
        }
    ));

    return C;
}


torch::Tensor matmul_cuda_shared_memory(
    torch::Tensor A,
    torch::Tensor B
){
    const int N = A.size(0), M = B.size(1);
    torch::Tensor C = torch::zeros({N, M}, A.options());
    // custom dtype
    // torch::zeros({N, F}, torch::dtype(kInt32).device(feats.device)); 
    const dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y -1) / threads.y);
    // launch a kernel function
    AT_DISPATCH_FLOATING_TYPES(A.type(), "matmul_cuda_shared_memory",
    ([&]
        {
            matmul_shared_memory_kernel<scalar_t><<<blocks, threads>>>(
                A.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(), // .packed_accessor is only used for tensor
                B.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                C.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
            );
        }
    ));

    return C;
}
