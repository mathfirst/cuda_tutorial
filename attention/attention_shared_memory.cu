#include<torch/extension.h>

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


#define Br 2
#define Bc 16
#define d 32

template<typename scalar_t>
__global__ void attention_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> Q,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> K,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> V,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> O,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> M
){
    __shared__ scalar_t Qi[Br][d];
    __shared__ scalar_t Oi[Br][d];
    __shared__ scalar_t Kj[Bc][d];
    __shared__ scalar_t Vj[Bc][d];
    __shared__ scalar_t Si[Br][Bc];
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
    int i, j, kq, kk, ks, ko, kmax, kl, kd, Q_col_idx, K_row_idx, K_col_idx, Q_row_idx, O_col_idx, V_col_idx;
    #pragma unroll
    for(i=0; i<Tr; ++i) // outer loop    
    {     
        Q_row_idx = i*Br + row_sub + Tr*Br*blockIdx.x;        
        if (Q_row_idx >= Tr*Br*(blockIdx.x+1)) return;
        l[row_sub] = 0.0;
        m[row_sub] = -1.0e-9;  
        #pragma unroll
        for(kq=0; kq<Tk; ++kq) // load Q from global memory into shared memory
        {
            // printf("k: %d, Tk: %d, Q[0][0]: %f\n", kq, Tk, Q[0][0]);
            Q_col_idx = col_sub + kq*Bc;   
            // printf("i: %d, Tr: %d, Tk: %d, Q_row_idx: %d, Q_col_idx: %d\n", i, Tr, Tk, Q_row_idx, Q_col_idx);
            if (Q_col_idx >= d) return;
            Qi[row_sub][Q_col_idx] = Q[Q_row_idx][Q_col_idx];
            // printf("i: %d, Tr: %d, Tk: %d, Q_row_idx: %d\n", i, Tr, Tk, Q_row_idx);
            // printf("Qi[%d][%d]: %f\n", row_sub, Q_col_idx, Qi[row_sub][Q_col_idx]);
            Oi[row_sub][Q_col_idx] = 0.0;            
        }
        __syncthreads();
        #pragma unroll
        for(j=0; j<Tc; ++j) // inner loop
        {            
            K_row_idx = j*Bc + col_sub; // K is a tall matrix and K^T is a wide matrix
            if (K_row_idx >= N) return;
            #pragma unroll
            for(kk=0; kk<Tk_inner; ++kk) // load K, V from global memory into shared memory
            {
                K_col_idx = row_sub + kk*Br;   
                if (K_col_idx >= d) return;
                Kj[col_sub][K_col_idx] = K[K_row_idx][K_col_idx];
                Vj[col_sub][K_col_idx] = V[K_row_idx][K_col_idx];
            }
            __syncthreads();
            // printf("Kj[%d][%d]: %f\n", row_sub, K_col_idx, Kj[row_sub][K_col_idx]);
            scalar_t tmpSum = 0.0;
            #pragma unroll
            for(ks=0; ks<d; ++ks)
            {
                tmpSum += Qi[row_sub][ks] * Kj[col_sub][ks];
            }
            Si[row_sub][col_sub] = tmpSum;
            __syncthreads();
            scalar_t m_prev = m[row_sub]; // m_{j-1}
            if (col_sub == 0)
            {
                #pragma unroll
                for(kmax=0; kmax<Bc; ++kmax)
                {
                    m[row_sub] = max(m[row_sub], Si[row_sub][kmax]);
                }                
            }
            __syncthreads(); // necessary
            Si[row_sub][col_sub] = exp(Si[row_sub][col_sub] - m[row_sub]); // compute P_i^j, we reuse Si to save memory          
            __syncthreads();
            if (col_sub == 0)
            {
                scalar_t tmpSum = 0.0;
                #pragma unroll
                for(kl=0; kl<Bc; ++kl)
                {
                    tmpSum += Si[row_sub][kl];                    
                }
                l[row_sub] = exp(m_prev - m[row_sub]) * l[row_sub] + tmpSum;
            }            
            __syncthreads();  
            #pragma unroll
            for(kd=0; kd<Tk; ++kd)
            {
                V_col_idx = col_sub + kd*Bc;
                if(V_col_idx >= d) return;
                scalar_t tmpSum = 0.0;
                #pragma unroll
                for(ko=0; ko<Bc; ++ko) // To calculate PV, load Vj first
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
            M[Q_row_idx] = l[row_sub];
        }
        #pragma unroll
        for(kq=0; kq<Tk; ++kq) // update O block by block
        {
            O_col_idx = col_sub + kq*Bc;   
            if (O_col_idx >= d) return;
            O[Q_row_idx][O_col_idx] = (1.0 / l[row_sub]) * Oi[row_sub][O_col_idx];  
            // if (Q_row_idx == 0)  
            //     printf("O[%d][%d]: %f, 1.0 / l[row_sub]: %f\n", Q_row_idx, O_col_idx, O[Q_row_idx][O_col_idx], (1.0 / l[row_sub]));          
        }
    }
}

template<typename scalar_t>
__global__ void attention_kernel_test(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> Q,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> K,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> V,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> O,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> M
){
    __shared__ scalar_t Qi[Br][d];
    __shared__ scalar_t Oi[Br][d];
    __shared__ scalar_t Kj[Bc][d];
    __shared__ scalar_t Vj[Bc][d];
    __shared__ scalar_t Si[Br][Bc];
    __shared__ scalar_t l[Br];
    __shared__ scalar_t m[Br];
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int col_sub = threadIdx.x; // Thread column within Csub
    const int row_sub = threadIdx.y; // Thread row within Csub
    const int N = Q.size(0);
    const int Tr = N / (Br * gridDim.x);
    const int Tc = N / Bc;
    const int Tk = d / Bc;
    const int Tk_inner = d / Br;
    int i, j, kq, kk, ks, ko, kmax, kl, kd, Q_col_idx, K_row_idx, K_col_idx, Q_row_idx, O_col_idx, V_col_idx;
    #pragma unroll
    for(i=0; i<Tr; ++i) // outer loop    
    {     
        Q_row_idx = i*Br + row_sub + Tr*Br*blockIdx.x;        
        if (Q_row_idx >= Tr*Br*(blockIdx.x+1)) return;
        l[row_sub] = 0.0;
        m[row_sub] = -1.0e-9;  
        #pragma unroll
        for(kq=0; kq<Tk; ++kq) // load Q from global memory into shared memory
        {
            // printf("k: %d, Tk: %d, Q[0][0]: %f\n", kq, Tk, Q[0][0]);
            Q_col_idx = col_sub + kq*Bc;   
            // printf("i: %d, Tr: %d, Tk: %d, Q_row_idx: %d, Q_col_idx: %d\n", i, Tr, Tk, Q_row_idx, Q_col_idx);
            if (Q_col_idx >= d) return;
            Qi[row_sub][Q_col_idx] = Q[Q_row_idx][Q_col_idx];
            // printf("i: %d, Tr: %d, Tk: %d, Q_row_idx: %d\n", i, Tr, Tk, Q_row_idx);
            // printf("Qi[%d][%d]: %f\n", row_sub, Q_col_idx, Qi[row_sub][Q_col_idx]);
            Oi[row_sub][Q_col_idx] = 0.0;            
        }
        __syncthreads();
        #pragma unroll
        for(j=0; j<Tc; ++j) // inner loop
        {            
            K_row_idx = j*Bc + col_sub; // K is a tall matrix and K^T is a wide matrix
            if (K_row_idx >= N) return;
            #pragma unroll
            for(kk=0; kk<Tk_inner; ++kk) // load K, V from global memory into shared memory
            {
                K_col_idx = row_sub + kk*Br;   
                if (K_col_idx >= d) return;
                Kj[col_sub][K_col_idx] = K[K_row_idx][K_col_idx];
                Vj[col_sub][K_col_idx] = V[K_row_idx][K_col_idx];
            }
            __syncthreads();
            // printf("Kj[%d][%d]: %f, row_sub: %d\n", col_sub, K_col_idx, Kj[col_sub][K_col_idx], row_sub);
            scalar_t tmpSum = 0.0;
            #pragma unroll
            for(ks=0; ks<d; ++ks)
            {
                tmpSum += Qi[row_sub][ks] * Kj[col_sub][ks];
            }
            Si[row_sub][col_sub] = tmpSum;
            __syncthreads();
            scalar_t m_prev = m[row_sub]; // m_{j-1}
            if (col_sub == 0)
            {
                #pragma unroll
                for(kmax=0; kmax<Bc; ++kmax)
                {
                    m[row_sub] = max(m[row_sub], Si[row_sub][kmax]);
                }                
            }
            __syncthreads();  
            Si[row_sub][col_sub] = exp(Si[row_sub][col_sub] - m[row_sub]); // compute P_i^j, we reuse Si to save memory          
            __syncthreads();
            if (col_sub == 0)
            {
                scalar_t tmpSum = 0.0;
                #pragma unroll
                for(kl=0; kl<Bc; ++kl)
                {
                    tmpSum += Si[row_sub][kl];                    
                }
                l[row_sub] = exp(m_prev - m[row_sub]) * l[row_sub] + tmpSum;
            }            
            __syncthreads();  
            // O_col_idx = j*Bc + col_sub;
            // O[Q_row_idx][O_col_idx] = tmpSum;
            #pragma unroll
            for(kd=0; kd<Tk; ++kd)
            {
                V_col_idx = col_sub + kd*Bc;
                if(V_col_idx >= d) return;
                scalar_t tmpSum = 0.0;
                #pragma unroll
                for(ko=0; ko<Bc; ++ko) // To calculate PV, load Vj first
                {
                    tmpSum += Si[row_sub][ko] * Vj[ko][V_col_idx]; // PV,                     
                }                      
                Oi[row_sub][V_col_idx] = exp(m_prev - m[row_sub]) * Oi[row_sub][V_col_idx] + tmpSum;                          
            }   
            __syncthreads();
        }   
        if(col_sub==0) // && M_idx < N
        {
            M[Q_row_idx] = l[row_sub];
        } 
        #pragma unroll
        for(kq=0; kq<Tk; ++kq) // update O block by block
        {
            O_col_idx = col_sub + kq*Bc;   
            if (O_col_idx >= d) return;
            O[Q_row_idx][O_col_idx] = (1.0 / l[row_sub]) * Oi[row_sub][O_col_idx];  
            // if (Q_row_idx == 0)  
            //     printf("O[%d][%d]: %f, 1.0 / l[row_sub]: %f\n", Q_row_idx, O_col_idx, O[Q_row_idx][O_col_idx], (1.0 / l[row_sub]));          
        }        
    }
}

std::vector<torch::Tensor> attention_Br_Bc(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
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
