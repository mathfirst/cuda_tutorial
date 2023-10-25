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
