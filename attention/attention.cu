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

#define Br 16
#define Bc 16
#define Bc2 (Bc / 2)  // used to define temporary Si and Pi
#define d 32

// template<typename scalar_t>
// __global__ void attn_kernel_fwd_half(
//     const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> Q,
//     const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> K,
//     const torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> V,
//     const float tau,
//     torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> O,
//     torch::PackedTensorAccessor<float, 1, torch::RestrictPtrTraits, size_t> M
// ){
//     __shared__ __half Qi[Br * d];
//     __shared__ __half Kj[Bc * d];
//     __shared__ float Oi[Br * d];
//     __shared__ float Vj[Bc * d];
//     __shared__ float Si[Br * Bc];
//     __shared__ float Si_tmp[Br * Bc2];
//     __shared__ float l[Br];
//     __shared__ float m[Br];
//     const int col = blockDim.x * blockIdx.x + threadIdx.x;
//     const int row = blockDim.y * blockIdx.y + threadIdx.y;
//     const int col_sub = threadIdx.x; // Thread column within Csub
//     const int row_sub = threadIdx.y; // Thread row within Csub
//     const int seq_len = Q.size(0);
//     const int Tr = seq_len / (Br * gridDim.x);
//     const int Tc = seq_len / Bc;
//     const int Q_block_row_idx = blockIdx;
//     for (int i=0; i < Tr; ++i)
//     {
//         int Q_row_idx = i * Br + row_sub + Tr * Br * Q_block_row_idx; 
//         if (Q_row_idx >= seq_len)
//         {
//             return; 
//         }
//         if (col_sub == 0)
//         {
//             l[row_sub] = 0.0;
//             m[row_sub] = -1.0e-9; 
//         }

//     }
// }

// std::vector<torch::Tensor> attn_half_test(
//     torch::Tensor Q,
//     torch::Tensor K,
//     torch::Tensor V,
//     const float tau,
//     const int iDev
// ){
//     // int iDev = 2;
// 	cudaError_t error = cudaSetDevice(iDev);
// 	if(error != cudaSuccess)
// 	{
// 		printf("failed to set GPU %d for computing.\n", iDev);
// 		exit(-1);
// 	}
// 	// else
// 	// {
// 	// 	printf("set GPU %d for computing.\n", iDev);
// 	// }
//     int maxbytes = 98304;
//     error = cudaFuncSetAttribute(attn_kernel_fwd_half<at::Half>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
//     if(error != cudaSuccess)
// 	{
//         printf("failed to configure Dynamic Shared Memory. ERROR: %s \n", cudaGetErrorString(error));
// 		exit(-1);
// 	}
//     const int N = Q.size(0);//, N = K.size(0);
//     torch::Tensor O = torch::empty({N, d}, V.options());
//     torch::Tensor M = torch::empty({N}, V.options());
//     // custom dtype
//     // torch::zeros({N, F}, torch::dtype(kInt32).device(feats.device)); 
//     const dim3 threads(Bc, Br);
//     const dim3 blocks(N / Br, 1);
//     // printf("Br: %d, Bc: %d\n", Br, Bc);
//     // launch a kernel function
//     AT_DISPATCH_FLOATING_TYPES_AND_HALF(V.type(), "attention_h",
//     ([&]
//         {
//             attn_kernel_fwd_half<at::Half><<<blocks, threads, maxbytes>>>(
//                 // .packed_accessor is only used for tensor
//                 Q.packed_accessor<at::Half, 2, torch::RestrictPtrTraits, size_t>(), 
//                 K.packed_accessor<at::Half, 2, torch::RestrictPtrTraits, size_t>(),
//                 V.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
//                 tau,
//                 O.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
//                 M.packed_accessor<float, 1, torch::RestrictPtrTraits, size_t>()
//             );
//         }
//     ));
//     cudaError_t err = cudaGetLastError();
// 	if (err != cudaSuccess) {
//   		printf("ERROR: %s \n", cudaGetErrorString(err));
// 	}
//     return {O, M};
// }