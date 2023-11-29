#include<torch/extension.h>
#include "cuda_fp16.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x "must be a Cuda tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x "must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


torch::Tensor matmul_cuda(
    torch::Tensor A,
    torch::Tensor B
);

torch::Tensor matmul_cuda_shared_memory(
    torch::Tensor A,
    torch::Tensor B
);

// torch::Tensor 
std::vector<torch::Tensor> attention_shared_memory(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    const float tau,
    const int iDev
);

std::vector<torch::Tensor> attention_Br_Bc(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    const float tau,
    const int iDev
);

std::vector<torch::Tensor> attention_half_test(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    const float tau,
    const int iDev
);

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
);

std::vector<torch::Tensor> attention_half_test2(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    const float tau,
    const int headDim,
    const int iDev
);