#include<torch/extension.h>

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
