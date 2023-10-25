#include<torch/extension.h>
#include "include/utils.h"

torch::Tensor matmul(
    torch::Tensor A, 
    torch::Tensor B
){
    CHECK_INPUT(A);
    CHECK_INPUT(B);    
    return matmul_cuda(A, B);
}

torch::Tensor matmul_shared_memory(
    torch::Tensor A, 
    torch::Tensor B
){
    CHECK_INPUT(A);
    CHECK_INPUT(B);    
    return matmul_cuda_shared_memory(A, B);
}


// torch::Tensor
std::vector<torch::Tensor> attention(
    torch::Tensor Q, 
    torch::Tensor K,
    torch::Tensor V,
    const int iDev
){
    CHECK_INPUT(Q);
    CHECK_INPUT(K);    
    CHECK_INPUT(V);
    return attention_shared_memory(Q, K, V, iDev);
}

std::vector<torch::Tensor> attention_test(
    torch::Tensor Q, 
    torch::Tensor K,
    torch::Tensor V,
    const int iDev
){
    CHECK_INPUT(Q);
    CHECK_INPUT(K);    
    CHECK_INPUT(V);
    return attention_Br_Bc(Q, K, V, iDev);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("matmul", &matmul);
    m.def("matmul_shared_memory", &matmul_shared_memory);
    m.def("attention_shared_memory", &attention);
    m.def("attention_test", &attention_test);
}