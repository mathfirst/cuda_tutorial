#include<torch/extension.h>
#include "include/utils.h"
#include "cuda_fp16.h"

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
    const float tau,
    const int iDev
){
    CHECK_INPUT(Q);
    CHECK_INPUT(K);    
    CHECK_INPUT(V);
    return attention_shared_memory(Q, K, V, tau, iDev);
}

std::vector<torch::Tensor> attention_test(
    torch::Tensor Q, 
    torch::Tensor K,
    torch::Tensor V,
    const float tau,
    const int iDev
){
    CHECK_INPUT(Q);
    CHECK_INPUT(K);    
    CHECK_INPUT(V);
    return attention_Br_Bc(Q, K, V, tau, iDev);
}

std::vector<torch::Tensor> attention_test_half(
    torch::Tensor Q, 
    torch::Tensor K,
    torch::Tensor V,
    const float tau,
    const int iDev
){
    CHECK_INPUT(Q);
    CHECK_INPUT(K);    
    CHECK_INPUT(V);
    return attention_half_test(Q, K, V, tau, iDev);
}

std::vector<torch::Tensor> attention_test_half2(
    torch::Tensor Q, 
    torch::Tensor K,
    torch::Tensor V,
    const float tau,
    const int headDim,
    const int iDev
){
    CHECK_INPUT(Q);
    CHECK_INPUT(K);    
    CHECK_INPUT(V);
    return attention_half_test2(Q, K, V, tau, headDim, iDev);
}

std::vector<torch::Tensor> attention_bwd_test(
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
    CHECK_INPUT(Q);
    CHECK_INPUT(K);    
    CHECK_INPUT(V);
    CHECK_INPUT(O);
    CHECK_INPUT(dO);    
    CHECK_INPUT(L);
    CHECK_INPUT(D);
    return attention_bwd(Q, K, V, O, dO, L, D, tau, iDev);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("matmul", &matmul);
    m.def("matmul_shared_memory", &matmul_shared_memory);
    m.def("attention_shared_memory", &attention);
    m.def("attention_test", &attention_test);
    m.def("attention_half", &attention_test_half);
    m.def("attention_half2", &attention_test_half2);
    m.def("attention_bwd", &attention_bwd_test);
}