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

// torch::Tensor trilinear_interpolation_fw(
//     torch::Tensor feats, 
//     torch::Tensor points
// ){
//     CHECK_INPUT(feats);
//     CHECK_INPUT(points);
//     return trilinear_fw_cu(feats, points);
// }

// torch::Tensor trilinear_interpolation_bw(
//     torch::Tensor dL_dfeats_interp, 
//     torch::Tensor feats, 
//     torch::Tensor points
// ){
//     CHECK_INPUT(dL_dfeats_interp);
//     CHECK_INPUT(feats);
//     CHECK_INPUT(points);
//     return trilinear_bw_cu(dL_dfeats_interp, feats, points);
// }

// void MatrixMulOnCPU(float *Q, float *K_tr, float *S, int N, int d)
// {
//     for(int i=0; i<N; ++i)
//     {
//         for(int j=0; j<N; ++j)
//         {
//             double tmpSum = 0;
//             for(int k=0; k < d; ++k)
//             {
//                 // A 2D array is stored in the computer's memory one row following another.
//                 tmpSum += Q[i*d + k] * K_tr[j*d + k]; 
//             }
//             S[i*d + j] = tmpSum;
//         }
//     }
// }


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("matmul", &matmul);
    m.def("matmul_shared_memory", &matmul_shared_memory);
}