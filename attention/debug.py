from typing import Any
import torch, time
import cppcuda_tutorial

if __name__ == "__main__":
    iDev = 3
    N = 128
    d = 32
    tau = 1.0 / d**0.5
    
    A = torch.rand(N, d, device=f'cuda:{iDev}', dtype=torch.float32)
    # A = torch.ones(N, d, device=f'cuda:{iDev}', dtype=torch.float32)
    B = A.t().contiguous()
    # C, M = cppcuda_tutorial.attention_test(A, A, A, tau, iDev)
    C, M = cppcuda_tutorial.attention_half(A.to(torch.float16), A.to(torch.float16), A, tau, iDev)
    tau = torch.Tensor([tau]).half().to(f'cuda:{iDev}')
    result = ((tau * A.half()) @ B.half()).float()
    # print(torch.allclose(C, result))
    # print(f"result: \n{result[:3, :3]}")
    # print(f"C: \n{C[:3, :3]}")
    # print(f"result: \n{result[-3:, -3:]}")
    # print(f"C: \n{C[-3:, -3:]}")
    M_gt = torch.max(result, dim=-1)[0]
    unnormalized_score = torch.exp(result - M_gt.unsqueeze(-1))
    l = torch.sum(unnormalized_score, dim=-1, keepdim=True)
    L = M_gt + torch.log(l.squeeze())
    print(l.shape, L.shape, M.shape, M_gt.shape)
    score = unnormalized_score / l
    result = score @ A
    print(torch.allclose(L.squeeze(), M, rtol=1e-04, atol=1e-04))
    print(torch.allclose(result, C, rtol=1e-04, atol=1e-04))
    print(L.squeeze())
    print(M)
    # N_ls = [512, 1024,]# 2048, 4096]
    # d_ls = [32, ]
    # for N in N_ls:
    #     for d in d_ls:
    #         A = torch.rand(N, d, device=f'cuda:{iDev}', dtype=torch.float32)
    #         B = A.t().contiguous()
    #         elapsed_t_custom = 0.0
    #         # t0 = time.time()
    #         for i in range(100):
    #             t0 = time.time()
    #             C, M = cppcuda_tutorial.attention_shared_memory(A, A, A, iDev)
    #             if i >= 50:
    #                 torch.cuda.synchronize()
    #                 elapsed_t_custom += time.time() - t0
            
    #         print(f"elapsed time of flash attention using custom cuda: {elapsed_t_custom:6.6}")
            
    #         # A_t = torch.transpose(A, 0, 1)
    #         # B = torch.rand(d, N, device='cuda')
    #         # V = torch.rand(N, d, device='cuda')
    #         # L = torch.rand(N, 1, device='cuda')  
    #         elapsed_t_pytorch = 0.0
    #         # t0 = time.time()
    #         for i in range(100):
    #             t0 = time.time()
    #             # C_gt = torch.mm(A, B) #A @ A_t
    #             # M_gt = torch.max(C_gt, dim=-1)[0]
    #             # P = torch.exp(C_gt-M_gt.unsqueeze(-1))
    #             # L = torch.sum(P, dim=-1)
    #             # O = (P / L.reshape(N, 1)) @ A
    #             O = torch.softmax(A@B, dim=-1) @ A
    #             if i >= 50:
    #                 torch.cuda.synchronize()
    #                 elapsed_t_pytorch += time.time() - t0
            
    #         print(f"elapsed time of flash attention using pytorch: {elapsed_t_pytorch:6.6}")
    #         correctness = torch.allclose(C, O)
    #         print(f"N: {N}, d: {d}, speedup: {elapsed_t_pytorch / elapsed_t_custom:2.2}, correctness: {correctness}")
    # C_gt = torch.mm(A, B) #A @ A_t
    # M_gt = torch.max(C_gt, dim=-1)[0]
    # # # print(M_gt.unsqueeze(-1).shape)
    # P = torch.exp(C_gt-M_gt.unsqueeze(-1))
    # L = torch.sum(P, dim=-1)
    # O = (P / L.reshape(N, 1)) @ A
    # # print(f"O shape: {O.shape}")
    # # print(L)
    # # print(M)
    # # print(torch.allclose(M, L))
    # # print(f"C_gt shape: {C_gt.shape}")
    # print(C[-3:, -6:])
    # print(O[-3:, -6:])
    # print(C[:3, :6])
    # print(O[:3, :6])
    # print(torch.allclose(C, O))
    # elapsed_t_cuda_custom = 0
    # for i in range(200):  
    #     t = time.time()
    #     result = cppcuda_tutorial.matmul(A, B)
    #     torch.cuda.synchronize()
    #     if i >= 100:
    #         elapsed_t_cuda_custom += time.time() - t
    # print(f"custom cuda time: {elapsed_t_cuda_custom/100:6.2}")
    # elapsed_t_cuda_shared_memory = 0
    # for i in range(200):  
    #     t = time.time()
    #     result_shared_memory = cppcuda_tutorial.matmul_shared_memory(A, B)
    #     torch.cuda.synchronize()
    #     if i >= 100:
    #         elapsed_t_cuda_shared_memory += time.time() - t
    # print(f"custom cuda time using shared memory: {elapsed_t_cuda_shared_memory/100:6.2}")
    # elapsed_t_cuda = 0
    # for i in range(200):        
    #     t = time.time()
    #     C = A @ B
    #     torch.cuda.synchronize()
    #     if i >= 100:
    #         elapsed_t_cuda += time.time() - t
    # print(f"cuda time: {elapsed_t_cuda/100:6.6}")
    # print(torch.allclose(result, C))
    # print(torch.allclose(result_shared_memory, C))
    
    # t = time.time()
    # out_cuda = cppcuda_tutorial.trilinear_interpolation(feats, points)
    # # print(out_cuda.shape)
    # torch.cuda.synchronize()
    # print("consumed time using cuda: %.4fs" % (time.time()-t))
    
    # out_pytorch = trilinear_interpolation_py(feats, points)
    # torch.cuda.synchronize()
    # print("consumed time using pytorch with gpu: %.4fs" % (time.time()-t))
    
    
    # feats_cpu = feats.cpu()
    # points_cpu = points.cpu()
    # t = time.time()
    # out_py = trilinear_interpolation_py(feats_cpu, points_cpu)
    # print("consumed time using cpu: %.4fs" % (time.time()-t))
    # print(torch.allclose(out_cuda.cpu(), out_py))
    # print(torch.allclose(out_pytorch.cpu(), out_py))
    # print(out_py[0, :5], out_cuda[0, :5])

    # feats = torch.ones(2, device='cuda')
    # point = torch.zeros(2, device='cuda')

    # out = cppcuda_tutorial.trilinear_interpolation(feats, point)
    # print(out)
    # data = feats = torch.rand(N, 8, F, device='cuda')
    # feats = data.clone().requires_grad_()
    # feats2 = data.clone().requires_grad_()
    # points = torch.rand(N, 3, device='cuda') * 2 - 1
    
    # t = time.time()
    # out_cuda = trilinear_interpolation_custom.apply(feats2, points)
    # loss_cuda = out_cuda.sum()
    # torch.cuda.synchronize()
    # print("consumed time fw using cuda: %.4fs" % (time.time()-t))
    
    # t = time.time()
    # out_pytorch = trilinear_interpolation_py(feats, points)
    # loss_pytorch = out_pytorch.sum()
    # torch.cuda.synchronize()
    # print("consumed time fw using pytorch with gpu: %.4fs" % (time.time()-t))
    
    
    # feats3 = data.clone().cpu().requires_grad_()
    # points_cpu = points.cpu()
    # t = time.time()
    # out_py = trilinear_interpolation_py(feats3, points_cpu)
    # loss_py = out_py.sum()
    # print("consumed time fw using cpu: %.4fs" % (time.time()-t))
    # print(torch.allclose(out_cuda.cpu(), out_py))
    # print(torch.allclose(out_pytorch.cpu(), out_py))
    
    # # test backward
    # t = time.time()
    # loss_cuda.backward()
    # torch.cuda.synchronize()
    # print("consumed time bw using cuda: %.4fs" % (time.time()-t))
    
    # t = time.time()
    # loss_pytorch.backward()
    # torch.cuda.synchronize()
    # print("consumed time bw using pytorch with gpu: %.4fs" % (time.time()-t))
    
    # points_cpu = points.cpu()
    # t = time.time()
    # loss_py.backward()
    # print("consumed time bw using cpu: %.4fs" % (time.time()-t))
    # print(torch.allclose(feats.grad, feats2.grad))
    # print(torch.allclose(feats.grad.cpu(), feats3.grad))