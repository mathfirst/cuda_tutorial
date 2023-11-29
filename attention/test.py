from typing import Any
import torch, time
import torch.nn.functional as F
import cppcuda_tutorial

if __name__ == "__main__":
    iDev = 7
    # N = 128
    # d = 32
    # A = torch.rand(N, d, device=f'cuda:{iDev}', dtype=torch.float32)
    # B = A.t().contiguous()
    # C, M = cppcuda_tutorial.attention_shared_memory(A, A, A, iDev)
    # result = cppcuda_tutorial.matmul(A, B)
    # O = torch.softmax(A@B, dim=-1) @ A
    # print(torch.allclose(C, O))
    # print(f"result: {result[:3, :3]}")
    N_ls = [1024,]
    d_ls = [32, ]
    tau = 1.0 / d_ls[0]**0.5
    print(f"tau: {tau:2.2}")
    for N in N_ls:
        for d in d_ls:
            A = torch.rand(N, d, device=f'cuda:{iDev}', dtype=torch.float32)
            # B = A.t().contiguous()            
            # A_t = torch.transpose(A, 0, 1)
            # B = torch.rand(d, N, device='cuda')
            # V = torch.rand(N, d, device='cuda')
            # L = torch.rand(N, 1, device='cuda')  
            elapsed_t_pytorch = 0.0
            # t0 = time.time()
            for i in range(100):
                # torch.cuda.empty_cache()
                # torch.cuda.reset_max_memory_allocated(device=f'cuda:{iDev}')
                # torch.cuda.synchronize() 
                t0 = time.time()
                # C_gt = torch.mm(A, B) #A @ A_t
                # M_gt = torch.max(C_gt, dim=-1)[0]
                # P = torch.exp(C_gt-M_gt.unsqueeze(-1))
                # L = torch.sum(P, dim=-1)
                # O = (P / L.reshape(N, 1)) @ A
                O = torch.softmax(tau * A.half() @ A.t().contiguous().half(), dim=-1) @ A.half()
                if i >= 50:
                    torch.cuda.synchronize()
                    elapsed_t_pytorch += time.time() - t0
            
            print(f"elapsed time of flash attention using pytorch: {elapsed_t_pytorch:6.6}")
            
            elapsed_t_official = 0.0
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):                               
                for i in range(100):
                    # torch.cuda.empty_cache()
                    # torch.cuda.reset_max_memory_allocated(device=f'cuda:{iDev}')
                    # torch.cuda.synchronize() 
                    start = time.time()
                    out = F.scaled_dot_product_attention(A.unsqueeze(0).unsqueeze(0).half(), A.unsqueeze(0).unsqueeze(0).half(), A.unsqueeze(0).unsqueeze(0).half(), dropout_p=0)
                    if i >= 50:
                        torch.cuda.synchronize()
                        elapsed_t_official += time.time() - start
            print(f"elapsed time of flash attention using official: {elapsed_t_official:6.6}")
            print(f"out: {out.shape}")
            print(torch.allclose(out, O, rtol=1e-3, atol=1e-3))
            elapsed_t_custom = 0.0
            # t0 = time.time()
            for i in range(100):
                # torch.cuda.empty_cache()
                # torch.cuda.reset_max_memory_allocated(device=f'cuda:{iDev}')
                torch.cuda.synchronize() 
                t0 = time.time()
                # C, M = cppcuda_tutorial.attention_half2(A.half(), A.half(), A.half(), tau, iDev)
                C, M = cppcuda_tutorial.attention_half2(A.half(), A.half(), A.half(), tau, d, iDev)
                # C, M = cppcuda_tutorial.attention_shared_memory(A, A, A, tau, iDev)
                if i >= 50:
                    torch.cuda.synchronize()
                    elapsed_t_custom += time.time() - t0
            
            print(f"elapsed time of flash attention using custom cuda: {elapsed_t_custom:6.6}")
            correctness = torch.allclose(C.float(), O.float(), rtol=1e-3, atol=1e-3)
            print(f"N: {N}, d: {d}, speedup: {elapsed_t_pytorch / elapsed_t_custom:2.2}, correctness: {correctness}")
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