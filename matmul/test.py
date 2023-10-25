from typing import Any
import torch, time
import cppcuda_tutorial

# def trilinear_interpolation_py(feats, points):
#     '''
#     inputs: 
#         feats: (N,8,F)
#         points: (N,3) local coordinate [-1,1]
        
#     outputs:
#         feats_interp: (N,F)
#     '''
#     u = (points[:, 0:1] + 1) / 2
#     v = (points[:, 1:2] + 1) / 2
#     w = (points[:, 2:3] + 1) / 2
#     a = (1 - v) * (1 - w)
#     b = (1 - v) * w
#     c = v * (1 - w)
#     d = v * w
    
#     feats_interp = (1 - u) * (a * feats[:, 0] + b * feats[:, 1] + c * feats[:, 2] + d * feats[:, 3]) +\
#                     u * (a * feats[:, 4] + b * feats[:, 5] + c * feats[:, 6] + d * feats[:, 7])
                    
#     return feats_interp


# class trilinear_interpolation_custom(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, feats, points):
#         feats_interp = cppcuda_tutorial.trilinear_interpolation_fw(feats, points)
#         ctx.save_for_backward(feats, points)
#         return feats_interp
    
#     def backward(ctx, dL_dfeats_interp):
#         feats, points = ctx.saved_tensors
#         dL_feats = cppcuda_tutorial.trilinear_interpolation_bw(dL_dfeats_interp.contiguous(), feats, points)
#         return dL_feats, None
    


if __name__ == "__main__":
    N = 1024
    d = 256
    A = torch.rand(N, d, device='cuda')
    B = A.t().contiguous()# torch.rand(d, N, device='cuda')
    # C = torch.empty(N, N, device='cuda')
    elapsed_t_cuda_custom = 0
    for i in range(200):  
        t = time.time()
        result = cppcuda_tutorial.matmul(A, B)
        torch.cuda.synchronize()
        if i >= 100:
            elapsed_t_cuda_custom += time.time() - t
    print(f"custom cuda time: {elapsed_t_cuda_custom/100:6.2}")
    elapsed_t_cuda_shared_memory = 0
    for i in range(200):  
        t = time.time()
        result_shared_memory = cppcuda_tutorial.matmul_shared_memory(A, B)
        torch.cuda.synchronize()
        if i >= 100:
            elapsed_t_cuda_shared_memory += time.time() - t
    print(f"custom cuda time using shared memory: {elapsed_t_cuda_shared_memory/100:6.2}")
    elapsed_t_cuda = 0
    for i in range(200):        
        t = time.time()
        C = A @ B
        torch.cuda.synchronize()
        if i >= 100:
            elapsed_t_cuda += time.time() - t
    print(f"cuda time: {elapsed_t_cuda/100:6.6}")
    print(torch.allclose(result, C))
    print(torch.allclose(result_shared_memory, C))
    
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