import torch, time
import cppcuda_tutorial

if __name__ == "__main__":
    iDev = 0
    N = 128
    d = 32
    tau = 1.0 / d**0.5
    
    A = torch.randn(N, d, device=f'cuda:{iDev}', dtype=torch.float16)
    # A[:2] = 0.0
    # A = torch.ones(N, d, device=f'cuda:{iDev}', dtype=torch.float32)
    B = A.t().contiguous()
    # C, M = cppcuda_tutorial.attention_test(A, A, A, tau, iDev)
    
    C, M = cppcuda_tutorial.attention_half2(A, A, A, tau, d, iDev)
    tau = torch.Tensor([tau]).half().to(f'cuda:{iDev}')
    result = ((tau * A) @ B).float()
    M_gt = torch.max(result, dim=-1)[0]
    unnormalized_score = torch.exp(result - M_gt.unsqueeze(-1))
    l = torch.sum(unnormalized_score, dim=-1, keepdim=True)
    L = M_gt + torch.log(l.squeeze())
    score = unnormalized_score / l
    result = score.float() @ A.float()
    print(torch.allclose(L, M, rtol=1e-03, atol=1e-02))
    # print(torch.allclose(M_gt, M, rtol=1e-03, atol=1e-03))
    print(torch.allclose(C, result.float(), rtol=1e-03, atol=1e-03))
    print(L[:3])
    # print(M_gt[:3])
    print(M[:3])
    print(C[-3:, -3:])
    print(result[-3:, -3:])
    print(C[:3, :3])
    print(result[:3, :3])
    # print(result-C)