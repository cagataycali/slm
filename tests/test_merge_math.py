"""NEW-1 regression: merge_experience must compose LoRA DELTAS, not factors.

CPU-only, no model download, <5s. Run: python3 tests/test_merge_math.py

The old code did A=sum(A_i), B=sum(B_i) which yields
(sum A_i)(sum B_i) = sum A_i B_i + CROSS TERMS — with independent random A_i
the cross terms are the same magnitude as the signal (rel err ~1.0).
The fix is rank concatenation (exact), re-compressed via thin-QR + small SVD
(optimal rank-r) when total rank exceeds the instance rank.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from slm.strands_model import SLM

def main():
    torch.manual_seed(0)
    d_in, d_out, r = 256, 2048, 16
    As = [torch.randn(d_in, r) * 0.01 for _ in range(2)]
    Bs = [torch.randn(r, d_out) * 0.5 for _ in range(2)]
    true = sum(a @ b for a, b in zip(As, Bs))

    # document the old bug's magnitude
    old = (As[0] + As[1]) @ (Bs[0] + Bs[1])
    err_old = ((old - true).norm() / true.norm()).item()
    assert err_old > 0.5, "factor-sum should be catastrophically wrong"

    # exact path (K <= r_out): must reproduce the true delta-sum
    A32, B32 = SLM._merge_factors(As, Bs, 32)
    err = (((A32 @ B32) - true).norm() / true.norm()).item()
    assert err < 1e-6, f"exact merge path rel err {err}"

    # compressed path (K > r_out): must equal the OPTIMAL rank-r approximation
    A16, B16 = SLM._merge_factors(As, Bs, 16)
    err16 = (((A16 @ B16) - true).norm() / true.norm()).item()
    U, S, Vh = torch.linalg.svd(true)
    err_opt = ((((U[:, :16] * S[:16]) @ Vh[:16]) - true).norm()
               / true.norm()).item()
    assert abs(err16 - err_opt) < 1e-4, f"SVD path not optimal: {err16} vs {err_opt}"

    print(f"MERGE MATH PASS — old-bug err {err_old:.3f}, "
          f"exact path {err:.2e}, compressed path optimal ({err16:.4f})")

if __name__ == "__main__":
    main()
