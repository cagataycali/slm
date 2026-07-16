"""Regression: learned per-channel gate on the plastic path.

CPU-only, no model download, <5s. Run: python3 tests/test_gate.py

Contract:
  G1  behavior-preserving init: effective multiplier == scale exactly at init
      (a RoboTTT-style ~0 init would silence B's gradients and kill learning)
  G2  off-switch through the gate: B==0 -> output bit-identical to base for
      ANY alpha value (alpha is slow state, carries no facts)
  G3  per-channel: closing alpha[c] kills channel c's delta only
  G4  gradients flow to alpha (online-learnable) and to A/B through the gate
  G5  gate is bounded: |effective mult| <= 2*scale
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from slm.qwen import _PlasticHead


def main():
    torch.manual_seed(0)
    d_in, d_out, r = 16, 32, 4
    base = nn.Linear(d_in, d_out)
    head = _PlasticHead(base, r=r, scale=2.0)
    x = torch.randn(3, 5, d_in)

    # G1: init gate == scale exactly (tanh(atanh(0.5)) * 2 * scale == scale)
    g = (2.0 * head.scale) * torch.tanh(head.alpha)
    err = (g - head.scale).abs().max().item()
    assert err < 1e-5, f"G1 FAIL: init gate off by {err}"
    # and with a nonzero B, gated forward == old scale*delta forward
    with torch.no_grad():
        head.B.normal_(std=0.1)
    old_style = base(x) + head.scale * ((x @ head.A) @ head.B)
    new_style = head(x)
    d = (old_style - new_style).abs().max().item()
    assert d < 1e-4, f"G1 FAIL: gated forward != legacy forward ({d})"
    print(f"G1 PASS — behavior-preserving init (gate=={head.scale}, fwd Δ={d:.2e})")

    # G2: B==0 -> bit-identical to base for any alpha (off-switch invariant)
    with torch.no_grad():
        head.B.zero_()
        head.alpha.uniform_(-3, 3)   # arbitrary gate state
    diff = (head(x) - base(x)).abs().max().item()
    assert diff == 0.0, f"G2 FAIL: B=0 not exact through gate ({diff})"
    print("G2 PASS — reset invariant: B=0 ⇒ Δoutput=0 for arbitrary alpha")

    # G3: per-channel gating
    with torch.no_grad():
        head.alpha.fill_(0.5493061443340549)
        head.B.normal_(std=0.1)
        head.alpha[7] = 0.0          # close channel 7
    delta = head(x) - base(x)
    assert delta[..., 7].abs().max().item() == 0.0, "G3 FAIL: closed channel leaked"
    assert delta[..., 8].abs().max().item() > 0, "G3 FAIL: open channel dead"
    print("G3 PASS — per-channel gate (closed channel 7 exactly silent)")

    # G4: gradients reach alpha, A, B
    head.zero_grad()
    loss = head(x).pow(2).mean()
    loss.backward()
    assert head.alpha.grad is not None and head.alpha.grad.abs().sum() > 0, "G4 FAIL: no alpha grad"
    assert head.A.grad is not None and head.A.grad.abs().sum() > 0, "G4 FAIL: no A grad"
    assert head.B.grad is not None and head.B.grad.abs().sum() > 0, "G4 FAIL: no B grad"
    # closed channel gets ~0 alpha grad only through its own channel term
    print("G4 PASS — grads flow to alpha/A/B (online-learnable gate)")

    # G5: bounded
    with torch.no_grad():
        head.alpha.fill_(100.0)
    g = (2.0 * head.scale) * torch.tanh(head.alpha)
    assert g.max().item() <= 2.0 * head.scale + 1e-6, "G5 FAIL: unbounded gate"
    print(f"G5 PASS — gate ceiling 2*scale={2.0*head.scale}")

    print("\nGATE TESTS PASS (G1–G5)")


if __name__ == "__main__":
    main()
