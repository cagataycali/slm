"""
slm.qwen — self-learning Qwen3-VL runtimes (optional extra: pip install self-learning-model[qwen]).

Two runtimes over a frozen (or merged strands-expert) Qwen3-VL-2B:

  StrandsPlasticQwen — the Strands-Agents-expert Qwen (post-tuned, merged) with a
      plastic LoRA head on lm_head that keeps learning at inference:
      surprise-gated SGD + EMA decay (bounded plasticity), provable off-switch.

Validated (see README.md / PROOF.md in the repo):
  * continual OOD stream: NLL drops online while base knowledge is retained
  * reset() restores the base exactly (Δlogits = 0)

Usage:
    from slm.qwen import StrandsPlasticQwen
    m = StrandsPlasticQwen.from_pretrained()            # default: strands-expert model
    print(m.chat("How do I create a custom tool in Strands Agents?"))
    m.observe(new_docs, learn=True)                     # self-learn after deployment
    m.reset()                                           # off-switch

Requires: torch, transformers (installed via the [qwen] extra). Private HF repos
need HF_TOKEN in the environment.
"""
import os

DEFAULT_MODEL = "cagataydev/strands-qwen3-vl-2b"


def _require_torch():
    try:
        import torch  # noqa
        import transformers  # noqa
    except ImportError as e:
        raise ImportError(
            "slm.qwen needs the optional deps: pip install 'self-learning-model[qwen]'"
        ) from e


class StrandsPlasticQwen:
    """Strands-expert Qwen3-VL-2B + fast plastic LoRA head (self-learning at inference)."""

    def __init__(self, model, tok, head, lr=8e-3, decay=0.98, k_gate=0.0):
        import torch
        self.model, self.tok, self.head = model, tok, head
        self.opt = torch.optim.SGD([head.A, head.B], lr=lr)
        self.decay, self.k_gate = decay, k_gate
        self.mean, self.beta = None, 0.9
        self.device = next(model.parameters()).device

    # ---------------- constructors ----------------
    @classmethod
    def from_pretrained(cls, model_id=DEFAULT_MODEL, device="cuda", r_fast=16,
                        token=None, **kw):
        """Load the merged strands-expert model (or any Qwen3-VL id) + attach
        the fast plastic head. Private repos: pass token= or set HF_TOKEN."""
        _require_torch()
        import torch
        import torch.nn as nn
        from transformers import AutoModelForImageTextToText, AutoProcessor
        token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        proc = AutoProcessor.from_pretrained(model_id, token=token)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=dtype, device_map=device, token=token)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        # attach fast plastic LoRA on lm_head
        head_name, head_mod = None, None
        for name, mod in model.named_modules():
            if name.endswith("lm_head") and isinstance(mod, nn.Linear):
                head_name, head_mod = name, mod
        head = _PlasticHead(head_mod, r=r_fast)
        parent = model
        *pth, last = head_name.split(".")
        for pp in pth:
            parent = getattr(parent, pp)
        setattr(parent, last, head)
        return cls(model, proc.tokenizer, head, **kw)

    # ---------------- fast self-learning ----------------
    def reset(self):
        """Wipe fast adaptation -> exactly the strands-expert base again."""
        import torch
        import torch.nn as nn
        with torch.no_grad():
            self.head.B.zero_()
            nn.init.normal_(self.head.A, std=0.01)
        self.mean = None

    def _nll(self, ids):
        import torch
        o = self.model(input_ids=ids)
        lg = o.logits[:, :-1, :]
        return torch.nn.functional.cross_entropy(
            lg.reshape(-1, lg.size(-1)).float(), ids[:, 1:].reshape(-1))

    def observe(self, text, learn=True, max_length=2048):
        """Predict `text`; if surprised, rewrite the fast weights (bounded).
        Returns the pre-update NLL (the surprise)."""
        import torch
        ids = self.tok(text, return_tensors="pt", truncation=True,
                       max_length=max_length).input_ids.to(self.device)
        if ids.shape[1] < 2:
            return None
        with torch.no_grad():
            e = self._nll(ids).item()
        fire = (self.mean is None) or (e > self.mean + self.k_gate * abs(self.mean))
        self.mean = e if self.mean is None else self.beta * self.mean + (1 - self.beta) * e
        if learn and fire:
            loss = self._nll(ids)
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([self.head.A, self.head.B], 1.0)
            self.opt.step()
            with torch.no_grad():
                self.head.B.mul_(self.decay)      # EMA decay = bounded plasticity
        return e

    # ---------------- chat ----------------
    def chat(self, user_msg, max_new_tokens=512, temperature=0.7):
        import torch
        msgs = [{"role": "user", "content": user_msg}]
        ids = self.tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                input_ids=ids, max_new_tokens=max_new_tokens,
                do_sample=temperature > 0, temperature=max(temperature, 1e-5),
                pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def _plastic_head_cls():
    import torch
    import torch.nn as nn

    class PlasticHead(nn.Module):
        """Fast LoRA on lm_head: y = base(x) + scale*(x A)B. Only A,B change."""
        def __init__(self, base, r=16, scale=2.0):
            super().__init__()
            self.base = base
            for p in base.parameters():
                p.requires_grad_(False)
            dev, dt = base.weight.device, base.weight.dtype
            self.A = nn.Parameter(torch.randn(base.in_features, r, device=dev, dtype=dt) * 0.01)
            self.B = nn.Parameter(torch.zeros(r, base.out_features, device=dev, dtype=dt))
            self.scale = scale

        def forward(self, x):
            return self.base(x) + self.scale * ((x @ self.A) @ self.B)

    return PlasticHead


def _PlasticHead(base, r=16, scale=2.0):
    return _plastic_head_cls()(base, r=r, scale=scale)
