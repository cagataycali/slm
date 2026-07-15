"""
slm.qwen — self-learning Qwen3-VL runtimes (needs torch + transformers, core deps of strands-slm).

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

Requires: torch, transformers (core deps of strands-slm). Private HF repos
need HF_TOKEN in the environment.
"""
import os
import re

DEFAULT_MODEL = "cagataydev/strands-qwen3-vl-2b"
GEMMA_MODEL = "cagataydev/strands-gemma4-e2b"

# assistant content span inside a chat template render — per model family.
# picked automatically by probing the tokenizer's rendered template.
_ASSISTANT_RES = [
    # Qwen: <|im_start|>assistant\n ... <|im_end|>
    re.compile(r"<\|im_start\|>assistant\n(.*?)(?:<\|im_end\|>|$)", re.DOTALL),
    # Gemma 4: <|turn>model\n ... <turn|>
    re.compile(r"<\|turn>model\n(.*?)(?:<turn\|>|$)", re.DOTALL),
    # Gemma 2/3: <start_of_turn>model\n ... <end_of_turn>
    re.compile(r"<start_of_turn>model\n(.*?)(?:<end_of_turn>|$)", re.DOTALL),
]
_ASSISTANT_RE = _ASSISTANT_RES[0]  # backward-compat default


def _pick_assistant_re(tok):
    """Probe the tokenizer's chat template to find the assistant-span regex."""
    try:
        probe = tok.apply_chat_template(
            [{"role": "user", "content": "q"},
             {"role": "assistant", "content": "PROBE_ANSWER"}], tokenize=False)
    except Exception:
        return _ASSISTANT_RES[0]
    for rx in _ASSISTANT_RES:
        m = rx.search(probe)
        if m and "PROBE_ANSWER" in m.group(1):
            return rx
    return _ASSISTANT_RES[0]


def _dequantize_qat(model):
    """Gemma-4 QAT mobile ships int8 QuantizedLinear wrappers. Swap them for
    dense bf16/fp32 nn.Linear so PEFT adapters attach and autograd flows.
    No-op on models without QuantizedLinear."""
    import torch
    import torch.nn as nn
    dtype = next(model.parameters()).dtype
    n = 0
    for _, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            if type(child).__name__ == "QuantizedLinear":
                W32 = child._dequantize_weights(torch.float32)
                W = W32.to(dtype)
                lin = nn.Linear(child.in_features, child.out_features,
                                bias=child.bias is not None, dtype=dtype)
                with torch.no_grad():
                    lin.weight.copy_(W)
                    if child.bias is not None:
                        lin.bias.copy_(child.bias.to(dtype))
                lin = lin.to(W32.device)
                setattr(mod, child_name, lin)
                n += 1
    return n


def _peft_base(model_id, token):
    """If model_id is a PEFT adapter repo, return its base model id, else None."""
    try:
        from huggingface_hub import hf_hub_download
        import json
        p = hf_hub_download(model_id, "adapter_config.json", token=token)
        return json.load(open(p)).get("base_model_name_or_path")
    except Exception:
        return None


def _require_torch():
    try:
        import torch  # noqa
        import transformers  # noqa
    except ImportError as e:
        raise ImportError(
            "slm.qwen needs torch + transformers: pip install strands-slm"
        ) from e


class StrandsPlasticQwen:
    """Strands-expert Qwen3-VL-2B + fast plastic LoRA head (self-learning at inference)."""

    def __init__(self, model, tok, head, lr=8e-3, decay=0.98, k_gate=0.0,
                 max_B_norm=None, neuromod=False, prompt_loss_weight=0.1):
        import threading
        import torch
        self.model, self.tok, self.head = model, tok, head
        # QA-17: concurrent observe() from OS threads corrupts the native
        # heap (interleaved backward/step). RLock serializes the learn path;
        # re-entrant so wrapped observes (deep decay) stay single-lock.
        self.learn_lock = threading.RLock()
        # keep the END of long documents (assistant answer lives there)
        self.tok.truncation_side = "left"
        self.opt = torch.optim.SGD([head.A, head.B], lr=lr)
        self.decay, self.k_gate = decay, k_gate
        self.mean, self.beta = None, 0.9
        self.var = None                    # EMA variance (neuromod)
        self.max_B_norm = max_B_norm       # optional hard bound
        self.neuromod = neuromod           # graded plasticity
        self.last_fired = False            # gate observability
        # prompt tokens learn at reduced weight (not hard -100 mask)
        # — hard masking destroyed prompt-familiarity binding (T6: 3/8 recall)
        self.prompt_loss_weight = prompt_loss_weight
        self.assistant_re = _ASSISTANT_RE   # overridden per-template in from_pretrained
        self.device = next(model.parameters()).device

    # ---------------- constructors ----------------
    @classmethod
    def from_pretrained(cls, model_id=DEFAULT_MODEL, device=None, r_fast=16,
                        token=None, **kw):
        """Load the merged strands-expert model (or any Qwen3-VL id) + attach
        the fast plastic head. Private repos: pass token= or set HF_TOKEN."""
        _require_torch()
        import torch
        import torch.nn as nn
        from transformers import AutoModelForImageTextToText, AutoProcessor
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32

        # adapter repo? (e.g. cagataydev/strands-gemma4-e2b) -> load base + merge
        base_id = _peft_base(model_id, token)
        weights_id = base_id or model_id

        try:
            proc = AutoProcessor.from_pretrained(weights_id, token=token)
            tok_ = getattr(proc, "tokenizer", proc)
        except Exception:
            from transformers import AutoTokenizer
            tok_ = AutoTokenizer.from_pretrained(weights_id, token=token)
            proc = tok_

        def _load(mid):
            try:
                return AutoModelForImageTextToText.from_pretrained(
                    mid, dtype=dtype, device_map=device, token=token)
            except Exception:
                from transformers import AutoModelForCausalLM
                return AutoModelForCausalLM.from_pretrained(
                    mid, dtype=dtype, device_map=device, token=token)

        model = _load(weights_id)

        if base_id is not None:
            # QAT bases (Gemma-4 mobile): dequantize so the adapter attaches
            n_deq = _dequantize_qat(model)
            if n_deq:
                model = model.to(device)
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, model_id, token=token)
            model = model.merge_and_unload()   # slow weights baked in, then frozen

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
        inst = cls(model, tok_, head, **kw)
        inst.assistant_re = _pick_assistant_re(tok_)
        return inst

    # ---------------- fast self-learning ----------------
    def reset(self):
        """Wipe fast adaptation -> exactly the strands-expert base again."""
        import torch
        import torch.nn as nn
        with torch.no_grad():
            self.head.B.zero_()
            nn.init.normal_(self.head.A, std=0.01)
        self.mean = None
        self.var = None
        self.last_fired = False

    def _nll(self, ids, weights=None):
        """Weighted token NLL. `weights` (same shape as ids) scales each
        target token's loss: 1.0 = full learning (assistant tokens),
        prompt_loss_weight = damped learning (prompt/boilerplate tokens).
        None -> uniform (plain LM loss)."""
        import torch
        o = self.model(input_ids=ids)
        lg = o.logits[:, :-1, :]
        tgt = ids[:, 1:]
        ce = torch.nn.functional.cross_entropy(
            lg.reshape(-1, lg.size(-1)).float(), tgt.reshape(-1),
            reduction="none")
        if weights is None:
            return ce.mean()
        w = weights[:, 1:].reshape(-1).to(ce.dtype).to(ce.device)
        return (ce * w).sum() / w.sum().clamp(min=1e-6)

    def _assistant_labels(self, ids, offsets, text, prompt_weight=None):
        """completion-weighted labels. The FINAL assistant span's tokens get
        weight 1.0; everything else (prompt, boilerplate, and the model's OWN
        earlier replies in the sliding window) gets prompt_loss_weight.
        Returns None when the text has no chat-template markers.

        prompt_weight overrides self.prompt_loss_weight for this call only
        (passed through instead of mutating shared state — concurrent turns
        raced on the old temp mutation).

        Earlier assistant spans are DAMPED, not full-weight: the transcript
        window re-presents each of the model's replies on several subsequent
        turns, and re-learning them at 1.0 amplified self-training."""
        import torch
        pw = self.prompt_loss_weight if prompt_weight is None else prompt_weight
        spans = [m.span(1) for m in self.assistant_re.finditer(text)]
        if not spans:
            return None
        full_spans = spans[-1:]              # only the newest reply learns fully
        keep = torch.zeros(ids.shape[1], dtype=torch.bool)
        off = offsets.tolist()
        for i, (s, e) in enumerate(off):
            if s == e:                       # special token
                continue
            for (as_, ae) in full_spans:
                if s < ae and e > as_:       # overlap with final assistant span
                    keep[i] = True
                    break
        if keep.sum() < 2:                   # nothing learnable -> no mask
            return None
        # the terminator special token right after the span also learns at
        # full weight — otherwise the model under-learns WHEN TO STOP and
        # rambles after learned responses.
        idxs = keep.nonzero().flatten().tolist()
        j = idxs[-1] + 1
        if j < ids.shape[1]:
            tid = int(ids[0, j])
            special = (off[j][0] == off[j][1]
                       or tid in getattr(self.tok, "all_special_ids", []))
            if special:
                keep[j] = True
        weights = torch.full((1, ids.shape[1]), float(pw))
        weights[0, keep] = 1.0
        return weights

    def _lr_scale(self, e):
        """graded plasticity — lr_t = lr * sigmoid(z-score of surprise)."""
        import math
        if not self.neuromod or self.mean is None or not self.var:
            return 1.0
        z = (e - self.mean) / (self.var ** 0.5 + 1e-6)
        return 1.0 / (1.0 + math.exp(-z))

    def observe(self, text, learn=True, max_length=2048, update_gate_stats=True,
                prompt_weight=None, force_fire=False):
        """Predict `text`; if surprised, rewrite the fast weights (bounded).

        single forward — the surprise NLL and the training loss share
        one forward pass (graph is discarded if the gate does not fire).
        pass update_gate_stats=False for rehearsal/replay so old, easy
        documents do not drag the EMA surprise mean down.

        Thread-safety (QA-17): the whole predict→gate→update sequence holds
        self.learn_lock — concurrent OS-thread observes corrupted the
        native heap (interleaved backward/step on shared params).

        Returns the pre-update NLL (the surprise)."""
        with self.learn_lock:
            return self._observe_locked(text, learn, max_length,
                                        update_gate_stats, prompt_weight,
                                        force_fire)

    def _observe_locked(self, text, learn, max_length, update_gate_stats,
                        prompt_weight, force_fire):
        import torch
        enc = self.tok(text, return_tensors="pt", truncation=True,
                       max_length=max_length, return_offsets_mapping=True)
        ids = enc.input_ids.to(self.device)
        if ids.shape[1] < 2:
            self.last_fired = False
            return None
        # prompt_weight=1.0 -> uniform loss (curated short docs need full
        # prompt binding for retrieval); default (None) -> self.prompt_loss_weight
        if prompt_weight is not None and prompt_weight >= 1.0:
            weights = None
        else:
            weights = self._assistant_labels(ids, enc.offset_mapping[0], text,
                                             prompt_weight=prompt_weight)

        if learn:
            loss = self._nll(ids, weights)               # one forward, with graph
            e = loss.item()
        else:
            with torch.no_grad():
                e = self._nll(ids, weights).item()

        # amb_E20b fix: curated channel (teach/revise) must bypass the gate —
        # repeated exposures of the same lesson fall below the EMA and the
        # gate starves legitimate curated learning (recall 0/8 at k_gate=0).
        fire = force_fire or (self.mean is None) or (
            e > self.mean + self.k_gate * abs(self.mean))
        if update_gate_stats:
            if self.mean is None:
                self.mean, self.var = e, 0.0
            else:
                d = e - self.mean
                self.mean = self.beta * self.mean + (1 - self.beta) * e
                self.var = self.beta * (self.var or 0.0) + (1 - self.beta) * d * d

        self.last_fired = bool(learn and fire)
        if learn and fire:
            scale = self._lr_scale(e)
            self.opt.zero_grad()
            loss.backward()
            # clip HEAD only (parity with validated recipe): deep params
            # were never clipped — clipping the joint norm throttled deep
            # fact-storage gradients and regressed fact recall (T6 2/8)
            torch.nn.utils.clip_grad_norm_([self.head.A, self.head.B], 1.0)
            if scale != 1.0:
                old = [g["lr"] for g in self.opt.param_groups]
                for g in self.opt.param_groups:
                    g["lr"] *= scale
                self.opt.step()
                for g, lr in zip(self.opt.param_groups, old):
                    g["lr"] = lr
            else:
                self.opt.step()
            with torch.no_grad():
                self.head.B.mul_(self.decay)      # EMA decay = bounded plasticity
                if self.max_B_norm is not None:   # hard Frobenius bound
                    n = self.head.B.norm()
                    if n > self.max_B_norm:
                        self.head.B.mul_(self.max_B_norm / n)
        elif learn:
            self.opt.zero_grad(set_to_none=True)  # discard unused graph
            del loss
        return e

    # ---------------- chat ----------------
    def chat(self, user_msg, max_new_tokens=512, temperature=0.7,
             enable_thinking=None):
        import torch
        msgs = [{"role": "user", "content": user_msg}]
        kw = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
        ids = self.tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt", **kw)
        if not torch.is_tensor(ids):        # newer transformers -> BatchEncoding
            ids = ids["input_ids"]
        ids = ids.to(self.device)
        gen_kwargs = dict(
            input_ids=ids, max_new_tokens=max_new_tokens,
            pad_token_id=self.tok.eos_token_id)
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature)
        else:
            gen_kwargs["do_sample"] = False
        with torch.no_grad():
            out = self.model.generate(**gen_kwargs)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def _plastic_head_cls():
    import torch
    import torch.nn as nn

    class PlasticHead(nn.Module):
        """Fast LoRA on lm_head: y = base(x) + scale*(x A)B. Only A,B change.

        A/B live in fp32 even when the base is bf16 — small SGD steps
        would otherwise round to zero."""
        def __init__(self, base, r=16, scale=2.0):
            super().__init__()
            self.base = base
            for p in base.parameters():
                p.requires_grad_(False)
            dev = base.weight.device
            self.A = nn.Parameter(torch.randn(base.in_features, r, device=dev,
                                              dtype=torch.float32) * 0.01)
            self.B = nn.Parameter(torch.zeros(r, base.out_features, device=dev,
                                              dtype=torch.float32))
            self.scale = scale

        def forward(self, x):
            y = self.base(x)
            delta = (x.to(torch.float32) @ self.A) @ self.B
            return y + self.scale * delta.to(y.dtype)

    return PlasticHead


def _PlasticHead(base, r=16, scale=2.0):
    return _plastic_head_cls()(base, r=r, scale=scale)
