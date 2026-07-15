"""
slm.strands_model — SLM: a Strands model provider whose weights CHANGE every agent turn.

    from strands import Agent
    from strands_tools import shell
    from slm import SLM

    model = SLM("cagataydev/strands-qwen3-vl-2b", plasticity="high")
    agent = Agent(tools=[shell], model=model)
    agent("...")     # <- this turn physically updates the model's fast weights

Mechanism (validated in README.md / demo.ipynb):
  frozen Qwen3-VL-2B (strands-expert) + plastic LoRA head, surprise-gated SGD +
  EMA decay. After every turn the full transcript is observed with learn=True.
  reset() -> bit-identical to base. save/load fast weights for persistence.
"""
import json
import logging
import os
from typing import Any, AsyncIterable, Optional

from strands.types.content import Messages
from strands.types.tools import ToolSpec
from strands.types.streaming import StreamEvent
from strands.models import Model

logger = logging.getLogger(__name__)

PLASTICITY = {
    #            lr,    decay,  r_fast, k_gate
    "off":      (0.0,   1.0,    16,  1e9),
    "low":      (8e-3,  0.98,   16,  0.0),     # production: retention Δ≈0
    "medium":   (2e-2,  0.995,  64, -10.0),
    "high":     (5e-2,  0.999,  128, -10.0),   # demo-validated: visible learning
}


class SLM(Model):
    """Self-learning Strands model: every turn updates the fast weights."""

    def __init__(self, model_id: str = "cagataydev/strands-qwen3-vl-2b",
                 device: str = "cuda", plasticity: str = "high",
                 placement: str = "deep", deep_blocks: int = 6, deep_r: int = 32,
                 learn_on_turn: bool = True, learn_epochs: int = 1,
                 max_tokens: int = 1024, temperature: float = 0.0,
                 token: Optional[str] = None, **kwargs):
        from .qwen import StrandsPlasticQwen
        lr, decay, r_fast, k_gate = PLASTICITY[plasticity]
        self._m = StrandsPlasticQwen.from_pretrained(
            model_id, device=device, r_fast=r_fast, lr=lr, decay=decay,
            k_gate=k_gate, token=token)
        # cycle-6 finding: plastic LoRA on q/v_proj of the last-k attention
        # blocks stores fact bindings ~4x more sample-efficiently and with far
        # less retention damage than the head alone.
        self.placement = placement
        self._deep_params = []
        if placement == "deep":
            # C8 finding: deep placement wants a cooler lr than the head —
            # 2e-2 learns with ZERO retention cost; 5e-2 pays ~0.8 NLL.
            deep_lr = min(lr, 2e-2) if plasticity == "high" else lr
            self._inject_deep(deep_blocks, deep_r, deep_lr, decay)
        self.plasticity = plasticity
        self.learn_on_turn = learn_on_turn and plasticity != "off"
        self.learn_epochs = learn_epochs
        self.config = {"model_id": model_id, "max_tokens": max_tokens,
                       "temperature": temperature, "plasticity": plasticity}
        self.turn_count = 0
        self.surprise_log = []          # (turn, pre-update NLL)
        self.replay_buffer = []         # past turn transcripts (for rehearsal)
        self.replay_k = int(kwargs.get("replay_k", 3))
        self.replay_cap = int(kwargs.get("replay_cap", 64))

    # ---------- strands Model interface ----------
    def update_config(self, **model_config: Any) -> None:
        self.config.update(model_config)

    def get_config(self) -> dict:
        return dict(self.config)

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        raise NotImplementedError("SLM does not support structured output yet")

    async def stream(self, messages: Messages,
                     tool_specs: Optional[list[ToolSpec]] = None,
                     system_prompt: Optional[str] = None,
                     **kwargs: Any) -> AsyncIterable[StreamEvent]:
        """Generate a reply; then LEARN from the full turn transcript."""
        import torch

        chat = self._to_chat(messages, system_prompt, tool_specs)
        ids = self._m.tok.apply_chat_template(
            chat, add_generation_prompt=True, return_tensors="pt",
            tools=self._tools_for_template(tool_specs)).to(self._m.device)

        temp = float(self.config.get("temperature", 0.0))
        with torch.no_grad():
            out = self._m.model.generate(
                input_ids=ids,
                max_new_tokens=int(self.config.get("max_tokens", 1024)),
                do_sample=temp > 0, temperature=max(temp, 1e-5),
                repetition_penalty=float(self.config.get("repetition_penalty", 1.1)),
                pad_token_id=self._m.tok.eos_token_id)
        text = self._m.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

        tool_use = self._parse_tool_call(text)

        yield {"messageStart": {"role": "assistant"}}
        if tool_use is not None:
            yield {"contentBlockStart": {"start": {"toolUse": {
                "toolUseId": tool_use["toolUseId"], "name": tool_use["name"]}}}}
            yield {"contentBlockDelta": {"delta": {"toolUse": {
                "input": json.dumps(tool_use["input"])}}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {"contentBlockStart": {"start": {}}}
            yield {"contentBlockDelta": {"delta": {"text": text}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
        yield {"metadata": {"usage": {
            "inputTokens": int(ids.shape[1]),
            "outputTokens": int(out.shape[1] - ids.shape[1]),
            "totalTokens": int(out.shape[1])},
            "metrics": {"latencyMs": 0}}}

        # ---- THE POINT: learn from this turn (with replay rehearsal) ----
        if self.learn_on_turn:
            import random as _random
            transcript = self._transcript(messages, text)
            e = None
            for _ in range(self.learn_epochs):
                e = self._m.observe(transcript, learn=True)
                # rehearse k random past transcripts to prevent interference
                # (cycle-3 finding: interleaving keeps old knowledge alive)
                if self.replay_buffer and self.replay_k > 0:
                    for past in _random.sample(
                            self.replay_buffer,
                            min(self.replay_k, len(self.replay_buffer))):
                        self._m.observe(past, learn=True)
            self.replay_buffer.append(transcript)
            if len(self.replay_buffer) > self.replay_cap:
                self.replay_buffer.pop(0)
            self.turn_count += 1
            if e is not None:
                self.surprise_log.append((self.turn_count, e))
                logger.debug("SLM turn %d: surprise %.3f (weights updated)",
                             self.turn_count, e)

    def _inject_deep(self, k_blocks: int, r: int, lr: float, decay: float):
        """Attach plastic LoRA to q_proj/v_proj of the last k attention blocks."""
        import torch
        import torch.nn as nn

        class _DeepLoRA(nn.Module):
            def __init__(self, base, r=32, scale=2.0):
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

        inner = self._m.model.model
        layers = (inner.language_model.layers
                  if hasattr(inner, "language_model") else inner.layers)
        n = len(layers)
        for li in range(max(0, n - k_blocks), n):
            attn = layers[li].self_attn
            for name in ("q_proj", "v_proj"):
                lora = _DeepLoRA(getattr(attn, name), r=r)
                setattr(attn, name, lora)
                self._deep_params += [lora.A, lora.B]
        # single optimizer over deep + head fast weights
        self._m.opt = torch.optim.SGD(
            self._deep_params + [self._m.head.A, self._m.head.B], lr=lr)
        self._deep_decay = decay
        # wrap observe to decay deep B matrices too
        _orig_observe = self._m.observe

        def observe_with_deep_decay(text, learn=True, max_length=2048):
            import torch as _t
            e = _orig_observe(text, learn=learn, max_length=max_length)
            if learn:
                with _t.no_grad():
                    for p in self._deep_params[1::2]:
                        p.mul_(self._deep_decay)
            return e

        self._m.observe = observe_with_deep_decay

    # ---------- self-learning API (beyond the Model interface) ----------
    def observe(self, text: str, learn: bool = True, epochs: int = 1):
        """Directly feed text to learn from (returns pre-update NLL)."""
        e = None
        for _ in range(epochs):
            e = self._m.observe(text, learn=learn)
        return e

    def teach(self, prompt: str, response: str, epochs: int = 3,
              rehearse: bool = True):
        """Curated learning: bind (future query -> desired response) directly.

        Cycle-11 finding: raw feedback dialogues teach 'comply when instructed',
        not the skill itself. The learnable unit must pair the bare task with
        the desired answer — this is experience curation. Renders through the
        real chat template (cycle-10 finding) so it transfers to inference.
        """
        chat = [{"role": "user", "content": prompt},
                {"role": "assistant", "content": response}]
        try:
            doc = self._m.tok.apply_chat_template(chat, tokenize=False)
        except Exception:
            doc = f"user: {prompt}\nassistant: {response}"
        # SUPERSESSION (C38): drop stale buffer lessons for the same prompt
        # with a different response — otherwise replay/consolidate rehearses
        # the old belief and fights the revision.
        prompt_key = prompt.strip()[:80]
        self.replay_buffer = [
            d for d in self.replay_buffer
            if not (prompt_key in d and response.strip()[:40] not in d)
        ]
        e = None
        for _ in range(epochs):
            e = self._m.observe(doc, learn=True)
        self.replay_buffer.append(doc)
        if len(self.replay_buffer) > self.replay_cap:
            self.replay_buffer.pop(0)
        return e

    def revise(self, prompt: str, old_response: str, new_response: str,
               steps: int = 14, lr: float = 1e-2):
        """Revise a deeply-consolidated belief: targeted unlearning.

        C45/C46 findings: in dense memories, teach() alone cannot displace a
        consolidated belief (P_old stayed 0.94 despite 8x exposure). This
        method alternates gradient ASCENT on the old (prompt -> old_response)
        binding with descent on the new one — flips the belief in ~14 steps
        with zero measured collateral, and survives serialization.

        WARNING (C46a): do NOT call consolidate() immediately after — the
        replay buffer's semantic neighbors can re-burn the old belief. This
        method purges buffer lessons containing the old response for this
        prompt automatically; still, prefer to let new turns accumulate before
        the next sleep.
        """
        import torch
        def _doc(u, a):
            return self._m.tok.apply_chat_template(
                [{"role": "user", "content": u},
                 {"role": "assistant", "content": a}], tokenize=False)
        params = self._deep_params + [self._m.head.A, self._m.head.B]
        opt = torch.optim.SGD(params, lr=lr)
        old_ids = self._m.tok(_doc(prompt, old_response),
                              return_tensors="pt").input_ids.to(self._m.device)
        new_ids = self._m.tok(_doc(prompt, new_response),
                              return_tensors="pt").input_ids.to(self._m.device)
        for _ in range(steps):
            loss = -self._m._nll(old_ids)          # ascent: forget old
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.5)
            opt.step()
            loss = self._m._nll(new_ids)           # descent: learn new
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.5)
            opt.step()
        # purge stale buffer lessons (prompt + old response) so future sleeps
        # don't re-burn the superseded belief
        pk, ok_ = prompt.strip()[:80], old_response.strip()[:40]
        before = len(self.replay_buffer)
        self.replay_buffer = [d for d in self.replay_buffer
                              if not (pk in d and ok_ in d)]
        # store the new binding as a lesson
        self.replay_buffer.append(_doc(prompt, new_response))
        return {"steps": steps, "purged": before - len(self.replay_buffer) + 1}

    def consolidate(self, epochs: int = 5, lr_boost: float = 1.0):
        """Sleep phase: replay the whole turn buffer to consolidate knowledge.

        Biological analogy: hippocampal replay during sleep. The agent
        accumulates experience cheaply during turns (low interference) and
        consolidates it into stronger weights offline.

        Returns mean surprise over the last replay epoch.
        """
        import random as _random
        if not self.replay_buffer:
            return None
        old_lrs = [g["lr"] for g in self._m.opt.param_groups]
        for g in self._m.opt.param_groups:
            g["lr"] = g["lr"] * lr_boost
        last = None
        try:
            for _ in range(epochs):
                docs = list(self.replay_buffer)
                _random.shuffle(docs)
                es = [self._m.observe(d, learn=True) for d in docs]
                es = [e for e in es if e is not None]
                last = sum(es) / max(len(es), 1)
        finally:
            for g, lr in zip(self._m.opt.param_groups, old_lrs):
                g["lr"] = lr
        return last

    def reset(self):
        """Wipe all test-time learning -> exactly the base model."""
        import torch
        import torch.nn as nn
        self._m.reset()
        with torch.no_grad():
            for i in range(0, len(self._deep_params), 2):
                nn.init.normal_(self._deep_params[i], std=0.01)
                self._deep_params[i + 1].zero_()
        self.turn_count = 0
        self.surprise_log = []
        self.replay_buffer = []

    def save_fast_weights(self, path: str):
        import torch
        torch.save({"A": self._m.head.A.detach().cpu(),
                    "B": self._m.head.B.detach().cpu(),
                    "deep": [p.detach().cpu() for p in self._deep_params],
                    "turn_count": self.turn_count,
                    "plasticity": self.plasticity,
                    "placement": self.placement,
                    "replay_buffer": list(self.replay_buffer)}, path)

    def load_experience_from_hf(self, repo: str = "cagataydev/self-learning-model",
                                filename: str = "slm_agent/marathon_experience.pt",
                                token: Optional[str] = None):
        """Download and load a published experience file from the HF Hub.

        One-liner to an experienced agent:
            m = SLM(...); m.load_experience_from_hf()
        """
        from huggingface_hub import hf_hub_download
        token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        path = hf_hub_download(repo, filename, token=token)
        self.load_fast_weights(path)
        return path

    def merge_experience(self, paths, strategy: str = "sum"):
        """Fleet learning: merge experience files from multiple agents.

        C27 finding: task-arithmetic SUM losslessly composes non-overlapping
        skills (LoRA deltas live in near-orthogonal subspaces); AVERAGE halves
        every skill below firing threshold. For CONFLICTING skills use
        strategy="relearn" (merges replay buffers and relearns).

        Args:
            paths: list of experience-file paths (from save_fast_weights)
            strategy: "sum" (default, lossless for disjoint skills) or
                      "relearn" (buffer merge + observe passes, handles conflicts)
        """
        import torch
        import random as _random
        import re as _re
        ckpts = [torch.load(p, map_location=self._m.device) for p in paths]

        # C41 guardrail: detect conflicting lessons (same user prompt, different
        # assistant response) across checkpoints. SUM on conflicts corrupts
        # (superposed deltas -> babble that smears into neighbors).
        def _pairs(buf):
            out = {}
            for d in buf:
                mm = _re.search(r"user\n(.+?)<\|im_end\|>.*?assistant\n(.+?)<\|im_end\|>",
                                d, _re.DOTALL)
                if mm:
                    out.setdefault(mm.group(1).strip(), set()).add(mm.group(2).strip())
            return out

        merged_pairs = {}
        for c in ckpts:
            for pr, answers in _pairs(c.get("replay_buffer", [])).items():
                merged_pairs.setdefault(pr, set()).update(answers)
        conflicts = [pr for pr, ans in merged_pairs.items() if len(ans) > 1]
        if conflicts and strategy == "sum":
            logger.warning(
                "merge_experience: %d conflicting lesson prompt(s) detected "
                "(e.g. %r) — SUM will corrupt these bindings (C41). "
                "Auto-switching to strategy='relearn'.",
                len(conflicts), conflicts[0][:60])
            strategy = "relearn"

        if strategy == "sum":
            with torch.no_grad():
                A = sum(c["A"] for c in ckpts)
                B = sum(c["B"] for c in ckpts)
                self._m.head.A.copy_(A.to(self._m.head.A.dtype))
                self._m.head.B.copy_(B.to(self._m.head.B.dtype))
                for i, p in enumerate(self._deep_params):
                    s = sum(c["deep"][i] for c in ckpts if c.get("deep"))
                    p.copy_(s.to(p.dtype).to(p.device))
        elif strategy == "relearn":
            self.reset()
            lessons = []
            for c in ckpts:
                lessons.extend(c.get("replay_buffer", []))
            _random.Random(0).shuffle(lessons)
            for _ in range(3):
                for d in lessons:
                    self._m.observe(d, learn=True)
            self.replay_buffer = lessons[-self.replay_cap:]
        else:
            raise ValueError(f"unknown strategy: {strategy}")
        # merge buffers regardless (for future consolidation)
        merged_buf = []
        for c in ckpts:
            merged_buf.extend(c.get("replay_buffer", []))
        if strategy == "sum":
            self.replay_buffer = merged_buf[-self.replay_cap:]
        return {"merged": len(paths), "strategy": strategy,
                "lessons": len(merged_buf), "conflicts": len(conflicts)}

    def load_fast_weights(self, path: str):
        import torch
        ckpt = torch.load(path, map_location=self._m.device)
        with torch.no_grad():
            self._m.head.A.copy_(ckpt["A"].to(self._m.head.A.dtype))
            self._m.head.B.copy_(ckpt["B"].to(self._m.head.B.dtype))
            for p, saved in zip(self._deep_params, ckpt.get("deep", [])):
                p.copy_(saved.to(p.dtype).to(p.device))
        self.turn_count = ckpt.get("turn_count", 0)
        self.replay_buffer = list(ckpt.get("replay_buffer", []))

    # ---------- helpers ----------
    @staticmethod
    def _content_to_text(content) -> str:
        if isinstance(content, str):
            return content
        parts = []
        for block in content:
            if "text" in block:
                parts.append(block["text"])
            elif "toolUse" in block:
                tu = block["toolUse"]
                parts.append(f"<tool_call>\n{json.dumps({'name': tu['name'], 'arguments': tu['input']})}\n</tool_call>")
            elif "toolResult" in block:
                tr = block["toolResult"]
                inner = []
                for c in tr.get("content", []):
                    if "text" in c:
                        inner.append(c["text"])
                    elif "json" in c:
                        inner.append(json.dumps(c["json"]))
                parts.append("\n".join(inner))
        return "\n".join(parts)

    def _to_chat(self, messages: Messages, system_prompt, tool_specs):
        chat = []
        if system_prompt:
            chat.append({"role": "system", "content": system_prompt})
        for msg in messages:
            role = msg["role"]
            text = self._content_to_text(msg.get("content", []))
            is_tool_result = any("toolResult" in b for b in msg.get("content", [])
                                 if isinstance(b, dict))
            if is_tool_result:
                chat.append({"role": "tool", "content": text})
            else:
                chat.append({"role": role, "content": text})
        return chat

    @staticmethod
    def _tools_for_template(tool_specs):
        if not tool_specs:
            return None
        tools = []
        for spec in tool_specs:
            tools.append({"type": "function", "function": {
                "name": spec["name"], "description": spec.get("description", ""),
                "parameters": spec["inputSchema"]["json"]}})
        return tools

    @staticmethod
    def _parse_tool_call(text: str):
        """Parse Qwen-style <tool_call>{...}</tool_call> from generated text."""
        import re, uuid
        mm = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
        if not mm:
            return None
        try:
            obj = json.loads(mm.group(1))
            return {"toolUseId": f"slm-{uuid.uuid4().hex[:8]}",
                    "name": obj["name"],
                    "input": obj.get("arguments", {})}
        except (json.JSONDecodeError, KeyError):
            return None

    def _transcript(self, messages: Messages, reply: str) -> str:
        """Render the turn in the REAL chat template so learning transfers to
        chat-format inference (cycle-10 finding: raw 'user:/assistant:' learning
        reaches P=0.89 but does NOT transfer to the templated argmax)."""
        chat = []
        for msg in messages[-6:]:
            text = self._content_to_text(msg.get("content", []))
            if text.strip():
                chat.append({"role": msg["role"], "content": text})
        chat.append({"role": "assistant", "content": reply})
        try:
            return self._m.tok.apply_chat_template(chat, tokenize=False)
        except Exception:
            return "\n".join(f"{c['role']}: {c['content']}" for c in chat)
