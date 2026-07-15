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
import hashlib
import json
import logging
import os
import random
import time
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
    # NOTE: k_gate=-10 => gate ~always open (learn every observe). Because the
    # surprise NLL and training loss share a single forward, this no longer
    # costs an extra forward pass.
    "medium":   (2e-2,  0.995,  64, -10.0),
    "high":     (5e-2,  0.999,  128, -10.0),   # demo-validated: visible learning
}


class SLM(Model):
    """Self-learning Strands model: every turn updates the fast weights."""

    def __init__(self, model_id: str = "cagataydev/strands-qwen3-vl-2b",
                 device: Optional[str] = None, plasticity: str = "high",
                 placement: str = "deep", deep_blocks: int = 6, deep_r: int = 32,
                 learn_on_turn: bool = True, learn_epochs: int = 1,
                 max_tokens: int = 1024, temperature: float = 0.0,
                 enable_thinking: Optional[bool] = None,
                 token: Optional[str] = None, **kwargs):
        from .qwen import StrandsPlasticQwen
        lr, decay, r_fast, k_gate = PLASTICITY[plasticity]
        # amb_E18 fix: honor explicit overrides — previously SLM(k_gate=...)
        # (and lr/decay) were silently swallowed by **kwargs, so callers got
        # the preset value without warning.
        lr = kwargs.pop("lr", lr)
        decay = kwargs.pop("decay", decay)
        k_gate = kwargs.pop("k_gate", k_gate)
        r_fast = kwargs.pop("r_fast", r_fast)
        self._m = StrandsPlasticQwen.from_pretrained(
            model_id, device=device, r_fast=r_fast, lr=lr, decay=decay,
            k_gate=k_gate, token=token,
            max_B_norm=kwargs.get("max_B_norm"),
            neuromod=kwargs.get("neuromod", False),
            prompt_loss_weight=kwargs.get("prompt_loss_weight", 0.1))
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
        self._enable_thinking = enable_thinking
        self.config = {"model_id": model_id, "max_tokens": max_tokens,
                       "temperature": temperature, "plasticity": plasticity,
                       "enable_thinking": enable_thinking}
        self.turn_count = 0
        # some chat templates (Gemma 4) silently DROP role="tool" messages —
        # probe once; if dropped, tool results are folded into a user turn.
        self._tool_role_ok = self._probe_tool_role()
        self.surprise_log = []          # (turn, pre-update NLL)
        self.replay_buffer = []         # past turn transcripts (for rehearsal)
        self.replay_k = int(kwargs.get("replay_k", 3))
        self.replay_cap = int(kwargs.get("replay_cap", 64))
        self._buffer_seen = 0           # reservoir-sampling counter (raw tier)
        self.audit_log = []             # SEC-1: {turn, nll, sha256, source}
        self._learn_lock = None         # concurrency guard (lazy asyncio.Lock)

    # ---------- strands Model interface ----------
    def update_config(self, **model_config: Any) -> None:
        self.config.update(model_config)

    def get_config(self) -> dict:
        return dict(self.config)

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        raise NotImplementedError("SLM does not support structured output yet")
        # unreachable yield makes this an async GENERATOR: strands drives it
        # with `async for`, so a plain coroutine raised a cryptic TypeError
        # ("requires __aiter__") instead of the NotImplementedError (QA-19)
        yield

    async def stream(self, messages: Messages,
                     tool_specs: Optional[list[ToolSpec]] = None,
                     system_prompt: Optional[str] = None,
                     **kwargs: Any) -> AsyncIterable[StreamEvent]:
        """Generate a reply; then LEARN from the full turn transcript."""
        import asyncio
        import torch

        chat = self._to_chat(messages, system_prompt, tool_specs)
        # QA-18: the HF fast tokenizer (Rust) is NOT thread-safe — concurrent
        # use from the event loop and a direct-API thread raises
        # RuntimeError('Already borrowed'). All tokenizer touches in stream()
        # hold the same RLock as the learn path.
        with self._m.learn_lock:
            try:
                ids = self._m.tok.apply_chat_template(
                    chat, add_generation_prompt=True, return_tensors="pt",
                    tools=self._tools_for_template(tool_specs), **self._tmpl_kw())
            except Exception:
                # some templates reject the tools= kwarg — degrade gracefully
                ids = self._m.tok.apply_chat_template(
                    chat, add_generation_prompt=True, return_tensors="pt",
                    **self._tmpl_kw())
        if not torch.is_tensor(ids):        # newer transformers -> BatchEncoding
            ids = ids["input_ids"]
        ids = ids.to(self._m.device)

        temp = float(self.config.get("temperature", 0.0))

        def _generate():
            # runs in a worker thread: hold the learn RLock so generation
            # never overlaps a direct-API thread's backward/step (QA-18)
            gen_kwargs = dict(
                input_ids=ids,
                max_new_tokens=int(self.config.get("max_tokens", 1024)),
                repetition_penalty=float(self.config.get("repetition_penalty", 1.1)),
                pad_token_id=self._m.tok.eos_token_id)
            if temp > 0:
                gen_kwargs.update(do_sample=True, temperature=temp)
            else:
                gen_kwargs["do_sample"] = False
            with self._m.learn_lock, torch.no_grad():
                return self._m.model.generate(**gen_kwargs)

        # serialize generate+learn across concurrent turns — the optimizer,
        # backward graph and plastic A/B are shared mutable state.
        if self._learn_lock is None:
            self._learn_lock = asyncio.Lock()

        # keep the event loop responsive
        _t_gen = time.time()
        async with self._learn_lock:
            out = await asyncio.to_thread(_generate)
        _latency_ms = int((time.time() - _t_gen) * 1000)
        with self._m.learn_lock:      # fast tokenizer: not thread-safe (QA-18)
            text = self._m.tok.decode(out[0, ids.shape[1]:],
                                      skip_special_tokens=True)

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
            "metrics": {"latencyMs": _latency_ms}}}

        # ---- THE POINT: learn from this turn (with replay rehearsal) ----
        if self.learn_on_turn:
            transcript = self._transcript(messages, text)

            def _learn():
                e = None
                for _ in range(self.learn_epochs):
                    e = self._m.observe(transcript, learn=True)
                    # rehearse k random past transcripts to prevent interference
                    # (cycle-3 finding: interleaving keeps old knowledge alive)
                    # rehearsal must NOT update the surprise EMA
                    if self.replay_buffer and self.replay_k > 0:
                        for past in random.sample(
                                self.replay_buffer,
                                min(self.replay_k, len(self.replay_buffer))):
                            self._m.observe(
                                self._entry_text(past), learn=True,
                                update_gate_stats=False,
                                prompt_weight=1.0 if self._entry_kind(past) == "curated" else None)
                return e

            # backward passes off the event loop (serialized with generation)
            async with self._learn_lock:
                e = await asyncio.to_thread(_learn)
            self._buffer_add(transcript, kind="raw", source="turn")
            self.turn_count += 1
            if e is not None:
                self.surprise_log.append((self.turn_count, e))
                # SEC-1 audit: content hash + source so a poisoned update can
                # be attributed (and its buffer entry located) after the fact
                self.audit_log.append({
                    "turn": self.turn_count, "nll": e,
                    "sha256": hashlib.sha256(transcript.encode()).hexdigest(),
                    "source": "turn"})
                logger.debug("SLM turn %d: surprise %.3f (weights updated)",
                             self.turn_count, e)

    @staticmethod
    def _entry(text: str, kind: str = "raw", prompt=None, response=None,
               source: str = "turn") -> dict:
        """Structured buffer entry: content hash + provenance for audit,
        exact prompt/response fields for supersession, kind for typed replay."""
        return {"text": text, "kind": kind, "prompt": prompt,
                "response": response,
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "source": source, "ts": time.time()}

    @staticmethod
    def _entry_text(d) -> str:
        return d["text"] if isinstance(d, dict) else d

    @staticmethod
    def _entry_kind(d) -> str:
        return d.get("kind", "raw") if isinstance(d, dict) else "raw"

    def _buffer_add(self, doc, kind: str = "raw", prompt=None, response=None,
                    source: str = "turn"):
        """Two-tier buffer.

        curated tier — guaranteed insertion, NEVER reservoir-evicted (a lesson
        curated via teach() at turn 1000 previously had a cap/seen chance of
        surviving); removed only by supersession/revise.
        raw tier — reservoir sampling over the raw slots, so every raw turn
        ever seen keeps equal survival probability.

        Deduplicates by content hash (tool-call turns re-appear inside the
        next turn's sliding window — no point double-buffering them)."""
        e = (self._entry(doc, kind, prompt, response, source)
             if isinstance(doc, str) else doc)
        if any(isinstance(d, dict) and d["sha256"] == e["sha256"]
               for d in self.replay_buffer):
            return e
        if e["kind"] == "curated":
            self.replay_buffer.append(e)
            if len(self.replay_buffer) > self.replay_cap:
                # over cap: evict the oldest RAW entry, never a curated one
                for i, d in enumerate(self.replay_buffer):
                    if self._entry_kind(d) != "curated":
                        del self.replay_buffer[i]
                        break
                else:
                    # all-curated buffer: nothing evictable — the curated tier
                    # grows past the cap by design (never evicted), but
                    # consolidate() cost now grows with every teach()
                    logger.warning(
                        "replay buffer: %d curated lessons exceed replay_cap="
                        "%d — curated entries are never evicted, so the "
                        "buffer (and consolidate() cost) will keep growing. "
                        "Raise replay_cap or prune via supersession.",
                        len(self.replay_buffer), self.replay_cap)
            return e
        # raw tier reservoir
        self._buffer_seen += 1
        raw_idx = [i for i, d in enumerate(self.replay_buffer)
                   if self._entry_kind(d) != "curated"]
        n_curated = len(self.replay_buffer) - len(raw_idx)
        raw_cap = max(self.replay_cap - n_curated, 1)
        if len(raw_idx) < raw_cap:
            self.replay_buffer.append(e)
        else:
            j = random.randrange(self._buffer_seen)
            if j < raw_cap:
                self.replay_buffer[raw_idx[j]] = e
        return e

    def _inject_deep(self, k_blocks: int, r: int, lr: float, decay: float):
        """Attach plastic LoRA to q_proj/v_proj of the last k attention blocks."""
        import torch
        import torch.nn as nn

        class _DeepLoRA(nn.Module):
            """A/B in fp32 even on bf16 bases."""
            def __init__(self, base, r=32, scale=2.0):
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

        inner = self._m.model.model
        layers = (inner.language_model.layers
                  if hasattr(inner, "language_model") else inner.layers)
        n = len(layers)
        # preferred: q_proj + v_proj. Some archs (Gemma 4 mobile) share KV
        # across layers and expose only q_proj/o_proj — probe what exists.
        probe = layers[-1].self_attn
        names = [nm for nm in ("q_proj", "v_proj") if hasattr(probe, nm)]
        if "v_proj" not in names and hasattr(probe, "o_proj"):
            names.append("o_proj")   # o_proj ~ value-path surrogate
        if not names:
            raise RuntimeError(
                f"deep placement: no attachable projections on "
                f"{type(probe).__name__} — use placement='head'")
        self._deep_proj_names = tuple(names)
        for li in range(max(0, n - k_blocks), n):
            attn = layers[li].self_attn
            for name in names:
                lora = _DeepLoRA(getattr(attn, name), r=r)
                setattr(attn, name, lora)
                self._deep_params += [lora.A, lora.B]
        # single optimizer over deep + head fast weights
        self._m.opt = torch.optim.SGD(
            self._deep_params + [self._m.head.A, self._m.head.B], lr=lr)
        self._deep_decay = decay
        # wrap observe to decay deep B matrices too
        _orig_observe = self._m.observe

        def observe_with_deep_decay(text, learn=True, max_length=2048,
                                    update_gate_stats=True, prompt_weight=None,
                                    force_fire=False):
            import torch as _t
            # QA-17: hold the (re-entrant) learn lock across observe AND the
            # deep decay — decaying outside it raced with another thread's
            # backward/step on the same deep params.
            with self._m.learn_lock:
                e = _orig_observe(text, learn=learn, max_length=max_length,
                                  update_gate_stats=update_gate_stats,
                                  prompt_weight=prompt_weight,
                                  force_fire=force_fire)
                # decay deep B only when the gate actually fired — the
                # same condition under which the head B decays. Previously deep
                # knowledge decayed on every learn=True call, so it faded faster
                # than head knowledge in low-plasticity mode.
                if learn and self._m.last_fired:
                    with _t.no_grad():
                        for p in self._deep_params[1::2]:
                            p.mul_(self._deep_decay)
            return e

        self._m.observe = observe_with_deep_decay

    # ---------- self-learning API (beyond the Model interface) ----------
    def observe(self, text: str, learn: bool = True, epochs: int = 1,
                update_gate_stats: bool = True):
        """Directly feed text to learn from (returns pre-update NLL)."""
        e = None
        for i in range(epochs):
            # only the FIRST exposure is a genuine surprise sample
            e = self._m.observe(text, learn=learn,
                                update_gate_stats=update_gate_stats and i == 0)
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
            with self._m.learn_lock:   # fast tokenizer: not thread-safe (QA-18)
                doc = self._m.tok.apply_chat_template(chat, tokenize=False,
                                                      **self._tmpl_kw())
        except Exception:
            doc = f"user: {prompt}\nassistant: {response}"
        # SUPERSESSION (C38): drop stale buffer lessons for the same prompt
        # with a different response — otherwise replay/consolidate rehearses
        # the old belief and fights the revision.
        pk, rk = prompt.strip(), response.strip()

        def _stale(d):
            if isinstance(d, dict) and d.get("prompt") is not None:
                return d["prompt"] == pk and (d.get("response") or "").strip() != rk
            t = self._entry_text(d)          # legacy string entries
            return pk[:80] in t and rk[:40] not in t
        self.replay_buffer = [d for d in self.replay_buffer if not _stale(d)]
        e = None
        for i in range(epochs):
            # repeated epochs on the same doc must not drag the EMA down.
            # prompt_weight=1.0: curated bindings need FULL prompt-familiarity
            # (weighted loss alone regressed fact recall T6 to 4/8)
            e = self._m.observe(doc, learn=True, update_gate_stats=(i == 0),
                                prompt_weight=1.0, force_fire=True)
        self._buffer_add(doc, kind="curated", prompt=pk, response=rk,
                         source="teach")
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
                 {"role": "assistant", "content": a}], tokenize=False,
                **self._tmpl_kw())

        def _enc(doc):
            # completion-only masking here too: ascent/descent only on assistant tokens
            enc = self._m.tok(doc, return_tensors="pt",
                              return_offsets_mapping=True)
            ids = enc.input_ids.to(self._m.device)
            labels = self._m._assistant_labels(ids, enc.offset_mapping[0], doc)
            return ids, labels

        params = self._deep_params + [self._m.head.A, self._m.head.B]
        opt = torch.optim.SGD(params, lr=lr)
        old_ids, old_lab = _enc(_doc(prompt, old_response))
        new_ids, new_lab = _enc(_doc(prompt, new_response))
        for _ in range(steps):
            loss = -self._m._nll(old_ids, old_lab)  # ascent: forget old
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.5)
            opt.step()
            loss = self._m._nll(new_ids, new_lab)   # descent: learn new
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.5)
            opt.step()
        # purge stale buffer lessons (prompt + old response) so future sleeps
        # don't re-burn the superseded belief
        pk, ok_ = prompt.strip(), old_response.strip()

        def _stale(d):
            if isinstance(d, dict) and d.get("prompt") is not None:
                return (d["prompt"] == pk
                        and (d.get("response") or "").strip() == ok_)
            t = self._entry_text(d)          # legacy string entries
            return pk[:80] in t and ok_[:40] in t
        before = len(self.replay_buffer)
        self.replay_buffer = [d for d in self.replay_buffer if not _stale(d)]
        # store the new binding as a protected lesson
        self._buffer_add(_doc(prompt, new_response), kind="curated",
                         prompt=pk, response=new_response.strip(),
                         source="revise")
        return {"steps": steps, "purged": before - len(self.replay_buffer) + 1}


    # ---------- probe & bind helpers (used by try_agent.py) ----------
    def ask(self, question: str, system_prompt: Optional[str] = None,
            max_new_tokens: int = 48) -> str:
        """Greedy, weights-only answer (no tools, no sampling).

        The cleanest way to see what the weights know — before/after teach().
        """
        import torch
        chat = ([{"role": "system", "content": system_prompt}]
                if system_prompt else [])
        chat.append({"role": "user", "content": question})
        ids = self._m.tok.apply_chat_template(
            chat, add_generation_prompt=True, return_tensors="pt",
            **self._tmpl_kw())
        if not torch.is_tensor(ids):
            ids = ids["input_ids"]
        ids = ids.to(self._m.device)
        with torch.no_grad():
            out = self._m.model.generate(
                input_ids=ids, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self._m.tok.eos_token_id)
        return self._m.tok.decode(out[0, ids.shape[1]:],
                                  skip_special_tokens=True)

    def prob(self, prompt: str, response: str) -> float:
        """Mean per-token P(response | prompt) under the templated chat."""
        import torch
        ids = self._m.tok.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True,
            return_tensors="pt", **self._tmpl_kw())
        if not torch.is_tensor(ids):
            ids = ids["input_ids"]
        ids = ids.to(self._m.device)
        ans = self._m.tok(response, return_tensors="pt",
                          add_special_tokens=False).input_ids.to(self._m.device)
        full = torch.cat([ids, ans], dim=1)
        with torch.no_grad():
            lp = torch.log_softmax(
                self._m.model(input_ids=full).logits[0].float(), -1)
        tot = sum(lp[i, full[0, i + 1]].item()
                  for i in range(ids.shape[1] - 1, full.shape[1] - 1))
        return float(torch.exp(torch.tensor(tot / max(ans.shape[1], 1))))

    def bind(self, prompt: str, response: str,
             system_prompt: Optional[str] = None, tool_specs=None,
             key: Optional[str] = None, max_rounds: int = 12,
             verbose: bool = True) -> bool:
        """Teach a fact until greedy generation flips. Returns True on success.

        Displaces a consolidated prior via revise() when needed, then repeats
        teach() across bare / system-prompt / tool-spec chat renders until
        ask() emits the key token (the most distinctive word of the response,
        auto-detected unless given). Stops at first hit — over-training babbles.
        """
        if key is None:
            words = [w.strip(".,;:!?") for w in response.split()]
            # priority: digit tokens (versions, codes) > ALL-CAPS > hyphenated
            digits = [w for w in words if any(c.isdigit() for c in w)]
            caps = [w for w in words if w.isupper() and len(w) > 2]
            hyph = [w for w in words if "-" in w and len(w) > 3]
            key = (digits or caps or hyph or words)[-1]
        g0 = self.ask(prompt, system_prompt)
        if key.lower() not in g0.lower() and len(g0.strip()) > 8:
            if verbose:
                logger.info("bind: displacing prior answer via revise(): %r",
                            g0[:60])
            self.revise(prompt, g0.strip(), response, steps=10)
        chat = ([{"role": "system", "content": system_prompt}]
                if system_prompt else [])
        chat += [{"role": "user", "content": prompt},
                 {"role": "assistant", "content": response}]
        docs = []
        try:
            docs.append(self._m.tok.apply_chat_template(
                chat, tokenize=False, **self._tmpl_kw()))
        except Exception:
            pass
        if tool_specs:
            try:
                docs.append(self._m.tok.apply_chat_template(
                    chat, tokenize=False,
                    tools=self._tools_for_template(tool_specs),
                    **self._tmpl_kw()))
            except Exception:
                pass
        for round_ in range(1, max_rounds + 1):
            self.teach(prompt, response, epochs=2)
            for d in docs:
                self.observe(d, learn=True, epochs=1, update_gate_stats=False)
            g = self.ask(prompt, system_prompt)
            hit = key.lower() in g.lower()
            if verbose:
                logger.info("bind round %2d: P=%.4f gen=%s%r", round_,
                            self.prob(prompt, response),
                            "HIT " if hit else "", g[:60])
            if hit:
                return True
        return False


    def learn_from_history(self, messages, system_prompt: Optional[str] = None,
                           tool_specs=None, epochs: int = 1,
                           chunk: int = 6) -> list:
        """Post-tune on a full conversation history (tool turns included).

        `messages` is a Strands-style Messages list — role + content blocks,
        including toolUse / toolResult blocks — OR a plain list of
        {"role": ..., "content": "<str>"} dicts. The history is rendered
        through the model's REAL chat template (with tool specs when given)
        in sliding windows of `chunk` messages, and each window is observed
        with learning on. This is how you turn dense agent traces into
        weight updates on the fly.

        C11 finding: raw transcripts teach FORM, not facts. So in addition
        to observing the raw windows, each (user prompt -> final assistant
        answer) pair is curated through teach() — the unit that actually
        binds facts. Tool-use turns stay in the raw windows so the model
        also learns the tool I/O shapes.

        Returns the list of per-window surprises (pre-update NLLs).
        """
        # normalize plain dicts into content-block form
        norm = []
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, str):
                content = [{"text": content}]
            norm.append({"role": msg["role"], "content": content})
        surprises = []
        for start in range(0, len(norm), max(chunk // 2, 1)):
            window = norm[start:start + chunk]
            if not window:
                break
            chat = self._to_chat(window, system_prompt, tool_specs)
            try:
                doc = self._m.tok.apply_chat_template(
                    chat, tokenize=False,
                    tools=self._tools_for_template(tool_specs),
                    **self._tmpl_kw())
            except Exception:
                doc = self._m.tok.apply_chat_template(chat, tokenize=False,
                                                      **self._tmpl_kw())
            e = self.observe(doc, learn=True, epochs=epochs)
            if e is not None:
                surprises.append(e)
            self._buffer_add(doc, kind="raw", source="history")
            if start + chunk >= len(norm):
                break
        # curate (user -> final assistant answer) pairs; skip tool-call turns
        for i, msg in enumerate(norm[:-1]):
            if msg["role"] != "user":
                continue
            user_text = " ".join(b.get("text", "") for b in msg["content"]
                                 if isinstance(b, dict)).strip()
            # find the LAST assistant message before the next user turn
            answer = None
            for nxt in norm[i + 1:]:
                if nxt["role"] == "user":
                    break
                if nxt["role"] == "assistant":
                    text = " ".join(b.get("text", "") for b in nxt["content"]
                                    if isinstance(b, dict)).strip()
                    if text and "<tool_call>" not in text:
                        answer = text
            if user_text and answer:
                # bind() = teach + revise-displacement until the greedy
                # generation actually flips (C45: teach alone can't displace
                # a consolidated prior)
                self.bind(user_text, answer, verbose=False, max_rounds=16)
        return surprises

    def consolidate(self, epochs: int = 5, lr_boost: float = 1.0):
        """Sleep phase: replay the whole turn buffer to consolidate knowledge.

        Biological analogy: hippocampal replay during sleep. The agent
        accumulates experience cheaply during turns (low interference) and
        consolidates it into stronger weights offline.

        Returns mean surprise over the last replay epoch.
        """
        if not self.replay_buffer:
            return None
        old_lrs = [g["lr"] for g in self._m.opt.param_groups]
        for g in self._m.opt.param_groups:
            g["lr"] = g["lr"] * lr_boost
        last = None
        try:
            for _ in range(epochs):
                docs = list(self.replay_buffer)
                random.shuffle(docs)
                # sleep replay must not pollute the wake surprise EMA.
                # ONLY curated lessons replay at full prompt weight — raw
                # transcripts keep the damped assistant weighting, otherwise
                # sleep burns tool output/boilerplate (and any injected text)
                # at 1.0 for epochs x buffer updates, re-opening the wake-time
                # poisoning damping.
                # amb_E20b fix: sleep replay of CURATED lessons bypasses the
                # gate (force_fire) — rehearsed lessons are low-surprise by
                # design, so a k>=0 gate silently skipped them and sleep did
                # not consolidate (gated recall stuck at 3/8).
                es = [self._m.observe(
                          self._entry_text(d), learn=True,
                          update_gate_stats=False,
                          prompt_weight=1.0 if self._entry_kind(d) == "curated" else None,
                          force_fire=self._entry_kind(d) == "curated")
                      for d in docs]
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
        self.audit_log = []
        self.replay_buffer = []
        self._buffer_seen = 0

    def save_fast_weights(self, path: str, include_transcripts: bool = True):
        """Persist the fast weights (and, optionally, the replay buffer).

        pass include_transcripts=False before PUBLISHING an experience
        file — the replay buffer contains verbatim user conversations.
        """
        import torch
        if include_transcripts:
            logger.warning(
                "save_fast_weights: replay buffer (verbatim transcripts) is "
                "included — use include_transcripts=False before publishing.")
        # QA-28: atomic write. torch.save truncates in place, so a crash /
        # disk-full mid-write destroyed the previous checkpoint at the same
        # path (agents overwrite one path every save). Write to a temp file
        # in the same directory, then os.replace (atomic on POSIX).
        tmp = f"{path}.tmp.{os.getpid()}"
        try:
            torch.save({"A": self._m.head.A.detach().cpu(),
                        "B": self._m.head.B.detach().cpu(),
                        "deep": [p.detach().cpu() for p in self._deep_params],
                        "turn_count": self.turn_count,
                        "plasticity": self.plasticity,
                        "placement": self.placement,
                        "replay_buffer": list(self.replay_buffer)
                                         if include_transcripts else []}, tmp)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise

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
        import re as _re
        # weights_only=True — no arbitrary pickle execution
        ckpts = [torch.load(p, map_location=self._m.device, weights_only=True)
                 for p in paths]

        # C41 guardrail: detect conflicting lessons (same user prompt, different
        # assistant response) across checkpoints. SUM on conflicts corrupts
        # (superposed deltas -> babble that smears into neighbors).
        _pair_res = [
            # Qwen
            _re.compile(r"<\|im_start\|>user\n(.+?)<\|im_end\|>.*?"
                        r"<\|im_start\|>assistant\n(.+?)(?:<\|im_end\|>|$)", _re.DOTALL),
            # Gemma 4
            _re.compile(r"<\|turn>user\n(.+?)<turn\|>.*?"
                        r"<\|turn>model\n(.+?)(?:<turn\|>|$)", _re.DOTALL),
            # Gemma 2/3
            _re.compile(r"<start_of_turn>user\n(.+?)<end_of_turn>.*?"
                        r"<start_of_turn>model\n(.+?)(?:<end_of_turn>|$)", _re.DOTALL),
            # raw fallback (template-less _transcript path)
            _re.compile(r"^user: (.+?)\nassistant: (.+?)$", _re.DOTALL),
        ]

        def _pairs(buf):
            out = {}
            for d in buf:
                if isinstance(d, dict) and d.get("prompt"):
                    out.setdefault(d["prompt"].strip(),
                                   set()).add((d.get("response") or "").strip())
                    continue
                text = d["text"] if isinstance(d, dict) else d
                for rx in _pair_res:
                    mm = rx.search(text)
                    if mm:
                        out.setdefault(mm.group(1).strip(),
                                       set()).add(mm.group(2).strip())
                        break
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
            # EXACT composition. Summing the LoRA FACTORS is wrong math:
            # (sum A_i)(sum B_i) = sum A_i B_i + sum_{i!=j} A_i B_j — each
            # agent draws its own random A, so the cross terms are the same
            # order of magnitude as the signal (measured rel. error ~1.0).
            # The exact composed delta is RANK CONCATENATION, re-compressed
            # to the instance rank via thin-QR + small SVD (_merge_factors);
            # the dense delta (d_in x vocab) is never materialized.
            for c in ckpts:
                if (c["A"].shape[0] != self._m.head.A.shape[0]
                        or c["B"].shape[1] != self._m.head.B.shape[1]):
                    raise ValueError(
                        "merge_experience: head shape mismatch — checkpoint "
                        f"A{tuple(c['A'].shape)}/B{tuple(c['B'].shape)} vs "
                        f"instance A{tuple(self._m.head.A.shape)}/"
                        f"B{tuple(self._m.head.B.shape)}")
                dl = c.get("deep", [])
                if dl and len(dl) != len(self._deep_params):
                    raise ValueError(
                        f"merge_experience: checkpoint has {len(dl)} deep "
                        f"tensors, instance has {len(self._deep_params)} — "
                        "merging requires identical placement/deep_blocks")
            with torch.no_grad():
                A, B = self._merge_factors(
                    [c["A"] for c in ckpts], [c["B"] for c in ckpts],
                    self._m.head.A.shape[1])
                hA, hB = self._m.head.A, self._m.head.B
                hA.copy_(A.to(hA.dtype).to(hA.device))
                hB.copy_(B.to(hB.dtype).to(hB.device))
                deep_lists = [c["deep"] for c in ckpts if c.get("deep")]
                if deep_lists:
                    for pi in range(0, len(self._deep_params), 2):
                        A, B = self._merge_factors(
                            [dl[pi] for dl in deep_lists],
                            [dl[pi + 1] for dl in deep_lists],
                            self._deep_params[pi].shape[1])
                        pA = self._deep_params[pi]
                        pB = self._deep_params[pi + 1]
                        pA.copy_(A.to(pA.dtype).to(pA.device))
                        pB.copy_(B.to(pB.dtype).to(pB.device))
        elif strategy == "relearn":
            # amb_E17/E17b finding: 3 plain observe passes UNDER-TRAIN the
            # merged lessons (0/6 recall). The proven recipe is the teach()
            # path itself: 4 shuffled passes x epochs=2 (chat-template render
            # + supersession + prompt_weight=1.0) then consolidate(3) — this
            # restored 6/6 recall with the conflict cleanly resolved
            # (last-writer-wins by shuffle order for conflicting prompts).
            self.reset()
            lessons = []
            for c in ckpts:
                lessons.extend(c.get("replay_buffer", []))
            lessons = [d if isinstance(d, dict)
                       else self._entry(d, "raw", source="legacy")
                       for d in lessons]
            rng = random.Random(0)
            for _ in range(4):
                order = list(lessons)
                rng.shuffle(order)
                for d in order:
                    if d.get("prompt") and d.get("response") is not None:
                        self.teach(d["prompt"], d["response"], epochs=2)
                    else:
                        for _e in range(2):
                            self._m.observe(
                                d["text"], learn=True,
                                update_gate_stats=False,
                                prompt_weight=(1.0 if d.get("kind") == "curated"
                                               else None))
            self.consolidate(epochs=3)
            self.replay_buffer = lessons[-self.replay_cap:]
        else:
            raise ValueError(f"unknown strategy: {strategy}")
        # merge buffers regardless (for future consolidation)
        merged_buf = []
        for c in ckpts:
            merged_buf.extend(
                d if isinstance(d, dict)
                else self._entry(d, "raw", source="legacy")
                for d in c.get("replay_buffer", []))
        if strategy == "sum":
            self.replay_buffer = merged_buf[-self.replay_cap:]
        return {"merged": len(paths), "strategy": strategy,
                "lessons": len(merged_buf), "conflicts": len(conflicts)}

    @staticmethod
    def _merge_factors(As, Bs, r_out):
        """Exact LoRA composition: delta = sum_i A_i B_i == A_cat @ B_cat
        (rank concatenation), re-compressed to rank r_out.

        When total rank K <= r_out the result is EXACT (zero-padded).
        Otherwise: thin-QR both stacked factors, SVD the small K x K core,
        keep the top-r_out singular triplets — the optimal rank-r_out
        approximation of the true sum, without ever materializing the
        dense (d_in x d_out) delta."""
        import torch
        As = [a.float() for a in As]
        Bs = [b.float() for b in Bs]
        A_cat = torch.cat(As, dim=1)            # [d_in, K]
        B_cat = torch.cat(Bs, dim=0)            # [K, d_out]
        K = A_cat.shape[1]
        if K <= r_out:
            A = torch.zeros(A_cat.shape[0], r_out, dtype=A_cat.dtype,
                            device=A_cat.device)
            B = torch.zeros(r_out, B_cat.shape[1], dtype=B_cat.dtype,
                            device=B_cat.device)
            A[:, :K] = A_cat
            B[:K, :] = B_cat
            return A, B
        Qa, Ra = torch.linalg.qr(A_cat)         # [d_in,K], [K,K]
        Qb, Rb = torch.linalg.qr(B_cat.t())     # [d_out,K], [K,K]
        U, S, Vh = torch.linalg.svd(Ra @ Rb.t())
        s = S[:r_out].clamp(min=0).sqrt()
        A = Qa @ U[:, :r_out] * s               # [d_in, r_out]
        B = (Qb @ Vh.t()[:, :r_out] * s).t()    # [r_out, d_out]
        return A, B

    def load_fast_weights(self, path: str):
        import torch
        # weights_only=True — tensors/str/primitives only, no pickle RCE
        ckpt = torch.load(path, map_location=self._m.device, weights_only=True)
        # shape/compat validation — fail loudly instead of cryptic copy_
        # errors or silently-truncating zips
        hA, hB = self._m.head.A, self._m.head.B
        if (tuple(ckpt["A"].shape) != tuple(hA.shape)
                or tuple(ckpt["B"].shape) != tuple(hB.shape)):
            raise ValueError(
                f"load_fast_weights: head rank mismatch — checkpoint "
                f"r={ckpt['A'].shape[1]} (A{tuple(ckpt['A'].shape)}) vs "
                f"instance r={hA.shape[1]} (A{tuple(hA.shape)}). "
                "Construct the SLM with the same plasticity/r_fast as the "
                "checkpoint.")
        deep_saved = ckpt.get("deep", [])
        if len(deep_saved) != len(self._deep_params):
            logger.warning(
                "load_fast_weights: checkpoint has %d deep tensors but this "
                "instance has %d — DEEP WEIGHTS SKIPPED (deep-stored "
                "knowledge will be missing). Match placement/deep_blocks to "
                "load them.", len(deep_saved), len(self._deep_params))
            deep_saved = []
        for meta in ("plasticity", "placement"):
            want = ckpt.get(meta)
            have = getattr(self, meta, None)
            if want and have and want != have:
                logger.warning(
                    "load_fast_weights: checkpoint %s=%r != instance %s=%r",
                    meta, want, meta, have)
        with torch.no_grad():
            hA.copy_(ckpt["A"].to(hA.dtype))
            hB.copy_(ckpt["B"].to(hB.dtype))
            for p, saved in zip(self._deep_params, deep_saved):
                p.copy_(saved.to(p.dtype).to(p.device))
        self.turn_count = ckpt.get("turn_count", 0)
        # legacy checkpoints stored plain strings — wrap into entries
        self.replay_buffer = [
            d if isinstance(d, dict) else self._entry(d, "raw", source="legacy")
            for d in ckpt.get("replay_buffer", [])]
        self._buffer_seen = len(self.replay_buffer)

    # ---------- helpers ----------
    def _tmpl_kw(self) -> dict:
        """Extra kwargs for apply_chat_template. enable_thinking toggles
        reasoning-mode on thinking models (Qwen3: False injects an empty
        <think></think> so generation skips chain-of-thought). None = leave
        the template at its default. Jinja ignores the kwarg on templates
        that don't use it (Gemma, Qwen3-VL), so it is always safe to pass."""
        if self._enable_thinking is None:
            return {}
        return {"enable_thinking": self._enable_thinking}

    def _probe_tool_role(self) -> bool:
        """True iff the chat template preserves role='tool' content."""
        try:
            r = self._m.tok.apply_chat_template(
                [{"role": "user", "content": "q"},
                 {"role": "assistant", "content": "a"},
                 {"role": "tool", "content": "TOOL_PROBE_XYZ"}], tokenize=False)
            return "TOOL_PROBE_XYZ" in r
        except Exception:
            return False

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
                if getattr(self, "_tool_role_ok", True):
                    chat.append({"role": "tool", "content": text})
                else:
                    # Gemma 4 template drops role="tool" — fold into user turn
                    chat.append({"role": "user",
                                 "content": f"[tool result]\n{text}"})
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
        """Parse a tool call from generated text.

        Supports Qwen (<tool_call>{json}</tool_call>) and Gemma 4
        (<|tool>call:name{args}<tool|>, with <|"|> quote tokens)."""
        import re
        import uuid
        # Qwen style
        mm = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
        if mm:
            try:
                obj = json.loads(mm.group(1))
                return {"toolUseId": f"slm-{uuid.uuid4().hex[:8]}",
                        "name": obj["name"],
                        "input": obj.get("arguments", {})}
            except (json.JSONDecodeError, KeyError):
                return None
        # Gemma 4 style
        mm = re.search(r"<\|tool>call:([\w.-]+)\s*(\{.*?\})\s*<tool\|>",
                       text, re.DOTALL)
        if mm:
            raw = mm.group(2).replace('<|"|>', '"')
            try:
                args = json.loads(raw)
            except json.JSONDecodeError:
                # keys may be unquoted in Gemma renders — best-effort repair
                try:
                    repaired = re.sub(r"([{,]\s*)([A-Za-z_][\w-]*)(\s*:)",
                                      r'\1"\2"\3', raw)
                    args = json.loads(repaired)
                except json.JSONDecodeError:
                    return None
            return {"toolUseId": f"slm-{uuid.uuid4().hex[:8]}",
                    "name": mm.group(1), "input": args}
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
            with self._m.learn_lock:   # fast tokenizer: not thread-safe (QA-18)
                return self._m.tok.apply_chat_template(chat, tokenize=False,
                                                       **self._tmpl_kw())
        except Exception:
            return "\n".join(f"{c['role']}: {c['content']}" for c in chat)
