# slm API reference

Two layers:

- **`StrandsPlasticQwen`** (`slm.qwen`) — the raw plastic model: load, chat, observe, reset.
- **`SLM`** (`slm.strands_model`) — a [Strands Agents](https://github.com/strands-agents) model
  provider wrapping it: agent turns learn automatically, plus the curated-learning API.
- **`slm_tools`** (`slm.tools`) — the `SLM` API surfaced as Strands `@tool` functions.

---

## SLM

```python
from slm import SLM

model = SLM(
    model_id="cagataydev/strands-qwen3-vl-2b",  # any HF causal-LM or PEFT adapter repo
    plasticity="high",       # "off" | "low" | "medium" | "high"
    placement="deep",        # "deep" attaches LoRA to q/v_proj of the last blocks
    deep_blocks=6, deep_r=32,
    learn_on_turn=True,      # agent turns update weights automatically
    max_tokens=1024, temperature=0.0,
    enable_thinking=None,    # False = skip chain-of-thought on Qwen3 thinking models
)
```

### Plasticity presets

| preset | lr | EMA decay | r_fast | behavior |
|---|---|---|---|---|
| `off` | 0 | 1.0 | 16 | frozen — no learning |
| `low` | 8e-3 | 0.98 | 16 | production: retention Δ≈0 |
| `medium` | 2e-2 | 0.995 | 64 | gate ~always open |
| `high` | 5e-2 | 0.999 | 128 | demo-validated visible learning |

### Probing — what do the weights know?

```python
model.ask("what is my name?")                     # greedy, no tools, no context
model.ask("...", system_prompt="Be brief.")       # under a system prompt
model.prob("what is my name?", "Cagatay")         # mean per-token P(answer | prompt)
```

`ask()` is the cleanest before/after check: no sampling, no tool specs, no
conversation history — if the answer is right, it came from the weights.

### Teaching

```python
# one-shot curated lesson (repeats epochs, full prompt weight)
model.teach("what is X?", "X is Y.", epochs=3)

# teach UNTIL the greedy generation actually flips (recommended)
ok = model.bind("what is X?", "X is Y.",
                key="Y",            # success token; auto-detected if omitted
                max_rounds=12)      # stops at first hit — over-training babbles

# flip a consolidated belief the model actively defends
model.revise("what is X?", old_response="X is Z.", new_response="X is Y.")

# free-form learning on raw text (one surprise-gated step)
surprise = model.observe("some text", learn=True)
```

`bind()` auto-displaces a consolidated prior via `revise()` when the model
generates a confident wrong answer, then teaches across bare / system-prompt /
tool-spec chat renders until the key token appears in greedy generation.

**Key-token caveat**: success is detected by substring match. Pick a `key`
that does **not** appear in the model's wrong prior answer.

### Learning from agent traces

```python
history = [
    {"role": "user", "content": "which host is the staging db on?"},
    {"role": "assistant", "content": '<tool_call>{"name": "shell", ...}</tool_call>'},
    {"role": "tool", "content": "10.0.4.7  BASILISK  # staging"},
    {"role": "assistant", "content": "The staging database runs on BASILISK."},
]
surprises = model.learn_from_history(history, epochs=2)
```

Renders the trace through the real chat template in sliding windows (teaches
tool-call FORM) and `bind()`s each (user → final answer) pair (teaches FACTS).
Accepts Strands `Messages` (content blocks, toolUse/toolResult) or plain
`{"role", "content"}` dicts.

### Consolidation & revision

```python
model.consolidate(epochs=5)     # sleep phase: replay the buffer, harden memories
```

Curated lessons replay at full prompt weight; raw transcripts stay damped.
Do **not** consolidate immediately after `revise()` — let new turns accumulate
first (the replay buffer's semantic neighbors can re-burn the old belief;
`revise` purges direct matches automatically).

### Persistence

```python
model.save_fast_weights("brain.pt")                          # includes replay buffer!
model.save_fast_weights("brain.pt", include_transcripts=False)  # safe to publish
model.load_fast_weights("brain.pt")   # validates rank/placement, fails loudly
model.reset()                         # bit-exact base model, Δlogits = 0
```

> ⚠️ The replay buffer contains **verbatim conversation transcripts**. Always
> pass `include_transcripts=False` before publishing an experience file.

`load_fast_weights` raises `ValueError` on head-rank mismatch (construct the
SLM with the same `plasticity`/`r_fast` as the checkpoint) and warns+skips
when deep-tensor counts differ.

### Fleet learning

```python
result = model.merge_experience(["agent1.pt", "agent2.pt"], strategy="sum")
```

- `"sum"` — exact LoRA composition via rank concatenation, re-compressed to
  the instance rank with thin-QR + SVD (optimal low-rank approximation; the
  dense delta is never materialized). Lossless for disjoint skills.
- `"relearn"` — merges replay buffers and relearns from scratch; use when
  checkpoints contain **conflicting** lessons (same prompt, different answers —
  detected automatically, SUM refuses and falls back).

### Observability

```python
model.surprise_log    # [(turn, pre-update NLL), ...] — watch it learn
model.audit_log       # [{turn, nll, sha256, source}, ...] — attribute updates
model.replay_buffer   # typed entries: {text, kind, prompt, response, sha256, source, ts}
```

Buffer entries are two-tier: `curated` lessons (from `teach`/`revise`) are
never reservoir-evicted; `raw` transcripts use reservoir sampling.

---

## slm_tools — agents that tune their own weights

```python
from strands import Agent
from slm import SLM, slm_tools

model = SLM(plasticity="high")
agent = Agent(model=model, tools=slm_tools(model))
agent("teach yourself that the deploy host is BASILISK, then verify")
```

| tool | wraps |
|---|---|
| `slm_teach(prompt, response)` | `bind()` — teach until greedy flips |
| `slm_probe(question)` | `ask()` — weights-only answer |
| `slm_learn_history(history_json, epochs)` | `learn_from_history()` |
| `slm_observe(text)` | `observe()` — one gated step |
| `slm_save(path)` / `slm_load(path)` | checkpoint I/O |
| `slm_reset()` | the off-switch |
| `slm_status()` | turn count, ‖B‖, buffer size, recent surprises |

---

## StrandsPlasticQwen (raw model)

```python
from slm import StrandsPlasticQwen

m = StrandsPlasticQwen.from_pretrained()   # default: cagataydev/strands-qwen3-vl-2b
m.chat("How do I create a custom tool in Strands Agents?")
m.observe(doc, learn=True)                 # returns pre-update NLL
m.reset()
```

`from_pretrained` handles: PEFT adapter repos (loads base + merges adapter),
Gemma QAT int8 dequantization, per-family assistant-span regex detection,
tool-role template probing, and `AutoModelForCausalLM` fallback for text-only
models.

Constructor knobs (usually set via `SLM` plasticity presets): `r_fast`, `lr`,
`decay`, `k_gate`, `max_B_norm` (Frobenius projection), `neuromod`
(surprise-scaled learning rate).
