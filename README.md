# slm — self-learning model

[![PyPI](https://img.shields.io/pypi/v/strands-slm.svg)](https://pypi.org/project/strands-slm/)
[![Python](https://img.shields.io/pypi/pyversions/strands-slm.svg)](https://pypi.org/project/strands-slm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20model-strands--qwen3--vl--2b-yellow)](https://huggingface.co/cagataydev/strands-qwen3-vl-2b)

**A model whose weights change while it runs.**
Predict, get surprised, rewrite a small bounded part of yourself, never forget the base.

Every LLM you have used is frozen at deployment. `slm` wraps a frozen
[Qwen3-VL-2B post-tuned on the strands-agents codebase](https://huggingface.co/cagataydev/strands-qwen3-vl-2b)
with a plastic layer that keeps learning at inference — with a provable off-switch.

> **For** agent builders and continual-learning researchers who want a model that
> adapts *after* deployment. **Runs on** one GPU (validated on an L40S; CPU works
> for smoke tests). ~21M plastic params (~1%) over a frozen 2.13B base.

```bash
pip install strands-slm
```

**Contents:** [Quickstart](#quickstart) · [Watch it learn](#watch-it-learn) · [Supported models](#supported-models) · [How it works](#how-it-works) · [Results](#results) · [API](#api) · [What we learned](#what-we-learned-building-it) · [Limitations](#honest-limitations) · [Reproduce](#reproduce-the-post-tune)

## Quickstart

As a [Strands Agents](https://github.com/strands-agents) model provider — every turn can change the weights:

```python
from strands import Agent
from strands_tools import shell
from slm import SLM

model = SLM("cagataydev/strands-qwen3-vl-2b")
agent = Agent(tools=[shell], model=model)

agent("use the shell tool to run: echo hello")   # this turn updated the weights
```

Prove it's the weights and not the context window:

```python
model.bind("what is my name?", "Your name is Cagatay.")  # teach until greedy flips
model.save_fast_weights("brain.pt")

# ... new process, fresh model, EMPTY context ...
model = SLM("cagataydev/strands-qwen3-vl-2b")
model.ask("what is my name?")          # doesn't know
model.load_fast_weights("brain.pt")
model.ask("what is my name?")          # "Your name is Cagatay." — from weights
model.reset()                          # forgotten, bit-exact base again
```

Or hand the learning controls to the agent itself:

```python
from slm import SLM, slm_tools

model = SLM()
agent = Agent(model=model, tools=slm_tools(model))
agent("teach yourself that the deploy host is BASILISK, then probe your weights to verify")
```

Or drive the learning loop directly:

```python
from slm import StrandsPlasticQwen

m = StrandsPlasticQwen.from_pretrained()
print(m.chat("How do I create a custom tool in Strands Agents?"))

for doc in your_stream:
    m.observe(doc, learn=True)   # predicts; if surprised, rewrites its fast weights
m.reset()                        # bit-exact back to the base
```

## Watch it learn

**[demo.ipynb](demo.ipynb)** — ask the model a question it cannot know, let it
read documents (pure inference), ask again — it knows. Then reset, and it
forgets. Executed outputs embedded; validated on an L40S:
P(correct) 0.09 → 0.74, greedy answers 3/3, reset Δlogits = 0.
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cagataycali/slm/blob/main/demo.ipynb)
· [view on nbviewer](https://nbviewer.org/github/cagataycali/slm/blob/main/demo.ipynb)

**[try_agent.py](try_agent.py)** — a 66-line REPL where every turn physically
updates the weights (`/teach`, `/ask`, `/reset`):

```bash
python try_agent.py
```

## Supported models

Any HF causal-LM (or PEFT adapter repo) works — the plastic layer attaches
generically. Validated end to end:

| model | params | notes |
|---|---|---|
| [`cagataydev/strands-qwen3-vl-2b`](https://huggingface.co/cagataydev/strands-qwen3-vl-2b) | 2B | **default** — Qwen3-VL-2B post-tuned on the strands-agents codebase (vision included) |
| [`cagataydev/strands-gemma4-e2b`](https://huggingface.co/cagataydev/strands-gemma4-e2b) | e2b | Gemma 4 QAT adapter repo — int8 layers dequantized so the plastic LoRA attaches |
| [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) | 0.6B | smallest validated; pass `enable_thinking=False` to skip chain-of-thought |

```python
SLM("cagataydev/strands-qwen3-vl-2b")               # strands expert (default)
SLM("cagataydev/strands-gemma4-e2b")                # gemma family
SLM("Qwen/Qwen3-0.6B", enable_thinking=False)       # tiny + fast
```

Per family, the loader auto-detects: assistant-span regex for the chat
template, tool-role support (folds tool results into user turns when the
template drops them), PEFT adapter merging, and QAT dequantization.

## How it works

```
frozen Qwen3-VL-2B            instinct — never updated, cannot forget
  + strands LoRA (merged)     slow: post-tuned strands-agents expertise
  + plastic LoRA              fast: ~21M params (~1%) over the frozen 2.13B,
                              updated on every observation at inference
  + surprise gate             learn only when prediction error spikes
  + EMA decay                 bounded plasticity — learns AND retains

loss = next-observation prediction error   (the free label from reality)
```

## Results

Measured on a single GPU, seed-replicated. The base model is never updated.

| claim | evidence |
|---|---|
| Domain expert | strands probe NLL 4.85 → 2.22, 8/8 probes improved |
| Learns while running | continual OOD stream NLL 6.18 → 5.37, pure inference |
| Does not forget | strands expertise after OOD learning: Δ −0.01 |
| Agent competence grows | held-out tasks 0/4 → 4/4 after 18 curated lessons, 5/5 seeds |
| Fact memory | 15/15 facts at 100% verbatim recall |
| Fleet merge (arithmetic) | exact delta composition (rel. err 1e-7 vs ~1.0 for naive factor-sum); **skill transfer after merging is NOT guaranteed** — see caveat below |
| Persistence | experience survives process death bit-exact |
| Provable off-switch | `reset()` is bit-identical to the base, Δlogits = 0 |
| Cost | +0.11–0.25 s/turn learning overhead |

The stability–plasticity dial, measured (OOD baseline NLL 4.23):

| lr | EMA decay | OOD gain | retention Δ | |
|---|---|---|---|---|
| 2e-3 | 0.98 | +0.05 | +0.00 | too timid |
| 8e-3 | 0.98 | +0.89 | +0.03 | the sweet spot |
| 1e-2 | none | +3.30 | +7.09 | forgets the base |

## API

| method | what it does |
|---|---|
| `SLM(model_id, plasticity="high", placement="deep")` | Strands provider; agent turns learn automatically |
| `.ask(question)` | greedy, weights-only answer — probe what the weights know |
| `.prob(prompt, response)` | P(response \| prompt) under the chat template |
| `.bind(prompt, response)` | teach until the greedy generation actually flips (auto-displaces consolidated priors via `revise`) |
| `.teach(prompt, response)` | curated lesson: bind a future query to a desired response |
| `.observe(text, learn=True)` | free-form learning; returns pre-update surprise (NLL) |
| `.learn_from_history(messages)` | post-tune on a full agent trace — tool inputs/outputs included |
| `.consolidate(epochs=5)` | sleep phase: replay the lesson buffer, harden weak memories |
| `.revise(prompt, old, new)` | targeted unlearning: flip a consolidated belief |
| `.save_fast_weights(path)` / `.load_fast_weights(path)` | persist or restore acquired experience |
| `.merge_experience(paths)` | compose agents' experience files. **Caveat (measured):** the delta arithmetic is exact and conflict detection works, but merged fact bindings from same-format experience can fail to transfer — the merged deltas are parameter-orthogonal yet the composed model may babble. Verify recall after merging; prefer `strategy="relearn"` + re-teaching for critical lessons |
| `.reset()` | the off-switch — exactly the base model again |
| `.surprise_log` | (turn, NLL) history — watch it learn |
| `.audit_log` | per-update content hash + provenance — attribute any poisoned update |
| `slm_tools(model)` | the whole API above as Strands `@tool` functions — agents tune their own weights |

Full reference with parameters and examples: **[docs/api.md](docs/api.md)** · styled version with a replay of the verified session: **[cagataycali.github.io/slm/api.html](https://cagataycali.github.io/slm/api.html)**.

> **Privacy**: `save_fast_weights` includes the replay buffer — verbatim
> conversation transcripts — by default. Pass `include_transcripts=False`
> before publishing an experience file.

## What we learned building it

1. Placement determines what can be learned: attention q/v LoRA stores bindings
   about 4x more sample-efficiently than the LM head.
2. There is a free-learning regime (deep placement, lr 2e-2, decay 0.999):
   skill acquisition at zero retention cost.
3. You retrieve in the format you learned — render lessons through the real
   chat template or the knowledge is invisible at inference.
4. Curation is the difference between experience and learning: raw feedback
   transcripts teach nothing; distilled (task → corrected response) pairs
   take held-out competence from 0/4 to 4/4.
5. Interleave or lose it: sequential lessons evict each other; replay makes
   them coexist. Sleep-style consolidation hardens weak memories.
6. Belief revision is a terminal operation: whatever is learned last in a
   semantic neighborhood wins — order lessons before the revision.
7. Raw transcripts teach FORM (tool-call syntax, formats); curated
   (prompt → answer) pairs teach FACTS. `learn_from_history` does both.
8. Tokenization shapes learnability: word-shaped facts (`BASILISK`) flip in
   one call; multi-digit strings (`88.1.21`) fragment into many tokens and
   take an order of magnitude more rounds to bind.

## Honest limitations

- A bolt-on linear memory degrades single-prompt in-context recall — softmax
  attention is already the better mechanism there. The win is persistent
  cross-sequence adaptation, which the context window cannot retain.
- About a third of naive test-time-training gains in the literature are pure
  calibration (even zero-information targets help an over-confident head).
  Our evals control for this with an information-ladder baseline.
- Composition (learned schema x unseen entity) plateaus near 67% at 2B.
- `bind()` verifies success by key-token match — pick a key that does NOT
  appear in the model's wrong prior answer, or you get a false positive.
- Freshly-bound facts can smear at minimal binding strength (right key
  token, loose frame) — a couple of extra teach rounds tightens it.
- **Merging is exact arithmetic, not guaranteed skill composition** — E7c
  (see `experiments/`): two agents' bind-trained deltas were near-orthogonal
  in parameter space, yet the SUM-merged model recalled 0/6 facts and emitted
  repetition loops. Verify recall after any merge.
- Replay protects lessons at a measured cost to general retention
  (+0.59 vs +0.16 NLL under a 12-doc interference stream) — two-tier replay
  chooses lesson retention; tune `replay_k` to your priorities.
- All findings are at 0.6B–2B scale; scaling behavior is unknown.

A full experimental treatment (ablations, negative results, cross-family
replication, latency) is in the paper draft under `paper/` with logs in
`experiments/`.

## Reproduce the post-tune

```bash
pip install "strands-slm[train]"
python scripts/build_corpus.py       # strands-agents repos -> corpus.jsonl
python scripts/train_lora.py --steps 1200 --bs 2 --accum 4 --lr 1e-4
python scripts/eval_strands.py       # base vs tuned probes
```

Private HF repos need `HF_TOKEN` in the environment, or pass `token=`.

## Citation

If you use `slm` in your research, please cite:

```bibtex
@software{slm2026,
  title  = {slm: a self-learning Strands-Agents model with a provable off-switch},
  author = {Cali, Cagatay},
  year   = {2026},
  url    = {https://github.com/cagataycali/slm}
}
```

## License

MIT — see [LICENSE](LICENSE).
