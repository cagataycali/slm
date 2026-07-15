# slm

**A model whose weights change while it runs.**
Predict, get surprised, rewrite a small bounded part of yourself, never forget the base.

Every LLM you have used is frozen at deployment. `slm` wraps a frozen
[Qwen3-VL-2B post-tuned on the strands-agents codebase](https://huggingface.co/cagataydev/strands-qwen3-vl-2b)
with a plastic layer that keeps learning at inference — with a provable off-switch.

```bash
pip install strands-slm
```

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

Or drive the learning loop directly:

```python
from slm import StrandsPlasticQwen

m = StrandsPlasticQwen.from_pretrained()
print(m.chat("How do I create a custom tool in Strands Agents?"))

for doc in your_stream:
    m.observe(doc, learn=True)   # predicts; if surprised, rewrites its fast weights
m.reset()                        # bit-exact back to the base
```

**See it happen: [demo.ipynb](demo.ipynb)** — ask the model a question it cannot
know, let it read documents (pure inference), ask again — it knows. Then reset,
and it forgets. Executed outputs and plots embedded; validated on an L40S:
P(correct) 0.09 → 0.74, greedy answers 3/3, reset Δlogits = 0.

## How it works

```
frozen Qwen3-VL-2B            instinct — never updated, cannot forget
  + strands LoRA (merged)     slow: post-tuned strands-agents expertise
  + plastic LoRA              fast: ~1.6M params over the frozen 2.13B,
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
| Fleet learning | two agents' experience files summed losslessly |
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
| `.teach(prompt, response)` | curated lesson: bind a future query to a desired response |
| `.observe(text, learn=True)` | free-form learning; returns pre-update surprise (NLL) |
| `.consolidate(epochs=5)` | sleep phase: replay the lesson buffer, harden weak memories |
| `.revise(prompt, old, new)` | targeted unlearning: flip a consolidated belief |
| `.save_fast_weights(path)` / `.load_fast_weights(path)` | persist or restore acquired experience |
| `.merge_experience(paths)` | fleet learning: compose multiple agents' experience files |
| `.reset()` | the off-switch — exactly the base model again |
| `.surprise_log` | (turn, NLL) history — watch it learn |

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

## Honest limitations

- A bolt-on linear memory degrades single-prompt in-context recall — softmax
  attention is already the better mechanism there. The win is persistent
  cross-sequence adaptation, which the context window cannot retain.
- About a third of naive test-time-training gains in the literature are pure
  calibration (even zero-information targets help an over-confident head).
  Our evals control for this with an information-ladder baseline.
- Composition (learned schema x unseen entity) plateaus near 67% at 2B.
- All findings are at 2B scale; scaling behavior is unknown.

## Reproduce the post-tune

```bash
pip install "strands-slm[train]"
python scripts/build_corpus.py       # strands-agents repos -> corpus.jsonl
python scripts/train_lora.py --steps 1200 --bs 2 --accum 4 --lr 1e-4
python scripts/eval_strands.py       # base vs tuned probes
```

Private HF repos need `HF_TOKEN` in the environment, or pass `token=`.

## License

MIT
