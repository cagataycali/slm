"""Meta-learn the plastic init W0 (A matrices) + per-site lr (RoboTTT-inspired).

RoboTTT meta-learns fast-weight init and update dynamics with gradients-of-
gradients over trajectory sequences. Our tractable translation:

  * A inits (head + deep)  -> FOMAML: run the REAL inner loop (observe() on a
    synthetic fact trace), compute the RETRIEVAL loss (NLL of probe answers
    through the chat template — the cycle-10 lesson baked into the objective),
    take its gradient at the adapted point, apply it to the init. First-order
    because observe()'s in-place SGD steps break the exact second-order graph.
  * per-site lr multipliers (head / deep) -> hill-climb on the same retrieval
    objective (lr needs second-order info for exact gradients; search is the
    honest first-order substitute).
  * B stays zero-init ALWAYS — reset() remains a provable off-switch. The
    meta state is SLOW model identity, not experience.

Output: artifacts/meta_state.pt {"head_A", "deep_A": [...], "lr_mults",
        "model_id", "r_fast", "deep_r", "deep_blocks", "retrieval_nll"}
Load with: SLM(..., meta_state="artifacts/meta_state.pt")

Usage: python scripts/meta_train.py --model Qwen/Qwen3-0.6B --iters 8
"""
import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

# retention anchors: base-knowledge snippets the meta objective must protect.
# (with B=0 the plastic delta is 0, so their baseline NLL is init-independent.)
RET_PROBES = [
    "from strands import Agent, tool",
    "from strands.models import BedrockModel",
    "from strands_tools import calculator, file_read, shell",
    "The quick brown fox jumps over the lazy dog.",
    "Water boils at 100 degrees Celsius at sea level.",
    "def add(a, b):\n    return a + b",
]

ENTITIES = [
    ("what is the name of the maintenance robot on deck {n}?", "It is called {v}.",
     ["RUSTY-{n}", "VOLT-{n}", "GEARBOX-{n}", "PISTON-{n}"]),
    ("what is our project codename? answer with just the name.", "{v}.",
     ["BLUEHERON", "IRONFERN", "SANDPIPER", "COLDBREW", "NIGHTJAR"]),
    ("who leads the oncall rotation? answer with just the name.", "{v}.",
     ["Marisol", "Kenji", "Adaeze", "Bjorn", "Priya"]),
    ("what is the artifact bucket called? answer with just the name.", "{v}.",
     ["crate-vault", "bin-harbor", "pkg-cellar", "jar-loft"]),
    ("what port does the collector listen on? answer with just the port.", "Port {v}.",
     ["61443", "50912", "48733", "59201"]),
    ("what is the deploy host? answer with just the hostname.", "{v}.",
     ["BASILISK", "MANTICORE", "WYVERN", "KRAKEN"]),
]


def make_trace(rng, tok, tmpl_kw, k_facts=3, style="static"):
    """One synthetic trace: k chat-templated fact docs + retrieval probes.

    style="static": each fact appears once, already correct.
    style="ad" (Algorithm Distillation): traces are IMPROVING histories —
      the same question first appears with a WRONG answer, then corrected.
      The meta objective (retrieval of the CORRECT answer) then rewards
      inits whose learning dynamics implement belief REVISION, not just
      first-write binding: the last write must win and the stale one fade.
    """
    picks = rng.sample(ENTITIES, k_facts)
    docs, probes = [], []
    for q_t, a_t, values in picks:
        n = rng.randint(3, 19)
        vals = [x.format(n=n) for x in values]
        v = rng.choice(vals)
        q = q_t.format(n=n)
        a = a_t.format(v=v)
        if style == "ad":
            wrong = rng.choice([x for x in vals if x != v] or ["UNKNOWN"])
            docs.append(tok.apply_chat_template(
                [{"role": "user", "content": q},
                 {"role": "assistant", "content": a_t.format(v=wrong)}],
                tokenize=False, **tmpl_kw))
        docs.append(tok.apply_chat_template(
            [{"role": "user", "content": q},
             {"role": "assistant", "content": a}], tokenize=False, **tmpl_kw))
        probes.append((q, a, v))
    if style == "ad":
        # wrong-then-right must be temporally ordered per fact but
        # interleaved across facts (AD: across-episodic improvement context)
        wrongs, rights = docs[0::2], docs[1::2]
        docs = wrongs + rights
    return docs, probes


def retrieval_loss(m, probes, with_graph=False):
    """Mean NLL of the correct answer given the probe question (templated).
    with_graph=True keeps the autograd graph to A params (FOMAML gradient)."""
    q_ = m._m
    total = None
    for q, a, _v in probes:
        doc = q_.tok.apply_chat_template(
            [{"role": "user", "content": q},
             {"role": "assistant", "content": a}], tokenize=False,
            **m._tmpl_kw())
        enc = q_.tok(doc, return_tensors="pt", return_offsets_mapping=True)
        ids = enc.input_ids.to(q_.device)
        w = q_._assistant_labels(ids, enc.offset_mapping[0], doc)
        nll = q_._nll(ids, w)
        total = nll if total is None else total + nll
        if not with_graph:
            total = total.detach()
    return total / len(probes)


def retention_loss(m, with_graph=False):
    """Mean NLL over base-knowledge anchors (uniform loss, no masking)."""
    q_ = m._m
    total = None
    for pr in RET_PROBES:
        ids = q_.tok(pr, return_tensors="pt").input_ids.to(q_.device)
        nll = q_._nll(ids)
        total = nll if total is None else total + nll
        if not with_graph:
            total = total.detach()
    return total / len(RET_PROBES)


def a_params(m):
    """All A-side plastic tensors: [head.A, deep A0, deep A1, ...]."""
    return [m._m.head.A] + m._deep_params[0::2]


def set_site_lrs(m, base_lr, mults):
    """Fresh optimizer with per-site lrs: head vs deep (+ alphas slow)."""
    head_group = [m._m.head.A, m._m.head.B]
    deep_group = list(m._deep_params)
    alpha_group = m._deep_alphas + [m._m.head.alpha]
    m._m.opt = torch.optim.SGD([
        {"params": head_group, "lr": base_lr * mults["head"]},
        {"params": deep_group, "lr": base_lr * mults["deep"]},
        {"params": alpha_group, "lr": base_lr * 0.1},
    ], lr=base_lr)


def load_init(m, meta_A):
    with torch.no_grad():
        for p, a in zip(a_params(m), meta_A):
            p.copy_(a)
        m._m.head.B.zero_()
        for p in m._deep_params[1::2]:
            p.zero_()
    m._m.mean = None
    m._m.var = None


def inner_run(m, docs, epochs=2):
    for _ in range(epochs):
        for d in docs:
            m._m.observe(d, learn=True, update_gate_stats=False,
                         prompt_weight=1.0, force_fire=True)


def evaluate(m, meta_A, mults, traces, base_lr, with_graph=False, lam=2.0):
    """Adapt on each trace from the meta init; outer objective =
    retrieval NLL + lam * retention NLL (retention term stops the meta loop
    from buying binding speed with base-knowledge damage — measured failure
    without it: probe NLL delta exploded +0.64 -> +3.76).
    Returns (objective, retrieval, retention[, grads])."""
    grads = [torch.zeros_like(a) for a in meta_A] if with_graph else None
    tot_obj = tot_ret = tot_keep = 0.0
    for docs, probes in traces:
        load_init(m, meta_A)
        set_site_lrs(m, base_lr, mults)
        inner_run(m, docs)
        if with_graph:
            r = retrieval_loss(m, probes, with_graph=True)
            k = retention_loss(m, with_graph=True)
            loss = r + lam * k
            for p in a_params(m):
                if p.grad is not None:
                    p.grad = None
            loss.backward()
            for g, p in zip(grads, a_params(m)):
                if p.grad is not None:
                    g += p.grad.detach()
            tot_obj += loss.item(); tot_ret += r.item(); tot_keep += k.item()
        else:
            with torch.no_grad():
                r = retrieval_loss(m, probes).item()
                k = retention_loss(m).item()
            tot_obj += r + lam * k; tot_ret += r; tot_keep += k
    n = len(traces)
    if with_graph:
        return tot_obj / n, tot_ret / n, tot_keep / n, [g / n for g in grads]
    return tot_obj / n, tot_ret / n, tot_keep / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--meta-lr", type=float, default=5e-3)
    ap.add_argument("--n-train", type=int, default=6)
    ap.add_argument("--n-val", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trace-style", choices=["static", "ad"], default="static",
                    help="ad = Algorithm-Distillation improving histories")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "artifacts", "meta_state.pt"))
    a = ap.parse_args()

    from slm import SLM
    t0 = time.time()

    def log(msg):
        print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

    log(f"loading {a.model} ...")
    m = SLM(a.model, plasticity="high", placement="deep",
            enable_thinking=False, max_tokens=32, replay_k=0)
    m.learn_on_turn = False
    base_lr = 2e-2                      # validated deep-placement fast lr
    rng = random.Random(a.seed)
    tmpl_kw = m._tmpl_kw()
    traces = [make_trace(rng, m._m.tok, tmpl_kw, style=a.trace_style)
              for _ in range(a.n_train)]
    val = [make_trace(rng, m._m.tok, tmpl_kw, style=a.trace_style)
           for _ in range(a.n_val)]

    meta_A = [p.detach().clone() for p in a_params(m)]
    mults = {"head": 1.0, "deep": 1.0}

    o0, r0, k0 = evaluate(m, meta_A, mults, val, base_lr)
    log(f"baseline (random init, uniform lr): val obj {o0:.4f} "
        f"(retrieval {r0:.4f}, retention {k0:.4f})")

    for it in range(1, a.iters + 1):
        # --- FOMAML step on A inits (objective incl. retention) ---
        tr_obj, tr_r, tr_k, grads = evaluate(m, meta_A, mults, traces,
                                             base_lr, with_graph=True)
        gn = sum(g.norm().item() ** 2 for g in grads) ** 0.5
        with torch.no_grad():
            for A0, g in zip(meta_A, grads):
                A0 -= a.meta_lr * g
        # --- hill-climb one lr multiplier per iter (alternate site) ---
        site = "head" if it % 2 else "deep"
        cur, _, _ = evaluate(m, meta_A, mults, traces, base_lr)
        for factor in (0.5, 2.0):
            trial = dict(mults)
            # cap at 2x: the outer objective's light inner budget under-
            # estimates retention damage from heavy loops (bind/teach) —
            # measured: uncapped search chose head x8, which cost +2.9 NLL
            # retention under bind() vs +0.77 at uniform. A init is the
            # clean win; lr search stays conservative.
            trial[site] = min(mults[site] * factor, 2.0)
            if trial[site] == mults[site]:
                continue
            o, _, _ = evaluate(m, meta_A, trial, traces, base_lr)
            if o < cur - 1e-4:
                mults, cur = trial, o
        ov, rv, kv = evaluate(m, meta_A, mults, val, base_lr)
        log(f"iter {it}: train obj {tr_obj:.4f} (ret {tr_r:.3f}/keep {tr_k:.3f}) "
            f"|g|={gn:.3f} lr_mults={mults} val obj {ov:.4f} "
            f"(ret {rv:.3f}/keep {kv:.3f})")

    of, rf, kf = evaluate(m, meta_A, mults, val, base_lr)
    log(f"final: val retrieval {r0:.4f} -> {rf:.4f}, "
        f"retention {k0:.4f} -> {kf:.4f} "
        f"({'IMPROVED' if rf < r0 and kf < k0 + 0.15 else 'check tradeoff'})")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save({"head_A": meta_A[0].cpu(),
                "deep_A": [t.cpu() for t in meta_A[1:]],
                "lr_mults": mults, "model_id": a.model,
                "r_fast": m._m.head.A.shape[1],
                "deep_r": m._deep_params[0].shape[1] if m._deep_params else 0,
                "deep_blocks": len(m._deep_params) // 4,  # 2 sites x (A,B)
                "base_lr": base_lr,
                "retrieval_nll": {"before": r0, "after": rf},
                "retention_nll": {"before": k0, "after": kf},
                "trace_style": a.trace_style}, a.out)
    log(f"saved meta state -> {a.out}")


if __name__ == "__main__":
    main()
