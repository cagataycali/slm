#!/usr/bin/env python
"""
Interactive self-learning agent — try it yourself.

A Strands Agent backed by SLM (strands-qwen3-vl-2b + plastic weights) with a
shell tool. Every turn you type physically updates the model's fast weights.

Run:
    BYPASS_TOOL_CONSENT=true .venv/bin/python try_agent.py

Try this experiment (use neutral facts — safety-refusal topics like "access
codes" are heavily consolidated in the base and fight back much harder):
    you> what is the name of the maintenance robot on deck 12?   # it doesn't know
    you> /teach what is the name of the maintenance robot on deck 12? | The maintenance robot on deck 12 is called RUSTY-9.
    you> what is the name of the maintenance robot on deck 12?   # now it KNOWS (from weights!)
    you> /reset                                                   # off-switch
    you> what is the name of the maintenance robot on deck 12?   # forgotten

Commands:
    /ask <question>                answer from WEIGHTS ONLY (no tools, greedy) —
                                   the cleanest way to see what it has learned
    /teach <prompt> | <response>   curated learning (repeats until it sticks)
    /observe <text>                learn from raw text (one surprise-gated step)
    /surprise                      show the surprise log (proof of weight updates)
    /save [path]                   save fast weights
    /load [path]                   load fast weights
    /reset                         wipe all learning -> exact base model
    /quit                          exit
Anything else is a normal agent turn (shell tool available) — and the model
learns from the transcript afterwards.

NOTE on agent turns vs /ask: when tool specs are in context, the model
correctly prefers "let me check via the shell tool" over answering from
memory — that's proper agent behaviour, not a learning failure. Use /ask to
probe what the weights know; use plain turns to watch it operate tools (and
learn from every transcript).
"""
import os
os.environ.setdefault("BYPASS_TOOL_CONSENT", "true")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from strands import Agent
from strands_tools import shell
from strands.tools.registry import ToolRegistry
from slm import SLM

_registry = ToolRegistry()
_registry.process_tools([shell])
TOOL_SPECS = _registry.get_all_tool_specs()

CKPT = "/tmp/slm_try_agent.pt"
SYSTEM = "You are a helpful assistant with shell access. Be brief."


def gen_as_agent(m, prompt, max_new=32):
    """Greedy generation under the SAME system prompt the agent uses."""
    ids = m.tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": prompt}],
        add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(ids):
        ids = ids["input_ids"]
    ids = ids.to(m.device)
    with torch.no_grad():
        out = m.model.generate(input_ids=ids, max_new_tokens=max_new,
                               do_sample=False, pad_token_id=m.tok.eos_token_id)
    return m.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def p_of(m, prompt, answer):
    """P(answer | prompt) under the templated chat distribution."""
    ids = m.tok.apply_chat_template([{"role": "user", "content": prompt}],
                                    add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(ids):
        ids = ids["input_ids"]
    ids = ids.to(m.device)
    ans = m.tok(answer, return_tensors="pt", add_special_tokens=False).input_ids.to(m.device)
    full = torch.cat([ids, ans], dim=1)
    with torch.no_grad():
        lp = torch.log_softmax(m.model(input_ids=full).logits[0].float(), -1)
    tot, n = 0.0, 0
    for i in range(ids.shape[1] - 1, full.shape[1] - 1):
        tot += lp[i, full[0, i + 1]].item(); n += 1
    return float(torch.exp(torch.tensor(tot / n)))


def main():
    print("loading strands-qwen3-vl-2b + plastic weights (plasticity=high, deep placement)...")
    model = SLM(plasticity="high", placement="deep", max_tokens=256,
                learn_epochs=1, replay_k=2)
    b_norm = lambda: model._m.head.B.norm().item()
    print(f"ready. fast params: head {tuple(model._m.head.B.shape)} + "
          f"{len(model._deep_params)//2} deep LoRAs. ||B||={b_norm():.4f}")
    print(__doc__.split("Commands:")[1])

    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue

        if q in ("/quit", "/exit", "/q"):
            break

        elif q.startswith("/ask "):
            question = q[len("/ask "):]
            g = gen_as_agent(model._m, question, max_new=48)
            print(f"weights> {g}")

        elif q == "/reset":
            model.reset()
            print(f"[reset] all fast weights wiped — bit-exact base model. ||B||={b_norm():.4f}")

        elif q == "/surprise":
            if not model.surprise_log:
                print("[surprise] no turns learned yet")
            for t, e in model.surprise_log[-15:]:
                print(f"  turn {t:3d}: surprise (pre-update NLL) = {e:.3f}")
            print(f"  ||B|| now = {b_norm():.4f}  (0 = base, >0 = learned)")

        elif q.startswith("/teach "):
            body = q[len("/teach "):]
            if "|" not in body:
                print("usage: /teach <prompt> | <response>")
                continue
            prompt, response = (s.strip() for s in body.split("|", 1))
            # key token = the most distinctive word of the response:
            # prefer code-looking tokens (digits/CAPS/hyphens), else last word
            words = response.replace(".", " ").replace(",", " ").split()
            distinctive = [w for w in words
                           if any(c.isdigit() for c in w)
                           or (w.isupper() and len(w) > 2)
                           or ("-" in w and len(w) > 3)]
            key = distinctive[-1] if distinctive else words[-1]
            p0 = p_of(model._m, prompt, response)
            g0 = gen_as_agent(model._m, prompt)
            print(f"[teach] before: P={p0:.4f}  gen={g0[:70]!r}")
            # if the model actively refuses/says something else, displace it
            # first with targeted unlearning (ascent on old, descent on new)
            if key.lower() not in g0.lower() and len(g0.strip()) > 8:
                print(f"[teach] displacing prior answer via revise() ...")
                model.revise(prompt, g0.strip(), response, steps=10)
            # teach THREE bindings so learning transfers to real agent turns:
            # bare, +system prompt, +system+tools (the agent's actual render)
            chat = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}]
            sys_doc = model._m.tok.apply_chat_template(chat, tokenize=False)
            try:
                tools_doc = model._m.tok.apply_chat_template(
                    chat, tokenize=False,
                    tools=model._tools_for_template(TOOL_SPECS))
            except Exception:
                tools_doc = None
            success = False
            for round_ in range(1, 13):
                model.teach(prompt, response, epochs=2)
                model.observe(sys_doc, learn=True, epochs=1,
                              update_gate_stats=False)
                if tools_doc:
                    model.observe(tools_doc, learn=True, epochs=1,
                                  update_gate_stats=False)
                p = p_of(model._m, prompt, response)
                g = gen_as_agent(model._m, prompt)
                hit = key.lower() in g.lower()
                print(f"[teach] round {round_:2d}: P={p:.4f}  ||B||={b_norm():.4f}  "
                      f"gen={'HIT ' if hit else ''}{g[:60]!r}")
                if hit:
                    success = True
                    break   # stop at first hit — over-training causes babble
            print(f"[teach] {'stuck! ask it yourself' if success else 'partial — try /teach again or /observe'}: {prompt!r}")

        elif q.startswith("/observe "):
            text = q[len("/observe "):]
            e = model.observe(text, learn=True)
            print(f"[observe] surprise={e:.3f}, gate fired={model._m.last_fired}, "
                  f"||B||={b_norm():.4f}")

        elif q.startswith("/save"):
            path = q.split(maxsplit=1)[1] if " " in q else CKPT
            model.save_fast_weights(path)
            print(f"[save] fast weights -> {path}")

        elif q.startswith("/load"):
            path = q.split(maxsplit=1)[1] if " " in q else CKPT
            model.load_fast_weights(path)
            print(f"[load] fast weights <- {path}   ||B||={b_norm():.4f}")

        else:
            # normal agent turn — fresh Agent each turn so answers come from
            # WEIGHTS, not from conversation context (that's the whole point)
            before = b_norm()
            agent = Agent(model=model, tools=[shell], callback_handler=None,
                          system_prompt="You are a helpful assistant with shell access. Be brief.")
            reply = agent(q)
            print(f"\nagent> {reply}")
            e = model.surprise_log[-1][1] if model.surprise_log else float("nan")
            print(f"[learned from this turn: surprise={e:.3f}, "
                  f"||B|| {before:.4f} -> {b_norm():.4f}]")

    print("bye — weights not saved unless you used /save")


if __name__ == "__main__":
    main()
