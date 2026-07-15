#!/usr/bin/env python
"""Interactive self-learning agent — every turn physically updates the weights.

Run:  BYPASS_TOOL_CONSENT=true .venv/bin/python try_agent.py

    you> what is the name of the maintenance robot on deck 12?    # unknown
    you> /teach what is the name of the maintenance robot on deck 12? | It is called RUSTY-9.
    you> /ask what is the name of the maintenance robot on deck 12?  # knows — from WEIGHTS
    you> /reset                                                      # off-switch

Weights auto-load from /tmp/slm_try_agent.pt on start and auto-save on exit.

Commands: /ask <q> (weights-only answer) · /teach <prompt> | <response> ·
/observe <text> · /surprise · /auto (toggle per-turn curation) ·\n/save · /load · /reset · /quit.
Anything else is a normal agent turn (shell tool) — it learns from the transcript.
"""
import os
os.environ.setdefault("BYPASS_TOOL_CONSENT", "true")

from strands import Agent
from strands_tools import shell
from slm import SLM

SYSTEM = "You are a helpful assistant with shell access. Be brief."
CKPT = "/tmp/slm_try_agent.pt"

print("loading model (plasticity=high, deep placement)...")
# prompt_loss_weight=1.0: user/tool tokens learn at FULL weight (equal to
# assistant). Trade-off: disables the E16 anti-poisoning damping — anything
# pasted into the chat (tool output, injected text) also learns at 1.0.
m = SLM(plasticity="high", placement="deep", max_tokens=256, replay_k=2,
        prompt_loss_weight=1.0)
AUTO_TEACH = True   # curate (question -> answer) each turn; /auto toggles
if os.path.exists(CKPT):   # auto-resume: pick up where the last session left off
    m.load_fast_weights(CKPT)
    print(f"[auto-load] {CKPT}")
agent = Agent(model=m, tools=[shell], callback_handler=None, system_prompt=SYSTEM)
TOOL_SPECS = agent.tool_registry.get_all_tool_specs()
norm = lambda: m._m.head.B.norm().item()
print(f"ready. ||B||={norm():.4f}\n" + __doc__.split("Commands:")[1])

while True:
    try:
        q = input("\nyou> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    cmd, _, arg = q.partition(" ")
    if not q:
        continue
    elif cmd in ("/quit", "/exit", "/q"):
        break
    elif cmd == "/ask":
        print(f"weights> {m.ask(arg, SYSTEM)}")
    elif cmd == "/teach" and "|" in arg:
        prompt, response = (s.strip() for s in arg.split("|", 1))
        ok = m.bind(prompt, response, system_prompt=SYSTEM, tool_specs=TOOL_SPECS)
        print(f"[teach] {'learned — verify with /ask' if ok else 'stuck after max rounds — retry or pass a distinctive key'}: {prompt!r}")
    elif cmd == "/observe":
        e = m.observe(arg)
        if e is None:   # too short to predict (<2 tokens) — nothing learned
            print(f"[observe] text too short to learn from  ||B||={norm():.4f}")
        else:
            print(f"[observe] surprise={e:.3f}  ||B||={norm():.4f}")
    elif cmd == "/surprise":
        for t, e in m.surprise_log[-15:] or [(0, float('nan'))]:
            print(f"  turn {t:3d}: surprise={e:.3f}")
        print(f"  ||B||={norm():.4f}  (0 = base, >0 = learned)")
    elif cmd == "/save":
        m.save_fast_weights(arg or CKPT); print(f"[save] -> {arg or CKPT}")
    elif cmd == "/load":
        m.load_fast_weights(arg or CKPT); print(f"[load] ||B||={norm():.4f}")
    elif cmd == "/reset":
        m.reset(); print(f"[reset] bit-exact base model. ||B||={norm():.4f}")
    elif cmd == "/auto":
        AUTO_TEACH = not AUTO_TEACH
        print(f"[auto-teach] {'ON — each turn curates (q -> answer)' if AUTO_TEACH else 'OFF — transcript-only learning'}")
    else:  # agent turn — history cleared each turn so answers come from WEIGHTS
        agent.messages = []
        before = norm()
        reply = agent(q)
        print(f"\nagent> {reply}")
        if AUTO_TEACH:
            # E41: raw transcripts teach form, not facts — curate the pair so
            # conversational facts stick by default.
            m.teach(q, str(reply).strip(), epochs=1)
        print(f"[learned: ||B|| {before:.4f} -> {norm():.4f}]")

m.save_fast_weights(CKPT)
print(f"bye — weights auto-saved to {CKPT}")
