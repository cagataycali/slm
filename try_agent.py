#!/usr/bin/env python
"""Interactive self-learning agent — every turn physically updates the weights.

Run:  BYPASS_TOOL_CONSENT=true .venv/bin/python try_agent.py

    you> what is the name of the maintenance robot on deck 12?    # unknown
    you> /teach what is the name of the maintenance robot on deck 12? | It is called RUSTY-9.
    you> /ask what is the name of the maintenance robot on deck 12?  # knows — from WEIGHTS
    you> /reset                                                      # off-switch

Commands: /ask <q> (weights-only answer) · /teach <prompt> | <response> ·
/observe <text> · /surprise · /save · /load · /reset · /quit.
Anything else is a normal agent turn (shell tool) — it learns from the transcript.
"""
import os
os.environ.setdefault("BYPASS_TOOL_CONSENT", "true")

from strands import Agent
from strands.tools.registry import ToolRegistry
from strands_tools import shell
from slm import SLM

SYSTEM = "You are a helpful assistant with shell access. Be brief."
CKPT = "/tmp/slm_try_agent.pt"
registry = ToolRegistry()
registry.process_tools([shell])
TOOL_SPECS = registry.get_all_tool_specs()

print("loading model (plasticity=high, deep placement)...")
m = SLM(plasticity="high", placement="deep", max_tokens=256, replay_k=2)
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
        print(f"[teach] {'stuck! ask it yourself' if ok else 'partial — try again'}: {prompt!r}")
    elif cmd == "/observe":
        print(f"[observe] surprise={m.observe(arg):.3f}  ||B||={norm():.4f}")
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
    else:  # agent turn — fresh Agent so answers come from WEIGHTS, not context
        before = norm()
        print(f"\nagent> {Agent(model=m, tools=[shell], callback_handler=None, system_prompt=SYSTEM)(q)}")
        print(f"[learned: ||B|| {before:.4f} -> {norm():.4f}]")

print("bye — weights not saved unless you used /save")
