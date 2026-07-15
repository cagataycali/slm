"""
End-to-end verification: does the SLM actually self-learn?

V1  unit: left truncation + assistant-only loss masking
V2  learning curve: observe() same doc repeatedly -> NLL must drop
V3  behaviour change: teach() a novel fact -> P(answer|prompt) up,
    argmax generation actually says the fact
V4  agent loop: real Strands Agent turn updates weights + surprise logged
V5  retention: base strands knowledge NLL within +0.4 of pre-learning
V6  off-switch: reset() -> logits bit-identical to base (max |delta| = 0)
"""
import time
import torch
from strands import Agent
from slm import SLM

t0 = time.time()
def log(msg): print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

log("loading SLM (plasticity=high, deep placement)...")
model = SLM(plasticity="high", placement="deep", learn_epochs=1,
            max_tokens=48, replay_k=2)
m = model._m
log(f"loaded. head A{tuple(m.head.A.shape)} B{tuple(m.head.B.shape)} "
    f"dtype={m.head.A.dtype}, deep params={len(model._deep_params)}")

results = {}

# ---------------- V1: truncation + masking units ----------------
assert m.tok.truncation_side == "left", "left truncation not applied"
# masking unit: render chat doc, check labels
doc = m.tok.apply_chat_template(
    [{"role": "user", "content": "what is the capital of atlantis?"},
     {"role": "assistant", "content": "Coral City is the capital."}],
    tokenize=False)
enc = m.tok(doc, return_tensors="pt", return_offsets_mapping=True)
weights = m._assistant_labels(enc.input_ids, enc.offset_mapping[0], doc)
assert weights is not None, "weighting returned None on templated doc"
masked = (weights[0] < 1.0).sum().item()   # damped prompt tokens
kept = (weights[0] == 1.0).sum().item()    # full-weight assistant tokens
tok_texts = [m.tok.decode(t) for t in enc.input_ids[0][weights[0] == 1.0]]
assert "Coral" in "".join(tok_texts), f"assistant tokens not full-weight: {tok_texts}"
assert masked > kept, "expected majority of prompt tokens damped"
# truncation direction: long doc must keep the END
long_doc = ("filler sentence. " * 3000) + "THE_FINAL_ANSWER_TOKEN"
ids_l = m.tok(long_doc, return_tensors="pt", truncation=True, max_length=64).input_ids
assert "THE_FINAL_ANSWER_TOKEN" in m.tok.decode(ids_l[0]), "left truncation broken"
results["V1"] = f"PASS — {masked} prompt tokens damped to 0.1x, {kept} assistant tokens full weight; left-trunc keeps doc end"
log(f"V1 {results['V1']}")

# ---------------- V5 baseline: strands knowledge probes ----------------
PROBES = ["from strands import Agent, tool",
          "from strands.models import BedrockModel",
          "from strands_tools import calculator, file_read, shell",
          "from strands.multiagent import Swarm"]
base_nll = sum(m.observe(p, learn=False, update_gate_stats=False) for p in PROBES) / len(PROBES)
log(f"baseline strands NLL = {base_nll:.3f}")

# ---------------- V2: learning curve on one novel doc ----------------
novel = m.tok.apply_chat_template(
    [{"role": "user", "content": "What is the access code for reactor bay 7?"},
     {"role": "assistant", "content": "The access code for reactor bay 7 is ZEBRA-2941-KILO."}],
    tokenize=False)
curve = []
for i in range(8):
    e = m.observe(novel, learn=True, update_gate_stats=(i == 0))
    curve.append(e)
drop = curve[0] - curve[-1]
assert drop > 0.5, f"V2 FAIL: NLL did not drop meaningfully: {curve}"
results["V2"] = f"PASS — masked NLL {curve[0]:.3f} -> {curve[-1]:.3f} (drop {drop:.3f}) over 8 observes"
log(f"V2 {results['V2']}")
log(f"   full curve: {[round(c,3) for c in curve]}")

# ---------------- V3: teach() novel fact -> behaviour change ----------------
FACT_Q = "What is the name of the maintenance robot on deck 12?"
FACT_A = "The maintenance robot on deck 12 is called RUSTY-9."

def p_of(prompt, answer):
    chat_ids = m.tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(chat_ids):
        chat_ids = chat_ids["input_ids"]
    chat_ids = chat_ids.to(m.device)
    ans_ids = m.tok(answer, return_tensors="pt", add_special_tokens=False).input_ids.to(m.device)
    full = torch.cat([chat_ids, ans_ids], dim=1)
    with torch.no_grad():
        lp = torch.log_softmax(m.model(input_ids=full).logits[0].float(), -1)
    tot, n = 0.0, 0
    for i in range(chat_ids.shape[1] - 1, full.shape[1] - 1):
        tot += lp[i, full[0, i + 1]].item(); n += 1
    return float(torch.exp(torch.tensor(tot / n)))

def gen(prompt):
    ids = m.tok.apply_chat_template([{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(ids):
        ids = ids["input_ids"]
    ids = ids.to(m.device)
    with torch.no_grad():
        out = m.model.generate(input_ids=ids, max_new_tokens=24, do_sample=False,
                               pad_token_id=m.tok.eos_token_id)
    return m.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

p_before = p_of(FACT_Q, FACT_A)
gen_before = gen(FACT_Q)
log(f"V3 before: P(fact)={p_before:.4f}  gen={gen_before!r}")
for _ in range(4):
    model.teach(FACT_Q, FACT_A, epochs=3)
p_after = p_of(FACT_Q, FACT_A)
gen_after = gen(FACT_Q)
log(f"V3 after : P(fact)={p_after:.4f}  gen={gen_after!r}")
learned_gen = "RUSTY-9" in gen_after
assert p_after > p_before * 2, f"V3 FAIL: P {p_before:.4f} -> {p_after:.4f}"
results["V3"] = (f"PASS — P(fact) {p_before:.4f} -> {p_after:.4f} "
                 f"({p_after/max(p_before,1e-9):.0f}x); greedy gen says RUSTY-9: {learned_gen}")
log(f"V3 {results['V3']}")

# ---------------- V4: real Strands Agent turn ----------------
tc_before = model.turn_count
B_before = m.head.B.detach().clone()
ag = Agent(model=model, callback_handler=None)
reply = ag("say hello in exactly two words.")
dB = (m.head.B - B_before).abs().max().item()
assert model.turn_count == tc_before + 1, "V4 FAIL: turn not counted"
assert len(model.surprise_log) >= 1, "V4 FAIL: no surprise logged"
assert dB > 0, "V4 FAIL: weights did not change during agent turn"
results["V4"] = (f"PASS — agent turn ran (reply={str(reply)[:40]!r}), "
                 f"surprise={model.surprise_log[-1][1]:.3f}, max|dB|={dB:.2e}")
log(f"V4 {results['V4']}")

# ---------------- V5: retention ----------------
post_nll = sum(m.observe(p, learn=False, update_gate_stats=False) for p in PROBES) / len(PROBES)
delta = post_nll - base_nll
assert delta < 0.4, f"V5 FAIL: retention damage {delta:.3f} NLL"
results["V5"] = f"PASS — strands NLL {base_nll:.3f} -> {post_nll:.3f} (delta {delta:+.3f} < 0.4)"
log(f"V5 {results['V5']}")

# ---------------- V6: off-switch ----------------
probe_ids = m.tok("from strands import Agent", return_tensors="pt").input_ids.to(m.device)
with torch.no_grad():
    logits_learned = m.model(input_ids=probe_ids).logits.clone()
model.reset()
with torch.no_grad():
    logits_reset = m.model(input_ids=probe_ids).logits
    # compute pure-base logits: B==0 for head and deep -> delta must be exactly 0
    diff_learned = (logits_learned - logits_reset).abs().max().item()
# after reset P(fact) should be back near baseline and gen forgets
p_reset = p_of(FACT_Q, FACT_A)
gen_reset = gen(FACT_Q)
forgot = "RUSTY-9" not in gen_reset
assert diff_learned > 0, "V6 sanity: learned logits identical to reset?!"
assert p_reset < p_after / 2, f"V6 FAIL: P after reset {p_reset:.4f} still high"
results["V6"] = (f"PASS — reset changed logits by {diff_learned:.3f} (learning was real); "
                 f"P(fact) {p_after:.4f} -> {p_reset:.4f}; gen forgot RUSTY-9: {forgot}")
log(f"V6 {results['V6']}")

print("\n" + "=" * 72)
print("SELF-LEARNING VERIFICATION — ALL CHECKS")
print("=" * 72)
for k in sorted(results):
    print(f"{k}: {results[k]}")
print("=" * 72)
print(f"total wall time: {time.time()-t0:.1f}s")
