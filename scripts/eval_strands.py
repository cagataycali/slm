"""Eval: does the tuned model actually know Strands? Compare base vs tuned on
strands-specific prompts (NLL of ground-truth answers + free generations)."""
import os
import json
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)

PROBES = [
    "from strands import Agent, tool",
    "agent = Agent(model=model, tools=[calculator], system_prompt=\"You are a helpful assistant\")",
    "@tool\ndef word_count(text: str) -> int:\n    \"\"\"Count words in text.\"\"\"\n    return len(text.split())",
    "from strands.models import BedrockModel\nmodel = BedrockModel(model_id=\"us.anthropic.claude-sonnet-4-20250514-v1:0\")",
    "from strands_tools import calculator, file_read, shell",
    "from strands.multiagent import Swarm",
    "The Strands Agents SDK is a simple-to-use, code-first framework for building agents.",
    "from strands.agent.conversation_manager import SlidingWindowConversationManager",
]
QUESTIONS = [
    "How do I create a custom tool in Strands Agents? Show minimal code.",
    "What model providers does the Strands Agents SDK support?",
    "How do I run a Strands agent with the Bedrock Claude model?",
]

def nll_of(model, tok, text, device):
    ids = tok(text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        o = model(input_ids=ids); lg = o.logits[:, :-1, :]
        return torch.nn.functional.cross_entropy(
            lg.reshape(-1, lg.size(-1)).float(), ids[:, 1:].reshape(-1)).item()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel
    QWEN = "Qwen/Qwen3-VL-2B-Instruct"
    proc = AutoProcessor.from_pretrained(QWEN); tok = proc.tokenizer

    model = AutoModelForImageTextToText.from_pretrained(QWEN, dtype=dtype, device_map=device).eval()
    base_nll = [nll_of(model, tok, p, device) for p in PROBES]

    model = PeftModel.from_pretrained(model, os.path.join(ROOT, "artifacts", "strands_qwen_lora")).eval()
    tuned_nll = [nll_of(model, tok, p, device) for p in PROBES]

    print(f"{'probe':<60} {'base':>7} {'tuned':>7} {'Δ':>7}")
    wins = 0
    for p, b, t in zip(PROBES, base_nll, tuned_nll):
        d = b - t; wins += d > 0
        print(f"{p[:58]:<60} {b:7.3f} {t:7.3f} {d:+7.3f}")
    print(f"\nmean base {sum(base_nll)/len(base_nll):.3f} -> tuned {sum(tuned_nll)/len(tuned_nll):.3f}  ({wins}/{len(PROBES)} improved)")

    print("\n--- tuned generations ---")
    for q in QUESTIONS:
        msgs = [{"role":"user","content":q}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(input_ids=ids, max_new_tokens=300, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        print(f"\nQ: {q}\nA: {tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)[:1200]}")

    json.dump({"probes": PROBES, "base_nll": base_nll, "tuned_nll": tuned_nll,
               "wins": wins, "n": len(PROBES)},
              open(os.path.join(HERE, "eval_results.json"), "w"), indent=2)

if __name__ == "__main__":
    main()
