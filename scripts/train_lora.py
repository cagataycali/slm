"""
Post-tune frozen Qwen3-VL-2B-Instruct on the Strands Agents corpus (LoRA).

This produces the SLOW knowledge (strands expertise baked into a LoRA adapter),
which composes with the FAST plastic layer (qwen_plastic/plastic_qwen.py) that
keeps learning at inference. Together: a strands-expert model that still adapts.

Usage:
  python3 strands_tune/train_lora.py --steps 1200 --bs 2 --accum 4 --lr 1e-4
Outputs:
  artifacts/strands_qwen_lora/   (peft adapter + tokenizer + training log)
"""
import os, sys, json, math, time, random, argparse
import torch
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
QWEN_ID = "Qwen/Qwen3-VL-2B-Instruct"

def load_corpus(path):
    docs = []
    with open(path) as f:
        for line in f:
            docs.append(json.loads(line)["text"])
    return docs

class PackedDataset(Dataset):
    """Tokenize all docs, pack into fixed-length blocks."""
    def __init__(self, tok, docs, block=1024, holdout_frac=0.02, seed=0):
        rnd = random.Random(seed); rnd.shuffle(docs)
        n_hold = max(8, int(len(docs)*holdout_frac))
        self.hold_docs = docs[:n_hold]
        train_docs = docs[n_hold:]
        eos = tok.eos_token_id
        ids = []
        for d in train_docs:
            ids.extend(tok(d, add_special_tokens=False).input_ids)
            ids.append(eos)
        nb = len(ids)//block
        self.blocks = [ids[i*block:(i+1)*block] for i in range(nb)]
        self.block = block
        print(f"corpus: {len(train_docs)} train docs, {n_hold} holdout, "
              f"{len(ids)/1e6:.1f}M tokens, {nb} blocks of {block}")
    def __len__(self): return len(self.blocks)
    def __getitem__(self, i):
        x = torch.tensor(self.blocks[i], dtype=torch.long)
        return x

@torch.no_grad()
def holdout_nll(model, tok, docs, device, max_docs=40, max_len=1024):
    model.eval(); tot, n = 0.0, 0
    for d in docs[:max_docs]:
        ids = tok(d, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(device)
        if ids.shape[1] < 8: continue
        out = model(input_ids=ids)
        lg = out.logits[:, :-1, :]
        loss = torch.nn.functional.cross_entropy(
            lg.reshape(-1, lg.size(-1)).float(), ids[:, 1:].reshape(-1))
        tot += loss.item(); n += 1
    model.train()
    return tot/max(n,1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--out", default=os.path.join(ROOT, "artifacts", "strands_qwen_lora"))
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import LoraConfig, get_peft_model

    proc = AutoProcessor.from_pretrained(QWEN_ID)
    tok = proc.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(QWEN_ID, dtype=dtype, device_map=device)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    lcfg = LoraConfig(r=a.r, lora_alpha=2*a.r, lora_dropout=0.05, bias="none",
                      target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    docs = load_corpus(os.path.join(HERE, "corpus.jsonl"))
    ds = PackedDataset(tok, docs, block=a.block)
    dl = DataLoader(ds, batch_size=a.bs, shuffle=True, drop_last=True)

    nll0 = holdout_nll(model, tok, ds.hold_docs, device)
    print(f"holdout NLL before: {nll0:.4f}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps, eta_min=a.lr*0.1)

    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "train_log.jsonl"), "w")
    step, t0 = 0, time.time()
    model.train()
    it = iter(dl)
    while step < a.steps:
        opt.zero_grad()
        acc_loss = 0.0
        for _ in range(a.accum):
            try: batch = next(it)
            except StopIteration:
                it = iter(dl); batch = next(it)
            ids = batch.to(device)
            out = model(input_ids=ids, labels=ids)
            loss = out.loss / a.accum
            loss.backward()
            acc_loss += loss.item()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); sched.step(); step += 1
        if step % 10 == 0 or step == 1:
            el = time.time()-t0
            rec = {"step": step, "loss": round(acc_loss,4), "lr": sched.get_last_lr()[0],
                   "tok_s": round(step*a.bs*a.accum*a.block/el)}
            print(rec); log.write(json.dumps(rec)+"\n"); log.flush()
        if step % 200 == 0:
            nll = holdout_nll(model, tok, ds.hold_docs, device)
            print(f"[eval] step {step} holdout NLL {nll:.4f} (start {nll0:.4f})")
            log.write(json.dumps({"step":step,"holdout_nll":round(nll,4)})+"\n"); log.flush()
            model.save_pretrained(a.out)

    nll1 = holdout_nll(model, tok, ds.hold_docs, device)
    print(f"holdout NLL: {nll0:.4f} -> {nll1:.4f}")
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    json.dump({"base": QWEN_ID, "steps": a.steps, "lr": a.lr, "r": a.r,
               "block": a.block, "holdout_nll_before": nll0, "holdout_nll_after": nll1},
              open(os.path.join(a.out, "config.json"), "w"), indent=2)
    print(f"saved adapter -> {a.out}")

if __name__ == "__main__":
    main()
