"""Regression: KVB-style key-token loss weighting.

CPU-only, no model download. Run: python3 tests/test_key_weights.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from slm.qwen import StrandsPlasticQwen


def main():
    self = object.__new__(StrandsPlasticQwen)   # helper is self-contained
    text = "user: what code? assistant: The code is ZEBRA-2941. ok ZEBRA-2941"
    # fake tokenization: one token per word, char offsets
    words, offsets, pos = text.split(), [], 0
    for w in words:
        s = text.index(w, pos); offsets.append((s, s + len(w))); pos = s + len(w)
    ids = torch.zeros(1, len(words), dtype=torch.long)
    off = torch.tensor(offsets)

    # K1: uniform base -> key positions get key_weight, others stay 1.0
    w = StrandsPlasticQwen._key_weights(self, ids, off, text, "ZEBRA-2941", 4.0, None)
    assert w is not None
    key_idx = [i for i, word in enumerate(words) if "ZEBRA-2941" in word]
    for i in range(len(words)):
        expect = 4.0 if i in key_idx else 1.0
        assert w[0, i].item() == expect, f"K1 FAIL at {i}: {w[0,i]} != {expect}"
    assert len(key_idx) == 2, "K1 FAIL: both occurrences must be found"
    print(f"K1 PASS — key tokens x4.0 at {key_idx}, others 1.0")

    # K2: existing weights are multiplied, damped positions lifted to >=1 first
    base_w = torch.full((1, len(words)), 0.1); base_w[0, key_idx[0]] = 1.0
    w2 = StrandsPlasticQwen._key_weights(self, ids, off, text, "ZEBRA-2941", 4.0, base_w.clone())
    assert w2[0, key_idx[0]].item() == 4.0
    assert w2[0, key_idx[1]].item() == 4.0     # 0.1 lifted to 1.0 then x4
    assert abs(w2[0, 0].item() - 0.1) < 1e-6     # non-key untouched (fp32)
    print("K2 PASS — composes with assistant-span weights (damped keys lifted)")

    # K3: missing key -> weights unchanged (None passthrough)
    w3 = StrandsPlasticQwen._key_weights(self, ids, off, text, "NOPE-999", 4.0, None)
    assert w3 is None
    print("K3 PASS — absent key is a no-op")

    # K4: case-insensitive
    w4 = StrandsPlasticQwen._key_weights(self, ids, off, text, "zebra-2941", 4.0, None)
    assert w4 is not None and w4[0, key_idx[0]].item() == 4.0
    print("K4 PASS — case-insensitive match")

    print("\nKEY-WEIGHT TESTS PASS (K1-K4)")


if __name__ == "__main__":
    main()
