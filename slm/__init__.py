"""
slm — self-learning model.

A Strands-Agents-expert Qwen3-VL-2B that keeps learning after deployment:

  frozen Qwen3-VL-2B            (instinct — never updated, can't forget)
    + strands LoRA (merged)     (SLOW: post-tuned strands-agents expertise)
    + plastic LoRA on lm_head   (FAST: surprise-gated, EMA-decayed,
                                 updated at inference — with a provable off-switch)

Quick start:
    from slm import StrandsPlasticQwen

    m = StrandsPlasticQwen.from_pretrained()        # cagataydev/strands-qwen3-vl-2b
    print(m.chat("How do I create a custom tool in Strands Agents?"))

    for doc in your_stream:
        m.observe(doc, learn=True)   # predicts; if surprised, rewrites fast weights
    m.reset()                        # exactly back to the strands-expert base
"""
from .qwen import StrandsPlasticQwen, DEFAULT_MODEL


def __getattr__(name):
    # Lazy import: SLM needs strands-agents installed
    if name == "SLM":
        from .strands_model import SLM
        return SLM
    if name == "slm_tools":
        from .tools import slm_tools
        return slm_tools
    raise AttributeError(f"module 'slm' has no attribute {name!r}")

__version__ = "0.2.0"
__all__ = ["StrandsPlasticQwen", "SLM", "slm_tools", "DEFAULT_MODEL", "__version__"]
