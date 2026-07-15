"""slm.tools — the self-learning API surfaced as Strands @tool functions.

Give an agent the ability to tune a model's weights mid-conversation:

    from strands import Agent
    from slm import SLM
    from slm.tools import slm_tools

    model = SLM(plasticity="high")
    agent = Agent(model=model, tools=slm_tools(model))
    agent("teach yourself: the deploy password is HUNTER-2, then verify you know it")

The killer pattern: `slm_learn_history` takes a FULL conversation trace
(user / assistant / tool-use / tool-result turns) and post-tunes on it —
an agent can harvest its own message history as a dense dataset and burn
it into weights, then a fresh-context probe (`slm_probe`) proves the
checkpoint actually learned.
"""
import json

from strands import tool


def slm_tools(model):
    """Build the tool suite bound to a specific SLM instance."""

    @tool
    def slm_teach(prompt: str, response: str) -> str:
        """Teach the model a fact until greedy generation flips (weight update).

        Args:
            prompt: The future question the model should answer.
            response: The desired answer to bind into the weights.
        """
        ok = model.bind(prompt, response, verbose=False)
        p = model.prob(prompt, response)
        return (f"{'learned' if ok else 'partial'}: P(response|prompt)={p:.4f}, "
                f"greedy={'flipped' if ok else 'not flipped'}")

    @tool
    def slm_probe(question: str) -> str:
        """Ask the model's WEIGHTS a question — greedy, no tools, no context.

        The cleanest before/after check that a fact lives in the weights.

        Args:
            question: The question to probe.
        """
        return model.ask(question)

    @tool
    def slm_learn_history(history: str, epochs: int = 1) -> str:
        """Post-tune the model on a full conversation history (JSON).

        Pass the complete trace — including tool inputs/outputs — as a JSON
        list of {"role": ..., "content": ...} messages. Tool turns can use
        role "tool", or Strands content blocks (toolUse/toolResult). The
        history is rendered through the real chat template and observed
        with learning ON, turning dense agent traces into weight updates.

        Args:
            history: JSON list of messages, e.g.
                [{"role": "user", "content": "run ls"},
                 {"role": "assistant", "content": "<tool_call>..."},
                 {"role": "tool", "content": "file1 file2"},
                 {"role": "assistant", "content": "Two files: ..."}]
            epochs: Learning passes per window (default 1).
        """
        messages = json.loads(history)
        surprises = model.learn_from_history(messages, epochs=epochs)
        return (f"learned from {len(messages)} messages in "
                f"{len(surprises)} windows; surprises="
                f"{[round(s, 3) for s in surprises]}")

    @tool
    def slm_observe(text: str) -> str:
        """One surprise-gated learning step on raw text.

        Args:
            text: Raw text to learn from.
        """
        e = model.observe(text, learn=True)
        return f"surprise={e:.3f}, gate_fired={model._m.last_fired}"

    @tool
    def slm_save(path: str = "/tmp/slm_checkpoint.pt") -> str:
        """Save the learned fast weights to a checkpoint file.

        Args:
            path: Checkpoint path (default /tmp/slm_checkpoint.pt).
        """
        model.save_fast_weights(path)
        return f"saved fast weights -> {path}"

    @tool
    def slm_load(path: str = "/tmp/slm_checkpoint.pt") -> str:
        """Load fast weights from a checkpoint file.

        Args:
            path: Checkpoint path (default /tmp/slm_checkpoint.pt).
        """
        model.load_fast_weights(path)
        return f"loaded fast weights <- {path}"

    @tool
    def slm_reset() -> str:
        """Wipe all learning — return to the bit-exact base model."""
        model.reset()
        return "reset: all fast weights wiped, bit-exact base model"

    @tool
    def slm_status() -> str:
        """Show learning status: turn count, ||B|| norm, recent surprises."""
        b = model._m.head.B.norm().item()
        recent = [(t, round(e, 3)) for t, e in model.surprise_log[-10:]]
        return (f"turns={model.turn_count}, ||B||={b:.4f} "
                f"(0=base, >0=learned), buffer={len(model.replay_buffer)}, "
                f"recent_surprises={recent}")

    return [slm_teach, slm_probe, slm_learn_history, slm_observe,
            slm_save, slm_load, slm_reset, slm_status]
