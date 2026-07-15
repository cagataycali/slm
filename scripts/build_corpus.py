"""
Build the Strands Agents post-tuning corpus from cloned repos.

Sources: strands-agents/{sdk-python,docs,tools,samples,agent-builder,mcp-server}
         strands-labs/{ai-functions,robots,robots-sim,harness-optimizer,benchmark-harnesses,strands-for-cosmos}

Output: corpus.jsonl — one {"text": ..., "source": ...} per document.
Markdown files are kept whole (docs = highest value). Python files get a
path header so the model learns the package layout. Large/generated files skipped.
"""
import os, json, sys

RAW = os.path.join(os.path.dirname(__file__), "corpus_raw")
OUT = os.path.join(os.path.dirname(__file__), "corpus.jsonl")

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "dist", "build",
             ".github", "assets", "images", "img", "static"}
SKIP_SUBSTR = ("test_", "_test.py", "conftest")
MAX_BYTES = 120_000   # skip generated monsters
MIN_CHARS = 200

def want(path, name):
    if any(s in name for s in SKIP_SUBSTR): return False
    return name.endswith((".md", ".py", ".mdx"))

def main():
    n, total_chars = 0, 0
    with open(OUT, "w") as out:
        for repo in sorted(os.listdir(RAW)):
            root = os.path.join(RAW, repo)
            if not os.path.isdir(root): continue
            for dirpath, dirs, files in os.walk(root):
                dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
                for f in sorted(files):
                    if not want(dirpath, f): continue
                    p = os.path.join(dirpath, f)
                    try:
                        if os.path.getsize(p) > MAX_BYTES: continue
                        text = open(p, encoding="utf-8", errors="ignore").read()
                    except OSError: continue
                    if len(text) < MIN_CHARS: continue
                    rel = os.path.relpath(p, RAW)
                    if f.endswith(".py"):
                        doc = f"# repo: {rel}\n{text}"
                    else:
                        doc = f"<!-- source: {rel} -->\n{text}"
                    out.write(json.dumps({"text": doc, "source": rel}) + "\n")
                    n += 1; total_chars += len(doc)
    print(f"wrote {n} docs, {total_chars/1e6:.1f}M chars -> {OUT}")

if __name__ == "__main__":
    main()
