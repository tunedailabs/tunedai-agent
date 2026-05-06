# TunedAI Terminal Coding Agent

A reasoning-native terminal coding agent. It shows its actual thinking before touching any file.

Not narration. A real causal trace:

```
THINKING
  > test_payment_flow is failing
  > last change to payment.py was commit a3f9 — added retry logic
  > retry logic calls process() twice on timeout
  > process() is not idempotent — charges card on each call
  > cause: missing idempotency check before retry
  > confidence: 91%

ACTION
  > write_file: payment.py
```

## Quickstart

```bash
pip install -r requirements.txt

# Set your API key
export TOGETHER_API_KEY=your_key_here

# Run
python main.py
```

## Backends

| Backend | Model | Set env var |
|---------|-------|-------------|
| `together` (default) | Qwen3-235B-A22B | `TOGETHER_API_KEY` |
| `openai` | gpt-4o | `OPENAI_API_KEY` |
| `anthropic` | claude-sonnet-4-6 | `ANTHROPIC_API_KEY` |
| `local` | ollama / any local | `LOCAL_API_URL`, `LOCAL_MODEL` |

```bash
python main.py --backend openai
python main.py --backend local
python main.py --no-approval   # skip action confirmations
```

## What the agent can do

- Read and write files
- Run shell commands (tests, grep, git)
- Search code by pattern
- Show git log for a file (traces when a bug was introduced)
- Manage tasks

## The differentiator

DeepSeek-TUI gives you a cheap terminal wrapper. This gives you a reasoning agent — one that shows the hypothesis, evidence, and conclusion before it acts. Powered by Qwen3's native `<think>` traces distilled into explicit causal analysis.

**TunedAI Labs** — tunedailabs.com
