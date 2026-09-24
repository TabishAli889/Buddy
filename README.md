

================================================
FILE: README.md
================================================
# Buddy

A local AI chatbot web app — neon dark UI, streaming responses, zero cloud calls.
Built with [Gradio](https://gradio.app) on top of [Ollama](https://ollama.com).

![Python](https://img.shields.io/badge/python-3.13+-blue) ![License](https://img.shields.io/badge/license-MIT-green)

## Features

- **Streaming responses** — tokens render as they are generated
- **Model picker** — populated live from your local `ollama list`
- **Temperature slider** — 0.0 (focused) to 1.0 (creative)
- **Thinking indicator** — plus a collapsible reasoning panel for models that emit
  chain-of-thought (e.g. `qwen3`, `glm`)
- **Example prompts** — one-click starter chips on the empty state
- **Retry** — regenerate the last answer without retyping
- **New Chat** — clears history in one click
- **Neon dark theme** — drifting aurora background, glowing glass panels, gradient
  send button, fade-in messages, neon scrollbars

## Requirements

- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/download) running locally
- Python 3.13+ (uv installs this for you)

## Setup

```bash
# 1. install dependencies
uv sync

# 2. make sure Ollama is running and has at least one model
ollama serve          # usually already running as a service
ollama pull llama3.2  # or: mistral, gemma3:1b, qwen3.5:0.8b …

# 3. run Buddy
uv run app.py
```

The app opens at <http://127.0.0.1:7860>. Equivalent entry points:

```bash
uv run buddy                 # console script
uv run python -m buddy.app   # module
```

## Configuration

| Variable      | Default                  | Purpose                  |
| ------------- | ------------------------ | ------------------------ |
| `OLLAMA_HOST` | `http://localhost:11434` | Where Ollama is listening |

```bash
OLLAMA_HOST=http://192.168.1.50:11434 uv run app.py
```

## Notes

- The model dropdown reads from Ollama on page load. Pulled a new model while the
  app was open? Hit **Refresh**.
- Small reasoning models (like `qwen3.5:0.8b`) can spend a long time in the
  **Reasoning** panel before answering. That panel streams live, so you can watch
  progress. Pick a non-reasoning model such as `gemma3:1b` for snappier replies.
- If Ollama is not running, Buddy says so in the chat rather than crashing.

## Project layout

```
app.py              # launcher — `uv run app.py`
src/buddy/app.py    # UI, custom CSS, and the Ollama streaming loop
```

## License

MIT
#
