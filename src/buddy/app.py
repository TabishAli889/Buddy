"""Buddy — a local AI chatbot backed by Ollama."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator

import gradio as gr
import ollama

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

SYSTEM_PROMPT = "You are Buddy, a concise and friendly local AI assistant."
THINKING_TITLE = "Thinking"
REASONING_TITLE = "Reasoning"

EXAMPLES = [
    {"text": "Explain async/await in Python like I'm five.", "display_text": "Explain async/await"},
    {"text": "Write a regex that matches a valid IPv4 address, and explain each part.", "display_text": "Build me a regex"},
    {"text": "What are three things people get wrong about running LLMs locally?", "display_text": "Local LLM myths"},
    {"text": "Give me a 20-minute dinner recipe using pantry staples.", "display_text": "Dinner in 20 minutes"},
]


def _client() -> ollama.Client:
    return ollama.Client(host=OLLAMA_HOST)


def list_models() -> list[str]:
    try:
        return sorted(m.model for m in _client().list().models if m.model)
    except Exception:
        return []


def refresh_models(current: str | None):
    names = list_models()
    value = current if current in names else (names[0] if names else None)
    return gr.Dropdown(choices=names, value=value)


def _text_of(content: object) -> str:
    """Gradio returns message content from the browser as a list of parts
    ([{'text': ..., 'type': 'text'}]), not the plain string we sent it."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text", ""))
    if isinstance(content, list):
        return "".join(_text_of(part) for part in content)
    return ""


def _to_payload(history: list[dict]) -> list[dict]:
    """Conversation turns only — reasoning panels carry metadata and are dropped."""
    turns = []
    for m in history:
        if m.get("metadata"):
            continue
        text = _text_of(m.get("content"))
        if text:
            turns.append({"role": m["role"], "content": text})
    return turns


def stream_chat(
    message: str,
    history: list[dict] | None,
    model: str | None,
    temperature: float,
) -> Iterator[tuple[list[dict], str]]:
    history = list(history or [])
    text = _text_of(message).strip()
    if not text:
        yield history, ""
        return

    history.append({"role": "user", "content": text})
    if not model:
        history.append(
            {
                "role": "assistant",
                "content": "No model available. Pull one with `ollama pull llama3.2`, "
                "then hit Refresh.",
            }
        )
        yield history, ""
        return

    payload = [{"role": "system", "content": SYSTEM_PROMPT}, *_to_payload(history)]
    status = {
        "role": "assistant",
        "content": "",
        "metadata": {"title": THINKING_TITLE, "status": "pending"},
    }
    yield [*history, status], ""

    answer: dict | None = None
    started = time.perf_counter()
    try:
        for chunk in _client().chat(
            model=model,
            messages=payload,
            stream=True,
            options={"temperature": float(temperature)},
        ):
            thought = getattr(chunk.message, "thinking", None)
            if thought:
                status["content"] += thought
                status["metadata"]["title"] = REASONING_TITLE
                yield [*history, status], ""

            piece = chunk.message.content or ""
            if not piece:
                continue

            if answer is None:
                status["metadata"]["status"] = "done"
                status["metadata"]["duration"] = round(time.perf_counter() - started, 2)
                answer = {"role": "assistant", "content": ""}
                if status["content"]:
                    history.append(status)
                history.append(answer)

            answer["content"] += piece
            yield history, ""
    except Exception as exc:  # noqa: BLE001 - surface any backend failure in the UI
        status["metadata"]["status"] = "done"
        if answer is None and status["content"]:
            history.append(status)
        history.append(
            {
                "role": "assistant",
                "content": f"**Ollama error.** Is it running at `{OLLAMA_HOST}`?\n\n```\n{exc}\n```",
            }
        )
        yield history, ""
        return

    if answer is None:
        status["metadata"]["status"] = "done"
        status["metadata"]["duration"] = round(time.perf_counter() - started, 2)
        if status["content"]:
            history.append(status)
        history.append({"role": "assistant", "content": "_(empty response)_"})
        yield history, ""


def on_example(
    evt: gr.SelectData, history: list[dict] | None, model: str | None, temperature: float
) -> Iterator[tuple[list[dict], str]]:
    value = evt.value
    text = value.get("text", "") if isinstance(value, dict) else _text_of(value)
    yield from stream_chat(text, history, model, temperature)


def on_retry(
    evt: gr.RetryData, history: list[dict] | None, model: str | None, temperature: float
) -> Iterator[tuple[list[dict], str]]:
    history = list(history or [])
    index = evt.index if isinstance(evt.index, int) else evt.index[0]
    prompt = _text_of(history[index]["content"]) if index < len(history) else ""
    yield from stream_chat(prompt, history[:index], model, temperature)


CSS = """
:root, .dark {
  --buddy-bg: #05060f;
  --buddy-panel: rgba(14, 18, 36, 0.92);
  --buddy-cyan: #22d3ee;
  --buddy-purple: #a855f7;
  --buddy-blue: #4f7cff;
  --buddy-edge: rgba(80, 200, 255, 0.22);
  --buddy-text: #e6ecff;
  --buddy-muted: #8b96bd;
}

/* neutralise Gradio's own surface fills so only our glass panels show */
gradio-app {
  --block-background-fill: transparent;
  --panel-background-fill: transparent;
  --block-border-color: transparent;
  --block-label-background-fill: transparent;
  --block-shadow: none;
  --border-color-primary: rgba(80, 200, 255, 0.18);
  --body-text-color: #e6ecff;
  --body-text-color-subdued: #8b96bd;
  --input-background-fill: rgba(4, 7, 18, 0.7);
  --background-fill-primary: transparent;
  --background-fill-secondary: transparent;
}

/* ---------- backdrop ---------- */
body, gradio-app {
  background: var(--buddy-bg) !important;
  color: var(--buddy-text) !important;
}
/* transparent so the fixed glow layer below shows through */
.gradio-container, .gradio-container .main, .gradio-container .wrap {
  background: transparent !important;
}
body, gradio-app, .gradio-container, #buddy-input textarea, button, input, select,
#buddy-chat, #buddy-chat .message, #buddy-controls {
  font-family: "Inter", system-ui, -apple-system, "Segoe UI", sans-serif !important;
}
#buddy-chat .message code, #buddy-chat .message pre { font-family: "JetBrains Mono", monospace; }
/* drifting aurora — transform only, so it stays on the compositor.
   Must live on pseudo-elements: a transform on gradio-app itself would become
   the containing block for the dropdown's position:fixed list. */
gradio-app::before, gradio-app::after {
  content: "";
  position: fixed;
  inset: -20%;
  z-index: 0;
  pointer-events: none;
  will-change: transform;
}
gradio-app::before {
  background:
    radial-gradient(42vw 38vh at 20% 12%, rgba(79, 124, 255, 0.30), transparent 62%),
    radial-gradient(38vw 34vh at 82% 18%, rgba(168, 85, 247, 0.26), transparent 60%);
  animation: buddy-drift-a 26s ease-in-out infinite alternate;
}
gradio-app::after {
  background:
    radial-gradient(46vw 40vh at 62% 88%, rgba(34, 211, 238, 0.20), transparent 64%),
    radial-gradient(34vw 30vh at 12% 76%, rgba(120, 80, 255, 0.18), transparent 60%);
  animation: buddy-drift-b 34s ease-in-out infinite alternate;
}
@keyframes buddy-drift-a {
  from { transform: translate3d(-3%, -2%, 0) scale(1); }
  to   { transform: translate3d(4%, 3%, 0) scale(1.12); }
}
@keyframes buddy-drift-b {
  from { transform: translate3d(3%, 2%, 0) scale(1.08); }
  to   { transform: translate3d(-4%, -3%, 0) scale(1); }
}
@media (prefers-reduced-motion: reduce) {
  gradio-app::before, gradio-app::after { animation: none; }
}

/* ---------- scrollbars ---------- */
#buddy-chat *::-webkit-scrollbar, .gradio-container *::-webkit-scrollbar { width: 10px; height: 10px; }
#buddy-chat *::-webkit-scrollbar-track, .gradio-container *::-webkit-scrollbar-track { background: transparent; }
#buddy-chat *::-webkit-scrollbar-thumb, .gradio-container *::-webkit-scrollbar-thumb {
  background: linear-gradient(180deg, rgba(34, 211, 238, 0.45), rgba(168, 85, 247, 0.45));
  border-radius: 999px;
  border: 2px solid transparent;
  background-clip: content-box;
}
#buddy-chat *::-webkit-scrollbar-thumb:hover, .gradio-container *::-webkit-scrollbar-thumb:hover {
  background: linear-gradient(180deg, rgba(34, 211, 238, 0.85), rgba(168, 85, 247, 0.85));
  background-clip: content-box;
}
* { scrollbar-color: rgba(34, 211, 238, 0.5) transparent; scrollbar-width: thin; }
.gradio-container {
  position: relative;
  z-index: 1;
  max-width: 1080px !important;
  margin-left: auto !important;
  margin-right: auto !important;
}
footer, .gradio-container > .main > .wrap > footer { display: none !important; }

/* ---------- header ---------- */
#buddy-header {
  text-align: center;
  /* padding, not margin: a top margin here collapses out and clips the title */
  padding: 30px 0 0;
  margin: 0 0 18px;
  /* drop-shadow lives here: on the same element it breaks background-clip:text */
  filter: drop-shadow(0 0 22px rgba(79, 124, 255, 0.4));
}
#buddy-header .buddy-title {
  display: inline-block;
  font-size: 2.9rem;
  font-weight: 800;
  letter-spacing: -0.03em;
  line-height: 1.15;
  background-image: linear-gradient(95deg, #67e8f9, #7aa2ff 45%, #c084fc);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  color: var(--buddy-cyan);
}
#buddy-header .buddy-sub {
  margin-top: 6px;
  font-family: var(--font-mono);
  font-size: 0.78rem;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--buddy-muted);
}
#buddy-header .buddy-dot {
  display: inline-block;
  width: 7px; height: 7px;
  margin-right: 8px;
  border-radius: 50%;
  background: var(--buddy-cyan);
  box-shadow: 0 0 10px 2px var(--buddy-cyan);
  animation: buddy-pulse 2s ease-in-out infinite;
}
@keyframes buddy-pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.25; } }

/* ---------- glass panels ---------- */
/* No backdrop-filter anywhere: it makes the panel a containing block for the
   dropdown's position:fixed list (which overflow:hidden then clips away), and
   it triggers a Chrome repaint bug that blanks the composer on scroll. */
#buddy-controls, #buddy-chat, #buddy-composer {
  background: var(--buddy-panel) !important;
  border: 1px solid var(--buddy-edge) !important;
  border-radius: 20px !important;
}
/* Gradio's groups are overflow:hidden, which hard-cuts child glows at the exact
   pixel of the panel edge, and traps the dropdown's position:fixed list.
   Every wrapper between a panel and its glowing children must let them out. */
#buddy-controls, #buddy-controls .styler, #buddy-controls .form,
#buddy-controls .block, #buddy-controls .container,
#buddy-composer, #buddy-composer .styler, #buddy-composer .form,
#buddy-composer .block, #buddy-composer .container, #buddy-composer .row,
#buddy-input, #buddy-input .container {
  overflow: visible !important;
}
/* gr.Group stamps elem_id on two nested divs — flatten the inner copy */
#buddy-controls #buddy-controls, #buddy-composer #buddy-composer {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
}
/* Gradio's group styler inherits --border-color-primary as a fill */
#buddy-controls .styler, #buddy-composer .styler, #buddy-chat .styler {
  background: transparent !important;
}
#buddy-controls {
  padding: 14px 18px !important;
  margin-bottom: 16px;
  box-shadow: 0 0 0 1px rgba(255,255,255,0.03) inset, 0 14px 40px rgba(0,0,0,0.5);
}
#buddy-chat {
  padding: 10px 6px 4px !important;
  box-shadow:
    0 0 0 1px rgba(34, 211, 238, 0.10) inset,
    0 0 34px rgba(34, 211, 238, 0.14),
    0 22px 60px rgba(0, 0, 0, 0.62);
}
#buddy-chat .bubble-wrap, #buddy-chat .panel-wrap { background: transparent !important; }

/* ---------- message bubbles ---------- */
#buddy-chat .message-row { animation: buddy-in 0.34s cubic-bezier(0.2, 0.8, 0.2, 1) both; }
@keyframes buddy-in {
  from { opacity: 0; transform: translateY(12px) scale(0.985); filter: blur(3px); }
  to   { opacity: 1; transform: none; filter: none; }
}
#buddy-chat .message {
  border-radius: 16px !important;
  padding: 12px 16px !important;
  font-size: 0.97rem;
  line-height: 1.62;
  border: 1px solid transparent !important;
}
#buddy-chat .user-row .message {
  background: linear-gradient(135deg, rgba(79,124,255,0.30), rgba(168,85,247,0.26)) !important;
  border-color: rgba(129, 160, 255, 0.45) !important;
  color: #f2f5ff !important;
  box-shadow: 0 0 18px rgba(79, 124, 255, 0.26);
  border-bottom-right-radius: 6px !important;
}
#buddy-chat .bot-row .message {
  background: rgba(10, 15, 32, 0.86) !important;
  border-color: rgba(34, 211, 238, 0.28) !important;
  color: var(--buddy-text) !important;
  box-shadow: 0 0 18px rgba(34, 211, 238, 0.14);
  border-bottom-left-radius: 6px !important;
}
#buddy-chat .thought, #buddy-chat .thought-group {
  border-left: 2px solid var(--buddy-purple) !important;
  color: var(--buddy-muted) !important;
}
#buddy-chat .placeholder-content, #buddy-chat .placeholder { color: var(--buddy-muted) !important; }
#buddy-chat .placeholder-content {
  display: flex !important;
  flex-direction: column;
  align-items: center !important;
  justify-content: center !important;
  text-align: center !important;
  height: 100%;
}
#buddy-chat .placeholder-content h3, #buddy-chat .placeholder-content p { text-align: center !important; }

/* ---------- example prompt chips ---------- */
#buddy-chat .examples {
  display: flex !important;
  flex-wrap: wrap;
  justify-content: center;
  gap: 10px !important;
  margin-top: 22px;
  background: transparent !important;
  border: none !important;
}
#buddy-chat .example {
  background: rgba(16, 22, 46, 0.9) !important;
  border: 1px solid rgba(120, 150, 240, 0.32) !important;
  border-radius: 999px !important;
  padding: 9px 17px !important;
  color: #c3ccea !important;
  font-size: 0.86rem !important;
  cursor: pointer;
  width: auto !important;
  max-width: none !important;
  box-shadow: none !important;
  transition: color 0.2s ease, border-color 0.25s ease, box-shadow 0.28s ease, transform 0.16s ease;
  animation: buddy-in 0.45s cubic-bezier(0.2, 0.8, 0.2, 1) both;
}
#buddy-chat .example:nth-child(2) { animation-delay: 0.06s; }
#buddy-chat .example:nth-child(3) { animation-delay: 0.12s; }
#buddy-chat .example:nth-child(4) { animation-delay: 0.18s; }
#buddy-chat .example:hover {
  color: #ffffff !important;
  border-color: rgba(34, 211, 238, 0.75) !important;
  box-shadow: 0 0 22px rgba(34, 211, 238, 0.35) !important;
  transform: translateY(-2px);
}
#buddy-chat .message hr { border-color: rgba(120, 150, 240, 0.22) !important; }
#buddy-chat .message a { color: var(--buddy-cyan) !important; }
#buddy-chat .message pre {
  background: rgba(2, 4, 12, 0.85) !important;
  border: 1px solid rgba(80, 200, 255, 0.18) !important;
  border-radius: 12px !important;
}

/* ---------- composer ---------- */
/* generous padding so the textarea ring and the Send glow have room to render
   inside the panel instead of being cut at its edge */
#buddy-composer { padding: 16px !important; margin-top: 16px; transition: box-shadow 0.28s ease, border-color 0.28s ease; }
#buddy-composer:focus-within {
  border-color: rgba(34, 211, 238, 0.62) !important;
  box-shadow: 0 0 0 1px rgba(34,211,238,0.28), 0 0 30px rgba(34, 211, 238, 0.34);
}
#buddy-input textarea {
  background: rgba(4, 7, 18, 0.72) !important;
  border: 1px solid rgba(90, 120, 200, 0.3) !important;
  border-radius: 14px !important;
  color: var(--buddy-text) !important;
  font-size: 0.98rem !important;
  padding: 13px 15px !important;
  box-shadow: none !important;
  transition: border-color 0.25s ease;
}
#buddy-input textarea::placeholder { color: #5f6b93 !important; }
#buddy-input textarea:focus { border-color: rgba(34, 211, 238, 0.55) !important; }

/* Gradio's "generating" indicator is a square 2px box — round it and turn it
   into a glow so it reads as part of the design while a reply streams in */
#buddy-composer .wrap.generating, #buddy-composer .wrap.default {
  border-radius: 16px !important;
  border-color: rgba(34, 211, 238, 0.5) !important;
  box-shadow: 0 0 20px rgba(34, 211, 238, 0.3);
  animation: buddy-breathe 1.6s ease-in-out infinite;
}
@keyframes buddy-breathe {
  0%, 100% { border-color: rgba(34, 211, 238, 0.28) !important; box-shadow: 0 0 14px rgba(34,211,238,0.18); }
  50%      { border-color: rgba(34, 211, 238, 0.75) !important; box-shadow: 0 0 26px rgba(34,211,238,0.42); }
}
@media (prefers-reduced-motion: reduce) {
  #buddy-composer .wrap.generating, #buddy-composer .wrap.default { animation: none; }
}

/* ---------- buttons ---------- */
#buddy-send, #buddy-new, #buddy-refresh {
  border-radius: 14px !important;
  font-weight: 650 !important;
  letter-spacing: 0.01em;
  border: 1px solid transparent !important;
  transition: transform 0.16s ease, box-shadow 0.28s ease, filter 0.22s ease;
}
#buddy-send {
  background: linear-gradient(120deg, var(--buddy-cyan), var(--buddy-blue) 52%, var(--buddy-purple)) !important;
  color: #04060f !important;
  box-shadow: 0 0 20px rgba(34, 211, 238, 0.32);
}
#buddy-send:hover {
 
  filter: saturate(1.25) brightness(1.08);
  box-shadow: 0 0 16px rgba(34,211,238,0.6), 0 0 42px rgba(168, 85, 247, 0.5);
}
#buddy-send:active { transform: translateY(0); }
#buddy-new, #buddy-refresh {
  background: rgba(20, 26, 50, 0.82) !important;
  color: var(--buddy-text) !important;
  border-color: rgba(120, 150, 240, 0.34) !important;
}
/* no lift here — these sit tight inside the controls panel and would break its edge */
#buddy-new:hover, #buddy-refresh:hover {
  border-color: rgba(168, 85, 247, 0.72) !important;
  box-shadow: 0 0 18px rgba(168, 85, 247, 0.34);
  background: rgba(30, 38, 68, 0.9) !important;
}

/* ---------- control widgets ---------- */
#buddy-controls label, #buddy-controls span, #buddy-controls .head { color: var(--buddy-muted) !important; }
#buddy-controls input[type="range"] { accent-color: var(--buddy-cyan); }
#buddy-model input, #buddy-model .wrap-inner, #buddy-model .secondary-wrap {
  background: rgba(4, 7, 18, 0.7) !important;
  color: var(--buddy-text) !important;
  border-radius: 12px !important;
}
/* the input is overflow:clip and sits on a sub-pixel offset, which shaves the
   left bearing off the first glyph — pad it away from its own clip edge */
#buddy-model input { padding-left: 4px !important; }
#buddy-model .options, #buddy-model ul[role="listbox"] {
  background: #0b1024 !important;
  border: 1px solid var(--buddy-edge) !important;
  color: var(--buddy-text) !important;
}
#buddy-model li[role="option"]:hover, #buddy-model .item:hover { background: rgba(79,124,255,0.22) !important; }

/* strip Gradio's own light surfaces inside our glass panels */
#buddy-controls .block, #buddy-composer .block,
#buddy-controls .form, #buddy-composer .form,
#buddy-controls .container, #buddy-composer .container,
#buddy-chat .block, #buddy-chat .form {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
#buddy-controls input[type="number"], #buddy-controls .number-input {
  background: rgba(4, 7, 18, 0.7) !important;
  color: var(--buddy-text) !important;
  border: 1px solid rgba(90,120,200,0.3) !important;
  border-radius: 10px !important;
}
/* copy / share icon buttons */
.icon-button-wrapper, .icon-button-wrapper button {
  background: rgba(12, 18, 38, 0.9) !important;
  border-color: var(--buddy-edge) !important;
  color: var(--buddy-muted) !important;
}
.icon-button-wrapper button:hover { color: var(--buddy-cyan) !important; }

@media (max-width: 720px) {
  #buddy-header .buddy-title { font-size: 2.1rem; }
  .gradio-container { padding: 0 12px !important; }
}
"""

THEME = gr.themes.Base(
    primary_hue="cyan",
    secondary_hue="purple",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
)

# Gradio picks light/dark from `?__theme=`; pin it before the app boots so no
# light-mode surfaces flash through the neon styling.
HEAD = """
<script>
(function () {
  try {
    var u = new URL(window.location.href);
    if (u.searchParams.get('__theme') !== 'dark') {
      u.searchParams.set('__theme', 'dark');
      window.location.replace(u.toString());
    }
  } catch (e) {}
  document.addEventListener('DOMContentLoaded', function () {
    document.body.classList.add('dark');
  });

  // Always open at the top: stop the browser restoring the previous scroll
  // position on reload, and focus the composer without scrolling to it.
  try { history.scrollRestoration = 'manual'; } catch (e) {}
  var tries = 0;
  var timer = setInterval(function () {
    var box = document.querySelector('#buddy-input textarea');
    if (box) {
      clearInterval(timer);
      window.scrollTo(0, 0);
      try { box.focus({ preventScroll: true }); } catch (e) { box.focus(); }
      // Gradio settles layout for a moment after mount
      setTimeout(function () { window.scrollTo(0, 0); }, 200);
    } else if (++tries > 120) {
      clearInterval(timer);
    }
  }, 50);
})();
</script>
"""

HEADER = """
<div id="buddy-header">
  <div class="buddy-title">Buddy</div>
  <div class="buddy-sub"><span class="buddy-dot"></span>local ai · powered by ollama</div>
</div>
"""


def build_ui() -> gr.Blocks:
    models = list_models()

    with gr.Blocks(title="Buddy", fill_height=True) as demo:
        gr.HTML(HEADER)

        with gr.Group(elem_id="buddy-controls"):
            with gr.Row():
                model = gr.Dropdown(
                    choices=models,
                    value=models[0] if models else None,
                    label="Model",
                    elem_id="buddy-model",
                    scale=4,
                )
                temperature = gr.Slider(
                    0.0, 1.0, value=0.7, step=0.05, label="Temperature", scale=3
                )
                refresh = gr.Button("Refresh", size="sm", scale=1, elem_id="buddy-refresh")
                new_chat = gr.Button("New Chat", size="sm", scale=1, elem_id="buddy-new")

        chat = gr.Chatbot(
            elem_id="buddy-chat",
            label=None,
            show_label=False,
            height="52vh",
            min_height=340,
            layout="bubble",
            placeholder="### Ask Buddy anything\nRunning fully local on your machine — nothing leaves this device.",
            group_consecutive_messages=False,
            examples=EXAMPLES,
            # default toolbar includes "share", which posts to Hugging Face Spaces
            # Discussions — it can't work for a local-only app
            buttons=["copy", "copy_all"],
        )

        with gr.Group(elem_id="buddy-composer"):
            with gr.Row():
                box = gr.Textbox(
                    placeholder="Message Buddy…",
                    show_label=False,
                    lines=1,
                    max_lines=6,
                    # focused from HEAD with preventScroll instead: plain autofocus
                    # makes the browser scroll the composer into view on load
                    autofocus=False,
                    elem_id="buddy-input",
                    scale=8,
                )
                send = gr.Button("Send", variant="primary", scale=1, elem_id="buddy-send")

        inputs = [box, chat, model, temperature]
        outputs = [chat, box]
        box.submit(stream_chat, inputs, outputs)
        send.click(stream_chat, inputs, outputs)
        chat.example_select(on_example, [chat, model, temperature], outputs)
        chat.retry(on_retry, [chat, model, temperature], outputs)
        new_chat.click(lambda: ([], ""), None, outputs)
        refresh.click(refresh_models, model, model)
        demo.load(refresh_models, model, model)

    return demo


def main() -> None:
    build_ui().launch(theme=THEME, css=CSS, head=HEAD, inbrowser=True)


if __name__ == "__main__":
    main()
