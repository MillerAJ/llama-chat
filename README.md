# llama-chat

A pretty terminal chat for any local model server.

Point it at wherever your model is listening and start talking. It works with anything that
speaks the common OpenAI-style API (`/v1/chat/completions`), which includes:

| Server | Usual port | Status |
|---|---|---|
| [vLLM](https://github.com/vllm-project/vllm) | 8000 | tested (Qwen3.8-27B) |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` | 8080 | expected to work, not yet tested |
| [Ollama](https://ollama.com) | 11434 | expected to work, not yet tested |
| [LM Studio](https://lmstudio.ai) | 1234 | expected to work, not yet tested |

## Quick start

```bash
git clone https://github.com/MillerAJ/llama-chat ~/llama-chat
cd ~/llama-chat
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python chat.py --url localhost:8000
```

Type a message, press Enter, and the reply streams in. Press `Ctrl+Q` to quit.

Optional shortcut: put the small `llama-chat` script on your `PATH` (for example copy it to
`~/bin`) and just type `llama-chat --url localhost:8000`.

## Options

| Option | Meaning |
|---|---|
| `--url` | where the server listens: `localhost:8080`, `192.168.1.50:8000`, or a full `http://...` address |
| `--model` | model name to use. If left out, llama-chat asks the server which model it is serving |
| `--api-key` | only if your server requires one (also read from `LLAMA_API_KEY`) |
| `--searx` | SearXNG address for web search (default `http://localhost:8888`, also read from `SEARXNG_URL`) |
| `--no-search` | turn web search off |
| `-c`, `--continue` | pick up the most recent saved conversation |
| `--incognito` | don't save this conversation |

The address is forgiving: `host:port`, `http://host:port` and `http://host:port/v1` all work.
`LLAMA_URL` sets the default address.

## Web search

The model can call a `web_search` tool when it needs current facts. llama-chat runs the search
against your [SearXNG](https://docs.searxng.org) server (JSON output must be enabled), hands the
top results back to the model, and the model answers from them. The bubble shows
"Searching the web: ..." while it works. The model needs native function calling (vLLM with
tool calling on, as ADAM-lite uses).

If SearXNG only listens on its own machine, tunnel to it first, for example
`ssh -L 8888:127.0.0.1:8888 motherbrain`, then the default address just works.

## Saved chats

Every conversation is saved after each answer as a small JSON file in `~/.llama-chat/chats/`
(set `LLAMA_CHAT_DATA` to move it). Start with `llama-chat -c` to continue the latest one: the
earlier messages are redrawn and the model remembers them. `--incognito` keeps a chat off disk.

## How the code works

Everything is in one file, [`chat.py`](chat.py), written to be read top to bottom. It uses
[Textual](https://textual.textualize.io) for the screen and [httpx](https://www.python-httpx.org)
to talk to the server. Five moves:

1. `normalize_url()` turns whatever you typed into the full address the API lives at.
2. `compose()` says what is on screen: a header, a message list, an input box, a footer.
3. `on_input_submitted()` runs when you press Enter. It shows your message and starts a reply.
4. `reply()` asks the server and streams the answer into the message bubble, piece by piece.
5. `CSS` is the look: colours and spacing. Change a colour, rerun, see it.

## Status

Early and deliberately small. It chats, streams, and remembers the conversation. Ideas for later:
a model switcher, live GPU meters, saved history, themes.
