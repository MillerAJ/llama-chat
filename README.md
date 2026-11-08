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

The address is forgiving: `host:port`, `http://host:port` and `http://host:port/v1` all work.
`LLAMA_URL` sets the default address.

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
