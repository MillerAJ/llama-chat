"""llama-chat: a pretty terminal chat for any OpenAI-compatible model server.

Works with anything that speaks the common "/v1/chat/completions" API:
llama.cpp (llama-server), vLLM, Ollama, LM Studio, and many more.
You only tell it where the server is listening:

    python chat.py --url localhost:8080
    python chat.py --url http://192.168.1.50:8000

How the code works, in five moves:
  1. normalize_url()     -> turns "host:port" into the full address the API lives at
  2. compose()           -> says what is on screen (header, message list, input box, footer)
  3. on_input_submitted  -> runs when you press Enter: shows your message, starts a reply
  4. reply()             -> asks the model server and streams the answer in, piece by piece
  5. CSS                 -> the colours and spacing (it really is just CSS)
"""
import argparse
import json
import os

import httpx
from rich.markdown import Markdown
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Footer, Header, Input, Static


# 1. Be forgiving about how people type the address -------------------------
def normalize_url(raw: str) -> str:
    """'host:8000', 'http://host:8000' or 'http://host:8000/v1'  ->  'http://host:8000/v1'"""
    url = raw.strip().rstrip("/")
    if "://" not in url:
        url = "http://" + url
    if not url.endswith("/v1"):
        url += "/v1"
    return url


class ChatApp(App):
    TITLE = "llama-chat"

    # 5. The look. Change a colour here and rerun: that is the whole loop.
    CSS = """
    Screen { background: #14141f; }
    #log   { padding: 1 2; }
    .you   { background: #2a3b63; color: #ffffff; padding: 0 2; margin: 1 0 0 10; }
    .bot   { background: #23233a; padding: 0 2; margin: 1 10 0 0; }
    Input  { margin: 0 2 1 2; }
    """

    def __init__(self, url: str, model: str | None = None, api_key: str | None = None):
        super().__init__()
        self.url = url
        self.model = model   # if None, we ask the server at startup
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.history = []    # the conversation so far, so the model remembers

    # 2. What is on screen ---------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(id="log")
        yield Input(placeholder="Type a message and press Enter   (Ctrl+Q quits)")
        yield Footer()

    async def on_mount(self) -> None:
        self.query_one(Input).focus()
        if self.model:                      # the user named the model: nothing to look up
            self.sub_title = f"{self.model}  @  {self.url}"
            return
        # Otherwise ask the server which model it is serving.
        try:
            async with httpx.AsyncClient(timeout=10, headers=self.headers) as client:
                r = await client.get(f"{self.url}/models")
            r.raise_for_status()
            self.model = r.json()["data"][0]["id"]
            self.sub_title = f"{self.model}  @  {self.url}"      # shows in the header bar
        except Exception as e:
            self.sub_title = f"can't reach {self.url}: {e}"

    # 3. You pressed Enter ---------------------------------------------------
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text or not self.model:
            return
        event.input.value = ""
        log = self.query_one("#log")
        await log.mount(Static(Text(text), classes="you"))   # your bubble
        bot = Static("...", classes="bot")                   # an empty bubble for the answer
        await log.mount(bot)
        log.scroll_end(animate=False)
        self.history.append({"role": "user", "content": text})
        self.reply(bot)

    # 4. Ask the model, stream the answer into the bubble --------------------
    @work(exclusive=True)
    async def reply(self, bot: Static) -> None:
        log = self.query_one("#log")
        answer = ""
        body = {"model": self.model, "messages": self.history, "stream": True}
        try:
            async with httpx.AsyncClient(timeout=None, headers=self.headers) as client:
                async with client.stream("POST", f"{self.url}/chat/completions", json=body) as r:
                    if r.status_code != 200:
                        raise RuntimeError(f"server answered HTTP {r.status_code}")
                    async for line in r.aiter_lines():
                        if not line.startswith("data: ") or line.endswith("[DONE]"):
                            continue
                        choices = json.loads(line[6:]).get("choices")
                        if not choices:
                            continue
                        answer += choices[0]["delta"].get("content") or ""
                        bot.update(Markdown(answer.lstrip()))   # redraw the bubble (lstrip drops leading blank lines)
                        log.scroll_end(animate=False)
        except Exception as e:
            bot.update(Text(f"Error: {e}", style="red"))
            return
        self.history.append({"role": "assistant", "content": answer})


def main() -> None:
    p = argparse.ArgumentParser(
        prog="llama-chat",
        description="A pretty terminal chat for any OpenAI-compatible server (llama.cpp, vLLM, Ollama, ...).",
    )
    p.add_argument("--url", default=os.environ.get("LLAMA_URL", "localhost:8000"),
                   help="where the server listens, e.g. localhost:8080 or http://192.168.1.50:8000 (default: localhost:8000)")
    p.add_argument("--model", help="model name to use (default: ask the server)")
    p.add_argument("--api-key", default=os.environ.get("LLAMA_API_KEY"),
                   help="API key, only if the server requires one")
    args = p.parse_args()
    ChatApp(normalize_url(args.url), args.model, args.api_key).run()


if __name__ == "__main__":
    main()
