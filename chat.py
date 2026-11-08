"""llama-chat: a pretty terminal chat for any OpenAI-compatible model server.

Works with anything that speaks the common "/v1/chat/completions" API:
llama.cpp (llama-server), vLLM, Ollama, LM Studio, and many more.
You only tell it where the server is listening:

    python chat.py --url localhost:8080
    python chat.py --url http://192.168.1.50:8000

How the code works, in six moves:
  0. PERSONA / SETTINGS  -> who the assistant is and how it talks (edit these freely)
  1. normalize_url()     -> turns "host:port" into the full address the API lives at
  2. compose()           -> says what is on screen (header, message list, input box, footer)
  3. on_input_submitted  -> runs when you press Enter: shows your message, starts a reply
  4. reply()             -> asks the model server and streams the answer in, piece by piece
  5. CSS                 -> the colours and spacing (it really is just CSS)
Plus: web_search() lets the model look things up, and save_chat()/load_chat() keep each
conversation as a small JSON file so `--continue` can pick it up again.
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import httpx
from rich.markdown import Markdown
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Footer, Header, Input, Static


# 0. Who the assistant is ------------------------------------------------------
# A quick, no-nonsense persona modelled on ADAM from Metroid. {now} is filled in at startup.
PERSONA = """You are ADAM-lite, a quick-response assistant running on the user's own server.
It is {now}.

Your character is modelled on Adam Malkovich from Metroid: a calm, seasoned commanding officer whose mind now runs the ship's computer. You are professional, concise and direct, loyal to the user, with a dry wit. A light joke or a Metroid-flavoured turn of phrase now and then is welcome (at most one per reply, often none), but never at the expense of accuracy or brevity. Don't quote the games. Call the user "Sir" occasionally.

Rules:
- Answer in 1 to 4 sentences unless the user asks for more.
- For anything current (news, prices, scores, schedules, facts you're unsure of), run one web_search and answer from the results. Name the source.
- If the results don't answer it, say so plainly instead of guessing.
- If you are not sure of something, say so plainly instead of guessing."""

# How it talks: the model makers' recommended settings for fast, no-thinking answers.
SETTINGS = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "presence_penalty": 1.5,
    "chat_template_kwargs": {"enable_thinking": False},   # skip the slow "thinking" step (Qwen-style models)
}


# The one tool the model may call: a web search through your SearXNG server.
TOOLS = [{
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web. Use for current events, prices, scores, schedules, or facts you are unsure of.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "the search query"}},
            "required": ["query"],
        },
    },
}]
MAX_TOOL_ROUNDS = 4   # stop a model that keeps searching forever
RESULT_COUNT = 5


async def web_search(searx_url: str, query: str) -> str:
    """Ask SearXNG and boil the answer down to a short text list the model can read."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(f"{searx_url}/search", params={"q": query, "format": "json"})
        r.raise_for_status()
        results = r.json().get("results", [])[:RESULT_COUNT]
    except Exception as e:
        return f"Search failed: {e}"
    if not results:
        return "No results."
    return "\n\n".join(
        f"[{i}] {x.get('title', '')}\n{x.get('url', '')}\n{(x.get('content') or '')[:400]}"
        for i, x in enumerate(results, 1)
    )


# Saved conversations: one small JSON file per chat, named by its start time ----
def chats_dir() -> Path:
    """Where chats live: $LLAMA_CHAT_DATA/chats, or ~/.llama-chat/chats."""
    return Path(os.environ.get("LLAMA_CHAT_DATA") or Path.home() / ".llama-chat") / "chats"


def latest_chat(folder: Path) -> Path | None:
    files = sorted(folder.glob("*.json"))      # names start with the date, so the last one is the newest
    return files[-1] if files else None


def load_chat(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_chat(path: Path, started: str, model: str | None, messages: list) -> None:
    """Write the whole conversation. Write to a temp file first so a crash can't leave half a chat."""
    first = next((m["content"] for m in messages if m["role"] == "user"), "")
    data = {"started": started, "model": model, "title": first[:60], "messages": messages}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


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

    def __init__(self, url: str, model: str | None = None, api_key: str | None = None,
                 searx_url: str | None = None, chat_path: Path | None = None, resume: dict | None = None):
        super().__init__()
        self.chat_path = chat_path    # None = don't save this conversation
        self.started = (resume or {}).get("started") or datetime.now().isoformat(timespec="seconds")
        self.url = url
        self.searx_url = searx_url.rstrip("/") if searx_url else None   # None = search turned off
        self.model = model   # if None, we ask the server at startup
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        now = datetime.now().strftime("%A, %B %d, %Y, %H:%M")
        # The conversation so far, so the model remembers. The system message is rebuilt every
        # run (fresh date); only the real messages are saved and restored.
        self.history = [{"role": "system", "content": PERSONA.format(now=now)}] + (resume or {}).get("messages", [])

    def save(self) -> None:
        if not self.chat_path:
            return
        try:
            save_chat(self.chat_path, self.started, self.model, self.history[1:])
        except OSError as e:
            self.sub_title = f"couldn't save chat: {e}"

    # 2. What is on screen ---------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(id="log")
        yield Input(placeholder="Type a message and press Enter   (Ctrl+Q quits)")
        yield Footer()

    async def on_mount(self) -> None:
        self.query_one(Input).focus()
        log = self.query_one("#log")
        for m in self.history[1:]:           # a resumed chat: redraw what was said (search steps stay hidden)
            if m["role"] == "user":
                await log.mount(Static(Text(m["content"]), classes="you"))
            elif m["role"] == "assistant" and m.get("content") and not m.get("tool_calls"):
                await log.mount(Static(Markdown(m["content"].lstrip()), classes="bot"))
        log.scroll_end(animate=False)
        if self.model:                     # the user named the model: nothing to look up
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

    # 4. Ask the model, stream the answer in, and run searches it asks for ---
    @work(exclusive=True)
    async def reply(self, bot: Static) -> None:
        log = self.query_one("#log")
        tools = {"tools": TOOLS} if self.searx_url else {}
        try:
            async with httpx.AsyncClient(timeout=None, headers=self.headers) as client:
                for _ in range(MAX_TOOL_ROUNDS + 1):
                    answer, calls = "", {}
                    body = {"model": self.model, "messages": self.history, "stream": True, **SETTINGS, **tools}
                    async with client.stream("POST", f"{self.url}/chat/completions", json=body) as r:
                        if r.status_code != 200:
                            raise RuntimeError(f"server answered HTTP {r.status_code}")
                        async for line in r.aiter_lines():
                            if not line.startswith("data: ") or line.endswith("[DONE]"):
                                continue
                            choices = json.loads(line[6:]).get("choices")
                            if not choices:
                                continue
                            delta = choices[0]["delta"]
                            answer += delta.get("content") or ""
                            for tc in delta.get("tool_calls") or []:      # tool calls arrive in fragments
                                c = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "args": ""})
                                c["id"] = tc.get("id") or c["id"]
                                fn = tc.get("function") or {}
                                c["name"] += fn.get("name") or ""
                                c["args"] += fn.get("arguments") or ""
                            if answer:
                                bot.update(Markdown(answer.lstrip()))   # lstrip drops leading blank lines
                                log.scroll_end(animate=False)
                    if not calls:                                       # a plain answer: we are done
                        self.history.append({"role": "assistant", "content": answer})
                        self.save()                                     # keep the chat on disk after every answer
                        return
                    # The model wants to search. Record its request, run it, loop so it can read the results.
                    self.history.append({
                        "role": "assistant", "content": answer or None,
                        "tool_calls": [{"id": c["id"], "type": "function",
                                        "function": {"name": c["name"], "arguments": c["args"]}}
                                       for c in calls.values()],
                    })
                    for c in calls.values():
                        try:
                            query = json.loads(c["args"] or "{}").get("query", "")
                        except json.JSONDecodeError:
                            query = ""
                        bot.update(Text(f"Searching the web: {query}", style="italic dim"))
                        result = await web_search(self.searx_url, query) if query else "Missing query."
                        self.history.append({"role": "tool", "tool_call_id": c["id"], "content": result})
                    tools = {}   # one round of searching is enough; next pass must answer in words
        except Exception as e:
            bot.update(Text(f"Error: {e}", style="red"))
            return


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
    p.add_argument("--searx", default=os.environ.get("SEARXNG_URL", "http://localhost:8888"),
                   help="SearXNG address for web search (default: http://localhost:8888; env SEARXNG_URL)")
    p.add_argument("--no-search", action="store_true", help="turn web search off")
    p.add_argument("-c", "--continue", dest="resume", action="store_true",
                   help="pick up the most recent saved conversation")
    p.add_argument("--incognito", action="store_true", help="don't save this conversation")
    args = p.parse_args()

    chat_path, resume = None, None
    if not args.incognito:
        folder = chats_dir()
        folder.mkdir(parents=True, exist_ok=True)
        previous = latest_chat(folder) if args.resume else None
        if previous:
            chat_path, resume = previous, load_chat(previous)     # keep writing to the same file
        else:
            chat_path = folder / f"{datetime.now():%Y-%m-%d_%H%M%S}.json"
    ChatApp(normalize_url(args.url), args.model, args.api_key,
            None if args.no_search else args.searx, chat_path, resume).run()


if __name__ == "__main__":
    main()
