"""fetch_door: the "read a web page" door for llama-chat's jail.

It listens on a Unix socket and answers ONE kind of question:

    GET /fetch?url=<address>[&start=<number>]   ->  the page as plain text

Everything the jailed chat can read from the internet goes through here, so this file is where the
rules live:
  - only http:// and https:// on ports 80/443 (default ports only)
  - the address must resolve to a PUBLIC internet address; localhost, your LAN, link-local and other
    private ranges are refused (this is what stops the model from poking your own network)
  - we connect to the address we checked (no DNS trick between "check" and "connect"),
    and every redirect is checked again from scratch
  - size, time and concurrency limits; no cookies, no login, no request bodies, GET only
  - the reply is text only (HTML is boiled down to its readable text)

Usage (the launcher does this for you):  python fetch_door.py /path/to/fetch.sock
"""
import asyncio
import ipaddress
import os
import re
import socket
import sys
from html.parser import HTMLParser
from urllib.parse import parse_qs, urlsplit

import httpx

PAGE_CHARS = 8000            # how much text one call returns
MAX_BYTES = 2_000_000        # never download more than this
TIMEOUT = 15                 # seconds per request
MAX_REDIRECTS = 5
MAX_CONCURRENT = 4
UA = "llama-chat/1.0 (personal homelab text reader; +https://github.com/MillerAJ/llama-chat)"
TEXT_TYPES = ("text/", "application/json", "application/xml", "application/xhtml", "application/rss",
              "application/atom", "application/ld+json")


# --- "is this address safe to visit?" -------------------------------------------
async def public_address(host: str, port: int) -> str:
    """Resolve host and return one address, but only if EVERY answer is a public internet address."""
    infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    addrs = []
    for *_, sockaddr in infos:
        ip = ipaddress.ip_address(sockaddr[0].split("%")[0])
        if getattr(ip, "ipv4_mapped", None):
            ip = ip.ipv4_mapped
        if not ip.is_global or ip.is_multicast:
            raise ValueError(f"{host} is not a public internet address, so I won't visit it")
        addrs.append(str(ip))
    if not addrs:
        raise ValueError(f"can't resolve {host}")
    return addrs[0]


# --- HTML -> readable text --------------------------------------------------------
class Reader(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "head", "template", "iframe", "nav", "footer", "aside", "form", "select"}
    LINE = {"br", "li", "tr", "div", "dt", "dd", "hr", "figcaption"}                      # a new line starts here
    PARA = {"p", "ul", "ol", "table", "section", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "blockquote"}  # blank line after
    MAIN = {"main", "article"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.all, self.main = [], []                  # the whole page, and just the <main>/<article> part
        self.title, self.skip, self.in_title, self.in_main = "", 0, False, 0

    def put(self, s):
        if not self.skip:
            self.all.append(s)
            if self.in_main:
                self.main.append(s)

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self.in_title = True
        elif tag in self.SKIP:
            self.skip += 1
        elif tag in self.MAIN:
            self.in_main += 1
        if tag in self.LINE or tag in self.PARA:
            self.put("\n")
        if tag in ("h1", "h2", "h3"):
            self.put("\n# ")
        elif tag == "li":
            self.put("- ")

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False
        elif tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag in self.MAIN:
            self.in_main = max(0, self.in_main - 1)
        if tag in self.PARA:
            self.put("\n\n")

    def handle_data(self, data):
        if self.in_title:
            self.title += data
        else:
            self.put(data)

    @staticmethod
    def clean(parts) -> str:
        t = "".join(parts)
        t = re.sub(r"[ \t\r\f\v\u00a0]+", " ", t)
        t = re.sub(r" ?\n ?", "\n", t)
        t = re.sub(r"\n{3,}", "\n\n", t)
        return t.strip()

    def text(self) -> str:
        main = self.clean(self.main)
        return main if len(main) >= 500 else self.clean(self.all)    # prefer the article; fall back to everything


def to_text(body: str, content_type: str) -> str:
    if "html" in content_type or body.lstrip()[:15].lower().startswith(("<!doctype", "<html")):
        r = Reader()
        r.feed(body)
        title = re.sub(r"\s+", " ", r.title).strip()
        return (f"Title: {title}\n\n" if title else "") + r.text()
    return body.strip()


# --- the actual fetch -------------------------------------------------------------
async def fetch_text(url: str) -> str:
    for _ in range(MAX_REDIRECTS + 1):
        u = httpx.URL(url)
        if u.scheme not in ("http", "https"):
            raise ValueError("only http and https addresses are allowed")
        if not u.host:
            raise ValueError("that isn't a web address")
        port = u.port or (443 if u.scheme == "https" else 80)
        if port not in (80, 443):
            raise ValueError("only the standard web ports (80 and 443) are allowed")
        ip = await public_address(u.host, port)
        host_header = u.host if u.port is None else f"{u.host}:{u.port}"
        async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=False, trust_env=False) as client:
            async with client.stream(
                "GET", u.copy_with(host=ip),
                headers={"Host": host_header, "User-Agent": UA, "Accept": "text/html,text/plain,*/*;q=0.5",
                         "Accept-Language": "en-US,en;q=0.8"},
                extensions={"sni_hostname": u.host},
            ) as r:
                if r.is_redirect and r.headers.get("location"):
                    url = str(u.join(r.headers["location"]))      # then loop: re-check the new address
                    continue
                if r.status_code >= 400:
                    raise ValueError(f"the site answered HTTP {r.status_code}")
                ctype = r.headers.get("content-type", "text/html").split(";")[0].strip().lower()
                if not ctype.startswith(TEXT_TYPES):
                    raise ValueError(f"I can only read text pages, not {ctype}")
                data = b""
                async for chunk in r.aiter_bytes():
                    data += chunk
                    if len(data) > MAX_BYTES:
                        data = data[:MAX_BYTES]
                        break
                body = data.decode(r.charset_encoding or r.encoding or "utf-8", errors="replace")
                return to_text(body, ctype)
    raise ValueError("too many redirects")


# --- a tiny HTTP server on a Unix socket --------------------------------------------
LIMIT = asyncio.Semaphore(MAX_CONCURRENT)


def reply(writer, code: int, text: str) -> None:
    body = text.encode("utf-8")
    head = f"HTTP/1.1 {code} {'OK' if code == 200 else 'Error'}\r\nContent-Type: text/plain; charset=utf-8\r\n" \
           f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n"
    writer.write(head.encode() + body)


async def handle(reader, writer):
    try:
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 10)
        line = head.split(b"\r\n", 1)[0].decode("latin-1")
        method, target, _ = line.split(" ", 2)
        parts = urlsplit(target)
        if method != "GET" or parts.path != "/fetch":
            reply(writer, 404, "Only GET /fetch is available here.")
            return
        q = parse_qs(parts.query)
        url = (q.get("url") or [""])[0].strip()
        if "://" not in url:
            url = "https://" + url
        try:
            start = max(0, int((q.get("start") or ["0"])[0]))
        except ValueError:
            start = 0
        try:
            async with LIMIT:
                text = await asyncio.wait_for(fetch_text(url), 40)
        except Exception as e:
            reply(writer, 200, f"Could not read {url}: {e}")
            return
        if not text:
            reply(writer, 200, "The page had no readable text (it may need JavaScript to show its content).")
            return
        chunk = text[start:start + PAGE_CHARS]
        if not chunk:
            reply(writer, 200, f"No more text: the page is only {len(text)} characters long.")
            return
        end = start + len(chunk)
        note = f"\n\n[Showing characters {start}-{end} of {len(text)}. Call fetch_url again with start={end} for more.]" \
            if end < len(text) else f"\n\n[End of page: {len(text)} characters.]"
        reply(writer, 200, chunk + note)
    except Exception as e:
        try:
            reply(writer, 400, f"Bad request: {e}")
        except Exception:
            pass
    finally:
        try:
            await writer.drain()
        except Exception:
            pass
        writer.close()


async def watch_parent():
    ppid = os.getppid()
    while os.getppid() == ppid:
        await asyncio.sleep(1)
    os._exit(0)


async def main(sock: str) -> None:
    server = await asyncio.start_unix_server(handle, path=sock)
    os.chmod(sock, 0o600)
    asyncio.create_task(watch_parent())
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
