"""Tiny byte relay: the jail's only door to the outside world (stdlib only).

  relay.py tcp2unix PORT SOCKET      (runs INSIDE the jail)  127.0.0.1:PORT  -> unix SOCKET
  relay.py unix2tcp SOCKET HOST PORT (runs OUTSIDE the jail) unix SOCKET     -> HOST:PORT (fixed target)
"""
import asyncio, os, sys


async def pipe(r, w):
    try:
        while data := await r.read(65536):
            w.write(data)
            await w.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        w.close()


def forward(open_upstream):
    async def handle(cr, cw):
        try:
            ur, uw = await open_upstream()
        except OSError:
            cw.close()
            return
        await asyncio.gather(pipe(cr, uw), pipe(ur, cw))
    return handle


async def watch_parent():                      # outside relay: exit when the launcher is gone
    ppid = os.getppid()
    while os.getppid() == ppid:
        await asyncio.sleep(1)
    os._exit(0)


async def main():
    mode = sys.argv[1]
    if mode == "tcp2unix":
        port, sock = int(sys.argv[2]), sys.argv[3]
        server = await asyncio.start_server(
            forward(lambda: asyncio.open_unix_connection(sock)), "127.0.0.1", port)
    else:
        sock, host, port = sys.argv[2], sys.argv[3], int(sys.argv[4])
        server = await asyncio.start_unix_server(
            forward(lambda: asyncio.open_connection(host, port)), sock)
        os.chmod(sock, 0o600)
        asyncio.create_task(watch_parent())
    await server.serve_forever()

asyncio.run(main())
