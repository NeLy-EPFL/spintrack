"""The page's server: FastAPI on a thread of its own, on localhost, behind a token.

`serve` binds the port asked for (`FIRST_PORT` by default), or the next free one after
it, and returns the page's link, with its token. On the default interface, reach it
from another machine through `ssh -L`. uvicorn runs off the main thread, so Ctrl-C
stays the run's.
"""

from __future__ import annotations

import os
import secrets
import socket
import threading
from collections.abc import Callable
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from spintrack.web.live import LiveView

STATIC = Path(__file__).parent / "static"
FIRST_PORT = 8300
PORTS = 100


def build_app(view: LiveView, controls=None, token: str | None = None) -> FastAPI:
    """The page and its API; `controls` (a `GuiSession`) adds the gui's routes."""

    def guard(token_: str = Query("", alias="token")) -> None:
        if token is not None and not secrets.compare_digest(token_, token):
            raise HTTPException(403, "this page's link has a token; use the full link")
        view.touch()

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    api = APIRouter(prefix="/api", dependencies=[Depends(guard)])

    @api.get("/state")
    def state(since: int = -1, log_since: int = -1) -> dict:
        return view.snapshot(since, log_since)

    @api.get("/image/{kind}")
    def image(kind: str) -> Response:
        found = view.image(kind)
        if found is None:
            raise HTTPException(404, f"no {kind} image yet")
        data, media_type = found
        return Response(
            data, media_type=media_type, headers={"Cache-Control": "no-store"}
        )

    @api.post("/stop")
    def stop() -> dict:
        view.request_stop()
        return {}

    if controls is not None:
        view.extra = controls.state
        controls.add_routes(api)
    app.include_router(api)

    @app.get("/")
    def page() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def _bind(host: str, port: int | None) -> socket.socket:
    first = port or FIRST_PORT
    ports = range(first, min(first + PORTS, 65536))
    for p in ports:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if os.name != "nt":  # as uvicorn binds: a port in TIME_WAIT is reusable
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, p))
        except OSError:
            sock.close()
            continue
        return sock
    raise OSError(f"no free port in {ports[0]}-{ports[-1]}")


def serve(
    view: LiveView,
    controls=None,
    *,
    port: int | None = None,
    host: str = "127.0.0.1",
) -> tuple[str, Callable[[], None]]:
    """Serve the page on a daemon thread: its link, and a function that stops it.

    Raises:
        OSError: If no port could be bound.
    """
    import uvicorn

    token = secrets.token_urlsafe(12)
    sock = _bind(host, port)
    app = build_app(view, controls, token)
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    thread = threading.Thread(
        target=server.run,
        kwargs={"sockets": [sock]},
        name="spintrack-page",
        daemon=True,
    )
    thread.start()

    def stop() -> None:
        server.should_exit = True
        thread.join(timeout=5.0)
        sock.close()

    # Bound to every interface, the page is reached by this machine's name.
    shown = socket.gethostname() if host in ("", "0.0.0.0", "::") else host
    return f"http://{shown}:{sock.getsockname()[1]}/?token={token}", stop
