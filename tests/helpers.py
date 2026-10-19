# Shared fakes and helpers for the agent-consensus test suite.
# Not a test module: unittest discovery only collects test_*.py files.
# Importing this first also puts the repo root on sys.path.
import contextlib
import io
import json
import os
import socket
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import ac_config
from ac_engine import run_discussion


def closed_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def patched_attrs(module: object, **values: object):
    # temporarily replace module attributes; always restored
    saved = {name: getattr(module, name) for name in values}
    try:
        for name, value in values.items():
            setattr(module, name, value)
        yield
    finally:
        for name, value in saved.items():
            setattr(module, name, value)


@contextlib.contextmanager
def fake_stdin(text: str):
    # swap sys.stdin for an in-memory stream; always restored
    with patched_attrs(sys, stdin=io.StringIO(text)):
        yield


@contextlib.contextmanager
def fake_argv(*args: str):
    # run the CLI entry point against a fixed argv; always restored
    with patched_attrs(sys, argv=["agent_consensus.py", *args]):
        yield


@contextlib.contextmanager
def captured_stdout():
    # redirect stdout into a buffer and yield it for assertions
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf


@contextlib.contextmanager
def temp_dir():
    # a self-deleting scratch directory; pair with enterContext or with
    with tempfile.TemporaryDirectory() as path:
        yield path


# setup-wizard stdin scripts: transport, then a blank answer for every prompt
# (defaults), then the max_rounds answer "3" (not a menu quit)
WIZARD_STDIN_OPENCODE = "opencode\n" + "\n" * 12 + "3\n"
WIZARD_STDIN_OPENAI = "openai\n" + "\n" * 5 * 3 + "3\n"


class _FakeHandler(BaseHTTPRequestHandler):
    # state lives on the FakeServer; the server carries it (see _FakeHTTPServer)
    @property
    def srv(self) -> "FakeServer":
        server = self.server
        assert isinstance(server, _FakeHTTPServer)
        return server.srv

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200, headers=None):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        srv = self.srv
        if self.path == "/session":
            self._json(srv.session_list)
        elif self.path == "/agent":
            self._json([{"name": n} for n in srv.agent_names])
        else:
            self._json({}, 404)

    def do_POST(self):
        srv = self.srv
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
        if srv.raw_response is not None:
            body, short = srv.raw_response
            srv.raw_response = None
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            # short > 0 declares more than is written: IncompleteRead
            self.send_header("Content-Length", str(len(body) + short))
            self.end_headers()
            self.wfile.write(body)
            return
        if srv.retry_codes:
            retry_headers = ({"Retry-After": srv.retry_after}
                             if srv.retry_after is not None else None)
            self._json({}, srv.retry_codes.pop(0), retry_headers)
        elif self.path == "/session":
            self._handle_session(srv, payload)
        elif self.path.endswith("/message"):
            self._handle_message(srv, payload)
        elif self.path.endswith("/v1/chat/completions"):
            self._handle_chat(srv, payload)
        else:
            self._json({}, 404)

    def _handle_session(self, srv, payload):
        srv._counter += 1
        sid = f"ses_fake{srv._counter:04d}test2018"
        srv.sessions.append({"id": sid, "title": payload.get("title")})
        self._json({"id": sid})

    def _handle_message(self, srv, payload):
        if srv.message_error:
            self._json({"info": {"error": srv.message_error}, "parts": []})
            return
        sid = self.path.split("/session/")[1].split("/")[0]
        mid = payload.get("messageID")
        hit = srv.responses.get((sid, mid)) if mid else None
        if hit is not None:
            # emulate the server dedupe: replay the stored answer;
            # a queued verdict must not be consumed twice
            srv.dupes.append({"sid": sid, "messageID": mid})
            self._json(hit)
            return
        sys_p = payload.get("system", "")
        if srv.reply_override is not None:
            text = srv.reply_override
        elif sys_p.startswith("You are a neutral judge"):
            text = srv.verdicts.pop(0) if srv.verdicts else "NO - no verdict queued."
        elif "PROPOSER" in sys_p:
            text = "Proposer: I defend the topic."
        else:
            text = "Critic: I challenge the topic."
        srv.msgs.append({"sid": sid,
                         "messageID": mid,
                         "agent": payload.get("agent"),
                         "system": sys_p,
                         "text": payload["parts"][0]["text"]})
        envelope = {"info": {"id": "m1"},
                    "parts": [{"type": "text", "text": text}]}
        if mid:
            srv.responses[(sid, mid)] = envelope
        self._json(envelope)

    def _handle_chat(self, srv, payload):
        srv.chats.append(payload)
        content = srv.reply_override if srv.reply_override is not None else "YES - agree"
        self._json({"choices": [{"message": {"content": content}}]})

    def do_DELETE(self):
        srv = self.srv
        srv.deletes.append(self.path)
        sid = self.path.rsplit("/", 1)[-1]
        srv.session_list = [s for s in srv.session_list if s.get("id") != sid]
        self._json({})


class _FakeHTTPServer(ThreadingHTTPServer):
    # carries the FakeServer so the handler can stay module-level and its
    # methods don't all count as statements of FakeServer.__init__
    def __init__(self, srv: "FakeServer") -> None:
        self.srv = srv
        super().__init__(("127.0.0.1", 0), _FakeHandler)


class FakeServer:
    # minimal stand-in for `opencode serve` + a chat-completions endpoint
    def __init__(self, agent_names=("debater",)):
        self.agent_names = list(agent_names)
        self.verdicts: list[str] = []
        self.session_list: list[dict] = []  # returned by GET /session (seeded orphans)
        self.sessions: list[dict] = []      # created via POST /session
        self.msgs: list[dict] = []
        self.dupes: list[dict] = []         # /message resends replayed by id
        self.responses: dict[tuple[str, str], dict] = {}  # (sid, messageID) -> envelope
        self.deletes: list[str] = []
        self.chats: list[dict] = []
        self.retry_codes: list[int] = []    # status codes to fail the next POSTs with
        self.retry_after: str | None = None  # Retry-After sent with those failures
        self.raw_response: tuple[bytes, int] | None = None  # (body, length shortfall)
        self.message_error: dict | None = None  # set: /message returns this envelope
        self.reply_override: str | None = None  # set: all model replies use this text
        self._counter = 0
        self.httpd = _FakeHTTPServer(self)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    @property
    def base(self):
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class FakeServerMixin(unittest.TestCase):
    # one line per test: a FakeServer whose shutdown joins the test cleanup
    def fake_server(self) -> FakeServer:
        srv = FakeServer()
        self.addCleanup(srv.stop)
        return srv


def oc_cfg(url, agent="debater", with_judge=True, max_rounds=1):
    cfg = {"max_rounds": max_rounds, "agents": {
        "proposer": {"name": "P", "transport": "opencode", "api_url": url,
                     "model": "opencode/x"},
        "critic": {"name": "C", "transport": "opencode", "api_url": url,
                   "model": "opencode/x"},
    }}
    if agent:
        cfg["agents"]["proposer"]["agent"] = agent
        cfg["agents"]["critic"]["agent"] = agent
    if with_judge:
        cfg["agents"]["judge"] = {"name": "J", "transport": "opencode",
                                  "api_url": url, "model": "opencode/x"}
    return cfg


def stateless_cfg(url):
    a = {"name": "A", "transport": "openai", "api_key": "k",
         "api_url": url, "model": "m"}
    return {"max_rounds": 1,
            "agents": {"proposer": dict(a), "critic": dict(a), "judge": dict(a)}}


class QuietRunMixin(unittest.TestCase):
    # redirects the transcript dir and captures run_discussion output
    def setUp(self):
        self._tdir = self.enterContext(temp_dir())
        # CONFIG_PATH is re-patched to its current value so cleanup restores
        # it even after tests point it at temp files
        self.enterContext(patched_attrs(
            ac_config, TRANSCRIPT_DIR=self._tdir,
            CONFIG_PATH=ac_config.CONFIG_PATH))

    def temp_config(self, name: str = "config.json") -> str:
        # point CONFIG_PATH at a fresh file under the test's temp dir;
        # setUp's cleanup restores the original afterwards
        ac_config.CONFIG_PATH = os.path.join(self._tdir, name)
        return ac_config.CONFIG_PATH

    def run_engine(self, cfg):
        with captured_stdout() as out:
            run_discussion(cfg, topic="test topic")
        return out.getvalue()

    def transcripts(self) -> dict[str, str]:
        # transcript file name -> full text, in name order
        files = {}
        for name in sorted(os.listdir(self._tdir)):
            if name.endswith(".txt"):
                with open(os.path.join(self._tdir, name), encoding="utf-8") as f:
                    files[name] = f.read()
        return files
