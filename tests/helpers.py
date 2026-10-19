# Shared fakes and helpers for the agent-consensus test suite.
# Not a test module: unittest discovery only collects test_*.py files.
# Importing this first also puts the repo root on sys.path.
import contextlib
import io
import json
import os
import shutil
import socket
import sys
import tempfile
import threading
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


class _FakeHandler(BaseHTTPRequestHandler):
    # state lives on the FakeServer; the server carries it (see _FakeHTTPServer)
    @property
    def srv(self) -> "FakeServer":
        server = self.server
        assert isinstance(server, _FakeHTTPServer)
        return server.srv

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
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
        if srv.retry_codes:
            self._json({}, srv.retry_codes.pop(0))
            return
        if self.path == "/session":
            srv._counter += 1
            sid = f"ses_fake{srv._counter:04d}test2018"
            srv.sessions.append({"id": sid, "title": payload.get("title")})
            self._json({"id": sid})
            return
        if self.path.endswith("/message"):
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
            if sys_p.startswith("You are a neutral judge"):
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
            return
        if self.path.endswith("/v1/chat/completions"):
            srv.chats.append(payload)
            self._json({"choices": [{"message": {"content": "YES - agree"}}]})
            return
        self._json({}, 404)

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
        self.message_error: dict | None = None  # set: /message returns this envelope
        self._counter = 0
        self.httpd = _FakeHTTPServer(self)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    @property
    def base(self):
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


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


class QuietRunMixin:
    # redirects the transcript dir and captures run_discussion output
    def setUp(self):
        self._tdir = tempfile.mkdtemp(prefix="ac_test_")
        self._orig_tdir = ac_config.TRANSCRIPT_DIR
        ac_config.TRANSCRIPT_DIR = self._tdir
        self._orig_cfg_path = ac_config.CONFIG_PATH

    def tearDown(self):
        ac_config.TRANSCRIPT_DIR = self._orig_tdir
        ac_config.CONFIG_PATH = self._orig_cfg_path
        shutil.rmtree(self._tdir, ignore_errors=True)

    def run_engine(self, cfg):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run_discussion(cfg, topic="test topic")
        return buf.getvalue()
