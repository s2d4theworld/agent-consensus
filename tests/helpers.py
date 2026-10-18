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


class FakeServer:
    # minimal stand-in for `opencode serve` + a chat-completions endpoint
    def __init__(self, agent_names=("debater",)):
        self.agent_names = list(agent_names)
        self.verdicts = []
        self.session_list = []   # returned by GET /session (seeded orphans)
        self.sessions = []       # created via POST /session
        self.msgs = []
        self.deletes = []
        self.chats = []
        self.retry_codes = []    # status codes to fail the next POSTs with
        self.message_error = None  # if set, /message returns this error envelope
        self._counter = 0
        srv = self

        class Handler(BaseHTTPRequestHandler):
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
                if self.path == "/session":
                    self._json(srv.session_list)
                elif self.path == "/agent":
                    self._json([{"name": n} for n in srv.agent_names])
                else:
                    self._json({}, 404)

            def do_POST(self):
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
                    sys_p = payload.get("system", "")
                    if sys_p.startswith("You are a neutral judge"):
                        text = srv.verdicts.pop(0) if srv.verdicts else "NO - no verdict queued."
                    elif "PROPOSER" in sys_p:
                        text = "Proposer: I defend the topic."
                    else:
                        text = "Critic: I challenge the topic."
                    srv.msgs.append({"sid": sid,
                                     "messageID": payload.get("messageID"),
                                     "agent": payload.get("agent"),
                                     "system": sys_p,
                                     "text": payload["parts"][0]["text"]})
                    self._json({"info": {"id": "m1"},
                                "parts": [{"type": "text", "text": text}]})
                    return
                if self.path.endswith("/v1/chat/completions"):
                    srv.chats.append(payload)
                    self._json({"choices": [{"message": {"content": "YES - agree"}}]})
                    return
                self._json({}, 404)

            def do_DELETE(self):
                srv.deletes.append(self.path)
                sid = self.path.rsplit("/", 1)[-1]
                srv.session_list = [s for s in srv.session_list if s.get("id") != sid]
                self._json({})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
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
