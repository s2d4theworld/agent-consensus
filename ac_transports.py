# agent-consensus — HTTP helpers and the two transports:
#   opencode = local `opencode serve` sessions (stateful: server keeps history,
#              the script only sends new messages)
#   openai   = any OpenAI-compatible /v1/chat/completions endpoint (stateless:
#              the script re-sends the full conversation each call)
import hashlib
import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from typing import Any

from ac_config import (
    AgentCfg,
    AGENT_LIST_TTL,
    MESSAGE_ID_TPL,
    RETRY_500_WAITS,
    RETRY_CONN_WAITS,
    RETRY_HTTP_CODES,
    RETRY_HTTP_WAITS,
    ROLE_JUDGE,
    SESSION_TITLE_PREFIX,
    SWEEP_MIN_AGE,
    TIMEOUT_DEFAULT,
    TIMEOUT_LIST,
    TIMEOUT_MESSAGE,
    TIMEOUT_SESSION,
    TS_FORMAT,
)
from ac_errors import TransportError

# ---------- HTTP helpers (retries + JSON) ----------
RETRY_POLICY: dict[str, tuple[int, ...]] = {
    "http": RETRY_HTTP_WAITS,   # 429/502/503: transient upstream errors
    "server": RETRY_500_WAITS,  # 500: often deterministic, retry briefly
    "conn": RETRY_CONN_WAITS,
}

def _retry_wait(kind: str, attempt: int) -> int | None:
    waits = RETRY_POLICY[kind]
    return waits[attempt] if attempt < len(waits) else None

def _retry_kind(code: int) -> str | None:
    if code in RETRY_HTTP_CODES:
        return "http"
    if code == 500:
        return "server"
    return None

def http_get_json(url: str, timeout: float) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

def post_json(url: str, payload: dict[str, Any],
              headers: dict[str, str] | None = None,
              timeout: float = TIMEOUT_DEFAULT) -> Any:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    attempts = max(len(waits) for waits in RETRY_POLICY.values()) + 1
    last_err: str | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:500]
            e.close()  # release the response socket; body is already read
            last_err = f"HTTP {e.code}: {body}"
            kind = _retry_kind(e.code)
            wait = _retry_wait(kind, attempt) if kind else None
            if wait is None:
                raise TransportError(last_err) from e
            print(f"    [retry {attempt+1}] {e.code}, waiting {wait}s...", flush=True)
            time.sleep(wait)
        except (TimeoutError, ConnectionError, urllib.error.URLError, socket.timeout) as e:
            last_err = f"connection: {e}"
            wait = _retry_wait("conn", attempt)
            if wait is None:
                raise TransportError(last_err) from e
            print(f"    [retry {attempt+1}] connection failed, waiting {wait}s...", flush=True)
            time.sleep(wait)
        except Exception as e:
            raise TransportError(f"request failed: {e}") from e
    raise TransportError(last_err or "request failed")

def strip_control_tokens(text: str) -> str:
    # Strip leaked control tokens from model output (e.g. <|constrain|>, <|channel|>)
    return re.sub(r"<\|[^|]*\|>", "", text).strip()

# ---------- Transport: OpenAI-compatible (stateless) ----------
def chat_completion(agent_key: str, agent_cfg: AgentCfg, system_prompt: str,
                    messages: list[dict[str, str]]) -> str:
    api_key = agent_cfg.get("api_key")
    if not api_key:
        env_name = f"CONSENSUS_API_KEY_{agent_key.upper()}"
        api_key = os.environ.get(env_name, "")
    if not api_key:
        raise TransportError(
            f"API key for agent '{agent_key}' is missing. Set it in "
            f"config.json or env {env_name}."
        )
    payload = {
        "model": agent_cfg["model"],
        "messages": [{"role": "system", "content": system_prompt}] + messages,
        # the judge should be deterministic; debaters benefit from some variety
        "temperature": agent_cfg.get("temperature", 0.0 if agent_key == ROLE_JUDGE else 0.7),
    }
    data = post_json(
        agent_cfg["api_url"],
        payload,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    msg = data["choices"][0]["message"]
    content = msg.get("content") or msg.get("reasoning_content") or ""
    return strip_control_tokens(content)

# ---------- Transport: opencode serve (stateful sessions) ----------
def oc_base_url(agent_cfg: AgentCfg) -> str:
    return agent_cfg["api_url"].rstrip("/")

def oc_session_create(agent_cfg: AgentCfg, title: str) -> str:
    model = agent_cfg["model"]
    provider, sep, model_id = model.partition("/")
    if not sep:
        provider, model_id = "opencode", model
    payload = {
        "model": {"id": model_id, "providerID": provider},
        "title": title,
    }
    data = post_json(oc_base_url(agent_cfg) + "/session", payload, timeout=TIMEOUT_SESSION)
    return data["id"]

def oc_send(agent_cfg: AgentCfg, session_id: str, system_prompt: str, text: str,
            agent_name: str | None = None, turn_index: int | None = None) -> str:
    payload = {
        "system": system_prompt,
        "parts": [{"type": "text", "text": text}],
    }
    if agent_name:
        payload["agent"] = agent_name
    if turn_index is not None:
        # content-derived id: a retry with the identical payload reuses the id
        # and the server dedupes it instead of appending the same turn twice
        # (verified); different content always gets a fresh id
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        payload["messageID"] = MESSAGE_ID_TPL.format(sid=session_id[4:16],
                                                     turn=turn_index,
                                                     digest=digest)
    data = post_json(
        f"{oc_base_url(agent_cfg)}/session/{session_id}/message",
        payload,
        timeout=TIMEOUT_MESSAGE,
    )
    info = data.get("info") or {}
    if info.get("error"):
        err = info["error"]
        detail = (err.get("data") or {}).get("message") or err.get("name") or "unknown error"
        raise TransportError(f"opencode: {detail}")
    parts = data.get("parts") or []
    out = "\n".join(p.get("text", "") for p in parts if p.get("type") == "text").strip()
    if not out:
        out = "\n".join(p.get("text", "") for p in parts if p.get("type") == "reasoning").strip()
    return strip_control_tokens(out)

def oc_delete_url(base_url: str, session_id: str) -> None:
    try:
        req = urllib.request.Request(
            f"{base_url}/session/{session_id}", method="DELETE"
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT_LIST):
            pass
    except Exception:
        pass

def oc_delete(agent_cfg: AgentCfg, session_id: str) -> None:
    oc_delete_url(oc_base_url(agent_cfg), session_id)

def _session_last_activity(session: dict, title: str) -> float:
    # server-reported last activity (ms epoch) wins; older servers without it
    # fall back to the creation time embedded in the title, and unparseable
    # titles count as old orphans
    stamp = session.get("time")
    updated = stamp.get("updated") if isinstance(stamp, dict) else None
    if isinstance(updated, (int, float)) and updated > 0:
        return updated / 1000.0
    try:
        return time.mktime(time.strptime(title.rsplit(" ", 1)[-1], TS_FORMAT))
    except ValueError:
        return 0.0

def oc_sweep(base_url: str, prefix: str = SESSION_TITLE_PREFIX,
             min_age: float = SWEEP_MIN_AGE) -> None:
    # best-effort: delete orphan sessions left by crashed runs. Sessions with
    # recent activity are skipped, so a live parallel run is not disturbed
    try:
        data = http_get_json(base_url + "/session", TIMEOUT_LIST)
    except Exception:
        return
    if not isinstance(data, list):
        return
    now = time.time()
    for s in data:
        title = s.get("title") or ""
        sid = s.get("id")
        if not (sid and title.startswith(prefix)):
            continue
        if now - _session_last_activity(s, title) >= min_age:
            oc_delete_url(base_url, sid)

def oc_agent_list(base_url: str, timeout: float = TIMEOUT_LIST) -> set[str] | None:
    # names of agents available on the server; None if unreachable
    try:
        data = http_get_json(base_url + "/agent", timeout)
    except Exception:
        return None
    names: set[str] = set()
    for a in data:
        if isinstance(a, dict) and isinstance(a.get("name"), str) and a["name"]:
            names.add(a["name"])
    return names

# base_url -> (checked_at, names or None)
_AGENT_LIST_CACHE: dict[str, tuple[float, set[str] | None]] = {}

def oc_agent_list_cached(base_url: str, timeout: float = TIMEOUT_LIST,
                         ttl: float = AGENT_LIST_TTL) -> set[str] | None:
    # avoids re-hitting the server on every menu redraw
    hit = _AGENT_LIST_CACHE.get(base_url)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    names = oc_agent_list(base_url, timeout)
    _AGENT_LIST_CACHE[base_url] = (time.time(), names)
    return names
