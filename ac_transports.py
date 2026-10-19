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
from email.utils import parsedate_to_datetime
from math import ceil
from typing import Any

from ac_config import (
    AgentCfg,
    AGENT_LIST_TTL,
    MESSAGE_ID_TPL,
    RETRY_500_WAITS,
    RETRY_AFTER_MAX,
    RETRY_CONN_WAITS,
    RETRY_HTTP_CODES,
    RETRY_HTTP_WAITS,
    SESSION_TITLE_PREFIX,
    SWEEP_MIN_AGE,
    SWEEP_TITLE_MIN_AGE,
    TIMEOUT_DEFAULT,
    TIMEOUT_LIST,
    TIMEOUT_MESSAGE,
    TIMEOUT_SESSION,
    TS_FORMAT,
    api_key_env,
    default_temperature,
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

def _retry_after(headers: Any) -> int | None:
    # a server-provided Retry-After (delta-seconds or HTTP-date) wins over the
    # policy table, but is capped so a broken or hostile value cannot stall
    # the CLI; None = no usable hint, use the policy table
    raw = ((headers.get("Retry-After") if headers else None) or "").strip()
    if not raw:
        return None
    delay: float
    try:
        delay = int(raw)
    except ValueError:
        try:
            delay = parsedate_to_datetime(raw).timestamp() - time.time()
        except (TypeError, ValueError):
            return None
    return max(0, min(ceil(delay), RETRY_AFTER_MAX))

def _read_json_response(resp: Any) -> Any:
    return json.loads(resp.read().decode("utf-8"))

def http_get_json(url: str, timeout: float) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return _read_json_response(resp)

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
                return _read_json_response(resp)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:500]
            e.close()  # release the response socket; body is already read
            last_err = f"HTTP {e.code}: {body}"
            kind = _retry_kind(e.code)
            wait = _retry_wait(kind, attempt) if kind else None
            if wait is None:
                raise TransportError(last_err) from e
            server_wait = _retry_after(e.headers)
            if server_wait is not None:
                wait = server_wait
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

def _json_object(data: Any, where: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise TransportError(f"unexpected response from {where}: expected a JSON object")
    return data

def _oc_error(data: dict[str, Any]) -> None:
    # opencode reports model/tool failures inside the message envelope
    info = data.get("info")
    err = info.get("error") if isinstance(info, dict) else None
    if not err:
        return
    if not isinstance(err, dict):
        raise TransportError(f"opencode: {err}")
    err_data = err.get("data")
    detail = ((err_data.get("message") if isinstance(err_data, dict) else None)
              or err.get("name") or "unknown error")
    raise TransportError(f"opencode: {detail}")

def _oc_text(parts: object, kind: str, where: str) -> str:
    # extract one part kind defensively: a malformed envelope must surface as
    # a transport error, not as an AttributeError from deep inside parsing
    if not isinstance(parts, list):
        raise TransportError(f"unexpected response from {where}: missing 'parts'")
    texts: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            raise TransportError(f"unexpected response from {where}: malformed part")
        text = part.get("text")
        if part.get("type") != kind:
            continue
        if not isinstance(text, str):
            raise TransportError(f"unexpected response from {where}: part has no text")
        texts.append(text)
    return "\n".join(texts).strip()

# ---------- Transport: OpenAI-compatible (stateless) ----------
def _api_key(agent_key: str, agent_cfg: AgentCfg) -> str:
    key = agent_cfg.get("api_key")
    if key:
        return key
    env_name = api_key_env(agent_key)
    key = os.environ.get(env_name, "")
    if key:
        return key
    raise TransportError(
        f"API key for agent '{agent_key}' is missing. Set it in "
        f"config.json or env {env_name}."
    )

def _chat_content(data: Any, where: str) -> str:
    choices = _json_object(data, where).get("choices")
    if not isinstance(choices, list) or not choices:
        raise TransportError(f"unexpected response from {where}: missing 'choices'")
    first = choices[0]
    msg = first.get("message") if isinstance(first, dict) else None
    if not isinstance(msg, dict):
        raise TransportError(f"unexpected response from {where}: missing 'choices[0].message'")
    content = msg.get("content") or msg.get("reasoning_content") or ""
    if not isinstance(content, str):
        raise TransportError(f"unexpected response from {where}: message content is not text")
    return content

def chat_completion(agent_key: str, agent_cfg: AgentCfg, system_prompt: str,
                    messages: list[dict[str, str]], role: str | None = None) -> str:
    # role decides the default temperature: a judging call must stay
    # deterministic even when it falls back to a debater's agent config
    role = role or agent_key
    api_key = _api_key(agent_key, agent_cfg)
    payload = {
        "model": agent_cfg["model"],
        "messages": [{"role": "system", "content": system_prompt}] + messages,
        # the judge should be deterministic; debaters benefit from some variety
        "temperature": agent_cfg.get("temperature", default_temperature(role)),
    }
    where = agent_cfg["api_url"]
    data = post_json(where, payload, headers={"Authorization": f"Bearer {api_key}"})
    content = strip_control_tokens(_chat_content(data, where))
    if not content:
        raise TransportError(f"empty response from model '{agent_cfg['model']}'")
    return content

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
    where = oc_base_url(agent_cfg) + "/session"
    data = post_json(where, payload, timeout=TIMEOUT_SESSION)
    session_id = _json_object(data, where).get("id")
    if not isinstance(session_id, str) or not session_id:
        raise TransportError(f"unexpected response from {where}: missing 'id'")
    return session_id

def _message_id(session_id: str, turn_index: int, text: str) -> str:
    # content-derived id: a retry with the identical payload reuses the id and
    # the server dedupes it instead of appending the same turn twice
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return MESSAGE_ID_TPL.format(sid=session_id[4:16], turn=turn_index, digest=digest)

def oc_send(agent_cfg: AgentCfg, session_id: str, system_prompt: str, text: str,
            agent_name: str | None = None, turn_index: int | None = None) -> str:
    payload = {
        "system": system_prompt,
        "parts": [{"type": "text", "text": text}],
    }
    if agent_name:
        payload["agent"] = agent_name
    if turn_index is not None:
        # content-derived: an identical retry dedupes, changed text is fresh
        payload["messageID"] = _message_id(session_id, turn_index, text)
    where = f"{oc_base_url(agent_cfg)}/session/{session_id}/message"
    data = _json_object(post_json(where, payload, timeout=TIMEOUT_MESSAGE), where)
    _oc_error(data)
    parts = data.get("parts")
    out = _oc_text(parts, "text", where)
    if not out:
        out = _oc_text(parts, "reasoning", where)
    out = strip_control_tokens(out)
    if not out:
        raise TransportError(f"opencode: empty response from model '{agent_cfg['model']}'")
    return out

def _delete_session(base_url: str, session_id: str) -> None:
    try:
        req = urllib.request.Request(
            f"{base_url}/session/{session_id}", method="DELETE"
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT_LIST):
            pass
    except Exception:
        pass

def oc_delete(agent_cfg: AgentCfg, session_id: str) -> None:
    _delete_session(oc_base_url(agent_cfg), session_id)

def _session_activity(session: dict, title: str) -> tuple[float, bool]:
    # (last activity, server-reported). The server-reported time.updated
    # refreshes on every message; the title only carries the creation stamp
    # of older servers, and unparseable titles count as old orphans
    stamp = session.get("time")
    updated = stamp.get("updated") if isinstance(stamp, dict) else None
    if isinstance(updated, (int, float)) and updated > 0:
        return updated / 1000.0, True
    try:
        return time.mktime(time.strptime(title.rsplit(" ", 1)[-1], TS_FORMAT)), False
    except ValueError:
        return 0.0, False

def oc_sweep(base_url: str, prefix: str = SESSION_TITLE_PREFIX,
             min_age: float = SWEEP_MIN_AGE,
             title_min_age: float = SWEEP_TITLE_MIN_AGE) -> None:
    # best-effort: delete orphan sessions left by crashed runs. Recent
    # server-reported activity protects a live parallel run; a title-only
    # session (old server) is judged against the much longer title_min_age
    # because its creation stamp never refreshes
    try:
        data = http_get_json(base_url + "/session", TIMEOUT_LIST)
    except Exception:
        return
    if not isinstance(data, list):
        return
    now = time.time()
    for s in data:
        if not isinstance(s, dict):
            continue
        title = s.get("title")
        sid = s.get("id")
        if not (isinstance(title, str) and isinstance(sid, str) and sid):
            continue
        if not title.startswith(prefix):
            continue
        last, reported = _session_activity(s, title)
        limit = min_age if reported else title_min_age
        if now - last >= limit:
            _delete_session(base_url, sid)

def oc_agent_list(base_url: str, timeout: float = TIMEOUT_LIST) -> set[str] | None:
    # names of agents available on the server; None if unreachable
    try:
        data = http_get_json(base_url + "/agent", timeout)
    except Exception:
        return None
    if not isinstance(data, list):
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
