# agent-consensus — HTTP helpers and the two transports:
#   opencode = local `opencode serve` sessions (stateful: server keeps history,
#              the script only sends new messages)
#   openai   = any OpenAI-compatible /v1/chat/completions endpoint (stateless:
#              the script re-sends the full conversation each call)
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
    Config,
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
    TIMEOUT_MENU,
    TIMEOUT_SESSION,
    is_opencode,
)
from ac_errors import TransportError

# ---------- HTTP helpers (retries + JSON) ----------
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
    attempts = max(len(RETRY_HTTP_WAITS), len(RETRY_CONN_WAITS)) + 1
    last_err: str | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:500]
            e.close()  # release the response socket; body is already read
            last_err = f"HTTP {e.code}: {body}"
            # transient upstream errors get the 45/90/135s backoff; a 500 is
            # often deterministic, so retry it once, briefly
            if e.code in RETRY_HTTP_CODES and attempt < len(RETRY_HTTP_WAITS):
                wait = RETRY_HTTP_WAITS[attempt]
            elif e.code == 500 and attempt < len(RETRY_500_WAITS):
                wait = RETRY_500_WAITS[attempt]
            else:
                raise TransportError(last_err) from e
            print(f"    [retry {attempt+1}] {e.code}, waiting {wait}s...", flush=True)
            time.sleep(wait)
        except (TimeoutError, ConnectionError, urllib.error.URLError, socket.timeout) as e:
            last_err = f"connection: {e}"
            if attempt >= len(RETRY_CONN_WAITS):
                raise TransportError(last_err) from e
            wait = RETRY_CONN_WAITS[attempt]
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
        # deterministic id (index of the first new turn in the session): a
        # retry after a lost response is deduped by the server instead of
        # appending the same turn twice (verified)
        payload["messageID"] = MESSAGE_ID_TPL.format(sid=session_id[4:16],
                                                     turn=turn_index)
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

def oc_sweep(base_url: str, prefix: str = SESSION_TITLE_PREFIX,
             min_age: float = SWEEP_MIN_AGE) -> None:
    # best-effort: delete orphan sessions left by crashed runs. Sessions with
    # recent activity are skipped, so a live parallel run is not disturbed;
    # without a usable timestamp the title's creation time is the fallback,
    # and unparseable titles count as old orphans.
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
        when = 0.0
        stamp = s.get("time")
        updated = stamp.get("updated") if isinstance(stamp, dict) else None
        if isinstance(updated, (int, float)) and updated > 0:
            when = updated / 1000.0  # ms epoch: last server-side activity
        else:
            try:  # older servers: fall back to the title's creation time
                when = time.mktime(time.strptime(title.rsplit(" ", 1)[-1], "%Y%m%d_%H%M%S"))
            except ValueError:
                pass
        if now - when >= min_age:
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

def restricted_agents_by_base(cfg: Config) -> dict[str, set[str]]:
    # {base_url: {agent names}} of restricted (websearch) agents per server
    by_base: dict[str, set[str]] = {}
    for a in cfg["agents"].values():
        if not is_opencode(a):
            continue
        name = a.get("agent")
        if name:
            by_base.setdefault(oc_base_url(a), set()).add(name)
    return by_base

def restricted_agent_lines(cfg: Config) -> list[str]:
    # menu status lines for the restricted (websearch) agents per server
    by_base = restricted_agents_by_base(cfg)
    if not by_base:
        return ["  Restricted agent: none configured (no websearch for debaters)"]
    lines = []
    for base, names in sorted(by_base.items()):
        avail = oc_agent_list_cached(base, timeout=TIMEOUT_MENU)
        for name in sorted(names):
            if avail is None:
                lines.append(f"  Restricted agent '{name}': {base} unreachable "
                             "(is `opencode serve` running?)")
            elif name in avail:
                lines.append(f"  Restricted agent '{name}': active on {base}")
            else:
                lines.append(f"  Restricted agent '{name}': missing on {base} "
                             "(run: python agent_consensus.py install-agent, then "
                             "restart opencode serve)")
    return lines
