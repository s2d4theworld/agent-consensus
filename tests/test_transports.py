# Tests for ac_transports: HTTP retry policy, agent-list cache, sweep guards
# and the transport error envelopes.
import time
import unittest
from typing import Any, Callable
from unittest import mock
from urllib.error import URLError

# helpers first: importing it puts the repo root on sys.path for ac_*
from helpers import (
    FakeServerMixin,
    captured_stdout,
    closed_port,
    env_var,
    oc_cfg,
)

import ac_config
import ac_transports
from ac_errors import TransportError


def stateless_agent(url: str = "http://127.0.0.1:1/v1/chat/completions",
                    ) -> ac_config.AgentCfg:
    # a minimal stateless agent; tests override what they exercise
    return {"name": "X", "transport": "openai", "api_key": "k", "model": "m",
            "api_url": url}


class TestPostJsonRetries(FakeServerMixin, unittest.TestCase):
    def setUp(self):
        # post_json prints retry notices; keep them out of the unittest output
        self.enterContext(captured_stdout())

    def test_post_json_retries_transient_error(self):
        srv = self.fake_server()
        srv.retry_codes.append(429)
        with mock.patch.object(ac_transports, "time") as fake_time:
            data = ac_transports.post_json(srv.base + "/session", {"title": "t"}, timeout=5)
        self.assertTrue(data["id"].startswith("ses_"))
        fake_time.sleep.assert_called_once_with(45)

    def test_post_json_retries_500_once_only(self):
        srv = self.fake_server()
        srv.retry_codes.extend([500, 500])
        with mock.patch.object(ac_transports, "time") as fake_time:
            with self.assertRaisesRegex(TransportError, "HTTP 500"):
                ac_transports.post_json(srv.base + "/session", {}, timeout=5)
        fake_time.sleep.assert_called_once_with(15)

    def test_post_json_honors_retry_after(self):
        srv = self.fake_server()
        srv.retry_codes.append(429)
        srv.retry_after = "7"
        with mock.patch.object(ac_transports, "time") as fake_time:
            ac_transports.post_json(srv.base + "/session", {"title": "t"}, timeout=5)
        fake_time.sleep.assert_called_once_with(7)

    def test_post_json_retry_budget_is_not_extended_by_hints(self):
        # once the policy is exhausted, a Retry-After hint must not add a
        # sleep after the last allowed request has already failed
        srv = self.fake_server()
        srv.retry_codes.extend([429, 429, 429, 429])
        srv.retry_after = "7"
        with mock.patch.object(ac_transports, "time") as fake_time:
            with self.assertRaisesRegex(TransportError, "HTTP 429"):
                ac_transports.post_json(srv.base + "/session", {}, timeout=5)
        self.assertEqual([call.args[0] for call in fake_time.sleep.call_args_list],
                         [7, 7, 7])

    def test_retry_after_parsing_and_cap(self):
        stamp = time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime(1_700_000_030))
        with mock.patch.object(ac_transports, "time") as fake_time:
            fake_time.time.return_value = 1_700_000_000.0
            self.assertEqual(ac_transports._retry_after({"Retry-After": stamp}), 30)
        self.assertEqual(ac_transports._retry_after({"Retry-After": "999999"}),
                         ac_config.RETRY_AFTER_MAX)
        # without a usable hint the caller's policy wait (default) survives
        self.assertEqual(ac_transports._retry_after({"Retry-After": "soon"}, 45), 45)
        self.assertEqual(ac_transports._retry_after({}, 45), 45)
        self.assertIsNone(ac_transports._retry_after({}))
        self.assertIsNone(ac_transports._retry_after(None))

    def test_post_json_does_not_retry_client_error(self):
        srv = self.fake_server()
        srv.retry_codes.append(400)
        with mock.patch.object(ac_transports, "time") as fake_time:
            with self.assertRaisesRegex(TransportError, "HTTP 400"):
                ac_transports.post_json(srv.base + "/session", {}, timeout=5)
        fake_time.sleep.assert_not_called()

    def test_post_json_retries_malformed_response(self):
        # a proxy error page instead of JSON is as transient as a reset
        srv = self.fake_server()
        srv.raw_response = (b"<html>bad gateway</html>", 0)
        with mock.patch.object(ac_transports, "time") as fake_time:
            data = ac_transports.post_json(srv.base + "/session", {"title": "t"}, timeout=5)
        self.assertTrue(data["id"].startswith("ses_"))
        fake_time.sleep.assert_called_once_with(20)

    def test_post_json_retries_truncated_response(self):
        # declared Content-Length > body: IncompleteRead, then a clean retry
        srv = self.fake_server()
        srv.raw_response = (b'{"id": "ses_truncated1test2018"}', 7)
        with mock.patch.object(ac_transports, "time") as fake_time:
            data = ac_transports.post_json(srv.base + "/session", {"title": "t"}, timeout=5)
        self.assertTrue(data["id"].startswith("ses_fake"))
        fake_time.sleep.assert_called_once_with(20)

    def test_post_json_retries_connection_failures(self):
        url = f"http://127.0.0.1:{closed_port()}/session"
        with mock.patch.object(ac_transports, "time") as fake_time:
            with self.assertRaisesRegex(TransportError, "connection:"):
                ac_transports.post_json(url, {}, timeout=0.5)
        self.assertEqual([call.args[0] for call in fake_time.sleep.call_args_list],
                         [20, 40, 60])


class TestTransports(FakeServerMixin, unittest.TestCase):
    def tearDown(self):
        ac_transports._AGENT_LIST_CACHE.clear()

    def assert_malformed(self, payloads: list[object], call: Callable[..., str],
                         *args: Any) -> None:
        # every payload must be rejected as an unexpected response
        for payload in payloads:
            with mock.patch.object(ac_transports, "post_json",
                                   return_value=payload):
                with self.assertRaisesRegex(TransportError, "unexpected response"):
                    call(*args)

    def test_agent_list_cache_ttl(self):
        calls = {"n": 0}

        def fake_list(_base_url, _timeout=ac_config.TIMEOUT_LIST):
            calls["n"] += 1
            return {"debater"}

        with mock.patch.object(ac_transports, "oc_agent_list", fake_list):
            ac_transports._AGENT_LIST_CACHE.clear()
            ac_transports.oc_agent_list_cached("http://x")
            ac_transports.oc_agent_list_cached("http://x")
            self.assertEqual(calls["n"], 1)
            ac_transports._AGENT_LIST_CACHE["http://x"] = (time.time() - 60, None)
            ac_transports.oc_agent_list_cached("http://x")
            self.assertEqual(calls["n"], 2)

    def test_agent_list_unreachable_returns_none(self):
        with mock.patch.object(ac_transports, "http_get_json",
                               side_effect=URLError("refused")):
            self.assertIsNone(ac_transports.oc_agent_list("http://127.0.0.1:1"))

    def test_chat_completion_requires_api_key(self):
        agent = stateless_agent()
        del agent["api_key"]
        with self.assertRaisesRegex(TransportError, "API key for agent 'nokeyrole'"):
            ac_transports.chat_completion("nokeyrole", agent, "sys", [])

    def test_chat_completion_reads_api_key_from_env(self):
        # an empty api_key means "read CONSENSUS_API_KEY_<ROLE>": the wizard
        # prints that name, so the transport must send that value as the token
        agent = stateless_agent("http://x/v1/chat/completions")
        agent["api_key"] = ""
        reply = {"choices": [{"message": {"content": "ok"}}]}
        with env_var(ac_config.api_key_env("proposer"), "env-secret"), \
                mock.patch.object(ac_transports, "post_json",
                                  return_value=reply) as post:
            text = ac_transports.chat_completion("proposer", agent, "sys", [])
        self.assertEqual(text, "ok")
        call = post.call_args
        assert call is not None
        self.assertEqual(call.kwargs["headers"]["Authorization"],
                         "Bearer env-secret")

    def test_oc_sweep_ignores_malformed_and_foreign_sessions(self):
        srv = self.fake_server()
        srv.session_list[:] = [{"id": "ses_keep0001test2018", "title": "other"}]
        ac_transports.oc_sweep(srv.base)
        self.assertEqual(srv.deletes, [])
        srv.session_list = {}  # type: ignore[assignment]  # malformed: must not crash the sweep
        ac_transports.oc_sweep(srv.base)
        self.assertEqual(srv.deletes, [])
        srv.session_list = ["junk"]  # type: ignore[list-item]  # malformed item
        ac_transports.oc_sweep(srv.base)
        self.assertEqual(srv.deletes, [])
        srv.session_list[:] = [{"id": "ses_keep0001test2018", "title": 5}]
        ac_transports.oc_sweep(srv.base)
        self.assertEqual(srv.deletes, [])

    def test_oc_sweep_prefers_last_activity(self):
        srv = self.fake_server()
        now_ms = int(time.time() * 1000)
        old_ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time() - 3600))
        srv.session_list[:] = [
            # live parallel run: old title, recent activity -> kept
            {"id": "ses_live0001test2018", "title": f"agent-consensus proposer {old_ts}",
             "time": {"created": now_ms - 3600_000, "updated": now_ms}},
            # crashed run: old title, stale activity -> swept
            {"id": "ses_dead0001test2018", "title": f"agent-consensus critic {old_ts}",
             "time": {"created": now_ms - 3600_000, "updated": now_ms - 3600_000}},
        ]
        ac_transports.oc_sweep(srv.base)
        self.assertNotIn("/session/ses_live0001test2018", srv.deletes)
        self.assertIn("/session/ses_dead0001test2018", srv.deletes)

    def test_oc_sweep_title_only_sessions_use_long_age(self):
        # an older server reports no time.updated: the creation stamp never
        # refreshes, so a long-running live session (older than min_age) must
        # survive, while a genuinely stale one is still swept
        srv = self.fake_server()
        running_ts = time.strftime("%Y%m%d_%H%M%S",
                                   time.localtime(time.time() - 3600))
        stale_ts = time.strftime(
            "%Y%m%d_%H%M%S",
            time.localtime(time.time() - ac_config.SWEEP_TITLE_MIN_AGE - 60))
        srv.session_list[:] = [
            {"id": "ses_running01test2018",
             "title": f"agent-consensus proposer {running_ts}"},
            {"id": "ses_stale0001test2018",
             "title": f"agent-consensus critic {stale_ts}"},
        ]
        ac_transports.oc_sweep(srv.base)
        self.assertNotIn("/session/ses_running01test2018", srv.deletes)
        self.assertIn("/session/ses_stale0001test2018", srv.deletes)

    def test_oc_send_error_envelope(self):
        srv = self.fake_server()
        srv.message_error = {"name": "ProviderError", "data": {"message": "boom"}}
        agent = oc_cfg(srv.base)["agents"]["proposer"]
        with self.assertRaisesRegex(TransportError, "opencode: boom"):
            ac_transports.oc_send(agent, "ses_fake0001test2018", "sys", "text", turn_index=0)

    def test_strip_control_tokens(self):
        # leaked control tokens (seen from real free models) must be removed
        # without touching ordinary text, pipes and angle brackets included
        self.assertEqual(ac_transports.strip_control_tokens(
            "  <|constrain|>YES - ok  "), "YES - ok")
        self.assertEqual(ac_transports.strip_control_tokens(
            "a <|channel|> b"), "a  b")
        self.assertEqual(ac_transports.strip_control_tokens(
            "100 < 200 | not a token"), "100 < 200 | not a token")

    def test_empty_model_reply_is_an_error(self):
        # empty output must fail loudly instead of becoming a silent blank turn
        srv = self.fake_server()
        srv.reply_override = ""
        agent = oc_cfg(srv.base)["agents"]["proposer"]
        with self.assertRaisesRegex(TransportError, "empty response"):
            ac_transports.oc_send(agent, "ses_fake0001test2018", "sys", "text", turn_index=0)
        chat_agent = stateless_agent(srv.base + "/v1/chat/completions")
        with self.assertRaisesRegex(TransportError, "empty response"):
            ac_transports.chat_completion("proposer", chat_agent, "sys", [])

    def test_chat_completion_rejects_malformed_response(self):
        agent = stateless_agent()
        payloads: list[object] = [
            [], {},
            {"choices": []},
            {"choices": [{}]},
            {"choices": [{"message": {"content": 5}}]},
        ]
        self.assert_malformed(payloads, ac_transports.chat_completion,
                              "proposer", agent, "sys", [])

    def test_chat_completion_ignores_reasoning_content(self):
        # empty content with reasoning_content set is a chain-of-thought
        # dead end, not an answer: the turn must fail as empty, never record
        # the model's internal monologue as reply text
        agent = stateless_agent("http://x/v1/chat/completions")
        payload = {"choices": [{"message": {"content": "",
                                            "reasoning_content": "thought it through"}}]}
        with mock.patch.object(ac_transports, "post_json", return_value=payload):
            with self.assertRaises(TransportError) as cm:
                ac_transports.chat_completion("proposer", agent, "sys", [])
        self.assertIn("empty response", str(cm.exception))

    def test_oc_session_create_rejects_malformed_response(self):
        agent = oc_cfg("http://x")["agents"]["proposer"]
        payloads: list[object] = [[], {}, {"id": 5}, {"id": ""}]
        self.assert_malformed(payloads, ac_transports.oc_session_create,
                              agent, "t")

    def test_oc_session_create_sends_provider_and_model(self):
        # "provider/model" splits into providerID/id; a slash-less model
        # defaults to the opencode provider
        srv = self.fake_server()
        agent = oc_cfg(srv.base)["agents"]["proposer"]
        ac_transports.oc_session_create(agent, "t")
        agent["model"] = "beam-free"
        ac_transports.oc_session_create(agent, "t")
        self.assertEqual([s["model"] for s in srv.sessions],
                         [{"id": "x", "providerID": "opencode"},
                          {"id": "beam-free", "providerID": "opencode"}])

    def test_oc_send_rejects_malformed_response(self):
        agent = oc_cfg("http://x")["agents"]["proposer"]
        payloads: list[object] = [
            {}, {"parts": [5]},
            {"parts": [{"type": "text"}]},
            {"parts": [{"type": "text", "text": 5}]},
        ]
        self.assert_malformed(payloads, ac_transports.oc_send,
                              agent, "ses_fake0001test2018", "sys", "text")

    def test_oc_send_ignores_reasoning_only_parts(self):
        # no text part at all, only reasoning parts: chain-of-thought is not
        # the reply; a turn that produced no text must fail as empty
        agent = oc_cfg("http://x")["agents"]["proposer"]
        payload = {"parts": [{"type": "reasoning", "text": "thinking..."}]}
        with mock.patch.object(ac_transports, "post_json", return_value=payload):
            with self.assertRaises(TransportError) as cm:
                ac_transports.oc_send(agent, "ses_fake0001test2018", "sys", "text")
        self.assertIn("opencode: empty response", str(cm.exception))

    def test_oc_send_message_id_is_content_derived(self):
        # identical resend (lost-response retry) -> replayed by the server;
        # different content -> fresh id, even at an index that was used before
        srv = self.fake_server()
        agent = oc_cfg(srv.base)["agents"]["proposer"]
        sid = "ses_fake0001test2018"
        ac_transports.oc_send(agent, sid, "sys", "same text", turn_index=0)
        ac_transports.oc_send(agent, sid, "sys", "same text", turn_index=0)
        ac_transports.oc_send(agent, sid, "sys", "other text", turn_index=0)
        self.assertEqual(len(srv.msgs), 2)
        self.assertEqual(len(srv.dupes), 1)
        ids = [m["messageID"] for m in srv.msgs]
        self.assertRegex(ids[0], r"^msg_fake0001test_t0_[0-9a-f]{8}$")
        self.assertNotEqual(ids[0], ids[1])


if __name__ == "__main__":
    unittest.main()
