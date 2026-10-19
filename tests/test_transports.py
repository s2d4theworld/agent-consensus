# Tests for ac_transports: HTTP retry policy, agent-list cache, sweep guards
# and the transport error envelopes.
import time
import unittest
from unittest import mock
from urllib.error import URLError

import ac_config
import ac_transports
from ac_errors import TransportError
from helpers import FakeServer, closed_port, oc_cfg


class TestPostJsonRetries(unittest.TestCase):
    def test_post_json_retries_transient_error(self):
        srv = FakeServer()
        self.addCleanup(srv.stop)
        srv.retry_codes.append(429)
        with mock.patch.object(ac_transports, "time") as fake_time:
            data = ac_transports.post_json(srv.base + "/session", {"title": "t"}, timeout=5)
        self.assertTrue(data["id"].startswith("ses_"))
        fake_time.sleep.assert_called_once_with(45)

    def test_post_json_retries_500_once_only(self):
        srv = FakeServer()
        self.addCleanup(srv.stop)
        srv.retry_codes.extend([500, 500])
        with mock.patch.object(ac_transports, "time") as fake_time:
            with self.assertRaisesRegex(TransportError, "HTTP 500"):
                ac_transports.post_json(srv.base + "/session", {}, timeout=5)
        fake_time.sleep.assert_called_once_with(15)

    def test_post_json_honors_retry_after(self):
        srv = FakeServer()
        self.addCleanup(srv.stop)
        srv.retry_codes.append(429)
        srv.retry_after = "7"
        with mock.patch.object(ac_transports, "time") as fake_time:
            ac_transports.post_json(srv.base + "/session", {"title": "t"}, timeout=5)
        fake_time.sleep.assert_called_once_with(7)

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
        srv = FakeServer()
        self.addCleanup(srv.stop)
        srv.retry_codes.append(400)
        with mock.patch.object(ac_transports, "time") as fake_time:
            with self.assertRaisesRegex(TransportError, "HTTP 400"):
                ac_transports.post_json(srv.base + "/session", {}, timeout=5)
        fake_time.sleep.assert_not_called()

    def test_post_json_retries_malformed_response(self):
        # a proxy error page instead of JSON is as transient as a reset
        srv = FakeServer()
        self.addCleanup(srv.stop)
        srv.raw_response = (b"<html>bad gateway</html>", 0)
        with mock.patch.object(ac_transports, "time") as fake_time:
            data = ac_transports.post_json(srv.base + "/session", {"title": "t"}, timeout=5)
        self.assertTrue(data["id"].startswith("ses_"))
        fake_time.sleep.assert_called_once_with(20)

    def test_post_json_retries_truncated_response(self):
        # declared Content-Length > body: IncompleteRead, then a clean retry
        srv = FakeServer()
        self.addCleanup(srv.stop)
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


class TestTransports(unittest.TestCase):
    def tearDown(self):
        ac_transports._AGENT_LIST_CACHE.clear()

    def test_agent_list_cache_ttl(self):
        calls = {"n": 0}

        def fake_list(_base_url, _timeout=ac_config.TIMEOUT_LIST):
            calls["n"] += 1
            return {"debater"}

        orig = ac_transports.oc_agent_list
        ac_transports.oc_agent_list = fake_list
        try:
            ac_transports._AGENT_LIST_CACHE.clear()
            ac_transports.oc_agent_list_cached("http://x")
            ac_transports.oc_agent_list_cached("http://x")
            self.assertEqual(calls["n"], 1)
            ac_transports._AGENT_LIST_CACHE["http://x"] = (time.time() - 60, None)
            ac_transports.oc_agent_list_cached("http://x")
            self.assertEqual(calls["n"], 2)
        finally:
            ac_transports.oc_agent_list = orig

    def test_agent_list_unreachable_returns_none(self):
        with mock.patch.object(ac_transports, "http_get_json",
                               side_effect=URLError("refused")):
            self.assertIsNone(ac_transports.oc_agent_list("http://127.0.0.1:1"))

    def test_chat_completion_requires_api_key(self):
        agent: ac_config.AgentCfg = {
            "name": "X", "transport": "openai", "model": "m",
            "api_url": "http://127.0.0.1:1/v1/chat/completions"}
        with self.assertRaisesRegex(TransportError, "API key for agent 'nokeyrole'"):
            ac_transports.chat_completion("nokeyrole", agent, "sys", [])

    def test_oc_sweep_ignores_malformed_and_foreign_sessions(self):
        srv = FakeServer()
        self.addCleanup(srv.stop)
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
        srv = FakeServer()
        self.addCleanup(srv.stop)
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
        srv = FakeServer()
        self.addCleanup(srv.stop)
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
        srv = FakeServer()
        self.addCleanup(srv.stop)
        srv.message_error = {"name": "ProviderError", "data": {"message": "boom"}}
        agent = oc_cfg(srv.base)["agents"]["proposer"]
        with self.assertRaisesRegex(TransportError, "opencode: boom"):
            ac_transports.oc_send(agent, "ses_fake0001test2018", "sys", "text", turn_index=0)

    def test_empty_model_reply_is_an_error(self):
        # empty output must fail loudly instead of becoming a silent blank turn
        srv = FakeServer()
        self.addCleanup(srv.stop)
        srv.reply_override = ""
        agent = oc_cfg(srv.base)["agents"]["proposer"]
        with self.assertRaisesRegex(TransportError, "empty response"):
            ac_transports.oc_send(agent, "ses_fake0001test2018", "sys", "text", turn_index=0)
        chat_agent: ac_config.AgentCfg = {
            "name": "X", "transport": "openai", "api_key": "k", "model": "m",
            "api_url": srv.base + "/v1/chat/completions"}
        with self.assertRaisesRegex(TransportError, "empty response"):
            ac_transports.chat_completion("proposer", chat_agent, "sys", [])

    def test_chat_completion_rejects_malformed_response(self):
        agent: ac_config.AgentCfg = {
            "name": "X", "transport": "openai", "api_key": "k", "model": "m",
            "api_url": "http://127.0.0.1:1/v1/chat/completions"}
        payloads: list[object] = [
            [], {},
            {"choices": []},
            {"choices": [{}]},
            {"choices": [{"message": {"content": 5}}]},
        ]
        for payload in payloads:
            with mock.patch.object(ac_transports, "post_json", return_value=payload):
                with self.assertRaisesRegex(TransportError, "unexpected response"):
                    ac_transports.chat_completion("proposer", agent, "sys", [])

    def test_oc_session_create_rejects_malformed_response(self):
        agent = oc_cfg("http://x")["agents"]["proposer"]
        payloads: list[object] = [[], {}, {"id": 5}, {"id": ""}]
        for payload in payloads:
            with mock.patch.object(ac_transports, "post_json", return_value=payload):
                with self.assertRaisesRegex(TransportError, "unexpected response"):
                    ac_transports.oc_session_create(agent, "t")

    def test_oc_send_rejects_malformed_response(self):
        agent = oc_cfg("http://x")["agents"]["proposer"]
        payloads: list[object] = [
            {}, {"parts": [5]},
            {"parts": [{"type": "text"}]},
            {"parts": [{"type": "text", "text": 5}]},
        ]
        for payload in payloads:
            with mock.patch.object(ac_transports, "post_json", return_value=payload):
                with self.assertRaisesRegex(TransportError, "unexpected response"):
                    ac_transports.oc_send(agent, "ses_fake0001test2018", "sys", "text")

    def test_oc_send_message_id_is_content_derived(self):
        # identical resend (lost-response retry) -> replayed by the server;
        # different content -> fresh id, even at an index that was used before
        srv = FakeServer()
        self.addCleanup(srv.stop)
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
