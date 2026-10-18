# Tests for ac_transports: HTTP retry policy, agent-list cache, sweep guards
# and the transport error envelopes.
import time
import unittest
from unittest import mock
from urllib.error import URLError

import ac_config
import ac_transports
from ac_errors import TransportError
from helpers import FakeServer, oc_cfg


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

    def test_restricted_agent_lines_states(self):
        srv = FakeServer()
        self.addCleanup(srv.stop)
        cfg = oc_cfg(srv.base)
        lines = ac_transports.restricted_agent_lines(cfg)
        self.assertEqual(lines, [f"  Restricted agent 'debater': active on {srv.base}"])

        srv.agent_names.clear()
        ac_transports._AGENT_LIST_CACHE.clear()
        lines = ac_transports.restricted_agent_lines(cfg)
        self.assertEqual(len(lines), 1)
        self.assertIn("missing on", lines[0])

        cfg_down = oc_cfg("http://127.0.0.1:1")
        ac_transports._AGENT_LIST_CACHE.clear()
        with mock.patch.object(ac_transports, "oc_agent_list", return_value=None):
            lines = ac_transports.restricted_agent_lines(cfg_down)
        self.assertIn("unreachable", lines[0])

        cfg_none = oc_cfg(srv.base, agent=None)
        self.assertEqual(ac_transports.restricted_agent_lines(cfg_none),
                         ["  Restricted agent: none configured (no websearch for debaters)"])

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

    def test_post_json_does_not_retry_client_error(self):
        srv = FakeServer()
        self.addCleanup(srv.stop)
        srv.retry_codes.append(400)
        with mock.patch.object(ac_transports, "time") as fake_time:
            with self.assertRaisesRegex(TransportError, "HTTP 400"):
                ac_transports.post_json(srv.base + "/session", {}, timeout=5)
        fake_time.sleep.assert_not_called()

    def test_chat_completion_requires_api_key(self):
        agent = {"name": "X", "transport": "openai", "model": "m",
                 "api_url": "http://127.0.0.1:1/v1/chat/completions"}
        with self.assertRaisesRegex(TransportError, "API key for agent 'nokeyrole'"):
            ac_transports.chat_completion("nokeyrole", agent, "sys", [])

    def test_oc_sweep_ignores_malformed_and_foreign_sessions(self):
        srv = FakeServer()
        self.addCleanup(srv.stop)
        srv.session_list[:] = [{"id": "ses_keep0001test2018", "title": "other"}]
        ac_transports.oc_sweep(srv.base)
        self.assertEqual(srv.deletes, [])
        srv.session_list = {}  # a non-list response must not crash the sweep
        ac_transports.oc_sweep(srv.base)
        self.assertEqual(srv.deletes, [])

    def test_oc_send_error_envelope(self):
        srv = FakeServer()
        self.addCleanup(srv.stop)
        srv.message_error = {"name": "ProviderError", "data": {"message": "boom"}}
        agent = oc_cfg(srv.base)["agents"]["proposer"]
        with self.assertRaisesRegex(TransportError, "opencode: boom"):
            ac_transports.oc_send(agent, "ses_fake0001test2018", "sys", "text", turn_index=0)


if __name__ == "__main__":
    unittest.main()
