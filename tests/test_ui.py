# Tests for ac_ui: setup wizard, prompts and menu status lines.
import contextlib
import io
import json
import os
import sys
import unittest
from unittest import mock

import ac_config
import ac_transports
import ac_ui
from helpers import FakeServer, QuietRunMixin, oc_cfg


class TestUi(QuietRunMixin, unittest.TestCase):
    def test_setup_wizard_writes_config(self):
        cfg_path = os.path.join(self._tdir, "config.json")
        ac_config.CONFIG_PATH = cfg_path
        stdin = "openai\n" + "\n" * 5 * 3 + "3\n"
        old_stdin = sys.stdin
        buf = io.StringIO()
        sys.stdin = io.StringIO(stdin)
        try:
            with contextlib.redirect_stdout(buf):
                ac_ui.setup_wizard()
        finally:
            sys.stdin = old_stdin
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.assertEqual(cfg["max_rounds"], 3)
        self.assertTrue(all(a["transport"] == "openai" for a in cfg["agents"].values()))
        self.assertEqual(cfg["agents"]["judge"]["temperature"], 0.0)

    def test_input_required_retries_on_empty(self):
        old_stdin = sys.stdin
        sys.stdin = io.StringIO("\n  \nx\n")
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                value = ac_ui.input_required("P: ")
        finally:
            sys.stdin = old_stdin
        self.assertEqual(value, "x")
        self.assertIn("(required)", buf.getvalue())

    def test_print_status_without_judge(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ac_ui.print_status(oc_cfg("http://127.0.0.1:1", with_judge=False, agent=None),
                               None)
        self.assertIn("Judge: none configured - proposer will judge", buf.getvalue())

    def test_restricted_agent_lines_states(self):
        srv = FakeServer()
        self.addCleanup(srv.stop)
        cfg = oc_cfg(srv.base)
        lines = ac_ui.restricted_agent_lines(cfg)
        self.assertEqual(lines, [f"  Restricted agent 'debater': active on {srv.base}"])

        srv.agent_names.clear()
        ac_transports._AGENT_LIST_CACHE.clear()
        lines = ac_ui.restricted_agent_lines(cfg)
        self.assertEqual(len(lines), 1)
        self.assertIn("missing on", lines[0])

        cfg_down = oc_cfg("http://127.0.0.1:1")
        ac_transports._AGENT_LIST_CACHE.clear()
        with mock.patch.object(ac_transports, "oc_agent_list", return_value=None):
            lines = ac_ui.restricted_agent_lines(cfg_down)
        self.assertIn("unreachable", lines[0])

        cfg_none = oc_cfg(srv.base, agent=None)
        self.assertEqual(ac_ui.restricted_agent_lines(cfg_none),
                         ["  Restricted agent: none configured (no websearch for debaters)"])


if __name__ == "__main__":
    unittest.main()
