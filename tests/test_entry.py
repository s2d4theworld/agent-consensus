# Tests for the CLI entry point: argument dispatch and menu behavior.
import json
import os
import unittest
from unittest import mock

import ac_config
import agent_consensus
from helpers import (
    FakeServerMixin,
    QuietRunMixin,
    captured_stdout,
    fake_argv,
    fake_stdin,
    oc_cfg,
)


class TestEntry(FakeServerMixin, QuietRunMixin, unittest.TestCase):
    def test_menu_quits_on_3(self):
        ac_config.CONFIG_PATH = os.path.join(self._tdir, "missing.json")
        with fake_stdin("3\n"), fake_argv(), captured_stdout() as out:
            agent_consensus.main()
        self.assertIn("No config yet", out.getvalue())
        self.assertIn("[2] Start discussion", out.getvalue())

    def test_menu_dispatches_setup_and_rejects_junk(self):
        # menu choice 1 must run the wizard; junk input must be reported, not
        # silently re-loop
        with mock.patch.object(agent_consensus, "setup_wizard",
                               return_value=True) as wizard:
            with fake_stdin("x\n1\n3\n"), fake_argv(), captured_stdout() as out:
                agent_consensus.main()
        wizard.assert_called_once_with()
        self.assertIn("Invalid choice.", out.getvalue())

    def test_menu_runs_discussion(self):
        # the primary menu action: choice 2 debates with the saved config,
        # waits for Enter, and returns to the menu
        srv = self.fake_server()
        ac_config.CONFIG_PATH = os.path.join(self._tdir, "config.json")
        with open(ac_config.CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(oc_cfg(srv.base), f)
        with fake_stdin("2\nA vs B\n\n3\n"), fake_argv(), \
                captured_stdout() as out:
            agent_consensus.main()
        text = out.getvalue()
        self.assertIn("Config found:", text)
        self.assertIn("Final status : NO CONSENSUS (hit max rounds)", text)
        self.assertIn("Press Enter to return to the menu...", text)

    def test_setup_aborts_on_eof(self):
        # `setup` with no stdin must not traceback, but must exit non-zero
        with fake_stdin(""), fake_argv("setup"), captured_stdout() as out, \
                self.assertRaises(SystemExit) as cm:
            agent_consensus.main()
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("Setup aborted", out.getvalue())

    def test_setup_save_failure_exits_1(self):
        with fake_argv("setup"), mock.patch.object(
                agent_consensus, "setup_wizard", return_value=False):
            with self.assertRaises(SystemExit) as cm:
                agent_consensus.main()
        self.assertEqual(cm.exception.code, 1)

    def test_install_agent_failure_exits_1(self):
        with fake_argv("install-agent"), mock.patch.object(
                agent_consensus, "ensure_agent_installed", return_value=False):
            with self.assertRaises(SystemExit) as cm:
                agent_consensus.main()
        self.assertEqual(cm.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
