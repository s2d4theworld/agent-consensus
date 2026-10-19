# Tests for the CLI entry point: argument dispatch and menu behavior.
import contextlib
import io
import os
import unittest
from unittest import mock

import ac_config
import agent_consensus
from helpers import QuietRunMixin, fake_argv, fake_stdin


class TestEntry(QuietRunMixin, unittest.TestCase):
    def test_menu_quits_on_3(self):
        ac_config.CONFIG_PATH = os.path.join(self._tdir, "missing.json")
        buf = io.StringIO()
        with fake_stdin("3\n"), fake_argv(), contextlib.redirect_stdout(buf):
            agent_consensus.main()
        out = buf.getvalue()
        self.assertIn("No config yet", out)
        self.assertIn("[2] Start discussion", out)

    def test_setup_aborts_on_eof(self):
        # `setup` with no stdin must not traceback, but must exit non-zero
        buf = io.StringIO()
        with fake_stdin(""), fake_argv("setup"), contextlib.redirect_stdout(buf):
            with self.assertRaises(SystemExit) as cm:
                agent_consensus.main()
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("Setup aborted", buf.getvalue())

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
