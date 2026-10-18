# Tests for the CLI entry point: argument dispatch and menu behavior.
import contextlib
import io
import os
import sys
import unittest

import ac_config
import agent_consensus
from helpers import QuietRunMixin


class TestEntry(QuietRunMixin, unittest.TestCase):
    def test_menu_quits_on_3(self):
        ac_config.CONFIG_PATH = os.path.join(self._tdir, "missing.json")
        old_stdin, old_argv = sys.stdin, sys.argv
        sys.stdin = io.StringIO("3\n")
        sys.argv = ["agent_consensus.py"]
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                agent_consensus.main()
        finally:
            sys.stdin, sys.argv = old_stdin, old_argv
        out = buf.getvalue()
        self.assertIn("No config yet", out)
        self.assertIn("[2] Start discussion", out)

    def test_setup_aborts_on_eof(self):
        # `setup` with no stdin must not traceback
        old_stdin, old_argv = sys.stdin, sys.argv
        sys.stdin = io.StringIO("")
        sys.argv = ["agent_consensus.py", "setup"]
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                agent_consensus.main()
        finally:
            sys.stdin, sys.argv = old_stdin, old_argv
        self.assertIn("Setup aborted", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
