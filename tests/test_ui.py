# Tests for ac_ui: setup wizard, prompts and menu status lines.
import contextlib
import io
import json
import os
import sys
import unittest

import ac_config
import ac_ui
from helpers import QuietRunMixin, oc_cfg


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


if __name__ == "__main__":
    unittest.main()
