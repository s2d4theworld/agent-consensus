# Tests for ac_ui: setup wizard, prompts and menu status lines.
import contextlib
import unittest
from typing import Any
from unittest import mock

# helpers first: importing it puts the repo root on sys.path for ac_*
from helpers import (
    FakeServerMixin,
    QuietRunMixin,
    WIZARD_STDIN_OPENAI,
    WIZARD_STDIN_OPENCODE,
    WIZARD_STDIN_OPENCODE_NO_AGENT,
    captured_stdout,
    fake_stdin,
    oc_cfg,
    read_json,
)

import ac_config
import ac_transports
import ac_ui


@contextlib.contextmanager
def wizard_run(answers: str):
    # drive the wizard on fake stdin/stdout; the installer is mocked so a
    # test can never write to the real HOME
    with mock.patch.object(ac_ui, "ensure_agent_installed",
                           return_value=True), \
            fake_stdin(answers), captured_stdout() as out:
        yield out


class TestUi(FakeServerMixin, QuietRunMixin, unittest.TestCase):
    def run_wizard(self, answers: str) -> tuple[Any, str]:
        # answer the wizard with a fresh CONFIG_PATH;
        # return the saved config and the captured output
        cfg_path = self.temp_config()
        with wizard_run(answers) as out:
            self.assertTrue(ac_ui.setup_wizard())
        return read_json(cfg_path), out.getvalue()

    def test_setup_wizard_writes_config(self):
        cfg, _ = self.run_wizard(WIZARD_STDIN_OPENAI)
        self.assertEqual(cfg["max_rounds"], 3)
        self.assertTrue(all(a["transport"] == "openai" for a in cfg["agents"].values()))
        self.assertEqual(cfg["agents"]["judge"]["temperature"], 0.0)

    def test_setup_wizard_reports_save_failure(self):
        # unwritable target (missing directory): wizard must return False
        self.temp_config("no/config.json")
        with wizard_run(WIZARD_STDIN_OPENAI) as out:
            self.assertFalse(ac_ui.setup_wizard())
        self.assertIn("cannot save", out.getvalue())

    def test_wizard_opencode_dash_omits_agent(self):
        # '-' at the restricted-agent prompt means "no agent": the saved role
        # must have no agent key, so it runs with the server's full tool set
        cfg, _ = self.run_wizard(WIZARD_STDIN_OPENCODE_NO_AGENT)
        for key in (ac_config.ROLE_PROPOSER, ac_config.ROLE_CRITIC,
                    ac_config.ROLE_JUDGE):
            self.assertNotIn("agent", cfg["agents"][key])

    def test_wizard_warns_on_unknown_transport(self):
        # a mistyped transport must warn and fall back to opencode
        answers = "x\n" + WIZARD_STDIN_OPENCODE.split("\n", 1)[1]
        cfg, out = self.run_wizard(answers)
        self.assertIn("unknown transport", out)
        self.assertTrue(all(a["transport"] == "opencode"
                            for a in cfg["agents"].values()))

    def test_input_required_retries_on_empty(self):
        with fake_stdin("\n  \nx\n"), captured_stdout() as out:
            value = ac_ui.input_required("P: ")
        self.assertEqual(value, "x")
        self.assertIn("(required)", out.getvalue())

    def test_wizard_rejects_invalid_temperature(self):
        # "abc" cannot even parse, "5"/"nan" parse but are not valid: all
        # must warn and fall back to the judge default, none may be saved
        for raw in ("abc", "5", "nan"):
            with fake_stdin(f"\n\n\n\n{raw}\n"), captured_stdout() as out:
                agent = ac_ui._prompt_agent(ac_config.ROLE_JUDGE,
                                            ac_config.TRANSPORT_OPENAI)
            self.assertEqual(agent["temperature"], 0.0)
            self.assertIn("invalid temperature", out.getvalue())

    def test_print_status_without_judge(self):
        with captured_stdout() as out:
            ac_ui.print_status(oc_cfg("http://127.0.0.1:1", with_judge=False, agent=None),
                               None)
        self.assertIn("Judge: none configured - proposer will judge", out.getvalue())

    def test_restricted_agent_lines_states(self):
        srv = self.fake_server()
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
