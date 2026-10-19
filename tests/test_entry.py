# Tests for the CLI entry point: argument dispatch and menu behavior.
import json
import unittest
from unittest import mock

# helpers first: importing it puts the repo root on sys.path for ac_*
from helpers import (
    FakeServerMixin,
    QuietRunMixin,
    WIZARD_STDIN_OPENCODE,
    fake_cli,
    oc_cfg,
)

import ac_engine
import ac_ui
import agent_consensus


class TestEntry(FakeServerMixin, QuietRunMixin, unittest.TestCase):
    def run_menu(self, stdin: str) -> str:
        # run the interactive menu with this stdin, return its full output
        with fake_cli(stdin) as out:
            agent_consensus.main()
        return out.getvalue()

    def write_config(self, cfg: object) -> None:
        with open(self.temp_config(), "w", encoding="utf-8") as f:
            json.dump(cfg, f)

    def write_oc_config(self) -> None:
        # save a config that points at this test's fake server
        self.write_config(oc_cfg(self.fake_server().base))

    def test_menu_quits_on_3(self):
        self.temp_config("missing.json")
        text = self.run_menu("3\n")
        self.assertIn("No config yet", text)
        self.assertIn("[2] Start discussion", text)

    def test_menu_interrupt_outside_input_exits_cleanly(self):
        # Ctrl+C while the status block is doing its agent-list lookup must
        # exit the menu like any other interrupt, not traceback
        with mock.patch.object(agent_consensus, "load_config",
                               side_effect=KeyboardInterrupt):
            text = self.run_menu("")
        self.assertIn("AGENT CONSENSUS", text)

    def test_menu_dispatches_setup_and_rejects_junk(self):
        # menu choice 1 must run the wizard; junk input must be reported, not
        # silently re-loop
        with mock.patch.object(agent_consensus, "setup_wizard",
                               return_value=True) as wizard:
            text = self.run_menu("x\n1\n3\n")
        wizard.assert_called_once_with()
        self.assertIn("Invalid choice.", text)

    def test_menu_runs_discussion(self):
        # the primary menu action: choice 2 debates with the saved config,
        # waits for Enter, and returns to the menu
        self.write_oc_config()
        text = self.run_menu("2\nA vs B\n\n3\n")
        self.assertIn("Config found:", text)
        self.assertIn("Final status : NO CONSENSUS (hit max rounds)", text)
        self.assertIn("Press Enter to return to the menu...", text)

    def test_menu_discussion_aborts_when_stdin_ends_at_the_topic_prompt(self):
        # redirected stdin ends at the topic prompt: the run aborts cleanly
        # and the menu survives to its next EOF
        self.write_oc_config()
        text = self.run_menu("2\n")
        self.assertIn("Discussion aborted: no input available", text)

    def test_menu_survives_interrupt_during_the_run(self):
        # Ctrl+C while the run waits for a topic: safe_run reports it and the
        # menu stays interactive (the trailing 3 quits it)
        self.write_oc_config()
        with mock.patch.object(ac_engine, "input_required",
                               side_effect=KeyboardInterrupt):
            text = self.run_menu("2\n3\n")
        self.assertIn("Interrupted.", text)

    def test_menu_reports_corrupt_config(self):
        # a hand-broken config.json must read as "no config", not traceback
        with open(self.temp_config(), "w", encoding="utf-8") as f:
            f.write("not json{")
        text = self.run_menu("2\n3\n")
        self.assertIn("cannot read", text)
        self.assertIn("Cannot start: no config", text)

    def test_menu_refuses_to_start_without_valid_config(self):
        # config present but failing validation: the menu reports the problem
        # and refuses to start instead of running with a broken config
        self.write_config({"max_rounds": 5})
        text = self.run_menu("2\n3\n")
        self.assertIn("Config invalid: missing 'agents' object", text)
        self.assertIn("Cannot start: missing 'agents' object", text)

    def test_setup_aborts_on_eof(self):
        # `setup` with no stdin must not traceback, but must exit non-zero
        with fake_cli("", "setup") as out, self.assertRaises(SystemExit) as cm:
            agent_consensus.main()
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("Setup aborted", out.getvalue())

    def test_setup_succeeds_when_agent_install_fails(self):
        # documented: the optional websearch-agent install may fail without
        # failing setup - the config is saved and runs fall back to no tools
        self.temp_config()
        with mock.patch.object(ac_ui, "ensure_agent_installed",
                               return_value=False) as install:
            with fake_cli(WIZARD_STDIN_OPENCODE, "setup") as out:
                agent_consensus.main()  # returns: no SystemExit
        install.assert_called_once_with()
        self.assertIn("Config saved:", out.getvalue())

    def test_setup_save_failure_exits_1(self):
        with fake_cli("", "setup"), mock.patch.object(
                agent_consensus, "setup_wizard", return_value=False):
            with self.assertRaises(SystemExit) as cm:
                agent_consensus.main()
        self.assertEqual(cm.exception.code, 1)

    def test_install_agent_failure_exits_1(self):
        with fake_cli("", "install-agent"), mock.patch.object(
                agent_consensus, "ensure_agent_installed", return_value=False):
            with self.assertRaises(SystemExit) as cm:
                agent_consensus.main()
        self.assertEqual(cm.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
