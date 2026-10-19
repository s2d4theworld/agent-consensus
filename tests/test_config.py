# Tests for ac_config: transport selection and config validation.
import json
import os
import unittest
from typing import Any, cast
from unittest import mock

# helpers first: importing it puts the repo root on sys.path for ac_*
from helpers import captured_stdout, oc_cfg, patched_attrs, read_json, temp_dir

import ac_config
from ac_config import (
    DEFAULT_MAX_ROUNDS,
    DEFAULT_TEMPERATURE_DEBATER,
    DEFAULT_TEMPERATURE_JUDGE,
    MAX_ROUNDS_LIMIT,
    ROLE_CRITIC,
    ROLE_JUDGE,
    ROLE_PROPOSER,
    coerce_max_rounds,
    default_temperature,
    is_opencode,
    role_agents,
    save_config,
    transport_of,
    valid_temperature,
    validate_config,
)


class TestAgentInstaller(unittest.TestCase):
    def setUp(self):
        self.tdir = self.enterContext(temp_dir())
        self.src = os.path.join(self.tdir, "debater.md")
        with open(self.src, "w", encoding="utf-8") as f:
            f.write("# agent\n")
        self.enterContext(patched_attrs(
            ac_config, AGENT_SRC=self.src,
            OPENCODE_AGENTS_DIR=os.path.join(self.tdir, "agents")))

    def test_installs_and_is_idempotent(self):
        # the second call must compare content, not mtime/hash-of-bytes: on
        # Windows the first write lands as CRLF and still reads back identical
        with captured_stdout() as out:
            self.assertTrue(ac_config.ensure_agent_installed())
            self.assertTrue(ac_config.ensure_agent_installed())
        self.assertIn("Websearch agent installed:", out.getvalue())
        self.assertIn("already installed", out.getvalue())
        with open(os.path.join(self.tdir, "agents", "debater.md"),
                  encoding="utf-8") as f:
            self.assertEqual(f.read(), "# agent\n")

    def test_missing_template_fails(self):
        self.enterContext(patched_attrs(
            ac_config, AGENT_SRC=os.path.join(self.tdir, "nope.md")))
        with captured_stdout() as out:
            self.assertFalse(ac_config.ensure_agent_installed())
        self.assertIn("Agent template missing", out.getvalue())

    def test_unwritable_target_fails(self):
        # a file where the agents dir should be: makedirs must fail cleanly
        blocker = os.path.join(self.tdir, "blocker")
        with open(blocker, "w", encoding="utf-8") as f:
            f.write("x")
        self.enterContext(patched_attrs(
            ac_config, OPENCODE_AGENTS_DIR=os.path.join(blocker, "agents")))
        with captured_stdout() as out:
            self.assertFalse(ac_config.ensure_agent_installed())
        self.assertIn("Could not install", out.getvalue())


class TestConfig(unittest.TestCase):
    def test_transport_defaults_to_openai(self):
        # minimal dicts stand in for arbitrary config junk
        self.assertEqual(transport_of(cast(Any, {})), "openai")
        self.assertFalse(is_opencode(cast(Any, {})))
        self.assertTrue(is_opencode(cast(Any, {"transport": "opencode"})))

    def test_validate_config(self):
        self.assertEqual(validate_config({}), "missing 'agents' object")
        self.assertIn("missing agent 'proposer'", validate_config(
            {"agents": {"critic": {"name": "x", "api_url": "u", "model": "m"}}}) or "")
        good = oc_cfg("http://x")
        self.assertIsNone(validate_config(good))
        bad = oc_cfg("http://x")
        del bad["agents"]["critic"]["model"]
        self.assertIn("missing 'model'", validate_config(bad) or "")
        bad_url = oc_cfg("http://x")
        bad_url["agents"]["proposer"]["api_url"] = "127.0.0.1:8645/v1"
        self.assertIn("api_url", validate_config(bad_url) or "")
        bad_transport = oc_cfg("http://x")
        bad_transport["agents"]["critic"]["transport"] = "opencod"
        self.assertIn("transport", validate_config(bad_transport) or "")

    def test_validate_config_rejects_non_string_fields(self):
        cases: list[tuple[str, object]] = [
            ("name", 1), ("api_url", 123), ("model", 7), ("api_key", 42),
        ]
        for field, value in cases:
            cfg = cast(Any, oc_cfg("http://x"))
            cfg["agents"]["critic"][field] = value
            self.assertIn(f"field '{field}' must be a string",
                          validate_config(cfg) or "", field)

    def test_validate_config_treats_blank_strings_as_missing(self):
        cfg = cast(Any, oc_cfg("http://x"))
        cfg["agents"]["critic"]["model"] = "   "
        self.assertIn("missing 'model'", validate_config(cfg) or "")

    def test_validate_config_ignores_unknown_roles(self):
        # unknown roles are warned about and ignored at run time: a stale or
        # malformed extra role must not block a valid config
        cfg = cast(Any, oc_cfg("http://x"))
        cfg["agents"]["backup"] = {"name": "B"}
        self.assertIsNone(validate_config(cfg))

    def test_validate_config_allows_non_dict_judge(self):
        # the engine falls back to the proposer's config for a judge that is
        # not an object, so validation must accept it too
        cfg = cast(Any, oc_cfg("http://x"))
        cfg["agents"]["judge"] = 5
        self.assertIsNone(validate_config(cfg))

    def test_default_temperature_by_role(self):
        self.assertEqual(default_temperature(ROLE_JUDGE),
                         DEFAULT_TEMPERATURE_JUDGE)
        self.assertEqual(default_temperature(ROLE_PROPOSER),
                         DEFAULT_TEMPERATURE_DEBATER)
        self.assertEqual(default_temperature(ROLE_CRITIC),
                         DEFAULT_TEMPERATURE_DEBATER)

    def test_role_agents_ignores_unknown_roles(self):
        cfg = cast(Any, oc_cfg("http://x"))
        cfg["agents"]["backup"] = {"name": "B", "api_url": "http://x",
                                   "model": "m"}
        self.assertEqual([key for key, _ in role_agents(cfg)],
                         [ROLE_PROPOSER, ROLE_CRITIC, ROLE_JUDGE])

    def test_max_rounds_coercion_is_total(self):
        self.assertEqual(coerce_max_rounds(float("inf")), DEFAULT_MAX_ROUNDS)
        self.assertEqual(coerce_max_rounds(float("-inf")), DEFAULT_MAX_ROUNDS)
        self.assertEqual(coerce_max_rounds(float("nan")), DEFAULT_MAX_ROUNDS)
        self.assertEqual(coerce_max_rounds("junk"), DEFAULT_MAX_ROUNDS)
        self.assertEqual(coerce_max_rounds(True), DEFAULT_MAX_ROUNDS)
        self.assertEqual(coerce_max_rounds(False), DEFAULT_MAX_ROUNDS)
        self.assertEqual(coerce_max_rounds(0), 1)
        self.assertEqual(coerce_max_rounds(3.9), 3)
        self.assertEqual(coerce_max_rounds(100000), MAX_ROUNDS_LIMIT)
        self.assertEqual(coerce_max_rounds("99"), 99)

    def test_load_config_tolerates_utf8_bom(self):
        # a Windows editor may save a hand-edited config.json with a BOM;
        # it must not read as broken
        tdir = self.enterContext(temp_dir())
        path = os.path.join(tdir, "config.json")
        with open(path, "w", encoding="utf-8-sig") as f:
            json.dump(oc_cfg("http://x"), f)
        with patched_attrs(ac_config, CONFIG_PATH=path):
            cfg = ac_config.load_config()
        self.assertIsNotNone(cfg)
        self.assertIn("agents", cfg or {})

    def test_save_config_reports_unwritable_path(self):
        with mock.patch("builtins.open", side_effect=OSError("read-only")), \
                captured_stdout() as out:
            self.assertFalse(save_config(oc_cfg("http://x")))
        self.assertIn("cannot save", out.getvalue())

    def test_save_config_keeps_previous_config_on_failure(self):
        # the atomic replace must not touch the old file when the write fails
        tdir = self.enterContext(temp_dir())
        path = os.path.join(tdir, "config.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"old": true}')
        with patched_attrs(ac_config, CONFIG_PATH=path), \
                mock.patch.object(ac_config.json, "dump",
                                  side_effect=OSError("disk full")), \
                captured_stdout():
            self.assertFalse(save_config(oc_cfg("http://x")))
        self.assertEqual(read_json(path), {"old": True})
        self.assertFalse(os.path.exists(path + ".tmp"))

    def test_save_config_reports_non_serializable_config(self):
        # the bool contract holds for any config, not just OSError paths
        tdir = self.enterContext(temp_dir())
        path = os.path.join(tdir, "config.json")
        with patched_attrs(ac_config, CONFIG_PATH=path), \
                captured_stdout() as out:
            self.assertFalse(save_config(cast(Any, {"agents": {1, 2}})))
        self.assertIn("cannot save", out.getvalue())
        self.assertFalse(os.path.exists(path))

    def test_temperature_validation(self):
        cases: list[tuple[Any, bool]] = [
            (0.0, True), (2.0, True), (0.7, True),
            (-0.1, False), (2.1, False), (True, False), ("0.7", False),
            (float("nan"), False), (float("inf"), False),
        ]
        for value, expected in cases:
            self.assertEqual(valid_temperature(value), expected, repr(value))
        bad_temp = oc_cfg("http://x")
        bad_temp["agents"]["critic"]["temperature"] = float("nan")
        self.assertIn("temperature", validate_config(bad_temp) or "")


if __name__ == "__main__":
    unittest.main()
