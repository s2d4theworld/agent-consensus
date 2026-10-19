# Tests for ac_config: transport selection and config validation.
import contextlib
import io
import unittest
from typing import Any, cast
from unittest import mock

from helpers import oc_cfg
from ac_config import (
    DEFAULT_MAX_ROUNDS,
    DEFAULT_TEMPERATURE_DEBATER,
    DEFAULT_TEMPERATURE_JUDGE,
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
        self.assertEqual(coerce_max_rounds(0), 1)
        self.assertEqual(coerce_max_rounds(3.9), 3)

    def test_save_config_reports_unwritable_path(self):
        with mock.patch("builtins.open", side_effect=OSError("read-only")):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertFalse(save_config(oc_cfg("http://x")))
        self.assertIn("cannot save", out.getvalue())

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
