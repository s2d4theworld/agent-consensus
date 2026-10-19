# Tests for ac_config: transport selection and config validation.
import unittest
from typing import Any, cast

from helpers import oc_cfg
from ac_config import is_opencode, transport_of, valid_temperature, validate_config


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
