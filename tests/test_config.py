# Tests for ac_config: transport selection and config validation.
import unittest

from helpers import oc_cfg
from ac_config import is_opencode, transport_of, validate_config


class TestConfig(unittest.TestCase):
    def test_transport_defaults_to_openai(self):
        self.assertEqual(transport_of({}), "openai")
        self.assertFalse(is_opencode({}))
        self.assertTrue(is_opencode({"transport": "opencode"}))

    def test_validate_config(self):
        self.assertEqual(validate_config({}), "missing 'agents' object")
        self.assertIn("missing agent 'proposer'", validate_config(
            {"agents": {"critic": {"name": "x", "api_url": "u", "model": "m"}}}))
        good = oc_cfg("http://x")
        self.assertIsNone(validate_config(good))
        bad = oc_cfg("http://x")
        del bad["agents"]["critic"]["model"]
        self.assertIn("missing 'model'", validate_config(bad))
        bad_url = oc_cfg("http://x")
        bad_url["agents"]["proposer"]["api_url"] = "127.0.0.1:8645/v1"
        self.assertIn("api_url", validate_config(bad_url))
        bad_transport = oc_cfg("http://x")
        bad_transport["agents"]["critic"]["transport"] = "opencod"
        self.assertIn("transport", validate_config(bad_transport))


if __name__ == "__main__":
    unittest.main()
