# Tests for ac_engine: parsing helpers and the debate engines
# (stateful opencode + stateless openai), including the fail-fast checks.
import contextlib
import io
import os
import time
import unittest
from typing import Any
from unittest import mock

import ac_engine
from ac_engine import build_prompts, parse_max_rounds, parse_verdict, run_discussion
from ac_errors import ServerUnreachable, TransportError
from helpers import FakeServerMixin, QuietRunMixin, closed_port, oc_cfg, stateless_cfg


class TestParseMaxRounds(unittest.TestCase):
    def test_parse_max_rounds(self):
        # deliberately malformed configs must not crash the parser
        cases: list[tuple[Any, int]] = [
            ({}, 10),
            ({"max_rounds": "3"}, 3),
            ({"max_rounds": 0}, 1),
            ({"max_rounds": "junk"}, 10),
        ]
        for cfg, expected in cases:
            self.assertEqual(parse_max_rounds(cfg), expected)


class TestParseVerdict(unittest.TestCase):
    def test_cases(self):
        cases = {
            "YES - agree": True,
            "yes": True,
            "NO.": False,
            "  No because": False,
            "maybe yes later": True,
            "Absolutely not": False,
            "": False,
            "NOtable difference": False,
            "YESmen not": False,
            "Final: YES": True,
        }
        for text, expected in cases.items():
            self.assertEqual(parse_verdict(text), expected, repr(text))


class TestBuildPrompts(unittest.TestCase):
    def test_prompts_are_keyed_by_debate_role(self):
        agents = oc_cfg("http://x")["agents"]
        prompts = build_prompts("topic", agents[ac_engine.ROLE_PROPOSER],
                                agents[ac_engine.ROLE_CRITIC],
                                {ac_engine.ROLE_PROPOSER: None,
                                 ac_engine.ROLE_CRITIC: None})
        self.assertEqual(sorted(prompts),
                         [ac_engine.ROLE_CRITIC, ac_engine.ROLE_PROPOSER])
        self.assertIn("PROPOSER", prompts[ac_engine.ROLE_PROPOSER])
        self.assertIn("CRITIC", prompts[ac_engine.ROLE_CRITIC])


class TestOpencodeEngine(FakeServerMixin, QuietRunMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.srv = self.fake_server()

    def test_consensus_after_two_rounds_and_cleanup(self):
        self.srv.verdicts[:] = ["NO - not yet.", "YES - agreed."]
        out = self.run_engine(oc_cfg(self.srv.base, max_rounds=3))
        self.assertIn("Final status : CONSENSUS", out)
        self.assertIn("[Judge] skipped in round 1", out)
        self.assertEqual(len(self.srv.sessions), 3)
        self.assertEqual(sorted(s["title"].split(" ")[1] for s in self.srv.sessions),
                         ["critic", "judge", "proposer"])
        self.assertEqual(len(self.srv.deletes), 3)
        tfiles = [f for f in os.listdir(self._tdir) if f.endswith(".txt")]
        self.assertEqual(len(tfiles), 1)
        with open(os.path.join(self._tdir, tfiles[0]), encoding="utf-8") as f:
            transcript = f.read()
        self.assertIn("STATUS: CONSENSUS", transcript)
        self.assertIn("JUDGE NOTE: YES - agreed.", transcript)

    def test_judge_failure_does_not_abort(self):
        # judge blows up in round 2, recovers in round 3
        with mock.patch.object(ac_engine.Discussion, "judge_now",
                               side_effect=[RuntimeError("boom"), (True, "YES - ok")]):
            out = self.run_engine(oc_cfg(self.srv.base, max_rounds=3))
        self.assertIn("[Judge] failed to evaluate: boom", out)
        self.assertIn("Final status : CONSENSUS", out)

    def test_all_judge_failures_report_no_verdict(self):
        # a run where the judge never answered must not be reported as a
        # genuine "no consensus" verdict
        with mock.patch.object(ac_engine.Discussion, "judge_now",
                               side_effect=RuntimeError("down")):
            out = self.run_engine(oc_cfg(self.srv.base, max_rounds=2))
        self.assertIn("[Judge] failed to evaluate: down", out)
        self.assertIn("NO VERDICT (judge unavailable)", out)
        self.assertNotIn("NO CONSENSUS", out)

    def test_abort_keeps_completed_turns_and_writes_footer(self):
        # a debater failure mid-round must not lose the turns that already
        # completed, and the transcript must still record the abort
        orig = ac_engine.Discussion._request_turn

        def flaky(disc, key, sys_prompt):
            if key == ac_engine.ROLE_CRITIC:
                raise TransportError("opencode: boom")
            return orig(disc, key, sys_prompt)

        buf = io.StringIO()
        with mock.patch.object(ac_engine.Discussion, "_request_turn", flaky):
            with contextlib.redirect_stdout(buf):
                with self.assertRaises(TransportError):
                    run_discussion(oc_cfg(self.srv.base, max_rounds=2),
                                   topic="test topic")
        self.assertIn(", 0 rounds", buf.getvalue())
        tfiles = [f for f in os.listdir(self._tdir) if f.endswith(".txt")]
        self.assertEqual(len(tfiles), 1)
        with open(os.path.join(self._tdir, tfiles[0]), encoding="utf-8") as f:
            transcript = f.read()
        self.assertIn("--- Round 1 ---\n[P]\nProposer: I defend the topic.\n", transcript)
        self.assertNotIn("[C]", transcript)
        self.assertIn("STATUS: ABORTED", transcript)
        self.assertIn("ABORTED: TransportError: opencode: boom", transcript)
        self.assertEqual(len(self.srv.deletes), 1)  # session cleaned up on abort

    def test_single_round_ends_unjudged(self):
        self.srv.verdicts[:] = ["NO - still apart."]
        out = self.run_engine(oc_cfg(self.srv.base, max_rounds=1))
        self.assertIn("[Judge] 1 round only, no judgement will be given", out)
        self.assertNotIn("[Judge:", out)
        self.assertIn("NO CONSENSUS (hit max rounds)", out)
        self.assertIn(", 1 round\n", out)

    def test_same_timestamp_transcripts_do_not_overwrite(self):
        with mock.patch.object(ac_engine.time, "strftime",
                               return_value="20260101_000000"):
            self.run_engine(oc_cfg(self.srv.base))
            self.run_engine(oc_cfg(self.srv.base))
        tfiles = sorted(f for f in os.listdir(self._tdir) if f.endswith(".txt"))
        self.assertEqual(tfiles, ["discussion_20260101_000000.txt",
                                  "discussion_20260101_000000_2.txt"])
        for name in tfiles:
            with open(os.path.join(self._tdir, name), encoding="utf-8") as f:
                self.assertIn("STATUS:", f.read())

    def test_missing_agent_warns_and_falls_back(self):
        self.srv.agent_names.clear()
        self.srv.verdicts[:] = ["YES - agree."]
        out = self.run_engine(oc_cfg(self.srv.base, max_rounds=2))
        self.assertIn("[Warn] agent 'debater' not on", out)
        self.assertIn("Final status : CONSENSUS", out)
        self.assertIsNone(self.srv.msgs[0]["agent"])

    def test_judge_gets_isolated_session_without_judge_role(self):
        self.srv.verdicts[:] = ["YES - agree."]
        self.run_engine(oc_cfg(self.srv.base, with_judge=False, max_rounds=2))
        self.assertEqual(len({s["id"] for s in self.srv.sessions}), 3)
        judge_msgs = [m for m in self.srv.msgs
                      if m["system"].startswith("You are a neutral judge")]
        prop_msgs = [m for m in self.srv.msgs if "PROPOSER" in m["system"]]
        self.assertEqual(len(judge_msgs), 1)
        self.assertNotEqual(judge_msgs[0]["sid"], prop_msgs[0]["sid"])

    def test_message_ids_are_deterministic(self):
        self.srv.verdicts[:] = ["YES - agree."]
        self.run_engine(oc_cfg(self.srv.base))
        for m in self.srv.msgs:
            self.assertRegex(m["messageID"],
                             rf"^msg_{m['sid'][4:16]}_t0_[0-9a-f]{{8}}$")
        self.assertEqual(len(self.srv.msgs), 2)  # round 1 is never judged

    def test_round2_message_id_differs_and_critic_reaches_proposer(self):
        # regression: st["sent"] was recorded before the agent's own turn was
        # appended, so round 2 reused the round-1 message id, the server
        # deduped it and the proposer replayed its opening verbatim
        self.srv.verdicts[:] = ["YES - agreed."]
        self.run_engine(oc_cfg(self.srv.base, max_rounds=2))
        prop = [m for m in self.srv.msgs if "PROPOSER" in m["system"]]
        self.assertEqual(len(prop), 2)
        sid = prop[0]["sid"][4:16]
        self.assertRegex(prop[0]["messageID"], rf"^msg_{sid}_t0_[0-9a-f]{{8}}$")
        self.assertRegex(prop[1]["messageID"], rf"^msg_{sid}_t1_[0-9a-f]{{8}}$")
        self.assertNotIn("Discussion topic:", prop[1]["text"])
        self.assertIn("Critic: I challenge the topic.", prop[1]["text"])

    def test_orphan_sessions_are_swept(self):
        sid = "ses_orphan0001test2018"
        fresh_sid = "ses_fresh0001test2018"
        fresh_ts = time.strftime("%Y%m%d_%H%M%S")
        self.srv.session_list.append({"id": sid, "title": "agent-consensus old"})
        self.srv.session_list.append(
            {"id": fresh_sid, "title": f"agent-consensus proposer {fresh_ts}"})
        self.srv.verdicts[:] = ["YES - agree."]
        self.run_engine(oc_cfg(self.srv.base))
        # unparseable title = old orphan -> deleted; fresh title -> left alone
        self.assertIn(f"/session/{sid}", self.srv.deletes)
        self.assertNotIn(f"/session/{fresh_sid}", self.srv.deletes)

    def test_extra_agent_roles_warn_and_are_not_probed(self):
        cfg = oc_cfg(self.srv.base)
        cfg["agents"]["backup"] = {"name": "B", "transport": "opencode",
                                   "api_url": f"http://127.0.0.1:{closed_port()}",
                                   "model": "opencode/x"}
        out = self.run_engine(cfg)
        self.assertIn("ignoring unknown agent role(s): backup", out)
        self.assertIn("NO CONSENSUS", out)

    def test_unreachable_server_fails_fast(self):
        # real refused TCP connect (the sweep is already covered elsewhere)
        cfg = oc_cfg(f"http://127.0.0.1:{closed_port()}")
        with mock.patch.object(ac_engine, "oc_sweep", return_value=None):
            with self.assertRaises(ServerUnreachable) as cm:
                run_discussion(cfg, topic="t")
        self.assertRegex(str(cm.exception), r"cannot reach .*opencode serve")

    def test_multiple_down_servers_all_listed(self):
        # connection handling itself is covered above: only the message matters
        cfg = oc_cfg("http://127.0.0.1:1")
        cfg["agents"]["critic"]["api_url"] = "http://127.0.0.1:2"
        with mock.patch.object(ac_engine, "oc_sweep", return_value=None), \
                mock.patch.object(ac_engine, "oc_agent_list", return_value=None):
            with self.assertRaises(ServerUnreachable) as cm:
                run_discussion(cfg, topic="t")
        self.assertEqual(str(cm.exception).count("cannot reach"), 1)
        self.assertEqual(str(cm.exception).count("http://127.0.0.1:"), 2)


class TestStatelessEngine(FakeServerMixin, QuietRunMixin, unittest.TestCase):
    def test_refused_endpoint_fails_fast(self):
        cfg = stateless_cfg(f"http://127.0.0.1:{closed_port()}/v1/chat/completions")
        with self.assertRaises(ServerUnreachable) as cm:
            run_discussion(cfg, topic="t")
        self.assertIn("connection refused", str(cm.exception))

    def test_malformed_port_skips_preflight(self):
        # the preflight must not crash on a nonnumeric port; the request itself
        # fails later with urllib's own error instead
        cfg = stateless_cfg("http://127.0.0.1:notaport/v1/chat/completions")
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(TransportError) as cm:
                run_discussion(cfg, topic="t")
        self.assertNotIn("connection refused", str(cm.exception))
        self.assertIn("request failed", str(cm.exception))

    def test_extra_stateless_role_is_not_probed(self):
        srv = self.fake_server()
        cfg = stateless_cfg(srv.base + "/v1/chat/completions")
        cfg["agents"]["backup"] = {
            "name": "B", "transport": "openai", "api_key": "k",
            "api_url": f"http://127.0.0.1:{closed_port()}/v1/chat/completions",
            "model": "m",
        }
        out = self.run_engine(cfg)
        self.assertIn("ignoring unknown agent role(s): backup", out)
        self.assertIn("NO CONSENSUS", out)

    def test_temperature_defaults(self):
        srv = self.fake_server()
        cfg = stateless_cfg(srv.base + "/v1/chat/completions")
        cfg["max_rounds"] = 2
        out = self.run_engine(cfg)
        self.assertIn("Final status : CONSENSUS", out)
        self.assertEqual([p["temperature"] for p in srv.chats],
                         [0.7, 0.7, 0.7, 0.7, 0.0])

    def test_judge_fallback_still_uses_judge_temperature(self):
        # without a judge role the judging call runs on the proposer's config,
        # but it is still a judging call: temperature 0.0, not 0.7
        srv = self.fake_server()
        cfg = stateless_cfg(srv.base + "/v1/chat/completions")
        cfg["max_rounds"] = 2
        del cfg["agents"]["judge"]
        out = self.run_engine(cfg)
        self.assertIn("Final status : CONSENSUS", out)
        self.assertEqual([p["temperature"] for p in srv.chats],
                         [0.7, 0.7, 0.7, 0.7, 0.0])


if __name__ == "__main__":
    unittest.main()
