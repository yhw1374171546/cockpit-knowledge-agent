# -*- coding: utf-8 -*-
"""多 Agent 协作测试：拆解、工具白名单、风险评级、冲突仲裁、安全评审。

    python agent/tests/test_multiagent.py
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent.agents import (ManualExpert, SafetyCritic, ServiceAdvisor,   # noqa: E402
                          VehicleControlAgent, assess_telemetry)
from agent.agents.base import ScopedPlanner                             # noqa: E402
from agent.kb import KnowledgeBase                                      # noqa: E402
from agent.obs import Tracer                                            # noqa: E402
from agent.orchestrator import Supervisor, detect_conflicts             # noqa: E402
from agent.protocols import (RISK_CRITICAL, RISK_OK, RISK_WARNING,      # noqa: E402
                             AgentResult, Handoff, SubTask)

KB = None
NORMAL_TELEMETRY = {
    "车型": "领克08 EM-P", "里程_km": 23860,
    "胎压_kPa": {"左前": 236, "右前": 241, "左后": 228, "右后": 239},
    "胎压告警": ["左后"], "剩余电量_%": 62, "续航_km": 148, "告警灯": [],
    "下次保养_km": 24000,
}
SEVERE_TELEMETRY = {
    **NORMAL_TELEMETRY,
    "胎压_kPa": {"左前": 236, "右前": 241, "左后": 148, "右后": 239},
    "告警灯": ["制动系统故障"],
}


def setUpModule():
    global KB
    KB = KnowledgeBase.load()


class TestDecompose(unittest.TestCase):
    def setUp(self):
        self.sup = Supervisor(KB, tracer=Tracer())

    def test_single_intent(self):
        plan = self.sup.decompose("怎么打开危险警告灯")
        self.assertEqual(plan.route, "single")
        self.assertEqual(plan.subtasks[0].agent, "manual_expert")

    def test_multi_intent_splits_by_expert(self):
        plan = self.sup.decompose("胎压报警了怎么办，另外座椅加热怎么关闭")
        self.assertEqual(plan.route, "multi")
        agents = {s.agent for s in plan.subtasks}
        self.assertEqual(agents, {"vehicle_control", "manual_expert"})

    def test_same_expert_subtasks_are_merged(self):
        """两个子任务若都归同一个专家，应合并，避免重复检索。"""
        plan = self.sup.decompose("这台车该保养了吗，顺便帮我预约到店")
        self.assertEqual(len(plan.subtasks), 1)
        self.assertEqual(plan.subtasks[0].agent, "service_advisor")

    def test_spec_question_routes_to_manual_expert(self):
        plan = self.sup.decompose("领克08的纯电续航是多少")
        self.assertEqual(plan.subtasks[0].agent, "manual_expert")


class TestToolWhitelist(unittest.TestCase):
    def setUp(self):
        self.sup = Supervisor(KB, tracer=Tracer())

    def test_specialists_only_see_their_tools(self):
        specs = self.sup.stats()["specialists"]
        self.assertEqual(set(specs["manual_expert"]), {"search_manual", "lookup_vehicle_spec"})
        self.assertEqual(set(specs["vehicle_control"]), {"get_vehicle_status", "search_manual"})
        self.assertEqual(set(specs["service_advisor"]),
                         {"get_maintenance_plan", "search_manual", "create_service_order"})
        # 车控专家不该持有写操作工具
        self.assertNotIn("create_service_order", specs["vehicle_control"])

    def test_subset_shares_confirmation_state(self):
        parent = self.sup.registry
        parent.confirm("create_service_order:更换机油机滤:2025-06-18")
        scoped = parent.subset(["create_service_order"], "svc")
        self.assertIn("create_service_order:更换机油机滤:2025-06-18", scoped.confirmed_actions)

    def test_scoped_planner_stays_in_lane(self):
        planner = ScopedPlanner("status", ("status",))
        self.assertEqual(planner.route("座椅加热怎么关闭"), "status")
        planner2 = ScopedPlanner("manual_qa", ("spec", "manual_qa"))
        self.assertEqual(planner2.route("领克08的电池容量是多少"), "spec")
        self.assertEqual(planner2.route("座椅加热怎么关闭"), "manual_qa")


class TestRiskAssessment(unittest.TestCase):
    def test_normal_vehicle_is_ok(self):
        telemetry = {**NORMAL_TELEMETRY, "胎压告警": [], "胎压_kPa": {"左前": 236, "右前": 241,
                                                                    "左后": 238, "右后": 239}}
        level, findings = assess_telemetry(telemetry)
        self.assertEqual(level, RISK_OK)

    def test_tire_pressure_alert_field_is_warning(self):
        level, findings = assess_telemetry(NORMAL_TELEMETRY)
        self.assertEqual(level, RISK_WARNING)
        self.assertTrue(any("胎压报警" in f for f in findings))

    def test_severe_underinflation_is_critical(self):
        level, findings = assess_telemetry(SEVERE_TELEMETRY)
        self.assertEqual(level, RISK_CRITICAL)
        self.assertTrue(any("严重亏气" in f for f in findings))

    def test_brake_warning_is_critical(self):
        telemetry = {**NORMAL_TELEMETRY, "胎压告警": [], "告警灯": ["制动系统故障"]}
        level, _ = assess_telemetry(telemetry)
        self.assertEqual(level, RISK_CRITICAL)


class TestConflictArbitration(unittest.TestCase):
    def test_safety_override_beats_permissive_manual_advice(self):
        results = [
            AgentResult("vehicle_control", "t1", answer="实时车况：左后胎压 148kPa 严重亏气",
                        risk_level=RISK_CRITICAL, findings=["左后胎压 148kPa 严重亏气"]),
            AgentResult("manual_expert", "t2",
                        answer="胎压低报警时，可在冷态充气后以 30 km/h 行驶几分钟解除报警。"),
        ]
        conflicts = detect_conflicts(results)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].kind, "safety_override")
        self.assertEqual(conflicts[0].winner, "vehicle_control")
        self.assertFalse(conflicts[0].escalated)

    def test_no_conflict_when_nothing_permissive(self):
        results = [
            AgentResult("vehicle_control", "t1", answer="实时车况正常", risk_level=RISK_OK),
            AgentResult("manual_expert", "t2", answer="请在中央显示屏中关闭座椅加热。"),
        ]
        self.assertEqual(detect_conflicts(results), [])

    def test_numeric_mismatch_is_escalated(self):
        results = [
            AgentResult("vehicle_control", "t1", answer="实时续航 148 km", risk_level=RISK_OK),
            AgentResult("manual_expert", "t2", answer="该车型续航 245 km", risk_level=RISK_OK),
        ]
        conflicts = detect_conflicts(results)
        self.assertTrue(any(c.kind == "numeric_mismatch" and c.escalated for c in conflicts))


class TestSafetyCritic(unittest.TestCase):
    def setUp(self):
        self.critic = SafetyCritic()

    def test_injects_directive_on_critical_risk(self):
        results = [AgentResult("vehicle_control", "t1", answer="左后胎压 148kPa 严重亏气",
                               risk_level=RISK_CRITICAL, findings=["左后胎压 148kPa 严重亏气"])]
        evidence = [{"text": "左后胎压 148kPa 严重亏气", "citation": "[手册 第1页]"}]
        verdict = self.critic.review("胎压报警了怎么办", "左后胎压 148kPa 严重亏气", results, evidence)
        self.assertEqual(verdict.verdict, "revise")
        self.assertIn("靠边停车", verdict.revised_answer)
        self.assertIn("领克中心", verdict.revised_answer)

    def test_strips_permissive_text_under_critical_risk(self):
        results = [AgentResult("vehicle_control", "t1", answer="x", risk_level=RISK_CRITICAL,
                               findings=["制动系统故障"])]
        answer = "请立即靠边停车。胎压低时可以继续低速行驶到最近的维修点。"
        verdict = self.critic.review("胎压报警", answer, results, [])
        self.assertEqual(verdict.verdict, "revise")
        self.assertNotIn("可以继续低速行驶", verdict.revised_answer)
        self.assertIn("请勿继续行驶", verdict.revised_answer)

    def test_blocks_on_injection_with_write_claim(self):
        results = [AgentResult("service_advisor", "t1", answer="已为您预约", injection_flagged=True)]
        verdict = self.critic.review("帮我预约", "已为您预约周六到店保养。", results, [])
        self.assertEqual(verdict.verdict, "block")
        self.assertIn("注入", " ".join(verdict.reasons))

    def test_blocks_ungrounded_answer(self):
        results = [AgentResult("manual_expert", "t1", answer="本车配备航空钛合金防撞梁。")]
        evidence = [{"text": "危险警告灯开关在方向盘下方。", "citation": "[手册 第1页]"}]
        verdict = self.critic.review("这车有什么配置", "本车配备航空钛合金防撞梁并支持水上漂移。",
                                     results, evidence)
        self.assertEqual(verdict.verdict, "block")

    def test_approves_grounded_answer(self):
        text = "危险警告灯开关在方向盘下方，按下开关即可打开危险警告灯。"
        results = [AgentResult("manual_expert", "t1", answer=text)]
        verdict = self.critic.review("怎么打开危险警告灯", text, results,
                                     [{"text": text, "citation": "[手册 第87页]"}])
        self.assertEqual(verdict.verdict, "approve")


class TestSupervisorRun(unittest.TestCase):
    def test_single_intent_end_to_end(self):
        sup = Supervisor(KB, tracer=Tracer())
        st = sup.run("怎么打开危险警告灯")
        self.assertEqual(st.status, "answered")
        self.assertIn("危险警告灯", st.answer)
        self.assertTrue(st.citations)

    def test_multi_intent_runs_parallel_specialists(self):
        sup = Supervisor(KB, tracer=Tracer(), telemetry=NORMAL_TELEMETRY)
        st = sup.run("胎压报警了怎么办，另外座椅加热怎么关闭")
        self.assertEqual(st.plan.route, "multi")
        agents = {r.agent for r in st.results}
        self.assertEqual(agents, {"vehicle_control", "manual_expert"})
        self.assertEqual(len(st.handoffs), 2)
        self.assertTrue(all(h.from_agent == "supervisor" for h in st.handoffs))

    def test_severe_risk_produces_safety_answer(self):
        sup = Supervisor(KB, tracer=Tracer(), telemetry=SEVERE_TELEMETRY)
        st = sup.run("胎压报警了怎么办")
        self.assertEqual(st.verdict.verdict, "revise")
        self.assertIn("靠边停车", st.answer)
        self.assertNotIn("可以继续低速行驶", st.answer)

    def test_write_action_requires_confirmation(self):
        sup = Supervisor(KB, tracer=Tracer())
        st = sup.run("帮我预约到店保养")
        self.assertIn(st.status, ("needs_confirmation", "refused", "answered"))
        # 未确认前不得出现"预约成功"这类既成事实表述
        self.assertNotIn("预约成功", st.answer)
        self.assertNotIn("已下单", st.answer)

    def test_refuses_off_topic(self):
        sup = Supervisor(KB, tracer=Tracer())
        st = sup.run("中国足球的队长是谁")
        self.assertEqual(st.answer.strip(), "无答案")

    def test_trace_and_budget_accounting(self):
        tracer = Tracer()
        sup = Supervisor(KB, tracer=tracer)
        st = sup.run("怎么打开危险警告灯")
        totals = tracer.totals()
        self.assertGreater(totals["tokens_total"], 0)
        self.assertGreaterEqual(totals["n_spans"], 2)
        self.assertGreater(st.tokens_total, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
