# -*- coding: utf-8 -*-
"""可回答性校验测试（answerability）。

要证明的核心命题：**接地通过 ≠ 回答了问题**。

以线上真实案例为基准：
    问「车内有冰箱吗」
    证据「12V 电源可以给电子设备（例如：音乐播放器、**车载冰箱**和移动手机）充电」
    → 检索没错、接地没错，但**没回答问题**（那是"能给冰箱供电"，不是"车里有冰箱"）。

四类判据都要覆盖：CONSUME / EXAMPLE / WARNING（→ 限定）与 CONFIG（→ 正常回答）。

    python agent/tests/test_answerability.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent.answerability import (  # noqa: E402
    ANSWERABLE,
    NEEDS_QUALIFICATION,
    NOT_ANSWERABLE,
    build_qualified_answer,
    check_answerability,
    classify_question,
    judge_occurrence,
)
from agent.graph import AgentConfig, AgentGraph  # noqa: E402
from agent.kb import KnowledgeBase  # noqa: E402
from agent.llm import RuleBasedPlannerLLM  # noqa: E402
from agent.tools import ToolRegistry  # noqa: E402

# 真实语料里的原文（第 129 页 12V 电源）
FRIDGE_EVIDENCE = (
    "车载12V电源通过 12 V电源插座，可以给电子设备（例如：音乐播放器、"
    "车载冰箱和移动手机）充电。安全出行前排扶手箱 12V电源插座警告！"
)
# 真实语料里的原文（第 298 页 对外供电）——注意这条**没有**"例如"，
# 所以"看引导词"的规则会漏掉，必须靠谓词语义判定。
MICROWAVE_EVIDENCE = (
    "车辆供电外出时，车辆可以作为电源对照明、小功率烤箱、微波炉等家用电器"
    "或其它新能源车辆进行供电。"
)
CONFIG_EVIDENCE = "本车标配车载冰箱，位于前排中央扶手箱内。"
WARNING_EVIDENCE = "请勿在行车记录仪附近放置磁性物品。"
SCOPE_EVIDENCE = "座椅通风为高配车型选装配置，需在购车时选配。"
SEAT_EVIDENCE = "您可以通过中央显示屏，设置驾驶员/副驾驶员侧座椅加热强度或关闭座椅加热功能。"


def ev(text: str, citation: str = "[train_a.pdf 第129页]") -> dict:
    return {"text": text, "citation": citation}


class TestClassifyQuestion(unittest.TestCase):
    def test_existence_questions(self):
        for question, entity in (
            ("车内有冰箱吗", "冰箱"),
            ("有没有车载冰箱", "车载冰箱"),
            ("是否有行车记录仪", "行车记录仪"),
            ("本车配备座椅通风吗", "座椅通风"),
        ):
            qtype, found, _ = classify_question(question)
            self.assertEqual(qtype, "existence", question)
            self.assertEqual(found, entity, question)

    def test_operational_question_is_not_constrained(self):
        """操作类问句（怎么关/怎么设置）不该被存在性判据干扰。"""
        for question in ("座椅加热怎么关闭", "如何设置空调温度", "胎压报警了怎么办"):
            qtype, _, _ = classify_question(question)
            self.assertEqual(qtype, "operational", question)

    def test_scope_question(self):
        qtype, entity, wants_all = classify_question("全系都支持座椅通风吗")
        self.assertEqual(qtype, "scope")
        self.assertIn("座椅通风", entity)
        self.assertTrue(wants_all)

    def test_premise_question_flagged(self):
        qtype, _, _ = classify_question("为什么胎压报警后可以继续高速行驶")
        self.assertEqual(qtype, "premise")


class TestPhraseVariants(unittest.TestCase):
    """问法变体：同一语义的口语说法必须都能识别出实体。

    这条来自用户实测——第一版只认 "有/有没有/配备"，
    问「车内自带冰箱吗」时被判成"操作类"直接放行，于是又答成了那段 12V 电源原文。
    **模式匹配永远追不上口语变体**，所以这批用例必须一直绿。
    """

    VARIANTS = {
        "车内有冰箱吗": "冰箱", "车里有冰箱吗": "冰箱", "车上有没有冰箱": "冰箱",
        "车内自带冰箱吗": "冰箱", "这车自带冰箱吗": "冰箱", "车上配了冰箱吗": "冰箱",
        "车内是否配有冰箱": "冰箱", "车辆含冰箱吗": "冰箱", "车内带冰箱吗": "冰箱",
        "车内有冰箱么": "冰箱", "本车有冰箱吗": "冰箱",
    }

    def test_phrase_variants_all_recognized(self):
        for question, entity in self.VARIANTS.items():
            qtype, found, _ = classify_question(question)
            self.assertEqual(qtype, "existence", question)
            self.assertEqual(found, entity, question)

    def test_adjective_question_is_not_forced(self):
        """兜底识别出的实体只被口头提及（中性语境）时不得触发限定回答。

        反例：「车内安静吗」→ 兜底抽出「安静」，而手册里恰好有"安静"二字，
        会给出"手册没有说明本车配备「安静」"这种荒唐回复。
        """
        result = check_answerability(
            "车内安静吗", [ev("静音行驶可营造安静的车内氛围。")])
        self.assertEqual(result.verdict, ANSWERABLE, result.reason)
        self.assertTrue(result.from_fallback)

    def test_fallback_entity_still_limits_when_context_is_concrete(self):
        """兜底实体命中"受电/举例"语境时仍要限定（这是兜底的价值）。

        注意用「车载冰箱吗」这种**不含占有动词**的说法才能真正走到兜底分支：
        「车内自带冰箱吗」已经被主模式（自带）覆盖，测不到兜底。
        """
        result = check_answerability("车载冰箱吗", [ev(FRIDGE_EVIDENCE)])
        self.assertEqual(result.verdict, NEEDS_QUALIFICATION, result.reason)
        self.assertTrue(result.from_fallback, "这条应走兜底路径")


class TestPredicateJudgement(unittest.TestCase):
    def test_consume_role(self):
        hit = judge_occurrence(FRIDGE_EVIDENCE, "冰箱")
        self.assertIsNotNone(hit)
        self.assertEqual(hit.role, "consume", hit.snippet)

    def test_config_role(self):
        self.assertEqual(judge_occurrence(CONFIG_EVIDENCE, "冰箱").role, "config")

    def test_warning_role(self):
        self.assertEqual(judge_occurrence(WARNING_EVIDENCE, "行车记录仪").role, "warning")


class TestCheckAnswerability(unittest.TestCase):
    """冰箱那条是本模块存在的**全部理由**，必须逐条锁死。"""

    def test_fridge_is_not_answerable_by_evidence(self):
        result = check_answerability("车内有冰箱吗", [ev(FRIDGE_EVIDENCE)])
        self.assertEqual(result.verdict, NEEDS_QUALIFICATION)
        self.assertEqual(result.entity, "冰箱")
        self.assertTrue(result.has_related_evidence, "证据里提到了它，应能给出限定回答")
        self.assertIn("受电", result.reason)
        self.assertTrue(result.evidence_citation, "限定回答必须能带出处")

    def test_microwave_without_lead_word_is_still_caught(self):
        """回归：第一版靠"例如/比如"引导词，这条证据**没有引导词**，会漏判。

        必须靠谓词语义（"作为电源对…供电"）判定为受电对象，而不是看引导词。
        """
        result = check_answerability("车内有微波炉吗", [ev(MICROWAVE_EVIDENCE)])
        self.assertEqual(result.verdict, NEEDS_QUALIFICATION, result.reason)
        self.assertNotIn("例如", MICROWAVE_EVIDENCE, "这条证据本来就没有引导词")

    def test_config_evidence_is_answerable(self):
        result = check_answerability("车内有冰箱吗", [ev(CONFIG_EVIDENCE)])
        self.assertEqual(result.verdict, ANSWERABLE)
        self.assertEqual(result.hits[0].role, "config")

    def test_warning_context_is_not_answerable(self):
        result = check_answerability("车内有行车记录仪吗", [ev(WARNING_EVIDENCE)])
        self.assertEqual(result.verdict, NEEDS_QUALIFICATION)

    def test_entity_absent_is_not_answerable(self):
        """证据里根本没提到 → 交给拒答逻辑（本模块不改这种情况）。"""
        result = check_answerability("车内有冰箱吗", [ev(SEAT_EVIDENCE)])
        self.assertEqual(result.verdict, NOT_ANSWERABLE)
        self.assertFalse(result.has_related_evidence)

    def test_scope_question_requires_qualification(self):
        """问"全系都有吗"，证据说"高配选装" → 必须限定。"""
        result = check_answerability("全系都支持座椅通风吗", [ev(SCOPE_EVIDENCE)])
        self.assertEqual(result.verdict, NEEDS_QUALIFICATION)
        self.assertTrue(result.scope_mismatch, result.reason)

    def test_operational_question_passes_through(self):
        result = check_answerability("座椅加热怎么关闭", [ev(SEAT_EVIDENCE)])
        self.assertEqual(result.verdict, ANSWERABLE)
        self.assertEqual(result.question_type, "operational")

    def test_config_wins_over_consumable_in_other_chunk(self):
        """同一问题两份证据：一份是受电、一份是配置声明 → 应判定可回答。"""
        result = check_answerability(
            "车内有冰箱吗", [ev(FRIDGE_EVIDENCE), ev(CONFIG_EVIDENCE, "[p1 第5页]")])
        self.assertEqual(result.verdict, ANSWERABLE)


class TestQualifiedAnswer(unittest.TestCase):
    def test_answer_states_absence_and_gives_source(self):
        result = check_answerability("车内有冰箱吗", [ev(FRIDGE_EVIDENCE)])
        answer = build_qualified_answer(result)
        self.assertIn("手册没有说明", answer)
        self.assertIn("第129页", answer)
        self.assertIn("并不代表", answer)
        # 必须以否定句开头——不能变成"车里有冰箱"这种肯定答复
        self.assertTrue(answer.startswith("手册没有说明"), answer)

    def test_scope_answer_points_out_optional_config(self):
        result = check_answerability("全系都支持座椅通风吗", [ev(SCOPE_EVIDENCE)])
        answer = build_qualified_answer(result)
        self.assertIn("不是全系标准配置", answer)


class TestAgentIntegration(unittest.TestCase):
    """接进 AgentGraph 之后的行为：status=limited，且不影响正常问答。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        path = os.path.join(cls._tmp.name, "ans_corpus.jsonl")
        rows = [
            {"id": 1, "text": FRIDGE_EVIDENCE, "source": "train_a.pdf", "page": 129},
            {"id": 2, "text": MICROWAVE_EVIDENCE, "source": "train_a.pdf", "page": 298},
            {"id": 3, "text": SEAT_EVIDENCE, "source": "train_a.pdf", "page": 114},
            # 刻意**不放** CONFIG_EVIDENCE：它会让"车内有冰箱吗"变成可回答，
            # 那是另一个场景（配置表里确实写了），由下面的 TestConfigCorpus 单独覆盖。
        ]
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        cls.kb = KnowledgeBase.load(path)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _agent(self, enabled: bool = True) -> AgentGraph:
        config = AgentConfig()
        config.enable_answerability = enabled
        return AgentGraph(RuleBasedPlannerLLM(), ToolRegistry(self.kb), config=config)

    def test_fridge_question_returns_limited_answer(self):
        state = self._agent().run("车内有冰箱吗")
        self.assertEqual(state.status, "limited", state.answer)
        self.assertIn("手册没有说明", state.answer)
        self.assertIn("第129页", state.answer)
        self.assertTrue(state.citations, "限定回答必须保留出处")
        self.assertEqual(state.answerability.get("verdict"), "needs_qualification")
        self.assertTrue(any(t.get("node") == "answerability" for t in state.trace),
                        "trace 里应记录可回答性判定，便于线上排查")

    def test_qualified_answer_puts_evidence_citation_first(self):
        state = self._agent().run("车内有冰箱吗")
        self.assertIn("第129页", state.citations[0],
                      "被引用的那条必须排在最前，否则用户找不到页码")

    def test_operational_question_unaffected(self):
        state = self._agent().run("座椅加热怎么关闭")
        self.assertEqual(state.status, "answered")

    def test_switch_can_disable_the_layer(self):
        """开关关掉后回到旧行为（这段话是"能关掉"的证明，也是可控性的证明）。"""
        state = self._agent(enabled=False).run("车内有冰箱吗")
        self.assertEqual(state.status, "answered")
        self.assertNotIn("手册没有说明", state.answer)


class TestConfigCorpus(unittest.TestCase):
    """配置表里确实写了"标配"时的行为：应当正常回答，**不能**被限定制裁掉。"""

    def test_configured_item_is_answered_normally(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path = os.path.join(tmp, "fit_corpus.jsonl")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"id": 4, "text": CONFIG_EVIDENCE,
                                     "source": "fit.pdf", "page": 5},
                                    ensure_ascii=False) + "\n")
            kb = KnowledgeBase.load(path)
            agent = AgentGraph(RuleBasedPlannerLLM(), ToolRegistry(kb),
                               config=AgentConfig())
            state = agent.run("车内有冰箱吗")
            self.assertEqual(state.status, "answered", state.answer)
            self.assertNotIn("手册没有说明", state.answer)


class TestSpecQuestion(unittest.TestCase):
    """数值题（"最快能开到多少"）：判据是"证据里有没有该参数的数值"。

    这条来自用户实测：问「车最快能开到多少」，旧代码把动词短语当实体，
    回了一句"手册没有明确说明本车支持「开到多少」"——荒唐。
    """

    HEADER = "技术资料350动力性能车型驱动方式最大设计车速（km/h）最大爬坡度"
    ROW = "（a）（%）MR6453DCH-EV02两驱20040（a）：车辆以坡底起步。"
    COMBINED = ("技术资料350动力性能车型 驱动方式最大设计车速（km/h）"
                "最大爬坡度（a）（%）MR6453DCH-EV02两驱20040")

    def test_page_number_is_not_a_value(self):
        """回归：页眉里的页码（350）紧挨着单位（km/h），曾被当成参数值。

        第一版没有剥页码，于是「车最快能开到多少」被判成 answerable，
        但实际上拿到的只是表头，一个数值都没有。
        """
        result = check_answerability("车最快能开到多少", [ev(self.HEADER)])
        self.assertEqual(result.verdict, NEEDS_QUALIFICATION, result.reason)

    def test_unit_before_and_after_number_both_count(self):
        for text in (self.COMBINED, "最大设计车速 200 km/h"):
            result = check_answerability("车最快能开到多少", [ev(text)])
            self.assertEqual(result.verdict, ANSWERABLE, text)

    def test_spec_question_never_extracts_verb_phrase(self):
        """「开到多少」不能变成实体（否则会冒出"支持「开到多少」"这种句子）。"""
        qtype, entity, _ = classify_question("车最快能开到多少")
        self.assertEqual(qtype, "spec")
        self.assertNotIn("开到", entity)
        self.assertEqual(entity, "最高车速")

    def test_quantified_question_without_value_is_qualified(self):
        result = check_answerability("车最快能开到多少", [ev(self.HEADER)])
        answer = build_qualified_answer(result)
        self.assertIn("具体数值", answer)
        self.assertNotIn("支持「", answer)


class TestCacheableStatus(unittest.TestCase):
    def test_limited_is_cacheable(self):
        """限定回答是确定性、带出处的结论，缓存它不会把错误固化。"""
        from agent.cache import cacheable_status

        self.assertTrue(cacheable_status("answered"))
        self.assertTrue(cacheable_status("limited"))

    def test_failures_are_not_cacheable(self):
        from agent.cache import cacheable_status

        for status in ("refused", "needs_confirmation", "error", "blocked", "max_steps"):
            self.assertFalse(cacheable_status(status), status)


if __name__ == "__main__":
    unittest.main(verbosity=2)
