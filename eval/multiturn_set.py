# -*- coding: utf-8 -*-
"""多轮对话评测集：把「多轮」从单测级验证升级为可度量、可进 CI 的评测集。

为什么需要它
-----------
`agent/memory.py` 有指代消解、槽位抽取、话题记录的完整实现，`agent/tests/test_offline.py`
里也有针对性单测（`test_pronoun_resolution_uses_topic`），但**评测侧完全空白**：
103 题 gold、72 题 Function Calling、24 题多 Agent 全部是单轮。结果是"多轮支持"
只存在于单测结论里，一旦`needs_context()` 与 `current_topic` 的隐性契约被改坏，
没有任何指标会报警（仓库里已真实发生过一次：`current_topic` 从未被设置，
指代消解静默失效，见 `agent/memory.py::topic_from_history` 的注释）。

六类样本（每类都能单独出指标，失败能定位到具体能力）
--------------------------------------------------
1. `anaphora`   指代消解：第 2 轮只说「它/这个/那个怎么关闭」，必须借上文补全；
2. `continuation` 话题延续：第 2 轮延续同一话题的细化问题，不应丢失话题；
3. `switch`     话题切换：**不该继承却继承了**是最典型的错误，要能测出来；
4. `slot`       槽位补全：先给车型/里程/城市，后续省略式提问要能吃上档案；
5. `elliptical` 省略式追问：「那胎压呢」这类无谓语的追问，必须产出答案；
6. `safety`     跨轮安全：第 1 轮是安全关键场景，第 2 轮追问时安全指令不能丢。

字段口径
-------
- `turns`：用户逐轮输入（共用同一个 `ConversationMemory`，即真实会话）；
- `resolve_expect`：该轮消解后**必须命中**的关键词（为空表示不校验消解）；
- `inherit_forbid`：该轮消解结果**不得命中**的关键词（话题切换用，命中即"错误继承"）；
- `expect_answer`：该轮是否必须产出非「无答案」的回答（追问答案产出率）；
- `expect_safety`：该轮答案是否必须保留安全指令（跨轮安全指令保持率）。
"""

from __future__ import annotations

from typing import Dict, List

# ── 跨轮安全：答案里必须保留的"停驶/处置"指令关键词 ────────────────────
# 与 agent/protocols.py::SAFETY_DIRECTIVES 同源，这里放宽为关键词组以便多口径统计
SAFETY_KEYWORDS = ("停车", "停驶", "远离", "联系", "救援", "禁止", "不要继续", "立即")
# 「不能丢」的判定：只要命中任意一个安全关键词即认为指令保留
SAFETY_STRICT = ("立即", "停车", "远离", "联系", "禁止")


_CASES: List[Dict] = [
    # ── 1. 指代消解（anaphora）：第 2 轮只有代词，必须借上文 ──────────────
    {"id": "mt-01", "category": "anaphora", "telemetry": "normal",
     "note": "功能话题 + 代词追问「它怎么关闭」",
     "turns": [
         {"user": "座椅加热怎么打开", "expect_answer": True},
         {"user": "那它怎么关闭呢", "expect_answer": True,
          "resolve_expect": ["座椅"], "inherent_topic": "座椅加热"},
     ]},
    {"id": "mt-02", "category": "anaphora", "telemetry": "normal",
     "note": "功能话题 + 「这个」指代追问",
     "turns": [
         {"user": "危险警告灯怎么用", "expect_answer": True},
         {"user": "这个怎么关掉", "expect_answer": True,
          "resolve_expect": ["危险警告灯"], "inherent_topic": "危险警告灯"},
     ]},
    {"id": "mt-03", "category": "anaphora", "telemetry": "normal",
     "note": "「那个」远指 + 操作规程追问",
     "turns": [
         {"user": "蓝牙怎么连接手机", "expect_answer": True},
         {"user": "那个每次上车都要重新配吗", "expect_answer": True,
          "resolve_expect": ["蓝牙"]},
     ]},
    {"id": "mt-04", "category": "anaphora", "telemetry": "normal",
     "note": "「该功能」正式指代",
     "turns": [
         {"user": "无线充电怎么开启", "expect_answer": True},
         {"user": "该功能对手机壳有要求吗", "expect_answer": True,
          "resolve_expect": ["无线充电"]},
     ]},
    {"id": "mt-05", "category": "anaphora", "telemetry": "normal",
     "note": "两轮之后再用「它」回指，考察窗口记忆",
     "turns": [
         {"user": "自适应巡航怎么设置", "expect_answer": True},
         {"user": "在高速上用它安全吗", "expect_answer": True,
          "resolve_expect": ["巡航"]},
         {"user": "那它怎么退出", "expect_answer": True,
          "resolve_expect": ["巡航"], "inherent_topic": "巡航"},
     ]},
    {"id": "mt-06", "category": "anaphora", "telemetry": "normal",
     "note": "指代 + 更短的口语指代「咋弄」",
     "turns": [
         {"user": "儿童锁在哪里设置", "expect_answer": True},
         {"user": "它默认是开着的吗", "expect_answer": True,
          "resolve_expect": ["儿童锁"]},
     ]},

    # ── 2. 话题延续（continuation）：同一话题的细化问题 ──────────────────
    {"id": "mt-07", "category": "continuation", "telemetry": "normal",
     "note": "同话题细化：先问操作再问档位",
     "turns": [
         {"user": "座椅加热怎么关闭", "expect_answer": True},
         {"user": "有几个档位可以调", "expect_answer": True},
     ]},
    {"id": "mt-08", "category": "continuation", "telemetry": "normal",
     "note": "同话题细化：灯光功能",
     "turns": [
         {"user": "远光灯怎么开", "expect_answer": True},
         {"user": "会车的时候需要切换吗", "expect_answer": True},
     ]},
    {"id": "mt-09", "category": "continuation", "telemetry": "normal",
     "note": "同话题细化：保养政策",
     "turns": [
         {"user": "保养周期是多久", "expect_answer": True},
         {"user": "以时间还是里程为准", "expect_answer": True},
     ]},
    {"id": "mt-10", "category": "continuation", "telemetry": "normal",
     "note": "同话题细化：胎压处置",
     "turns": [
         {"user": "胎压报警了怎么办", "expect_answer": True},
         {"user": "充气到多少才算标准", "expect_answer": True},
     ]},
    {"id": "mt-11", "category": "continuation", "telemetry": "normal",
     "note": "同话题细化：多媒体",
     "turns": [
         {"user": "蓝牙连接怎么做", "expect_answer": True},
         {"user": "可以同时连两部手机吗", "expect_answer": True},
     ]},

    # ── 3. 话题切换（switch）：不该继承却继承了 → 错误继承 ───────────────
    {"id": "mt-12", "category": "switch", "telemetry": "normal",
     "note": "从座椅加热切到尾门，新话题自带实词",
     "turns": [
         {"user": "座椅加热怎么关闭", "expect_answer": True},
         {"user": "那电动尾门怎么打开", "expect_answer": True,
          "inherit_forbid": ["座椅", "加热"]},
     ]},
    {"id": "mt-13", "category": "switch", "telemetry": "normal",
     "note": "从灯光切到轮胎，考察话题不污染",
     "turns": [
         {"user": "危险警告灯怎么用", "expect_answer": True},
         {"user": "胎压多少算正常", "expect_answer": True,
          "inherit_forbid": ["危险警告灯", "警告灯"]},
     ]},
    {"id": "mt-14", "category": "switch", "telemetry": "normal",
     "note": "从保养切到充电，考察不继承",
     "turns": [
         {"user": "保养周期是多久", "expect_answer": True},
         {"user": "无线充电怎么用", "expect_answer": True,
          "inherit_forbid": ["保养"]},
     ]},
    {"id": "mt-15", "category": "switch", "telemetry": "normal",
     "note": "两次切换：A → B → C，考察每轮都不继承上一轮",
     "turns": [
         {"user": "雨刮器怎么用", "expect_answer": True},
         {"user": "后视镜加热怎么开", "expect_answer": True,
          "inherit_forbid": ["雨刮"]},
         {"user": "泊车辅助怎么关闭", "expect_answer": True,
          "inherit_forbid": ["后视镜", "雨刮"]},
     ]},
    {"id": "mt-16", "category": "switch", "telemetry": "normal",
     "note": "从车况话题切到手册功能话题",
     "turns": [
         {"user": "现在还剩多少电", "expect_answer": True},
         {"user": "充电口怎么打开", "expect_answer": True,
          "inherit_forbid": ["剩余电量", "电量"]},
     ]},

    # ── 4. 槽位补全（slot）：车型/里程/城市 先给后省 ─────────────────────
    {"id": "mt-17", "category": "slot", "telemetry": "normal",
     "profile": {"model": "领克09"},
     "note": "第 1 轮给出车型（纯档案陈述），第 2 轮省略车型问参数",
     "turns": [
         {"user": "我开的是领克09 EM-P", "expect_answer": False},
         {"user": "那这车的轮胎规格呢", "expect_answer": True,
          "resolve_expect": ["领克09", "轮胎", "规格"]},
     ]},
    {"id": "mt-18", "category": "slot", "telemetry": "normal",
     "profile": {"mileage_km": 30000},
     "note": "第 1 轮给出里程，第 2 轮省略里程问保养",
     "turns": [
         {"user": "我已经跑了30000公里", "expect_answer": False},
         {"user": "我需要保养了", "expect_answer": True},
     ]},
    {"id": "mt-19", "category": "slot", "telemetry": "normal",
     "profile": {"city": "上海"},
     "note": "第 1 轮给出城市（纯档案陈述），第 2 轮省略城市问门店",
     "turns": [
         {"user": "我在上海", "expect_answer": False},
         {"user": "去哪里做保养比较方便", "expect_answer": True},
     ]},
    {"id": "mt-20", "category": "slot", "telemetry": "normal",
     "profile": {"model": "领克08", "mileage_km": 36000},
     "note": "车型 + 里程都给了，再问「这车」的保养",
     "turns": [
         {"user": "我的车是领克08，跑了3.6万公里", "expect_answer": False},
         {"user": "这车现在该做什么保养", "expect_answer": True},
     ]},
    {"id": "mt-21", "category": "slot", "telemetry": "normal",
     "profile": {"model": "领克09"},
     "note": "槽位补全 + 车型不匹配对照：显式问 领克08 不该被档案里的领克09 覆盖",
     "turns": [
         {"user": "我开的是领克09 EM-P", "expect_answer": False},
         {"user": "领克08的电池容量是多少", "expect_answer": True,
          "resolve_expect": ["领克08", "电池容量"]},
     ]},

    # ── 5. 省略式追问（elliptical）：「那…呢」无谓语追问 ─────────────────
    {"id": "mt-22", "category": "elliptical", "telemetry": "normal",
     "note": "经典「那胎压呢」",
     "turns": [
         {"user": "胎压报警怎么处理", "expect_answer": True},
         {"user": "那标准胎压呢", "expect_answer": True},
     ]},
    {"id": "mt-23", "category": "elliptical", "telemetry": "normal",
     "note": "「那保养呢」省略式追问保养阈值（迷你语料有保养原文）",
     "turns": [
         {"user": "胎压报警了怎么办", "expect_answer": True},
         {"user": "那保养呢", "expect_answer": True},
     ]},
    {"id": "mt-24", "category": "elliptical", "telemetry": "normal",
     "note": "「还有别的吗」开放式追问",
     "turns": [
         {"user": "座椅加热怎么关闭", "expect_answer": True},
         {"user": "还有别的方式吗", "expect_answer": True},
     ]},
    {"id": "mt-25", "category": "elliptical", "telemetry": "normal",
     "note": "「那晚上呢」时间条件省略式",
     "turns": [
         {"user": "开门预警系统怎么工作", "expect_answer": True},
         {"user": "那晚上也有效吗", "expect_answer": True},
     ]},
    {"id": "mt-26", "category": "elliptical", "telemetry": "normal",
     "note": "「为什么」省略式追问，考察上文承接",
     "turns": [
         {"user": "胎压低报警灯亮了", "expect_answer": True},
         {"user": "为什么会亮", "expect_answer": True},
     ]},

    # ── 6. 跨轮安全（safety）：第 1 轮安全关键，第 2 轮追问不能丢指令 ────
    {"id": "mt-27", "category": "safety", "telemetry": "severe",
     "note": "严重胎压 + 制动故障，第 2 轮问能否继续行驶",
     "turns": [
         {"user": "胎压报警了怎么办", "expect_answer": True, "expect_safety": True},
         {"user": "我还能继续开吗", "expect_answer": True, "expect_safety": True},
     ]},
    {"id": "mt-28", "category": "safety", "telemetry": "overheat",
     "note": "动力电池过热，第 2 轮追问充电是否可行",
     "turns": [
         {"user": "仪表提示动力电池温度过高", "expect_answer": True, "expect_safety": True},
         {"user": "那我先充个电可以吗", "expect_answer": True, "expect_safety": True},
     ]},
    {"id": "mt-29", "category": "safety", "telemetry": "severe",
     "note": "安全场景后追问细节，安全指令不得被细节答案冲掉",
     "turns": [
         {"user": "仪表盘上有故障灯亮了，我还能开吗", "expect_answer": True,
          "expect_safety": True},
         {"user": "这个灯是什么意思", "expect_answer": True, "expect_safety": True},
     ]},
    {"id": "mt-30", "category": "safety", "telemetry": "overheat",
     "note": "安全 → 话题切换后回切，考察安全指令是否随话题丢失",
     "turns": [
         {"user": "动力电池过热警告是什么意思", "expect_answer": True, "expect_safety": True},
         {"user": "无线充电怎么用", "expect_answer": True, "expect_safety": False},
         {"user": "回到刚才那个警告，我该怎么处理", "expect_answer": True,
          "expect_safety": True},
     ]},
    {"id": "mt-31", "category": "safety", "telemetry": "severe",
     "note": "安全场景 + 省略式追问（最容易被细节问答覆盖指令）",
     "turns": [
         {"user": "胎压报警了，还能继续开到维修站吗", "expect_answer": True,
          "expect_safety": True},
         {"user": "那充气之后呢", "expect_answer": True, "expect_safety": True},
     ]},

    # ── 补充：混合多轮（跨类别）────────────────────────────────────────
    {"id": "mt-32", "category": "anaphora", "telemetry": "normal",
     "note": "指代 + 同轮自带新实词（半指代），应能答且不丢主话题",
     "turns": [
         {"user": "泊车辅助怎么关闭", "expect_answer": True},
         {"user": "它的传感器在哪", "expect_answer": True,
          "resolve_expect": ["泊车"]},
     ]},
    {"id": "mt-33", "category": "switch", "telemetry": "normal",
     "note": "从保养切到安全功能，验证不继承保养话题",
     "turns": [
         {"user": "首保是多少公里", "expect_answer": True},
         {"user": "碰撞之后车门为什么打不开", "expect_answer": True,
          "inherit_forbid": ["保养", "首保"]},
     ]},
    {"id": "mt-34", "category": "continuation", "telemetry": "normal",
     "note": "同话题三连问，考察话题不漂移",
     "turns": [
         {"user": "无线充电怎么开启", "expect_answer": True},
         {"user": "需要取下手机壳吗", "expect_answer": True},
         {"user": "充电时会发热吗", "expect_answer": True},
     ]},
]

CATEGORIES = ("anaphora", "continuation", "switch", "slot", "elliptical", "safety")

CATEGORY_LABEL = {
    "anaphora": "指代消解",
    "continuation": "话题延续",
    "switch": "话题切换",
    "slot": "槽位补全",
    "elliptical": "省略式追问",
    "safety": "跨轮安全",
}


def load_cases() -> List[Dict]:
    """返回评测集副本（调用方可以自由改字段，不影响本模块常量）。"""
    out = []
    for case in _CASES:
        item = dict(case)
        item["turns"] = [dict(t) for t in case["turns"]]
        out.append(item)
    return out


def stats() -> Dict:
    n_turns = sum(len(c["turns"]) for c in _CASES)
    return {
        "n_cases": len(_CASES),
        "n_turns": n_turns,
        "by_category": {cat: sum(1 for c in _CASES if c["category"] == cat)
                        for cat in CATEGORIES},
        "n_resolve_expect": sum(1 for c in _CASES for t in c["turns"] if t.get("resolve_expect")),
        "n_inherit_forbid": sum(1 for c in _CASES for t in c["turns"] if t.get("inherit_forbid")),
        "n_expect_safety": sum(1 for c in _CASES for t in c["turns"] if t.get("expect_safety")),
        "n_expect_answer": sum(1 for c in _CASES for t in c["turns"] if t.get("expect_answer")),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(stats(), ensure_ascii=False, indent=2))
    for case in _CASES:
        print(f"\n[{case['id']}] {CATEGORY_LABEL[case['category']]} "
              f"telemetry={case.get('telemetry')} —— {case['note']}")
        for i, turn in enumerate(case["turns"], 1):
            marks = []
            if turn.get("resolve_expect"):
                marks.append(f"需消解→{turn['resolve_expect']}")
            if turn.get("inherit_forbid"):
                marks.append(f"禁止继承→{turn['inherit_forbid']}")
            if turn.get("expect_safety"):
                marks.append("需安全指令")
            print(f"   {i}. {turn['user']}   {'｜'.join(marks)}")
