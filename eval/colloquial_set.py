# -*- coding: utf-8 -*-
"""口语化问法测试集：补齐「103 题全是手册术语」这个已承认的评测短板。

问题背景
-------
现有 103 题 gold（`data/gold.json`）与 72 题 Function Calling 集**全部使用手册术语**
（"危险警告灯""座椅加热""保养周期"），而真实车主说的是"双闪""靠背太热""多少公里换机油"。
后果有两个：
1. 检索指标被系统性高估——用手册原词去检索手册原文，任何分词器都能命中；
2. **query 改写实验测不出增益**：`agent/graph.py::_reformulate` 与
   `RuleBasedPlannerLLM._normalize_query` 里有口语→术语映射表，但因为没有口语测试集，
   "改写到底有没有用"从来没有被量过（`eval/agent_report.md` 里的改写消融同样建立在术语题上）。

本测试集的设计
-------------
每条样本标注：
- `colloquial`：车主真实口语问法（评测的输入）；
- `manual_term`：它**对应哪条手册术语**（人工标注，可审计）；
- `expect_keywords`：命中的知识块里**必须出现**的关键词（用 `kb.chunks` 原文核对过）；
- `rewrite_terms`：现有改写表（`_normalize_query` 的 mapping）里能命中的词，
  用于解释"改写为什么有/没有生效"；
- `note`：该条考察的口语化类型。

⚠️ 关键词是"词面锚点"而非唯一正确答案：只要 top_k 命中块里出现了锚点，就算检索到了
对应知识点；这是可复现、可审计的代理指标（与 `agent/eval_agent.py` 的 context recall 同源）。
"""

from __future__ import annotations

from typing import Dict, List

# 现有改写表的键（agent/llm.py::RuleBasedPlannerLLM._normalize_query 的 mapping）
KNOWN_REWRITE_KEYS = ("靠背太热", "靠背发烫", "屁股热", "怎么关", "打不开",
                      "亮黄灯", "空调不凉", "没电了", "刹车")


_CASES: List[Dict] = [
    # ── 座椅 / 空调 ────────────────────────────────────────────────────
    {"id": "cq-01", "manual_term": "座椅加热", "colloquial": "靠背太热了怎么关掉",
     "expect_keywords": ["座椅"], "rewrite_terms": ["靠背太热", "怎么关"],
     "note": "身体部位口语 + 口语动词"},
    {"id": "cq-02", "manual_term": "座椅加热", "colloquial": "屁股底下那个加热咋关",
     "expect_keywords": ["座椅"], "rewrite_terms": ["屁股热"],
     "note": "部位口语 + 方言化动词「咋关」"},
    {"id": "cq-03", "manual_term": "座椅加热", "colloquial": "车里那个加热座椅在哪调",
     "expect_keywords": ["座椅"], "rewrite_terms": [],
     "note": "倒装口语（加热座椅）"},
    {"id": "cq-04", "manual_term": "空调制冷", "colloquial": "空调制冷不凉",
     "expect_keywords": ["空调"], "rewrite_terms": ["空调不凉"],
     "note": "状态描述式口语（不凉）"},
    {"id": "cq-05", "manual_term": "空调开启", "colloquial": "车里太热了怎么开空调",
     "expect_keywords": ["空调"], "rewrite_terms": [],
     "note": "从体感出发，不带术语"},

    # ── 灯光 / 危险警告灯 ──────────────────────────────────────────────
    {"id": "cq-06", "manual_term": "危险警告灯", "colloquial": "双闪怎么打开",
     "expect_keywords": ["危险警告灯", "危险"], "rewrite_terms": [],
     "note": "行业黑话「双闪」"},
    {"id": "cq-07", "manual_term": "危险警告灯", "colloquial": "车坏在路上了要开什么灯",
     "expect_keywords": ["危险警告灯", "危险"], "rewrite_terms": [],
     "note": "场景描述式，完全不含术语"},
    {"id": "cq-08", "manual_term": "危险警告灯", "colloquial": "紧急时候按哪个灯",
     "expect_keywords": ["危险警告灯", "危险"], "rewrite_terms": [],
     "note": "场景 + 疑问指代"},
    {"id": "cq-09", "manual_term": "远光灯", "colloquial": "晚上开车灯怎么切",
     "expect_keywords": ["远光"], "rewrite_terms": [],
     "note": "只说场景（晚上），不说灯型"},

    # ── 轮胎 / 胎压 ────────────────────────────────────────────────────
    {"id": "cq-10", "manual_term": "胎压过低报警", "colloquial": "轮胎报警灯亮了咋办",
     "expect_keywords": ["胎压", "轮胎"], "rewrite_terms": [],
     "note": "把「胎压」说成「轮胎」"},
    {"id": "cq-11", "manual_term": "胎压过低报警", "colloquial": "仪表上那个轮胎形状的灯亮了",
     "expect_keywords": ["胎压", "轮胎"], "rewrite_terms": ["亮黄灯"],
     "note": "用图形描述替代术语"},
    {"id": "cq-12", "manual_term": "标准胎压", "colloquial": "轮胎没气了还能开吗",
     "expect_keywords": ["胎压", "轮胎"], "rewrite_terms": [],
     "note": "现象式口语（没气）"},

    # ── 保养 ───────────────────────────────────────────────────────────
    {"id": "cq-13", "manual_term": "保养周期", "colloquial": "车多久要做一次保养",
     "expect_keywords": ["保养"], "rewrite_terms": [],
     "note": "时间口语化"},
    {"id": "cq-14", "manual_term": "保养项目（机油机滤）", "colloquial": "多少公里要换机油",
     "expect_keywords": ["保养", "机油"], "rewrite_terms": [],
     "note": "以具体零件代指保养"},
    {"id": "cq-15", "manual_term": "首次保养", "colloquial": "首保啥时候去",
     "expect_keywords": ["保养"], "rewrite_terms": [],
     "note": "圈内简称「首保」"},

    # ── 多媒体 / 充电 ──────────────────────────────────────────────────
    {"id": "cq-16", "manual_term": "蓝牙连接", "colloquial": "手机怎么连车上蓝牙",
     "expect_keywords": ["蓝牙"], "rewrite_terms": [],
     "note": "语序口语化"},
    {"id": "cq-17", "manual_term": "蓝牙连接", "colloquial": "车里怎么放歌不用插线",
     "expect_keywords": ["蓝牙", "无线充电", "充电"], "rewrite_terms": [],
     "note": "需求描述式，零术语"},
    {"id": "cq-18", "manual_term": "无线充电", "colloquial": "手机放上去就能充电吗",
     "expect_keywords": ["无线充电", "充电"], "rewrite_terms": [],
     "note": "动作描述替代术语"},
    {"id": "cq-19", "manual_term": "充电口盖", "colloquial": "充电盖打不开",
     "expect_keywords": ["充电口", "充电"], "rewrite_terms": ["打不开"],
     "note": "省略「口」+ 故障口吻"},

    # ── 车身 / 尾门 / 机舱 ─────────────────────────────────────────────
    {"id": "cq-20", "manual_term": "电动尾门", "colloquial": "后备箱怎么电动打开",
     "expect_keywords": ["尾门"], "rewrite_terms": [],
     "note": "「后备箱」vs 手册「尾门/后备厢」"},
    {"id": "cq-21", "manual_term": "电动尾门", "colloquial": "后尾门打不开",
     "expect_keywords": ["尾门"], "rewrite_terms": ["打不开"],
     "note": "口语复合词「后尾门」"},
    {"id": "cq-22", "manual_term": "前机舱盖", "colloquial": "引擎盖怎么打开",
     "expect_keywords": ["机舱盖", "机舱"], "rewrite_terms": [],
     "note": "「引擎盖」vs 手册「前机舱盖」"},
    {"id": "cq-23", "manual_term": "前机舱盖", "colloquial": "前盖怎么开",
     "expect_keywords": ["机舱盖", "机舱"], "rewrite_terms": [],
     "note": "极简口语"},

    # ── 雨刮 / 后视镜 ──────────────────────────────────────────────────
    {"id": "cq-24", "manual_term": "雨刮器", "colloquial": "雨刷怎么喷水",
     "expect_keywords": ["雨刮"], "rewrite_terms": [],
     "note": "「雨刷」vs 手册「雨刮器」"},
    {"id": "cq-25", "manual_term": "雨刮器", "colloquial": "下雨天玻璃看不清怎么办",
     "expect_keywords": ["雨刮"], "rewrite_terms": [],
     "note": "现象式描述"},
    {"id": "cq-26", "manual_term": "外后视镜加热", "colloquial": "后视镜起雾了怎么办",
     "expect_keywords": ["后视镜"], "rewrite_terms": [],
     "note": "现象 + 部件名"},
    {"id": "cq-27", "manual_term": "外后视镜加热", "colloquial": "镜子看不清怎么除雾",
     "expect_keywords": ["后视镜", "除霜"], "rewrite_terms": [],
     "note": "省略「后视」"},

    # ── 安全功能 ───────────────────────────────────────────────────────
    {"id": "cq-28", "manual_term": "儿童锁", "colloquial": "小孩坐后面怕乱开门怎么办",
     "expect_keywords": ["儿童锁", "儿童"], "rewrite_terms": [],
     "note": "以场景代指功能"},
    {"id": "cq-29", "manual_term": "儿童锁", "colloquial": "后排门从里面打不开",
     "expect_keywords": ["儿童锁", "儿童"], "rewrite_terms": ["打不开"],
     "note": "故障现象式（其实是儿童锁已启用）"},
    {"id": "cq-30", "manual_term": "安全带未系提醒", "colloquial": "安全带没系会响吗",
     "expect_keywords": ["安全带"], "rewrite_terms": [],
     "note": "口语疑问"},
    {"id": "cq-31", "manual_term": "开门预警系统（DOW）", "colloquial": "下车开门有车过来会提醒吗",
     "expect_keywords": ["开门预警"], "rewrite_terms": [],
     "note": "场景描述，零术语"},
    {"id": "cq-32", "manual_term": "开门预警系统（DOW）", "colloquial": "门没关好会提醒吗",
     "expect_keywords": ["开门预警", "车门"], "rewrite_terms": [],
     "note": "近似但不同义（语义漂移样本）"},
    {"id": "cq-33", "manual_term": "防盗系统", "colloquial": "车被偷了会响吗",
     "expect_keywords": ["防盗"], "rewrite_terms": [],
     "note": "结果导向式提问"},
    {"id": "cq-34", "manual_term": "防盗系统", "colloquial": "锁车之后有人开门会报警吗",
     "expect_keywords": ["防盗"], "rewrite_terms": [],
     "note": "场景 + 时间条件"},
    {"id": "cq-35", "manual_term": "碰撞后处置", "colloquial": "撞车之后门打不开",
     "expect_keywords": ["碰撞"], "rewrite_terms": ["打不开"],
     "note": "「撞车」vs 手册「碰撞」"},
    {"id": "cq-36", "manual_term": "碰撞后处置", "colloquial": "出事故了怎么办",
     "expect_keywords": ["碰撞", "安全气囊"], "rewrite_terms": [],
     "note": "极短场景问"},

    # ── 驾驶辅助 / 泊车 ────────────────────────────────────────────────
    {"id": "cq-37", "manual_term": "自适应巡航", "colloquial": "高速上那个自动跟车怎么用",
     "expect_keywords": ["巡航"], "rewrite_terms": [],
     "note": "功能描述替代术语（自动跟车）"},
    {"id": "cq-38", "manual_term": "自适应巡航", "colloquial": "定速巡航咋开",
     "expect_keywords": ["巡航"], "rewrite_terms": [],
     "note": "近似术语（定速 vs 自适应）"},
    {"id": "cq-39", "manual_term": "泊车辅助", "colloquial": "倒车雷达怎么关",
     "expect_keywords": ["泊车"], "rewrite_terms": ["怎么关"],
     "note": "「倒车雷达」vs 手册「泊车辅助传感器」"},
    {"id": "cq-40", "manual_term": "泊车辅助", "colloquial": "停车的时候老报警怎么关掉",
     "expect_keywords": ["泊车"], "rewrite_terms": ["怎么关"],
     "note": "现象 + 口语动词"},
    {"id": "cq-41", "manual_term": "泊车辅助", "colloquial": "这车的自动泊车怎么用",
     "expect_keywords": ["泊车"], "rewrite_terms": [],
     "note": "「自动泊车」营销词 vs 手册「泊车辅助」"},

    # ── 高压系统 / 仪表 ────────────────────────────────────────────────
    {"id": "cq-42", "manual_term": "动力电池过热警告", "colloquial": "电池过热了怎么办",
     "expect_keywords": ["动力电池", "过热"], "rewrite_terms": [],
     "note": "省略「动力」"},
    {"id": "cq-43", "manual_term": "动力电池过热警告", "colloquial": "仪表说电池温度高",
     "expect_keywords": ["动力电池", "过热"], "rewrite_terms": [],
     "note": "转述仪表提示"},
    {"id": "cq-44", "manual_term": "组合仪表", "colloquial": "仪表盘上那些灯都是啥意思",
     "expect_keywords": ["组合仪表", "仪表"], "rewrite_terms": [],
     "note": "泛指式提问"},
    {"id": "cq-45", "manual_term": "危险警告灯", "colloquial": "车抛锚了怎么提醒后车",
     "expect_keywords": ["危险警告灯", "危险"], "rewrite_terms": [],
     "note": "需求式长问句"},
    {"id": "cq-46", "manual_term": "蓝牙连接", "colloquial": "手机连车机老是断",
     "expect_keywords": ["蓝牙"], "rewrite_terms": [],
     "note": "故障口吻（口语化最难的一类）"},
]

COLLOQUIAL_KINDS = ("黑话", "场景描述", "现象描述", "语序口语", "近似术语")


def load_cases() -> List[Dict]:
    return [dict(c) for c in _CASES]


def stats() -> Dict:
    return {
        "n_cases": len(_CASES),
        "n_with_rewrite_term": sum(1 for c in _CASES if c["rewrite_terms"]),
        "n_manual_terms": len({c["manual_term"] for c in _CASES}),
        "by_manual_term": {},
    }


if __name__ == "__main__":
    import json

    print(json.dumps(stats(), ensure_ascii=False, indent=2))
    for case in _CASES:
        rw = "改写命中" if case["rewrite_terms"] else "改写不命中"
        print(f"{case['id']}  {case['colloquial']:<24} → {case['manual_term']:<22} "
              f"[{rw}] {case['expect_keywords']}  # {case['note']}")
