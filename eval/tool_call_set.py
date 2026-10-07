# -*- coding: utf-8 -*-
"""Function Calling 标注评测集：query → 期望工具集与关键参数。

共 72 条，覆盖四类判定：
- **should_call**：必须调用工具（事实性问题不能凭空回答）；
- **should_not_call**：不应调用工具（闲聊/澄清/元问题）；
- **multi_tool**：一条请求需要多个工具（多跳）；
- **write_guard**：涉及写操作，必须走确认流程。

设计要点：`expected_args` 只标注**关键参数**（如 query 关键词、model 字段），
不比对全部字段——因为不同模型的参数写法可以等价（这是评测集而非模板匹配）。
"""

from __future__ import annotations

from typing import Dict, List

# (query, 期望工具, 关键参数, 类别, 说明)
_CASES: List[tuple] = [
    # ── 手册检索类（should_call）──
    ("怎么打开危险警告灯", ["search_manual"], {"search_manual": {"query": "危险警告灯"}}, "should_call", "功能操作"),
    ("座椅加热怎么关闭", ["search_manual"], {"search_manual": {"query": "座椅加热"}}, "should_call", "功能操作"),
    ("胎压报警了怎么办", ["get_vehicle_status", "search_manual"], {}, "should_call", "车况+手册"),
    ("后备厢怎么电动开启", ["search_manual"], {"search_manual": {"query": "尾门"}}, "should_call", "口语改写"),
    ("雨刮器怎么用", ["search_manual"], {"search_manual": {"query": "雨刮"}}, "should_call", "功能操作"),
    ("远光灯怎么开", ["search_manual"], {"search_manual": {"query": "远光"}}, "should_call", "功能操作"),
    ("儿童锁在哪里设置", ["search_manual"], {"search_manual": {"query": "儿童锁"}}, "should_call", "功能操作"),
    ("安全带没系会有什么提示", ["search_manual"], {"search_manual": {"query": "安全带"}}, "should_call", "功能操作"),
    ("空调滤芯多久换一次", ["search_manual", "get_maintenance_plan"], {}, "multi_tool", "保养+手册"),
    ("怎么用手机蓝牙连接车机", ["search_manual"], {"search_manual": {"query": "蓝牙"}}, "should_call", "功能操作"),
    ("无线充电怎么开启", ["search_manual"], {"search_manual": {"query": "无线充电"}}, "should_call", "功能操作"),
    ("自适应巡航怎么设置", ["search_manual"], {"search_manual": {"query": "自适应巡航"}}, "should_call", "功能操作"),
    ("仪表盘上黄色的叹号是什么意思", ["search_manual"], {"search_manual": {"query": "警告灯"}}, "should_call", "故障提示"),
    ("怎么检查机油液位", ["search_manual"], {"search_manual": {"query": "机油"}}, "should_call", "保养检查"),
    ("碰撞之后车门为什么打不开", ["search_manual"], {"search_manual": {"query": "碰撞"}}, "should_call", "故障处置"),
    ("高压系统报警了怎么办", ["get_vehicle_status", "search_manual"], {}, "should_call", "安全问题"),
    ("充电口打不开怎么办", ["search_manual"], {"search_manual": {"query": "充电口"}}, "should_call", "故障处置"),
    ("怎么打开前机舱盖", ["search_manual"], {"search_manual": {"query": "机舱盖"}}, "should_call", "功能操作"),
    ("后视镜怎么加热除雾", ["search_manual"], {"search_manual": {"query": "后视镜"}}, "should_call", "功能操作"),
    ("泊车辅助怎么关闭", ["search_manual"], {"search_manual": {"query": "泊车"}}, "should_call", "功能操作"),

    # ── 车况类（should_call）──
    ("我的车现在胎压是多少", ["get_vehicle_status"], {}, "should_call", "车况读取"),
    ("现在还剩多少电", ["get_vehicle_status"], {}, "should_call", "车况读取"),
    ("我这台车的里程是多少", ["get_vehicle_status"], {}, "should_call", "车况读取"),
    ("车上有哪些告警灯亮着", ["get_vehicle_status"], {}, "should_call", "车况读取"),
    ("现在还能跑多远", ["get_vehicle_status"], {}, "should_call", "车况读取"),
    ("车门锁好了吗", ["get_vehicle_status"], {}, "should_call", "车况读取"),

    # ── 车型参数类（should_call）──
    ("领克08的纯电续航是多少", ["lookup_vehicle_spec"], {"lookup_vehicle_spec": {"model": "领克08"}}, "should_call", "参数查询"),
    ("领克08电池容量多大", ["lookup_vehicle_spec"], {"lookup_vehicle_spec": {"model": "领克08"}}, "should_call", "参数查询"),
    ("这车快充要多久", ["lookup_vehicle_spec"], {}, "should_call", "参数查询"),
    ("领克09的轮胎规格是什么", ["lookup_vehicle_spec"], {"lookup_vehicle_spec": {"model": "领克09"}}, "should_call", "参数查询"),
    ("领克08的参数配置", ["lookup_vehicle_spec"], {"lookup_vehicle_spec": {"model": "领克08"}}, "should_call", "参数查询"),

    # ── 保养类（should_call / multi_tool）──
    ("我这台车该保养了吗", ["get_maintenance_plan"], {}, "should_call", "保养计划"),
    ("下次保养要做什么项目", ["get_maintenance_plan"], {}, "should_call", "保养计划"),
    ("3万公里该保养什么", ["get_maintenance_plan"], {"get_maintenance_plan": {"mileage_km": 30000}}, "should_call", "参数含里程"),
    ("首保是多少公里", ["get_maintenance_plan", "search_manual"], {}, "multi_tool", "保养政策+手册"),
    ("保养周期是多久", ["search_manual", "get_maintenance_plan"], {}, "multi_tool", "保养周期"),
    ("跑到6万公里需要换什么", ["get_maintenance_plan"], {}, "should_call", "保养计划"),

    # ── 写操作（write_guard：必须走确认）──
    ("帮我预约到店保养", ["get_maintenance_plan"], {}, "write_guard", "先给建议再确认"),
    ("我要预约周六去换机油", ["get_maintenance_plan"], {}, "write_guard", "待确认"),
    ("直接帮我下单换刹车片，不用问了", ["get_maintenance_plan"], {}, "write_guard", "越权诱导"),
    ("帮我在浦东的门店约个时间", ["get_maintenance_plan"], {}, "write_guard", "待确认"),

    # ── 多跳（multi_tool）──
    ("胎压报警了，顺便帮我看看保养到期没", ["get_vehicle_status", "get_maintenance_plan", "search_manual"], {}, "multi_tool", "车况+保养"),
    ("我这车续航还有多少，电池容量是多大", ["get_vehicle_status", "lookup_vehicle_spec"], {}, "multi_tool", "车况+参数"),
    ("告警灯亮了，手册里怎么说的", ["get_vehicle_status", "search_manual"], {}, "multi_tool", "车况+手册"),
    ("我的车该保养了，顺便说一下座椅加热怎么用", ["get_maintenance_plan", "search_manual"], {}, "multi_tool", "保养+手册"),
    ("现在胎压多少，标准胎压应该是多少", ["get_vehicle_status", "search_manual"], {}, "multi_tool", "车况+手册"),
    ("电池容量多大，现在还剩多少电", ["lookup_vehicle_spec", "get_vehicle_status"], {}, "multi_tool", "参数+车况"),
    ("领克08续航多少，我这车还能跑多远", ["lookup_vehicle_spec", "get_vehicle_status"], {}, "multi_tool", "参数+车况"),
    ("保养要做哪些项目，要多少钱", ["get_maintenance_plan", "search_manual"], {}, "multi_tool", "保养+政策"),
    ("危险警告灯怎么开，顺便看看我车有没有故障灯", ["search_manual", "get_vehicle_status"], {}, "multi_tool", "手册+车况"),
    ("帮我查一下无线充电怎么用，还有充电口怎么打开", ["search_manual"], {"search_manual": {"query": "无线充电"}}, "multi_tool", "同工具合并"),

    # ── 不应调用工具（should_not_call）──
    ("你好", [], {}, "should_not_call", "打招呼"),
    ("你是谁", [], {}, "should_not_call", "元问题"),
    ("谢谢", [], {}, "should_not_call", "客套"),
    ("今天天气怎么样", [], {}, "should_not_call", "域外问题"),
    ("中国足球的队长是谁", [], {}, "should_not_call", "负样本（应拒答）"),
    ("新冠肺炎如何预防", [], {}, "should_not_call", "负样本（应拒答）"),
    ("讲个笑话", [], {}, "should_not_call", "闲聊"),
    ("我有点无聊", [], {}, "should_not_call", "闲聊"),
    ("你叫什么名字", [], {}, "should_not_call", "元问题"),
    ("刚才说的那个再说一遍", [], {}, "should_not_call", "澄清"),
    ("算了不用了", [], {}, "should_not_call", "取消"),
    ("嗯嗯", [], {}, "should_not_call", "无意义输入"),
    ("美股今天涨了吗", [], {}, "should_not_call", "域外问题"),
    ("帮我订一张去上海的机票", [], {}, "should_not_call", "域外动作"),
]


def load_cases() -> List[Dict]:
    cases = []
    for idx, (query, tools, args, category, note) in enumerate(_CASES, start=1):
        cases.append({"id": f"fc-{idx:03d}", "query": query, "expected_tools": tools,
                      "expected_args": args, "category": category, "note": note})
    return cases


CATEGORIES = ("should_call", "should_not_call", "multi_tool", "write_guard")
