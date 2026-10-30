# -*- coding: utf-8 -*-
"""车主画像卡库：程序化组合生成 100 张卡（确定性、零成本、可复现）。

组合空间：4 车型 × 5 城市（含门店对） × 2 漂移方向 × 3 偏好 × 2 红线 = 240，
取前 100 张（种子固定，跨机器可复现——生成物可进 CI 对比）。

每张卡包含：静态档案（车型/牌照区域）、门店（初始 + 漂移后）、功能偏好、
车主红线、工单日期——这些都是**生成对话的因变量**，也是 gold 事实的来源。
"""

from __future__ import annotations

from typing import Any, Dict, List

MODELS = ("领克08 EM-P", "领克09 EM-P", "领克05 EM-P", "领克01 EM-P")
CITIES = ("上海", "杭州", "深圳", "广州", "成都")
CENTERS = {
    "上海": "领克中心(徐汇店)", "杭州": "领克中心(杭州西湖店)",
    "深圳": "领克中心(深圳南山店)", "广州": "领克中心(广州天河店)",
    "成都": "领克中心(成都武侯店)",
}
ALT_CENTERS = {
    "上海": "领克中心(浦东金桥店)", "杭州": "领克中心(杭州滨江店)",
    "深圳": "领克中心(深圳福田店)", "广州": "领克中心(广州越秀店)",
    "成都": "领克中心(成都锦江店)",
}
PLATES = {"上海": "沪C", "杭州": "浙A", "深圳": "粤B", "广州": "粤A", "成都": "川A"}
PREFS = (("座椅加热", "关闭"), ("空调", "调低"), ("氛围灯", "开启"))
# 红线短语：与抽取句式"以后千万别{动词短语}"兼容（DO_NOT_CUE 需要 不要/别/关闭 等动词开头）
REDLINES = ("关闭车身稳定系统", "关闭车门未关提醒")


def build_cards(n: int = 100, seed: int = 42) -> List[Dict[str, Any]]:
    """程序化生成 n 张画像卡（顺序固定、种子固定 → 生成物完全可复现）。"""
    cards: List[Dict[str, Any]] = []
    i = 0
    for model in MODELS:
        for city in CITIES:
            for drift in (False, True):
                for pref_feature, pref_action in PREFS:
                    if len(cards) >= n:
                        return cards
                    i += 1
                    card: Dict[str, Any] = {
                        "model": model,
                        "city": city,
                        "plate": PLATES[city],
                        "center": CENTERS[city],
                        "drift": drift,
                        # 漂移方向：换到同城另一家店（KU/TR 题型的事实演化）
                        "alt_center": ALT_CENTERS[city],
                        "pref_feature": pref_feature,
                        "pref_action": pref_action,
                        "redline": REDLINES[i % len(REDLINES)],
                        # ABS 题的假前提：与真实车型不同的另一款车
                        "abs_model": "领克09 EM-P" if model != "领克09 EM-P" else "领克01 EM-P",
                    }
                    cards.append(card)
    return cards
