# -*- coding: utf-8 -*-
"""extract 规则抽取器测试：v1 句式 + 新增的漂移改口与否定修正（v1 的两大盲区）。"""

from __future__ import annotations

from agent.profile.extract import extract_rules


def _keys(text: str) -> dict:
    return {c.key: c.value for c in extract_rules(text)}


def test_v1_basic_patterns_still_work():
    out = _keys("我住在上海，我的车是领克 08，我一般去领克中心(徐汇店)，以后座椅加热都关掉")
    assert out["city"] == "上海"
    assert out["vehicle_model"] == "领克08"          # 归一化：领克 08 → 领克08
    assert out["usual_center"] == "领克中心(徐汇店)"
    assert out["pref:座椅加热"] == "关闭"


def test_drift_correction_extracts_new_store():
    """漂移改口：v1 只抽第一句（"我一般去"两次出现时以第一次为准或都抽），
    v2 需要把"不对，我一般去 X"识别为改口——抽取层抽到两条，consolidate 层按新者覆盖。"""
    out = extract_rules("我一般去领克中心(徐汇店)，以后都去这家。不对，我一般去领克中心(杭州西湖店)")
    centers = [c.value for c in out if c.key == "usual_center"]
    assert "领克中心(杭州西湖店)" in centers          # 改口后的门店必须被抽到


def test_negation_correction_is_extracted():
    """否定修正："不是那个店，是浦东店"——v1 的 CENTER_CUE 完全抽不到（0 命中）。"""
    out = _keys("不是那个店，是浦东金桥店")
    assert out.get("usual_center") == "浦东金桥店"


def test_plate_region_and_constraint():
    out = _keys("我车牌是沪C，记住以后千万别给我关闭车身稳定系统")
    assert out["plate_region"] == "沪C"
    assert "关闭" in out["do_not"]


def test_no_center_when_negation_absent():
    assert "usual_center" not in _keys("今天天气怎么样，帮我约个保养")


def test_pref_action_normalized():
    out = _keys("以后氛围灯都开")
    assert out["pref:氛围灯"] == "开启"
