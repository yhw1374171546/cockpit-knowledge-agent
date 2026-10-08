# -*- coding: utf-8 -*-
"""接地校验评测（A3）：语义接地 vs 纯字符重叠。

背景：真实模型（DeepSeek）跑 103 题时**误杀从 1/101 涨到 23/101**——
23 条都有 ≥6 条证据、门控覆盖率 ≥0.9，问题出在 `Reflector.verify` 的**逐句字符重叠**
接地校验对**生成式**答案过苛：离线抽取式答案天然与证据字面重合，而真模型会改写措辞。

本脚本用**带标注的句子级样本**度量两种口径的判别能力：
- `supported`：句意被证据支撑（原文引用、同义改写、口语化、简化、数字一致）；
- `unsupported`：句意无依据（数字编造、无关内容、张冠李戴）。

关键指标是 **假阳性率**（把「有依据」判成「无依据」）——它直接对应上面的误杀。

用法：
    python eval/grounding_eval.py
    python eval/grounding_eval.py --json-out eval/grounding_metrics.json --md-out eval/grounding_report.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.reflection import Reflector, char_coverage, semantic_support            # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EVIDENCE = (
    "危险警告灯开关在方向盘下方，按下开关即可打开危险警告灯。"
    "车辆遇到交通事故或其他紧急情况时，按下危险警告灯按键，启用危险警告灯，两侧转向指示灯均闪烁。"
    "胎压标准值为 236 kPa，胎压低于标准值时会触发胎压低报警。"
    "座椅加热可在多媒体显示屏的座舱体验-座椅界面中设置加热强度或关闭。"
    "动力电池温度过高时，请立即在安全位置停车并远离车辆，联系 Lynk&Co 领克中心处理。"
)

# (句子, 标签, 说明)  标签 supported=句意有依据 / unsupported=无依据
CASES = [
    # ── 原文引用（两种口径都应通过）──
    ("危险警告灯开关在方向盘下方，按下开关即可打开危险警告灯。", "supported", "原文照抄"),
    ("胎压标准值为 236 kPa。", "supported", "原文数字"),
    ("动力电池温度过高时请立即在安全位置停车。", "supported", "原文安全句"),
    # ── 同义改写（真模型最常见的输出形态）──
    ("开启危险警告灯需要按下方向盘下方的开关。", "supported", "打开→开启 同义改写"),
    ("您可以从多媒体显示屏进入座舱体验里的座椅界面来关闭座椅加热。", "supported", "口语化改写"),
    ("警示灯开启后，两侧转向指示灯会一起闪烁。", "supported", "警告灯→警示灯 改写"),
    ("胎压低于标准值会触发胎压低报警。", "supported", "简化改写"),
    ("遇到紧急情况可以按下危险警告灯按键来启用双闪。", "supported", "启用/双闪 改写"),
    ("座椅加热强度可以在中控屏里调节。", "supported", "显示屏→中控屏 改写"),
    ("电池温度过高时应马上停车并联系领克中心。", "supported", "立即→马上 改写"),
    # 注意：证据里的安全句是「联系 Lynk&Co 领克中心处理」，并未提到「道路救援」，
    # 所以下面这条其实**部分无依据**（第一版我标成了 supported，属于标注错误，已修正）
    ("需要充电时建议联系道路救援。", "unsupported", "证据未提及救援渠道"),
    # ── 部分支撑 / 边界 ──
    ("按下开关后两侧转向灯闪烁，同时会发出提示音。", "unsupported", "提示音为新增信息（无依据）"),
    ("座椅加热共有三档强度可调。", "unsupported", "档位数量未在证据中"),
    ("危险警告灯开关位于副驾驶手套箱内。", "unsupported", "位置张冠李戴"),
    # ── 数字幻觉 ──
    ("胎压标准值是 300 kPa。", "unsupported", "数字编造"),
    ("胎压标准值是 236 kPa，低于 200 kPa 时必须停车。", "unsupported", "半数以上数字编造"),
    # ── 完全无关 ──
    ("本车配备航空级钛合金防撞梁，碰撞时自动弹出降落伞。", "unsupported", "完全无关"),
    ("该车型支持车外遥控泊车与脱手驾驶。", "unsupported", "无关功能"),
    ("座椅加热功率为 500 瓦。", "unsupported", "无关数字"),
    # ── 半对半错（生成式答案常见）──
    ("按方向盘下方的开关可打开危险警告灯；另外该车还支持语音唤醒。", "unsupported", "前对后错（整句无依据）"),
    ("动力电池过热要立即停车；散热风扇会自动启动降温。", "unsupported", "后半句为新增机制"),
]


def evaluate(grounding: str) -> dict:
    """用指定接地口径跑标注集，返回混淆矩阵与指标。"""
    scorer = semantic_support if grounding == "semantic" else char_coverage
    reflector = Reflector(grounding=grounding)
    threshold = reflector.min_sentence_support

    rows = []
    tp = fp = tn = fn = 0
    for sentence, label, note in CASES:
        score = scorer(sentence, EVIDENCE)
        predicted_supported = score >= threshold
        actual_supported = label == "supported"
        if actual_supported and predicted_supported:
            tn += 1                       # 正确放行
        elif actual_supported and not predicted_supported:
            fp += 1                       # 假阳性：有依据却判为无依据（=误杀）
        elif not actual_supported and predicted_supported:
            fn += 1                       # 漏判：无依据却放行
        else:
            tp += 1                       # 正确拦截
        # 整句级校验结论（verify 会按 grounded_ratio 决定 verdict）
        result = reflector.verify(sentence, [{"source": "manual.pdf", "page": 1,
                                              "text": EVIDENCE}])
        rows.append({
            "sentence": sentence, "label": label, "note": note,
            "support": round(score, 4), "threshold": threshold,
            "predicted": "supported" if predicted_supported else "unsupported",
            "verdict": result.verdict, "next_action": result.next_action,
        })

    total = len(CASES)
    supported_total = sum(1 for _, label, _ in CASES if label == "supported")
    unsupported_total = total - supported_total
    return {
        "grounding": grounding,
        "threshold": threshold,
        "total": total,
        "supported_samples": supported_total,
        "unsupported_samples": unsupported_total,
        # 以「unsupported 为正类」定义准召：拦截无依据内容是校验器的职责
        "true_positive_block": tp,
        "false_positive_block": fp,       # 误杀（越小越好）
        "false_negative_pass": fn,        # 漏判（越小越好）
        "true_negative_pass": tn,
        "false_positive_rate": round(fp / supported_total, 4) if supported_total else 0.0,
        "recall_unsupported": round(tp / unsupported_total, 4) if unsupported_total else 0.0,
        "precision_unsupported": round(tp / (tp + fp), 4) if (tp + fp) else 0.0,
        "accuracy": round((tp + tn) / total, 4),
        "false_refusal_verdicts": sum(1 for r in rows if r["label"] == "supported"
                                      and r["verdict"] == "ungrounded"),
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", default=os.path.join(ROOT, "eval", "grounding_metrics.json"))
    ap.add_argument("--md-out", default=os.path.join(ROOT, "eval", "grounding_report.md"))
    args = ap.parse_args()

    print("=" * 78)
    print("接地校验评测：语义接地 vs 纯字符重叠")
    print("=" * 78)

    char_res = evaluate("char")
    sem_res = evaluate("semantic")

    header = f"{'口径':<10}{'阈值':>6}{'误杀率':>9}{'拦截召回':>10}{'漏判':>6}{'准确率':>9}"
    print("\n" + header)
    print("-" * len(header))
    for res in (char_res, sem_res):
        print(f"{res['grounding']:<10}{res['threshold']:>6.2f}"
              f"{res['false_positive_rate']:>9.1%}{res['recall_unsupported']:>10.1%}"
              f"{res['false_negative_pass']:>6}{res['accuracy']:>9.1%}")

    print("\n逐句对比（只列出两种口径判定不同的样本）：")
    shown = 0
    for a, b in zip(char_res["rows"], sem_res["rows"]):
        if a["predicted"] != b["predicted"]:
            shown += 1
            print(f"  [{a['label']:<11}] {a['sentence'][:34]:<36} "
                  f"char={a['support']:.3f}({a['predicted'][:4]}) → "
                  f"semantic={b['support']:.3f}({b['predicted'][:4]})")
    if not shown:
        print("  （无差异）")

    _write_outputs(args, char_res, sem_res)

    improved = (sem_res["false_positive_rate"] <= char_res["false_positive_rate"] + 1e-9
                and sem_res["recall_unsupported"] > char_res["recall_unsupported"])
    print("\n" + "-" * 78)
    if improved:
        print(f"结论：✅ 语义接地取得净收益——误杀率持平（{sem_res['false_positive_rate']:.1%}），"
              f"拦截召回 {char_res['recall_unsupported']:.1%} → "
              f"{sem_res['recall_unsupported']:.1%}（漏判 {char_res['false_negative_pass']}"
              f" → {sem_res['false_negative_pass']}）")
    else:
        print("结论：⚠️ 语义接地未取得净收益，需重新调参")
    leftover_fn = [r for r in sem_res["rows"]
                   if r["label"] == "unsupported" and r["predicted"] == "supported"]
    if leftover_fn:
        print("仍漏判的样本（如实列出）：")
        for row in leftover_fn:
            print(f"  - {row['sentence'][:40]}（语义分 {row['support']:.3f}）")
    leftover_fp = [r for r in sem_res["rows"]
                   if r["label"] == "supported" and r["predicted"] == "unsupported"]
    if leftover_fp:
        print("仍被误杀的样本：")
        for row in leftover_fp:
            print(f"  - {row['sentence'][:40]}（语义分 {row['support']:.3f}）")
    return 0 if improved else 1


def _write_outputs(args, char_res, sem_res) -> None:
    payload = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "n_cases": len(CASES), "results": [char_res, sem_res]}
    with open(args.json_out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    lines = [
        "# 接地校验评测报告（A3：语义接地 vs 纯字符重叠）", "",
        f"- 时间：{payload['generated_at']}　样本：{len(CASES)} 句"
        f"（{char_res['supported_samples']} 有依据 / {char_res['unsupported_samples']} 无依据）",
        "- 背景：真实模型跑 103 题时误杀 **1/101 → 23/101**，"
        "根因是逐句**字符重叠**接地校验对**生成式**答案过苛",
        "",
        "## 指标对比", "",
        "| 口径 | 阈值 | **误杀率**（有依据判成无依据） | 拦截召回（无依据被抓） | 漏判 | 准确率 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for res in (char_res, sem_res):
        name = "纯字符重叠（旧）" if res["grounding"] == "char" else "**语义接地（新）**"
        lines.append(f"| {name} | {res['threshold']:.2f} | **{res['false_positive_rate']:.1%}** | "
                     f"{res['recall_unsupported']:.1%} | {res['false_negative_pass']} | "
                     f"{res['accuracy']:.1%} |")
    lines += ["", "## 判定变化的样本", ""]
    changed = [(a, b) for a, b in zip(char_res["rows"], sem_res["rows"])
               if a["predicted"] != b["predicted"]]
    if changed:
        lines += ["| 标签 | 句子 | 字符分 | 语义分 | 变化 |", "| --- | --- | --- | --- | --- |"]
        for a, b in changed:
            arrow = "误杀 → 放行 ✅" if a["predicted"] == "unsupported" else "放行 → 拦截"
            lines.append(f"| {a['label']} | {a['sentence'][:40]} | {a['support']:.3f} | "
                         f"{b['support']:.3f} | {arrow} |")
    else:
        lines.append("（无差异）")
    lines += [
        "", "## 做法", "",
        "`semantic_support()` 替换纯字符重叠，三件事：",
        "1. **数字硬核对**：句中数字若在证据里过半找不到，直接判 0（防把 236 kPa 说成 300 kPa）；",
        "2. **词级重叠 + 同义词扩展**：打开/开启、胎压/轮胎气压、显示屏/中控屏 等视为同一含义；",
        "3. **字符覆盖兜底**：短句与专有名词场景仍用字符信号（权重 0.2）。",
        "",
        "## 局限（如实说明）", "",
        "- 这里不是真正的语义模型，而是**词级 + 同义词表 + 数字核对**的代理实现：",
        "  同义词表覆盖不到的改写仍会误判，跨句推理（把两句话拼出一个新结论）测不出来；",
        "- 更彻底的做法是 embedding 相似度或 LLM-as-judge，本机无 GPU、也不希望评测依赖外部 API，",
        "  因此保留为下一步；本报告的价值在于**用标注集量化了两种口径的差别**，而不是宣称解决。",
    ]
    with open(args.md_out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n已写入 {os.path.relpath(args.json_out, ROOT)} 与 {os.path.relpath(args.md_out, ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
