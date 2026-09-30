# -*- coding: utf-8 -*-
"""为「原始分块」的 8,785 个文档块补全页码元数据（对齐法）。

动机
----
`kb/build_kb.py` 是**从头重建**知识库：好处是原生带页码，代价是缺少 pdfplumber 字体块策略，
检索覆盖率从 0.8347 掉到 0.8215。本脚本走另一条路：**保留原始分块结果**（保持检索质量），
通过「块首 24 字在全文中的位置」反查页码，把溯源能力附加到已验证的块上。

做法
----
1. 用 PyPDF2 逐页取文本并做归一化（去空白/逗号/制表符），拼成全文并按页码记录偏移区间；
2. 取每个块归一化后的前 24 字作为指纹，扫描全文一次即可命中（哈希表 O(1) 查找）；
3. 命中 → 用二分查找定位所属页；未命中 → 标记 page=None 并计入未对齐统计。

用法：
    python kb/attach_pages.py            # 输出 kb/chunks.jsonl（原始分块 + 页码）
"""

from __future__ import annotations

import bisect
import hashlib
import json
import os
import re
import statistics
import sys
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SOURCE = os.path.join(ROOT, "all_text.txt")
DEFAULT_PDF = os.path.join(ROOT, "data", "train_a.pdf")
DEFAULT_OUT = os.path.join(ROOT, "kb", "chunks.jsonl")
DEFAULT_STATS = os.path.join(ROOT, "kb", "kb_paged_stats.json")

FP_LEN = 24


def norm(text: str) -> str:
    return re.sub(r"[\s,，\t]", "", text or "")


def load_source_chunks(path: str) -> List[str]:
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if len(text) >= 5:
                chunks.append(text)
    return chunks


def build_page_index(pdf_path: str) -> Tuple[str, List[int], List[int]]:
    """返回 (归一化全文, 页起始偏移列表, 页码列表)。"""
    from PyPDF2 import PdfReader

    big_parts: List[str] = []
    page_starts: List[int] = []
    pages: List[int] = []
    cursor = 0
    for page_no, page in enumerate(PdfReader(pdf_path).pages, start=1):
        text = norm(page.extract_text() or "")
        if not text:
            continue
        page_starts.append(cursor)
        pages.append(page_no)
        big_parts.append(text)
        cursor += len(text)
    return "".join(big_parts), page_starts, pages


def locate(fingerprints: Dict[str, int], big: str) -> Dict[int, int]:
    """单次扫描全文，返回 {块下标: 首次出现位置}。"""
    found: Dict[int, int] = {}
    limit = len(big) - FP_LEN
    for i in range(limit):
        hit = fingerprints.get(big[i:i + FP_LEN])
        if hit is not None and hit not in found:
            found[hit] = i
    return found


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SOURCE
    out = os.path.join(ROOT, "kb", "chunks.jsonl")
    raw_chunks = load_source_chunks(source)
    print(f"[info] 原始块 {len(raw_chunks)} 个（来源 {os.path.basename(source)}）")

    # 1) 指纹表：块下标 → 归一化前 24 字
    fingerprints: Dict[str, int] = {}
    for idx, text in enumerate(raw_chunks):
        fp = norm(text)[:FP_LEN]
        if len(fp) >= 12 and fp not in fingerprints:
            fingerprints[fp] = idx
    print(f"[info] 有效指纹 {len(fingerprints)} 个")

    # 2) 逐页全文 + 单次扫描定位
    big, page_starts, pages = build_page_index(DEFAULT_PDF)
    print(f"[info] 归一化全文 {len(big)} 字，覆盖 {len(pages)} 页")
    found = locate(fingerprints, big)
    print(f"[info] 命中 {len(found)}/{len(raw_chunks)} 个块")

    # 3) 位置 → 页码
    chunks = []
    with_page = 0
    for idx, text in enumerate(raw_chunks):
        page: Optional[int] = None
        pos = found.get(idx)
        if pos is not None:
            slot = bisect.bisect_right(page_starts, pos) - 1
            if slot >= 0:
                page = pages[slot]
                with_page += 1
        chunks.append({
            "id": idx,
            "text": text,
            "source": "train_a.pdf",
            "page": page,
            "header": None,
            "strategy": "original",
            "fingerprint": hashlib.md5(text.encode("utf-8")).hexdigest()[:12],
        })

    # 4) 去重校验（原始 list 判重是 O(n²)，这里用 set 复核重复率）
    digests = {c["fingerprint"] for c in chunks}
    lengths = sorted(len(c["text"]) for c in chunks)
    stats = {
        "source_file": os.path.basename(source),
        "n_chunks": len(chunks),
        "unique_chunks": len(digests),
        "duplicate_rate": round(1 - len(digests) / len(chunks), 4) if chunks else 0,
        "with_page": with_page,
        "page_coverage": round(with_page / len(chunks), 4) if chunks else 0,
        "total_chars": sum(lengths),
        "avg_len": round(statistics.mean(lengths), 1) if lengths else 0,
        "p50": lengths[len(lengths) // 2] if lengths else 0,
        "p90": lengths[int(len(lengths) * 0.9)] if lengths else 0,
        "max": lengths[-1] if lengths else 0,
    }
    with open(out, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    json.dump(stats, open(DEFAULT_STATS, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\n[out] {out}")


if __name__ == "__main__":
    main()
