# -*- coding: utf-8 -*-
"""知识库构建（带元数据版）。

相对原始 pdf_parse.py 的改进：
1. **保留元数据**：每个块记录 page / header / strategy，答案因此可以给出处（原实现把
   header、pageid 传进 Datafilter 后丢弃，导致无法溯源）；
2. **O(1) 去重**：用 set + 内容 hash 判重，替代原实现的 list `in` 线性扫描（O(n²)）；
3. **多策略并行**：字体块解析（pdfplumber，可选）/ 滑窗交叠 / 规则切分，统一去重后落盘 JSONL；
4. **输出统计**：块数、长度分位、每策略贡献，直接可写进 README。

用法：
    python kb/build_kb.py --pdf data/train_a.pdf --out kb/chunks.jsonl
    python kb/build_kb.py --no-block        # 跳过 pdfplumber 字体块策略（更快）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
from typing import Dict, Iterable, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NOISE_PATTERNS = ("....................", "目录")
HEADER_TOP_RANGE = (17, 20)


def _clean_line(text: str) -> str:
    text = text.strip().strip("\n")
    if any(p in text for p in NOISE_PATTERNS):
        return ""
    if text.isdigit():
        return ""
    return text.replace(",", "").replace("\t", "")


class KnowledgeBuilder:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.seen = set()                 # 内容 hash → O(1) 去重
        self.chunks: List[Dict] = []
        self.strategy_stats: Dict[str, Dict[str, int]] = {}

    # ── 工具 ──
    def _add(self, text: str, page: Optional[int], header: Optional[str], strategy: str,
             min_len: int = 6, max_len: int = 4096) -> bool:
        text = (text or "").strip()
        if len(text) < min_len or len(text) > max_len:
            return False
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()
        if digest in self.seen:
            return False
        self.seen.add(digest)
        self.chunks.append({
            "id": len(self.chunks),
            "text": text,
            "source": os.path.basename(self.pdf_path),
            "page": page,
            "header": header,
            "strategy": strategy,
        })
        st = self.strategy_stats.setdefault(strategy, {"added": 0, "duplicated": 0})
        st["added"] += 1
        return True

    def _reject(self, strategy: str, n: int = 1) -> None:
        st = self.strategy_stats.setdefault(strategy, {"added": 0, "duplicated": 0})
        st["duplicated"] += n

    # ── 逐页文本（PyPDF2） ──
    def page_texts(self) -> List[Tuple[int, str]]:
        from PyPDF2 import PdfReader

        out = []
        for idx, page in enumerate(PdfReader(self.pdf_path).pages, start=1):
            raw = page.extract_text() or ""
            lines = [_clean_line(w) for w in raw.split("\n")]
            text = "".join(l for l in lines if l)
            out.append((idx, text))
        return out

    # ── 策略一：字体块解析（pdfplumber，可选） ──
    def parse_block(self, max_seq_values: Iterable[int] = (1024, 512)) -> None:
        try:
            import pdfplumber  # type: ignore
        except Exception:
            print("[skip] pdfplumber 不可用，跳过字体块策略", file=sys.stderr)
            return

        with pdfplumber.open(self.pdf_path) as pdf:
            for pageno, page in enumerate(pdf.pages, start=1):
                try:
                    words = page.extract_words(use_text_flow=True, extra_attrs=["size"])
                except Exception:
                    continue
                header = self._page_header(page)
                if not words:
                    continue
                seq, last_size = "", None
                for i, w in enumerate(words):
                    token, size = w.get("text", ""), w.get("size", 0)
                    if token in ("□", "•", "·"):
                        continue
                    if token in ("警告！", "注意！", "说明！"):
                        for ms in max_seq_values:
                            self._flush(seq, pageno, header, "block", ms)
                        seq = ""
                        continue
                    if last_size is not None and abs(size - last_size) < 1e-5:
                        seq = (seq + token) if seq else token
                    else:
                        last_size = size
                        if 0 < len(seq) < 15:
                            seq += token
                        else:
                            for ms in max_seq_values:
                                self._flush(seq, pageno, header, "block", ms)
                            seq = token
                for ms in max_seq_values:
                    self._flush(seq, pageno, header, "block", ms)

    def _page_header(self, page) -> Optional[str]:
        try:
            words = page.extract_words()
        except Exception:
            return None
        if not words:
            return None
        for w in words:
            text = w.get("text", "")
            if any(p in text for p in NOISE_PATTERNS):
                return None
            top = w.get("top", -1)
            if HEADER_TOP_RANGE[0] < top < HEADER_TOP_RANGE[1]:
                return text
        return words[0].get("text")

    def _flush(self, seq: str, page: Optional[int], header: Optional[str],
               strategy: str, max_seq: int) -> None:
        """把累积文本按 max_seq 拆成块（超长时按分隔符再切）。"""
        seq = (seq or "").strip()
        if len(seq) < 6:
            return
        if len(seq) <= max_seq:
            if not self._add(seq, page, header, strategy):
                self._reject(strategy)
            return
        for sep in ("■", "•", "\t", "。"):
            if sep in seq:
                parts = seq.split(sep)
                break
        else:
            parts = [seq]
        buf = ""
        for part in parts:
            part = part.replace("\n", "")
            if 5 < len(part) < max_seq:
                if not self._add(part, page, header, strategy):
                    self._reject(strategy)
            elif len(part) >= max_seq:
                for i in range(0, len(part), max_seq):
                    if not self._add(part[i:i + max_seq], page, header, strategy):
                        self._reject(strategy)
            buf = ""

    # ── 策略二：滑窗交叠（带页码） ──
    def parse_sliding(self, kernels: Iterable[int] = (256, 512), stride: int = 1) -> None:
        """与原始 SlidingWindow 语义一致：窗口长满 kernel 即吐出一个块，
        然后从头部滑出一句（overlap = 窗口长度 - 1 句），保证跨页语义连续。
        差异：页码不再丢失——块记为窗口首句所在页。
        """
        page_texts = self.page_texts()
        sentences: List[Tuple[str, int]] = []
        for page, text in page_texts:
            for s in text.split("。"):
                s = s.strip()
                if len(s) >= 4:
                    sentences.append((s, page))
        if not sentences:
            return
        for kernel in kernels:
            cur, slow = "", 0
            for fast in range(len(sentences)):
                s, _pg = sentences[fast]
                if cur and len(cur + s) > kernel:
                    if cur.strip():
                        if not self._add(cur + s + "。", sentences[slow][1], None, "slide"):
                            self._reject("slide")
                    head = sentences[slow][0]
                    cur = cur[len(head) + 1:] if len(cur) > len(head) + 1 else ""
                    slow = min(slow + 1, fast)
                cur = cur + s + "。"
            if cur.strip():
                if not self._add(cur, sentences[slow][1], None, "slide"):
                    self._reject("slide")

    # ── 策略三：逐页规则切分 ──
    def parse_page_rule(self, max_seq_values: Iterable[int] = (256, 512)) -> None:
        for page, page_content in self.page_texts():
            if len(page_content) < 6:
                continue
            for max_seq in max_seq_values:
                if len(page_content) <= max_seq:
                    if not self._add(page_content, page, None, "page-rule"):
                        self._reject("page-rule")
                    continue
                cur = ""
                for sentence in page_content.split("。"):
                    if len(cur + sentence) > max_seq and len(cur) >= 6:
                        if not self._add(cur, page, None, "page-rule"):
                            self._reject("page-rule")
                        cur = sentence
                    else:
                        cur += sentence
                if len(cur) >= 6:
                    if not self._add(cur, page, None, "page-rule"):
                        self._reject("page-rule")

    # ── 主流程 ──
    def build(self, use_block: bool = True) -> Dict:
        if use_block:
            self.parse_block()
        self.parse_sliding()
        self.parse_page_rule()
        lengths = sorted(len(c["text"]) for c in self.chunks)

        def pct(p: float) -> int:
            if not lengths:
                return 0
            return lengths[min(len(lengths) - 1, int(p * len(lengths)))]

        stats = {
            "pdf": os.path.basename(self.pdf_path),
            "n_chunks": len(self.chunks),
            "total_chars": sum(lengths),
            "avg_len": round(statistics.mean(lengths), 1) if lengths else 0,
            "p50": pct(0.5), "p90": pct(0.9), "p99": pct(0.99),
            "min": lengths[0] if lengths else 0, "max": lengths[-1] if lengths else 0,
            "with_page": sum(1 for c in self.chunks if c["page"] is not None),
            "with_header": sum(1 for c in self.chunks if c["header"]),
            "strategies": {k: v for k, v in self.strategy_stats.items()},
        }
        return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=os.path.join(ROOT, "data", "train_a.pdf"))
    ap.add_argument("--out", default=os.path.join(ROOT, "kb", "chunks.jsonl"))
    ap.add_argument("--stats", default=os.path.join(ROOT, "kb", "kb_stats.json"))
    ap.add_argument("--no-block", action="store_true", help="跳过 pdfplumber 字体块策略")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    builder = KnowledgeBuilder(args.pdf)
    stats = builder.build(use_block=not args.no_block)

    with open(args.out, "w", encoding="utf-8") as f:
        for c in builder.chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    json.dump(stats, open(args.stats, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\n[out] {args.out}\n[stats] {args.stats}")


if __name__ == "__main__":
    main()
