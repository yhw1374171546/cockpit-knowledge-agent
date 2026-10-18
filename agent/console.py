# -*- coding: utf-8 -*-
"""控制台输出的"抗畸"处理：保证脚本在**中文 Windows（GBK 控制台）**上既跑得完、又不乱码。

为什么需要
----------
两个"消费者"要的编码不一样：

| 场景 | sys.stdout 是终端(tty) | 应该用 | 原因 |
| --- | --- | --- | --- |
| 人在控制台看 | 是 | **GBK（系统默认）** | Python 走 Windows 控制台 API，中文天然正常 |
| 被重定向到文件 / 管道 | 否 | **UTF-8** | 下游工具（CI、编辑器、解析脚本）默认按 UTF-8 读 |

而 Windows 中文控制台代码页是 **936（GBK）**，`✅ ❌ ⚠️ 🚀` 这些符号 **GBK 编不出来**，
于是 `print("✅ 全部通过")` 会抛：

    UnicodeEncodeError: 'gbk' codec can't encode character '\\u2705'

后果很隐蔽：**脚本其实全跑对了，却在最后一行崩掉、退出码变 1**。
（本项目就踩过：253 项测试全过，但用户看到的是 traceback + exit 1。）

四种组合要分清
--------------
| 设置 | Python 输出 | 控制台解码 | 结果 |
| --- | --- | --- | --- |
| 不设 `PYTHONIOENCODING` | GBK | GBK | ✅ 正常 |
| 设了 `PYTHONIOENCODING=utf-8` 但没 `chcp 65001` | UTF-8 | GBK | ❌ 乱码（`鍏ㄩ噺娴嬭瘯`） |
| `chcp 65001` + `PYTHONIOENCODING=utf-8` | UTF-8 | UTF-8 | ✅ 正常 |
| 不设，但输出含 GBK 编不出的符号 | GBK + 崩 | — | ❌ UnicodeEncodeError |

`harden_stdout()` 同时解决第一和第四个问题：**tty 用原生编码 + errors=replace（不崩），
非 tty 强制 UTF-8（下游可解析）**。它幂等、且在任何流都不支持 reconfigure 时静默跳过。
"""

from __future__ import annotations

import sys


def harden_stdout() -> None:
    """按"是否终端"选择编码，并把编不出的字符替换而非抛异常。

    - **终端**：保持系统默认编码（中文 Windows 就是 GBK），只是把不可编码字符换成 `?`，
      这样 `✅` 会显示成 `?` 但**不会崩**；
    - **重定向/管道**：强制 UTF-8，方便 CI、日志、编辑器与解析脚本统一处理。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            is_tty = bool(stream.isatty())
        except Exception:                              # noqa: BLE001
            is_tty = False
        try:
            if is_tty:
                stream.reconfigure(errors="replace")            # type: ignore[attr-defined]
            else:
                stream.reconfigure(encoding="utf-8",            # type: ignore[attr-defined]
                                   errors="replace")
        except Exception:                              # noqa: BLE001
            pass


# ── ASCII 安全的状态标记：任何控制台都一定能打印 ──
OK = "[OK]"
FAIL = "[FAIL]"
WARN = "[!]"
