# -*- coding: utf-8 -*-
"""Alembic 迁移环境（BE5）。

要点（为什么这样配）
--------------------
1. **URL 从服务配置读**，不写死在 `alembic.ini`：CI、本地、容器三种环境用同一份脚本，
   只需给 `SERVICE_DATABASE_URL`；`alembic.ini` 里的 `sqlalchemy.url` 只留空占位。
2. **`target_metadata` 指向服务的 `Base.metadata`**：autogenerate 才能对比出差异。
3. **`compare_type=True`**：字段类型变化也会被检出（否则"列类型改了但迁移没生成"很常见）。
4. **`render_as_batch=True`**：SQLite 不支持 `ALTER COLUMN`，batch 模式会重建表——
   本项目默认开发库就是 SQLite，不加这个选项改列会直接失败。
"""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# 让 migrations/ 能 import 到仓库根下的 agent 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.service.config import Settings  # noqa: E402
from agent.service.db import Base  # noqa: E402

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ── 数据库 URL：命令行 > 环境变量 > 服务默认 ──
# 命令行用法：alembic -x db_url=sqlite:///tmp.db upgrade head
db_url = context.get_x_argument(as_dictionary=True).get("db_url")
if not db_url:
    db_url = os.environ.get("SERVICE_DATABASE_URL") or Settings.from_env().database_url
# 百分号转义：alembic 的 URL 会走 configparser 插值，密码里含 % 时不做转义会报错
config.set_main_option("sqlalchemy.url", db_url.replace("%", "%%"))

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """离线模式：只生成 SQL，不连库（CI 里用它校验迁移脚本能否正确解析）。"""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            render_as_batch=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
