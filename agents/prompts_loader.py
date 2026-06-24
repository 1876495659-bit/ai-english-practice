"""
Prompt 模板加载器

从 prompts/ 目录加载 .txt 模板文件，支持 Jinja2 风格的 {{variable}} 替换。
所有 Node 通过此模块加载 prompt，禁止在 Node 代码中硬编码 prompt 字符串。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(name: str, **kwargs: str) -> str:
    """
    加载 prompt 模板并填充变量。

    Args:
        name: 模板文件名（不含 .txt 后缀），如 "conversation"
        **kwargs: 模板变量，如 scenario="interview", turn=1

    Returns:
        填充后的 prompt 字符串

    Raises:
        FileNotFoundError: 模板文件不存在
    """
    template_path = _PROMPTS_DIR / f"{name}.txt"
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")

    content = template_path.read_text(encoding="utf-8")

    # 简单的 {{variable}} 替换
    for key, value in kwargs.items():
        content = content.replace("{{" + key + "}}", str(value))

    return content.strip()
