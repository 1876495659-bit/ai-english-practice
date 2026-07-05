"""
公共工具函数 - AI 英语口语陪练系统

从 LangGraph messages 列表中提取最新用户输入的通用函数。
兼容 dict 格式和 LangChain BaseMessage 格式（LangGraph 1.x add_messages reducer）。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def extract_latest_user_input(messages: list[Any]) -> str:
    """
    从 LangGraph messages 列表中提取最新的用户输入。

    兼容两种消息格式：
    1. dict: {"role": "user", "content": "..."}
    2. BaseMessage: .type = "human"/"ai", .content = "..."

    遍历顺序：从最新消息向前查找，找到第一个 user/human 角色的消息即返回。

    Args:
        messages: LangGraph State 中的 messages 字段

    Returns:
        用户输入文本，未找到则返回空字符串
    """
    for msg in reversed(messages):
        if isinstance(msg, dict):
            if msg.get("role") == "user":
                return msg.get("content", "").strip()
        else:
            # BaseMessage 格式
            role = getattr(msg, "type", None) or getattr(msg, "_getType", lambda: "")()
            content = getattr(msg, "content", None) or getattr(msg, "_content", None)
            if role in ("human", "user") and content:
                return str(content).strip()
            # 额外兜底：某些 LangChain 版本用 _content 属性
            if content and role in ("human", "user"):
                return str(content).strip()
    return ""
