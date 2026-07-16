"""
Session Management — 会话管理公共模块

从 api/main.py 提取，供 REST API 和 WebSocket 共享。
每个 session_id 映射到一个 thread_id（LangGraph checkpoint 键）。
实际状态由 LangGraph Checkpointer（SQLite）持久化。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# 内存映射：session_id -> thread_id
# 实际状态由 LangGraph Checkpointer（SQLite）持久化
_sessions: dict[str, str] = {}
_next_session_id = 0


def get_or_create_session() -> tuple[str, str]:
    """
    获取或创建会话。

    查找逻辑：
    1. 遍历已注册的 session，检查对应 thread 是否有活跃状态
    2. 如果找到活跃 session，直接返回
    3. 否则创建新 session

    Returns:
        (session_id, thread_id) 对
    """
    global _next_session_id

    # 查找活跃会话
    for sid, tid in _sessions.items():
        if _is_session_active(tid):
            return sid, tid

    # 创建新会话
    _next_session_id += 1
    session_id = f"session_{_next_session_id}"
    thread_id = f"thread_{session_id}"
    _sessions[session_id] = thread_id

    logger.info(f"[Session] Created new session: {session_id} (thread_id={thread_id})")
    return session_id, thread_id


def _is_session_active(thread_id: str) -> bool:
    """
    检查某个 thread 是否有活跃状态。

    简化逻辑：只要 _sessions 字典中有该 thread_id 的映射，
    即认为会话活跃（实际状态由 LangGraph Checkpointer 管理）。
    """
    return thread_id in _sessions.values()


def make_initial_state() -> dict[str, Any]:
    """创建初始状态"""
    return {
        "scenario": "daily",
        "difficulty": "medium",
        "level": "intermediate",
        "scenario_goal": "提升日常英语交流的流利度和自然度",
        "ai_reply": "",
        "correction": {},
        "score": {},
        "metadata": {},
        "turn": 0,
        "retry_count": 0,
        "max_retries": 3,
        "session_active": True,
        "messages": [],
        "skill_progress": {
            "total_turns": 0,
            "avg_score": 0.0,
            "error_frequency": {},
            "weakest_dimension": "",
            "strongest_dimension": "",
            "improvement_trajectory": [],
        },
    }


def remove_session(session_id: str) -> str | None:
    """
    删除会话，返回被移除的 thread_id。

    Args:
        session_id: 要删除的会话 ID

    Returns:
        thread_id 或被移除的 None
    """
    return _sessions.pop(session_id, None)


def list_sessions() -> dict[str, str]:
    """列出所有活跃会话（session_id → thread_id）"""
    return dict(_sessions)


def get_active_session_ids() -> list[str]:
    """获取所有活跃会话 ID 列表"""
    return list(_sessions.keys())
