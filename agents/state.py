"""
LangGraph State Definition - AI 英语口语陪练系统

这是整个系统的唯一数据载体。
所有 Agent Node 通过读写 State 进行通信，Node 之间零耦合。

架构原则：
- State 是唯一的共享内存
- Node 只返回增量更新 dict（reducer 模式）
- 新增 Agent Node 只需在 State 中添加对应字段
"""

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


# ============================================================================
# Reducer 类型别名
# ============================================================================

# add_messages 将新消息追加到现有消息列表
# 这是 LangGraph 内置的消息管理器
Messages = Annotated[list[dict], add_messages]


# ============================================================================
# State 类型定义
# ============================================================================


class EnglishTutorState(TypedDict, total=False):
    """
    AI 英语口语陪练系统 - LangGraph 统一状态

    所有 Agent Node 通过读写此 State 进行数据交换。
    新增 Node 只需在 State 中添加对应字段。

    字段说明：
    - messages: 对话历史（由 LangGraph add_messages reducer 管理）
    - scenario/difficulty/level: 场景配置
    - ai_reply: Conversation Node 产出
    - correction: Correction Node 产出
    - score: Scoring Node 产出
    - metadata: 扩展字段
    - session_active: 会话控制标志
    """

    # ===== 对话历史（LangGraph 消息管理）=====
    messages: Messages

    # ===== 场景配置 =====
    scenario: str  # 场景标识: interview/restaurant/travel/meeting/daily
    difficulty: str  # 难度: easy/medium/hard
    level: str  # 用户水平: beginner/intermediate/advanced
    scenario_goal: str  # 场景目标描述

    # ===== Agent 产出 =====
    ai_reply: str  # Conversation Node 生成的 AI 回复
    correction: dict  # Correction Node 的结构化纠错结果
    score: dict  # Scoring Node 的四维评分结果

    # ===== 元数据 =====
    metadata: dict[str, Any]  # 扩展字段 (场景名称/角色/难度描述等)

    # ===== 会话控制 =====
    session_active: bool  # 会话是否活跃
