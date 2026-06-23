"""
LangGraph State Definition - AI 英语口语陪练系统

这是整个系统的唯一数据载体。
所有 Agent Node 通过读写 State 进行通信，Node 之间零耦合。

架构原则：
- State 是唯一的共享内存
- Node 只返回增量更新 dict（reducer 模式）
- 新增 Agent Node 只需在 State 中添加对应字段
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


# ============================================================================
# Reducer 类型别名
# ============================================================================

# add_messages 将新消息追加到现有消息列表
# 这是 LangGraph 内置的消息管理器
Messages = Annotated[list[dict], add_messages]


# ============================================================================
# 子结构定义
# ============================================================================


class ErrorItem(TypedDict, total=False):
    """单个错误项"""
    type: str  # grammar | vocabulary | style | punctuation
    issue: str  # 问题描述
    suggestion: str  # 修正建议


class CorrectionResult(TypedDict, total=False):
    """纠错结果结构"""
    original: str  # 用户原始输入
    errors: list[ErrorItem]  # 检测到的错误列表
    corrected: str  # 修正后的表达
    suggestion: str  # 更地道的表达建议
    explanation: str  # 错误解释（中文）
    has_errors: bool  # 是否有错误
    polish_level: str  # basic | enhanced | advanced


class ScoreResult(TypedDict, total=False):
    """评分结果结构"""
    scores: dict[str, float]  # fluency/grammar/vocabulary/naturalness
    total: float  # 综合评分
    feedback_en: str  # 英文反馈
    feedback_zh: str  # 中文反馈
    strengths: list[str]  # 优点
    improvements: list[str]  # 改进建议


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
    - turn: 当前对话轮次
    - ai_reply: Conversation Node 产出
    - correction: Correction Node 产出的结构化纠错结果
    - score: Scoring Node 产出的四维评分结果
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

    # ===== 会话控制 =====
    turn: int  # 当前对话轮次（从 0 开始）
    session_active: bool  # 会话是否活跃

    # ===== Agent 产出 =====
    ai_reply: str  # Conversation Node 生成的 AI 回复
    correction: CorrectionResult  # Correction Node 的结构化纠错结果
    score: ScoreResult  # Scoring Node 的四维评分结果

    # ===== 元数据 =====
    metadata: dict[str, Any]  # 扩展字段 (场景名称/角色/难度描述等)
