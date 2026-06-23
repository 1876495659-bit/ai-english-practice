"""
LangGraph StateGraph Builder - AI 英语口语陪练系统

核心职责：
1. 定义 StateGraph 的所有 Node
2. 配置 Node 之间的边（顺序执行）
3. 编译图为可执行的 CompiledGraph

架构：
    ┌──────────────┐
    │  entry_point  │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │   scenario   │  ← 场景初始化 / 开场白
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ conversation │  ← 生成 AI 回复
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │  correction  │  ← 语法纠错 + 表达优化
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │   scoring    │  ← 四维评分
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │     END      │
    └──────────────┘

数据流：所有 Node 通过 State 读写，Node 之间零耦合。
"""

from __future__ import annotations

from agents.scenario_node import scenario_node
from agents.conversation_node import conversation_node
from agents.correction_node import correction_node
from agents.scoring_node import scoring_node
from agents.state import EnglishTutorState

from langgraph.graph import END, StateGraph

# ============================================================================
# 图构建
# ============================================================================


def build_graph() -> StateGraph:
    """
    构建 LangGraph StateGraph

    Returns:
        已编译的 LangGraph 应用
    """
    # 使用 TypedDict 作为 State 类型，获得完整类型安全
    graph = StateGraph(EnglishTutorState)

    # 注册所有 Node
    graph.add_node("scenario", scenario_node)
    graph.add_node("conversation", conversation_node)
    graph.add_node("correction", correction_node)
    graph.add_node("scoring", scoring_node)

    # 配置边（顺序执行）
    graph.set_entry_point("scenario")
    graph.add_edge("scenario", "conversation")
    graph.add_edge("conversation", "correction")
    graph.add_edge("correction", "scoring")
    graph.add_edge("scoring", END)

    # 编译图
    app = graph.compile()

    return app


# ============================================================================
# 全局图实例（单例懒加载）
# ============================================================================

_app_instance: StateGraph | None = None


def get_graph() -> StateGraph:
    """
    获取编译好的 LangGraph 应用实例（单例）

    Returns:
        编译好的图
    """
    global _app_instance
    if _app_instance is None:
        _app_instance = build_graph()
    return _app_instance


def reset_graph() -> None:
    """重置图实例（用于测试或热重载）"""
    global _app_instance
    _app_instance = None
