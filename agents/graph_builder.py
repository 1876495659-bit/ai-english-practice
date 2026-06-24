"""
LangGraph StateGraph Builder - AI 英语口语陪练系统

核心职责：
1. 定义 StateGraph 的所有 Node
2. 配置 Node 之间的边（含条件分支）
3. 编译图为可执行的 CompiledGraph
4. 启用 Checkpointer 实现状态持久化（Stage 6）

架构（Stage 5 - 自适应学习）：
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
    │   scoring    │  ← 四维评分 + 条件路由（返回 Command）
    └──────┬───────┘
           │
      ┌────┴────┐
      ▼         ▼
  conversation  END

条件路由逻辑（由 scoring_node 通过 Command 控制）：
- grammar < 6 或 fluency < 6 → 回到 conversation 重新练习
- retry_count >= max_retries → 强制 END
- 其他情况 → END

数据流：所有 Node 通过 State 读写，Node 之间零耦合。

持久化（Stage 6）：
- 使用 SQLite 实现 checkpointer（生产级 session 持久化）
- thread_id 作为 session 的唯一键
- 每次 invoke 传入 config 即可恢复/保存状态
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from agents.conversation_node import conversation_node
from agents.correction_node import correction_node
from agents.scenario_node import scenario_node
from agents.scoring_node import scoring_node
from agents.state import EnglishTutorState

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

logger = logging.getLogger(__name__)

# ============================================================================
# Checkpointer 工厂
# ============================================================================


def _create_sqlite_checkpointer():
    """
    尝试创建 SQLite Checkpointer（生产级持久化）。
    如果 sqlite 不可用则回退到 MemorySaver。

    Returns:
        AsyncSqliteSaver 实例（非上下文管理器），或 None（回退到 MemorySaver）
    """
    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        db_path = os.environ.get(
            "CHECKPOINT_DB_PATH",
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "..",
                "data",
                "checkpoints.db",
            ),
        )
        # 确保数据目录存在
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        # from_conn_string 返回上下文管理器，我们提前 enter 它
        ctx = AsyncSqliteSaver.from_conn_string(db_path)
        saver = ctx.__enter__()
        logger.info(f"[Checkpointer] SQLite checkpointer enabled: {db_path}")
        return saver
    except ImportError:
        logger.warning(
            "[Checkpointer] langgraph-checkpoint-sqlite not installed. "
            "Falling back to MemorySaver (non-persistent)."
        )
        return None
    except Exception as e:
        logger.warning(
            f"[Checkpointer] Failed to init SQLite checkpointer ({e}). "
            "Falling back to MemorySaver."
        )
        return None


def _create_memory_checkpointer() -> MemorySaver:
    """创建内存 checkpointer（非持久化，用于测试/开发）"""
    logger.info("[Checkpointer] MemorySaver enabled (ephemeral, non-persistent)")
    return MemorySaver()


# 全局 checkpointer 实例
_sqlite_saver: Any = None
_memory_saver: Optional[MemorySaver] = None


def get_checkpointer():
    """
    获取 checkpointer 实例（单例懒加载）。

    优先级：
    1. SQLite（如果可用）
    2. MemorySaver（fallback）

    Returns:
        checkpointer 实例，或 None（都不可用）
    """
    global _sqlite_saver, _memory_saver

    if _sqlite_saver is not None:
        return _sqlite_saver

    # 尝试 SQLite
    _sqlite_saver = _create_sqlite_checkpointer()
    if _sqlite_saver is not None:
        return _sqlite_saver

    # 回退到 MemorySaver
    if _memory_saver is None:
        _memory_saver = _create_memory_checkpointer()
    return _memory_saver


# ============================================================================
# 图构建
# ============================================================================


def build_graph(checkpointer=None) -> StateGraph:
    """
    构建 LangGraph StateGraph

    关键设计：
    - scoring_node 返回 Command 对象来控制路由
    - checkpointer 启用状态持久化（Stage 6）

    Args:
        checkpointer: 可选的 checkpointer 实例。
                      如果为 None，自动创建（SQLite 优先，MemorySaver 回退）。

    Returns:
        已编译的 StateGraph（带 checkpointer）
    """
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

    # scoring → END（默认边）
    # 条件路由由 scoring_node 内部的 Command 控制
    graph.add_edge("scoring", END)

    # 获取 checkpointer
    cp = checkpointer or get_checkpointer()

    # 编译图（带 checkpointer）
    app = graph.compile(checkpointer=cp)

    if cp:
        logger.info(
            "[GraphBuilder] Checkpointer attached — state will be persisted "
            f"via {type(cp).__name__}"
        )
    else:
        logger.warning(
            "[GraphBuilder] No checkpointer available — "
            "state is ephemeral (in-memory only)"
        )

    return app


# ============================================================================
# 全局图实例（单例懒加载）
# ============================================================================

_app_instance: Optional[StateGraph] = None


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


def reset_checkpointer() -> None:
    """重置 checkpointer 单例（用于测试）"""
    global _sqlite_saver, _memory_saver
    _sqlite_saver = None
    _memory_saver = None
