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

持久化（Stage 9 - SQLite 生产化）：
- 使用 AsyncSqliteSaver 实现真正的会话持久化
- 数据库文件：data/checkpoints.db
- thread_id 作为 session 的唯一键
- 每次 invoke 传入 config 即可恢复/保存状态
- 连接生命周期由 FastAPI lifespan 管理（api/main.py）
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

from agents.conversation_node import conversation_node
from agents.correction_node import correction_node
from agents.scenario_node import scenario_node
from agents.scoring_node import scoring_node
from agents.state import EnglishTutorState

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


# ============================================================================
# Checkpointer 工厂
# ============================================================================

# 全局变量：SQLite 连接和 saver（由 FastAPI lifespan 在启动时初始化）
_sqlite_conn = None
_sqlite_saver = None
_memory_saver: Optional[MemorySaver] = None


def _get_db_path() -> str:
    """获取 SQLite 数据库文件路径"""
    # 允许通过环境变量覆盖默认路径
    env_path = os.environ.get("CHECKPOINT_DB_PATH")
    if env_path:
        return env_path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_dir = os.path.join(project_root, "data")
    os.makedirs(db_dir, exist_ok=True)
    return os.path.join(db_dir, "checkpoints.db")


async def _create_sqlite_checkpointer() -> "AsyncSqliteSaver | None":
    """
    创建 SQLite Checkpointer（生产级持久化）。

    关键设计：
    - 手动管理 aiosqlite 连接生命周期（不通过 from_conn_string context manager）
    - 连接在应用启动时创建，关闭时释放
    - 自动建表（setup）

    Returns:
        AsyncSqliteSaver 实例，失败则返回 None
    """
    global _sqlite_conn, _sqlite_saver

    try:
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    except ImportError:
        logger.warning(
            "[Checkpointer] aiosqlite not installed. "
            "Install with: pip install aiosqlite"
        )
        return None

    db_path = _get_db_path()
    logger.info(f"[Checkpointer] SQLite path: {db_path}")

    try:
        # 1. 创建异步连接（不通过 context manager，保持连接打开）
        _sqlite_conn = await aiosqlite.connect(db_path)

        # 2. 创建 saver 实例
        _sqlite_saver = AsyncSqliteSaver(_sqlite_conn)

        # 3. 初始化表结构
        await _sqlite_saver.setup()

        logger.info("[Checkpointer] SQLite Checkpointer enabled (persistent)")
        return _sqlite_saver
    except Exception as e:
        logger.error(f"[Checkpointer] Failed to create SQLite checkpointer: {e}")
        if _sqlite_conn:
            await _sqlite_conn.close()
            _sqlite_conn = None
        return None


async def close_sqlite_checkpointer() -> None:
    """
    关闭 SQLite 连接（应用退出时调用）。

    这是 SQLite checkpointer 的清理函数，应在 FastAPI lifespan shutdown
    阶段调用，确保连接正确释放。
    """
    global _sqlite_conn, _sqlite_saver

    if _sqlite_conn:
        try:
            await _sqlite_conn.close()
            logger.info("[Checkpointer] SQLite connection closed")
        except Exception as e:
            logger.warning(f"[Checkpointer] Error closing SQLite: {e}")
        finally:
            _sqlite_conn = None
            _sqlite_saver = None


def _create_memory_checkpointer() -> MemorySaver:
    """创建内存 checkpointer（非持久化，用于测试/开发）"""
    logger.info("[Checkpointer] MemorySaver enabled (ephemeral, non-persistent)")
    return MemorySaver()


def get_sqlite_checkpointer() -> Optional["AsyncSqliteSaver"]:
    """
    获取 SQLite checkpointer 实例（由 lifespan 初始化后调用）。

    Returns:
        AsyncSqliteSaver 或 None
    """
    return _sqlite_saver


def get_checkpointer():
    """
    获取 checkpointer 实例（单例懒加载）。

    优先级：
    1. SQLite（如果已初始化）
    2. MemorySaver（fallback，用于测试或未初始化的场景）

    Returns:
        checkpointer 实例
    """
    global _memory_saver

    # 优先使用 SQLite（如果已初始化）
    if _sqlite_saver is not None:
        return _sqlite_saver

    # 回退到 MemorySaver（测试/开发环境）
    if _memory_saver is None:
        _memory_saver = _create_memory_checkpointer()
    return _memory_saver


def reset_checkpointer() -> None:
    """重置 checkpointer 单例（用于测试）"""
    global _sqlite_saver, _memory_saver
    _sqlite_saver = None
    _memory_saver = None


# ============================================================================
# 图构建
# ============================================================================


def build_graph(checkpointer=None) -> CompiledStateGraph:
    """
    构建并编译 LangGraph StateGraph

    关键设计：
    - scoring_node 返回 Command 对象来控制路由
    - checkpointer 启用状态持久化（Stage 6/9）

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

_app_instance: Optional[CompiledStateGraph] = None


def get_graph() -> CompiledStateGraph:
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
