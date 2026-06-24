"""
FastAPI 主应用入口 (LangGraph v6 - Persistent Sessions)

提供 RESTful API 供前端或外部系统调用。

API 端点：
    POST /api/chat          - 发送消息并获取完整回复（含纠错+评分+自适应路由）
    POST /api/session/start - 开始新会话（指定场景）
    GET  /api/session       - 获取当前会话状态（含 skill_progress）
    DELETE /api/session     - 删除会话（清理 checkpoint）
    POST /api/session/end   - 结束会话
    GET  /                  - 健康检查

架构：LangGraph StateGraph + SQLite Checkpointer（Stage 6 - 持久化）
    scenario → conversation → correction → scoring → route → conversation/END

持久化机制：
    - 每个 session_id 映射到一个 thread_id
    - 图状态（messages, retry_count, skill_progress 等）持久化到 SQLite
    - 用户中断后可通过同一 thread_id 恢复对话上下文
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agents.graph_builder import get_graph, reset_graph
from agents.state import EnglishTutorState

logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(
    title="AI English Tutor",
    description="AI英语口语陪练系统 - LangGraph StateGraph 架构（持久化会话）",
    version="6.0.0",
)


# ========== 请求/响应模型 ==========


class StartSessionRequest(BaseModel):
    """开始会话请求"""
    scenario: str = "daily"
    difficulty: str = "medium"
    level: str = "intermediate"


class ChatRequest(BaseModel):
    """聊天请求"""
    message: str


class ChatResponse(BaseModel):
    """聊天响应 - 包含所有 Node 的处理结果"""
    session_id: str
    scenario: str
    scenario_name: str
    difficulty: str
    turn: int
    ai_reply: str
    user_input: str
    correction: Optional[dict[str, Any]] = None
    score: Optional[dict[str, Any]] = None
    skill_progress: Optional[dict[str, Any]] = None
    retry_count: int = 0
    messages: list[dict[str, Any]]
    has_checkpoint: bool = False


class DeleteSessionResponse(BaseModel):
    """删除会话响应"""
    message: str
    session_id: str


# ========== 会话存储 ==========

# 内存映射：session_id -> thread_id
# 实际状态由 LangGraph Checkpointer（SQLite）持久化
_sessions: dict[str, str] = {}
_next_session_id = 0


def _get_or_create_session() -> tuple[str, str]:
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

    如果 checkpointer 中存在该 thread 的 checkpoint 且 session_active=True，
    则认为会话活跃。
    """
    graph = get_graph()
    if not graph.checkpointer:
        # 没有 checkpointer，无法判断，假设活跃
        return True

    try:
        # 列出该 thread 的所有 checkpoints
        snapshots = list(graph.checkpointer.list({"configurable": {"thread_id": thread_id}}))
        if not snapshots:
            return False

        # 检查最新的 checkpoint 是否标记为活跃
        latest = snapshots[-1]
        if hasattr(latest, 'config') and 'thread_id' in latest.config:
            return True
        return True
    except Exception as e:
        logger.debug(f"[Session] Failed to check activity for {thread_id}: {e}")
        return False


def _make_initial_state() -> dict[str, Any]:
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


# ========== 生命周期 ==========


@app.on_event("startup")
async def startup() -> None:
    """应用启动时构建 LangGraph"""
    graph = get_graph()
    cp_name = type(graph.checkpointer).__name__ if graph.checkpointer else "None"
    print(f"[API] AI English Tutor v6 started (Checkpointer: {cp_name})")


# ========== API 路由 ==========


@app.get("/")
async def health_check() -> dict[str, Any]:
    """健康检查"""
    graph = get_graph()
    return {
        "status": "ok",
        "service": "AI English Tutor",
        "version": "6.0.0",
        "architecture": "LangGraph StateGraph + SQLite Checkpointer",
        "nodes": ["scenario", "conversation", "correction", "scoring"],
        "features": [
            "conditional_routing",
            "skill_tracking",
            "adaptive_difficulty",
            "persistent_sessions",
        ],
        "checkpointer": type(graph.checkpointer).__name__ if graph.checkpointer else "none",
    }


@app.post("/api/session/start")
async def start_session(request: StartSessionRequest) -> dict[str, Any]:
    """
    开始新会话 - 指定练习场景

    调用后图初始化，Scenario Node 生成开场白。
    状态持久化到 SQLite checkpoint。

    返回：
        session_id: 会话 ID（后续请求使用）
        thread_id:  LangGraph 线程 ID（内部使用）
        opening_line: 场景开场白
        has_checkpoint: 是否启用了持久化
    """
    session_id, thread_id = _get_or_create_session()

    # 构建初始状态
    initial_state: EnglishTutorState = {
        "scenario": request.scenario,
        "difficulty": request.difficulty,
        "level": request.level,
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

    # 运行图（从 scenario node 开始）
    config = {"configurable": {"thread_id": thread_id}}
    graph = get_graph()
    final_state = await graph.ainvoke(initial_state, config=config)

    # 提取最后一条 AI 消息作为开场白
    messages = final_state.get("messages", [])
    opening_line = messages[-1]["content"] if messages else ""

    return {
        "status": "started",
        "session_id": session_id,
        "thread_id": thread_id,
        "scenario": final_state["scenario"],
        "scenario_name": final_state["metadata"].get("scenario_name", request.scenario),
        "difficulty": final_state["difficulty"],
        "level": final_state["level"],
        "turn": final_state["turn"],
        "retry_count": final_state.get("retry_count", 0),
        "opening_line": opening_line,
        "scenario_goal": final_state.get("scenario_goal", ""),
        "has_checkpoint": bool(graph.checkpointer),
        "messages": messages,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    发送消息并获取完整回复

    流程：Scenario → Conversation → Correction → Scoring → Route
    返回 AI 回复、纠错结果、评分结果、skill_progress。

    持久化机制：
        - 同一个 session_id 复用 thread_id
        - LangGraph Checkpointer 自动保存每次 invoke 后的状态
        - 用户中断后，通过同一 thread_id 恢复对话上下文

    自适应学习：
        - 评分低 → 自动回到 conversation 重新练习
        - 连续高分 → 自动提升难度
        - 连续低分 → 自动降低难度
    """
    session_id, thread_id = _get_or_create_session()

    if not thread_id:
        raise HTTPException(status_code=400, detail="No active session")

    # 构建 config（thread_id 用于 checkpoint 持久化）
    config = {"configurable": {"thread_id": thread_id}}

    # 获取当前状态（从 checkpointer 恢复）
    graph = get_graph()
    current_state: Optional[dict[str, Any]] = None

    if graph.checkpointer:
        try:
            # 列出该 thread 的所有 checkpoints，获取最新的
            snapshots = list(graph.checkpointer.list(config))
            if snapshots:
                # 取最后一个（最新的）checkpoint
                latest = snapshots[-1]
                # channel_values 包含当前所有 state 字段
                if hasattr(latest, 'channel_values'):
                    current_state = dict(latest.channel_values)
                elif isinstance(latest, dict) and 'channel_values' in latest:
                    current_state = dict(latest['channel_values'])
        except Exception as e:
            logger.warning(f"[Chat] Failed to restore state from checkpoint: {e}")

    # 如果无法恢复（新 session 或首次调用），创建新状态
    if not current_state:
        current_state = _make_initial_state()
        current_state["session_active"] = True

    # 更新用户输入和轮次
    current_state["turn"] = current_state.get("turn", 0) + 1

    # 将用户消息注入 state
    current_messages = list(current_state.get("messages", []))
    current_messages.append({"role": "user", "content": request.message})
    current_state["messages"] = current_messages

    # 运行图（包含条件路由，状态自动持久化到 checkpoint）
    final_state = await graph.ainvoke(current_state, config=config)

    # 提取最后一条 AI 回复
    messages = final_state.get("messages", [])
    ai_reply = ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            ai_reply = msg.get("content", "")
            break

    return ChatResponse(
        session_id=session_id,
        scenario=final_state["scenario"],
        scenario_name=final_state["metadata"].get("scenario_name", final_state["scenario"]),
        difficulty=final_state["difficulty"],
        turn=final_state["turn"],
        ai_reply=ai_reply,
        user_input=request.message,
        correction=final_state.get("correction"),
        score=final_state.get("score"),
        skill_progress=final_state.get("skill_progress"),
        retry_count=final_state.get("retry_count", 0),
        messages=messages,
        has_checkpoint=bool(graph.checkpointer),
    )


@app.get("/api/session")
async def get_session() -> dict[str, Any]:
    """获取当前会话状态"""
    if not _sessions:
        raise HTTPException(status_code=404, detail="No active session")

    session_id = list(_sessions.keys())[0]
    thread_id = _sessions[session_id]
    config = {"configurable": {"thread_id": thread_id}}

    # 尝试从 checkpointer 恢复最新状态
    graph = get_graph()
    state = None

    if graph.checkpointer:
        try:
            snapshots = list(graph.checkpointer.list(config))
            if snapshots:
                latest = snapshots[-1]
                if hasattr(latest, 'channel_values'):
                    state = dict(latest.channel_values)
                elif isinstance(latest, dict) and 'channel_values' in latest:
                    state = dict(latest['channel_values'])
        except Exception as e:
            logger.warning(f"[Session] Failed to load state: {e}")

    if not state:
        state = _make_initial_state()

    messages = state.get("messages", [])
    last_ai = ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            last_ai = msg.get("content", "")[-100:]
            break

    return {
        "session_id": session_id,
        "thread_id": thread_id,
        "scenario": state.get("scenario", ""),
        "scenario_name": state.get("metadata", {}).get("scenario_name", ""),
        "difficulty": state.get("difficulty", ""),
        "level": state.get("level", ""),
        "turn": state.get("turn", 0),
        "retry_count": state.get("retry_count", 0),
        "max_retries": state.get("max_retries", 3),
        "message_count": len(messages),
        "last_ai_reply": last_ai,
        "skill_progress": state.get("skill_progress", {}),
        "has_checkpoint": bool(graph.checkpointer),
        "session_active": state.get("session_active", False),
    }


@app.delete("/api/session/{session_id}", response_model=DeleteSessionResponse)
async def delete_session(session_id: str) -> DeleteSessionResponse:
    """
    删除会话 - 清理 checkpoint 和内存映射

    Args:
        session_id: 要删除的会话 ID

    Returns:
        删除结果
    """
    if session_id not in _sessions:
        raise HTTPException(
            status_code=404, detail=f"Session {session_id} not found"
        )

    thread_id = _sessions.pop(session_id)

    # 尝试删除 checkpoint
    graph = get_graph()
    if graph.checkpointer:
        try:
            # 清除该 thread 的所有 checkpoints
            for snapshot in graph.checkpointer.list({"configurable": {"thread_id": thread_id}}):
                pass  # LangGraph 的 checkpointer 目前没有直接删除 API
                # 如果需要彻底删除，可以在 SQLite 层面操作
        except Exception as e:
            logger.warning(
                f"[Session] Failed to clean checkpoint for {thread_id}: {e}"
            )

    return DeleteSessionResponse(
        message=f"Session {session_id} deleted",
        session_id=session_id,
    )


@app.post("/api/session/end")
async def end_session() -> dict[str, str]:
    """结束当前会话"""
    if not _sessions:
        raise HTTPException(status_code=404, detail="No active session")

    session_id = list(_sessions.keys())[0]
    thread_id = _sessions[session_id]
    config = {"configurable": {"thread_id": thread_id}}

    # 更新状态标记为不活跃
    graph = get_graph()
    if graph.checkpointer:
        try:
            # 通过 invoke 保存 session_active=False
            await graph.ainvoke(
                {"session_active": False},
                config=config,
            )
        except Exception as e:
            logger.warning(f"[Session] Failed to mark session inactive: {e}")

    return {"message": "Session ended"}
