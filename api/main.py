"""
FastAPI 主应用入口 (LangGraph v3)

提供 RESTful API 供前端或外部系统调用。

API 端点：
    POST /api/chat          - 发送消息并获取完整回复（含纠错+评分）
    POST /api/session/start - 开始新会话（指定场景）
    GET  /api/session       - 获取当前会话状态
    POST /api/session/end   - 结束会话
    GET  /                  - 健康检查

架构：LangGraph StateGraph
    scenario -> conversation -> correction -> scoring -> END
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Optional

from agents.graph_builder import get_graph
from agents.state import EnglishTutorState

# 创建 FastAPI 应用
app = FastAPI(
    title="AI English Tutor",
    description="AI英语口语陪练系统 - LangGraph StateGraph 架构",
    version="3.0.0",
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
    scenario: str
    scenario_name: str
    difficulty: str
    turn: int
    ai_reply: str
    user_input: str
    correction: Optional[dict[str, Any]] = None
    score: Optional[dict[str, Any]] = None
    messages: list[dict[str, Any]]


# ========== 会话存储 ==========

# 使用内存存储会话状态（key: session_id, value: state dict）
_sessions: dict[str, dict[str, Any]] = {}
_next_session_id = 0


def _get_or_create_session() -> tuple[str, dict[str, Any]]:
    """获取或创建会话"""
    global _next_session_id
    if not _sessions:
        _next_session_id += 1
        session_id = f"session_{_next_session_id}"
        _sessions[session_id] = _make_initial_state()
        return session_id, _sessions[session_id]

    # 使用第一个活跃会话
    session_id = list(_sessions.keys())[0]
    state = _sessions[session_id]
    if not state.get("session_active", False):
        # 会话已结束，创建新会话
        _next_session_id += 1
        new_id = f"session_{_next_session_id}"
        _sessions[new_id] = _make_initial_state()
        return new_id, _sessions[new_id]
    return session_id, state


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
        "session_active": True,
        "messages": [],
    }


# ========== 生命周期 ==========


@app.on_event("startup")
async def startup() -> None:
    """应用启动时构建 LangGraph"""
    graph = get_graph()
    print(f"[API] AI English Tutor v3 started (LangGraph architecture)")


# ========== API 路由 ==========


@app.get("/")
async def health_check() -> dict[str, Any]:
    """健康检查"""
    return {
        "status": "ok",
        "service": "AI English Tutor",
        "version": "3.0.0",
        "architecture": "LangGraph StateGraph",
        "nodes": ["scenario", "conversation", "correction", "scoring"],
    }


@app.post("/api/session/start")
async def start_session(request: StartSessionRequest) -> dict[str, Any]:
    """
    开始新会话 - 指定练习场景

    调用后图初始化，Scenario Node 生成开场白。
    """
    # 创建新会话
    session_id, state = _get_or_create_session()

    # 更新场景配置
    state["scenario"] = request.scenario
    state["difficulty"] = request.difficulty
    state["level"] = request.level
    state["turn"] = 0
    state["session_active"] = True
    state["messages"] = []
    state["correction"] = {}
    state["score"] = {}

    # 运行图（从 scenario node 开始）
    graph = get_graph()
    final_state = graph.invoke(state)

    # 提取最后一条 AI 消息作为开场白
    messages = final_state.get("messages", [])
    opening_line = messages[-1]["content"] if messages else ""

    return {
        "status": "started",
        "session_id": session_id,
        "scenario": final_state["scenario"],
        "scenario_name": final_state["metadata"].get("scenario_name", request.scenario),
        "difficulty": final_state["difficulty"],
        "level": final_state["level"],
        "turn": final_state["turn"],
        "opening_line": opening_line,
        "scenario_goal": final_state.get("scenario_goal", ""),
        "messages": messages,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    发送消息并获取完整回复

    流程：Scenario -> Conversation -> Correction -> Scoring
    返回 AI 回复、纠错结果、评分结果。
    """
    session_id, state = _get_or_create_session()

    if not state.get("session_active", False):
        raise HTTPException(
            status_code=400, detail="No active session. Start a new session first."
        )

    # 更新用户输入和轮次
    state["turn"] += 1

    # 将用户消息注入 state
    current_messages = list(state.get("messages", []))
    current_messages.append({"role": "user", "content": request.message})
    state["messages"] = current_messages

    # 运行图
    graph = get_graph()
    final_state = graph.invoke(state)

    # 提取最后一条 AI 回复
    messages = final_state.get("messages", [])
    ai_reply = ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            ai_reply = msg.get("content", "")
            break

    return ChatResponse(
        scenario=final_state["scenario"],
        scenario_name=final_state["metadata"].get("scenario_name", final_state["scenario"]),
        difficulty=final_state["difficulty"],
        turn=final_state["turn"],
        ai_reply=ai_reply,
        user_input=request.message,
        correction=final_state.get("correction"),
        score=final_state.get("score"),
        messages=messages,
    )


@app.get("/api/session")
async def get_session() -> dict[str, Any]:
    """获取当前会话状态"""
    if not _sessions:
        raise HTTPException(status_code=404, detail="No active session")

    session_id = list(_sessions.keys())[0]
    state = _sessions[session_id]

    if not state.get("session_active", False):
        raise HTTPException(status_code=404, detail="Session ended")

    messages = state.get("messages", [])
    last_ai = ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            last_ai = msg.get("content", "")[-100:]
            break

    return {
        "scenario": state.get("scenario", ""),
        "scenario_name": state.get("metadata", {}).get("scenario_name", ""),
        "difficulty": state.get("difficulty", ""),
        "level": state.get("level", ""),
        "turn": state.get("turn", 0),
        "message_count": len(messages),
        "last_ai_reply": last_ai,
        "session_active": state.get("session_active", False),
    }


@app.post("/api/session/end")
async def end_session() -> dict[str, str]:
    """结束当前会话"""
    for state in _sessions.values():
        state["session_active"] = False

    return {"message": "Session ended"}
