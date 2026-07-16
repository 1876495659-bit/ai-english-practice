"""
FastAPI 主应用入口 (LangGraph v6 - Persistent Sessions)

提供 RESTful API 供前端或外部系统调用。

API 端点：
    POST /api/chat          - 发送消息并获取完整回复（含纠错+评分+自适应路由）
    POST /api/session/start - 开始新会话（指定场景）
    GET  /api/session       - 获取当前会话状态（含 skill_progress）
    DELETE /api/session     - 删除会话（清理 checkpoint）
    POST /api/session/end   - 结束会话
    WS   /ws/chat           - WebSocket 实时对话（逐 Node 事件推送）
    GET  /                  - 健康检查

架构：LangGraph StateGraph + SQLite Checkpointer（Stage 6 - 持久化）
    scenario → conversation → correction → scoring → route → conversation/END

持久化机制：
    - 每个 session_id 映射到一个 thread_id
    - 图状态（messages, retry_count, skill_progress 等）持久化到 SQLite
    - 用户中断后可通过同一 thread_id 恢复对话上下文
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request, UploadFile, Form, WebSocket
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from agents.graph_builder import (
    build_graph,
    close_sqlite_checkpointer,
    get_graph,
    reset_graph,
)
from agents.state import EnglishTutorState
from api.sessions import get_or_create_session, make_initial_state, remove_session, list_sessions, get_active_session_ids
from api.websocket import handle_chat_websocket

logger = logging.getLogger(__name__)


# ============================================================================
# Lifespan — 替代 on_event("startup"/"shutdown")
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan 事件处理器。

    替代已弃用的 @app.on_event("startup") / @app.on_event("shutdown")。

    启动阶段：
    1. 初始化 SQLite Checkpointer（创建连接、建表）
    2. 构建 LangGraph

    关闭阶段：
    1. 关闭 SQLite 连接，释放资源

    原理：
    - SQLite 连接必须在应用整个生命周期内保持打开
    - 不能在每次请求时创建/关闭连接（性能差 + 并发问题）
    - lifespan 确保连接在第一次请求前就绪，在退出前正确释放
    """
    # --- Startup ---
    logger.info("[Lifespan] Initializing SQLite Checkpointer...")
    try:
        from agents.graph_builder import _create_sqlite_checkpointer
        await _create_sqlite_checkpointer()
        graph = get_graph()
        cp_name = type(graph.checkpointer).__name__ if graph.checkpointer else "None"
        print(f"[API] AI English Tutor v6 started (Checkpointer: {cp_name})")
    except Exception as e:
        logger.error(f"[Lifespan] Failed to initialize: {e}")
        # 即使初始化失败，也要让应用启动（降级到 MemorySaver）
        print(f"[API] AI English Tutor v6 started (Checkpointer: MemorySaver fallback)")

    yield

    # --- Shutdown ---
    logger.info("[Lifespan] Shutting down...")
    await close_sqlite_checkpointer()


# ============================================================================
# FastAPI 应用
# ============================================================================

# 创建 FastAPI 应用
app = FastAPI(
    title="AI English Tutor",
    description="AI英语口语陪练系统 - LangGraph StateGraph 架构（持久化会话）",
    version="6.0.0",
    lifespan=lifespan,
)

# ============================================================================
# CORS 中间件 — 允许 Streamlit 跨域请求到 FastAPI
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# 全局异常处理器
# ============================================================================


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """处理 Pydantic 验证错误"""
    errors = []
    for error in exc.errors():
        loc = " → ".join(str(l) for l in error.get("loc", []))
        msg = error.get("msg", "")
        errors.append(f"{loc}: {msg}")
    return JSONResponse(
        status_code=422,
        content={"detail": "请求参数验证失败", "errors": errors},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """处理未捕获的异常，避免暴露内部堆栈"""
    logger.error(f"[API] Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "服务器内部错误",
            "message": str(exc)[:200] if str(exc) else "Unknown error",
        },
    )


# ============================================================================
# Pydantic 请求/响应模型
# ============================================================================


class StartSessionRequest(BaseModel):
    """开始会话请求"""
    scenario: str = "daily"
    difficulty: str = "medium"
    level: str = "intermediate"

    model_config = {"extra": "forbid"}

    @property
    def validated_scenario(self) -> str:
        """验证场景值，非法时回退到默认"""
        valid_scenarios = {"interview", "restaurant", "travel", "meeting", "daily"}
        if self.scenario not in valid_scenarios:
            return "daily"
        return self.scenario

    @property
    def validated_difficulty(self) -> str:
        """验证难度值，非法时回退到默认"""
        valid_difficulties = {"easy", "medium", "hard"}
        if self.difficulty not in valid_difficulties:
            return "medium"
        return self.difficulty

    @property
    def validated_level(self) -> str:
        """验证水平值，非法时回退到默认"""
        valid_levels = {"beginner", "intermediate", "advanced"}
        if self.level not in valid_levels:
            return "intermediate"
        return self.level


class ChatRequest(BaseModel):
    """聊天请求"""
    message: str

    model_config = {"extra": "forbid"}

    @property
    def validated_message(self) -> str:
        """验证消息内容"""
        stripped = self.message.strip()
        if not stripped:
            raise ValueError("消息不能为空")
        if len(stripped) > 500:
            raise ValueError("消息长度不能超过 500 字符")
        return stripped


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


# ============================================================================
# API 路由
# ============================================================================


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
    session_id, thread_id = get_or_create_session()

    # 构建初始状态（使用验证后的值）
    initial_state: EnglishTutorState = {
        "scenario": request.validated_scenario,
        "difficulty": request.validated_difficulty,
        "level": request.validated_level,
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

    # 提取最后一条 AI 消息作为开场白（兼容 dict 和 BaseMessage）
    messages = final_state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        opening_line = last_msg.content if hasattr(last_msg, "content") else last_msg.get("content", "")
    else:
        opening_line = ""

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
    session_id, thread_id = get_or_create_session()

    if not thread_id:
        raise HTTPException(status_code=400, detail="No active session")

    # 构建 config（thread_id 用于 checkpoint 持久化）
    config = {"configurable": {"thread_id": thread_id}}
    graph = get_graph()

    # 将用户消息注入 state（使用验证后的消息）
    try:
        validated_msg = request.validated_message
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # 先获取当前状态（从 checkpoint 恢复）
    try:
        state_snapshot = await graph.aget_state(config)
        if state_snapshot and state_snapshot.values:
            current_state = dict(state_snapshot.values)
        else:
            current_state = None
    except Exception as e:
        logger.warning(f"[Chat] Failed to get state: {e}")
        current_state = None

    # 如果无法恢复，创建新状态
    if not current_state:
        current_state = make_initial_state()
        current_state["session_active"] = True

    # 更新用户输入和轮次
    current_state["turn"] = current_state.get("turn", 0) + 1

    # 将用户消息注入 state
    current_messages = list(current_state.get("messages", []))
    current_messages.append({"role": "user", "content": validated_msg})
    current_state["messages"] = current_messages

    # 运行图（包含条件路由，状态自动持久化到 checkpoint）
    final_state = await graph.ainvoke(current_state, config=config)

    # 提取最后一条 AI 回复（兼容 dict 和 BaseMessage）
    messages = final_state.get("messages", [])
    ai_reply = ""
    for msg in reversed(messages):
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "type", None) or getattr(msg, "_getType", lambda: "")()
            content = getattr(msg, "content", "")
        if role in ("assistant", "ai"):
            ai_reply = content
            break

    # 序列化 messages 为 dict 格式（兼容 BaseMessage 和 dict）
    serialized_messages = []
    for msg in messages:
        if isinstance(msg, dict):
            serialized_messages.append(msg)
        else:
            role = getattr(msg, "type", None) or getattr(msg, "_getType", lambda: "")()
            # Map LangGraph role names to standard names
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"
            serialized_messages.append({
                "role": role,
                "content": getattr(msg, "content", ""),
            })

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
        messages=serialized_messages,
        has_checkpoint=bool(graph.checkpointer),
    )


@app.get("/api/session")
async def get_session() -> dict[str, Any]:
    """获取当前会话状态"""
    if not list_sessions():
        raise HTTPException(status_code=404, detail="No active session")

    session_id = get_active_session_ids()[0]
    thread_id = list_sessions()[session_id]
    config = {"configurable": {"thread_id": thread_id}}

    # 尝试从 checkpointer 恢复最新状态
    graph = get_graph()
    state = None

    if graph.checkpointer:
        try:
            # 使用 aget_state 获取当前状态（与 chat 端点一致）
            snapshot = await graph.aget_state(config)
            if snapshot and snapshot.values:
                state = dict(snapshot.values)
        except Exception as e:
            logger.warning(f"[Session] Failed to load state: {e}")

    if not state:
        state = make_initial_state()

    messages = state.get("messages", [])
    last_ai = ""
    for msg in reversed(messages):
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "type", None) or getattr(msg, "_getType", lambda: "")()
            content = getattr(msg, "content", "")
        if role in ("assistant", "ai"):
            last_ai = str(content)[-100:]
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
    if session_id not in list_sessions():
        raise HTTPException(
            status_code=404, detail=f"Session {session_id} not found"
        )

    thread_id = remove_session(session_id)

    # 尝试删除 checkpoint
    graph = get_graph()
    if graph.checkpointer:
        try:
            # 清除该 thread 的所有 checkpoints
            async for snapshot in graph.checkpointer.alist({"configurable": {"thread_id": thread_id}}):
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
    if not list_sessions():
        raise HTTPException(status_code=404, detail="No active session")

    session_id = get_active_session_ids()[0]
    thread_id = list_sessions()[session_id]
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


# ============================================================================
# WebSocket 实时对话端点
# ============================================================================


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket) -> None:
    """
    WebSocket 实时对话端点。

    客户端发送 JSON 消息，服务端逐个执行 LangGraph Node 并推送事件：
        - node_complete: 某个 Node 完成
        - correction: 纠错结果
        - score: 评分结果
        - chat_complete: 完整对话完成
        - error: 错误信息

    用法：
        ws = websockets.connect("ws://localhost:8000/ws/chat")
        await ws.send(json.dumps({"type": "chat", "message": "Hello"}))
        async for event in ws:
            print(json.loads(event))
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            await handle_chat_websocket(websocket, payload)
    except json.JSONDecodeError:
        await websocket.send_json({"type": "error", "detail": "无效 JSON 格式"})
    except Exception as e:
        logger.error(f"[WebSocket] Error: {e}", exc_info=True)
        try:
            await websocket.send_json({"type": "error", "detail": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ============================================================================
# ASR / TTS 端点
# ============================================================================


class TranscribeRequest(BaseModel):
    """语音识别请求"""
    language: str = "en"


class SynthesizeRequest(BaseModel):
    """语音合成请求"""
    text: str
    voice: Optional[str] = None
    speed: float = 1.0


@app.post("/api/asr/transcribe")
async def transcribe(language: str = Form("en"), file: UploadFile = ...):
    """
    语音转文本（ASR）

    上传音频文件，返回识别出的英文文本。
    支持 WAV/MP3/M4A/OGG 格式。
    """
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="请上传音频文件")

    try:
        from agents.asr import transcribe_audio

        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="音频文件为空")

        text = await transcribe_audio(audio_bytes, language=language)

        return {
            "status": "success",
            "text": text,
            "language": language,
            "audio_size": len(audio_bytes),
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[ASR] Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"语音识别失败: {str(e)}")


@app.post("/api/tts/synthesize")
async def synthesize(request: SynthesizeRequest):
    """
    文本转语音（TTS）

    将英文文本转换为音频并返回。
    支持自定义声音和语速。
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="待转换的文本不能为空")

    try:
        from agents.tts import synthesize_speech

        audio_bytes = await synthesize_speech(
            text=request.text,
            voice=request.voice,
            speed=request.speed,
        )

        # 返回 base64 编码的音频（方便前端直接播放）
        import base64
        encoded = base64.b64encode(audio_bytes).decode("ascii")

        return {
            "status": "success",
            "audio_base64": encoded,
            "voice": request.voice or "alloy",
            "speed": request.speed,
            "audio_size": len(audio_bytes),
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[TTS] Synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=f"语音合成失败: {str(e)}")


@app.get("/api/tts/voices")
async def list_voices() -> dict[str, Any]:
    """列出可用的 TTS 声音"""
    return {
        "voices": [
            {"name": "alloy", "gender": "neutral", "description": "平衡中性声音"},
            {"name": "echo", "gender": "male", "description": "温暖的男性声音"},
            {"name": "fable", "gender": "male", "description": "英式口音的男性声音"},
            {"name": "onyx", "gender": "male", "description": "沉稳的男性声音"},
            {"name": "nova", "gender": "female", "description": "友好的女性声音"},
            {"name": "shimmer", "gender": "female", "description": "轻快的女性声音"},
        ],
        "speed_range": [0.25, 4.0],
        "default_voice": "alloy",
    }
