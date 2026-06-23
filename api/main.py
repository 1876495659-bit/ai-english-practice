"""
FastAPI 主应用入口 (v2)

提供 RESTful API 供前端或外部系统调用。

API 端点：
    POST /api/chat          - 发送消息并获取完整回复（含纠错+评分）
    POST /api/session/start - 开始新会话（指定场景）
    GET  /api/session       - 获取当前会话状态
    POST /api/session/end   - 结束会话
    GET  /                  - 健康检查
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from agents.orchestrator import Orchestrator
from agents.base_agent import MessageContext

# 创建 FastAPI 应用
app = FastAPI(
    title="AI English Tutor",
    description="AI英语口语陪练系统 - 多Agent协作架构 v2",
    version="2.0.0",
)

# 全局 Orchestrator 实例
orchestrator = Orchestrator()


# ========== 请求/响应模型 ==========

class StartSessionRequest(BaseModel):
    """开始会话请求"""
    scenario: str = "daily"        # interview/restaurant/travel/meeting/daily
    difficulty: str = "medium"     # easy/medium/hard
    level: str = "intermediate"    # beginner/intermediate/advanced


class ChatRequest(BaseModel):
    """聊天请求"""
    message: str


class ChatResponse(BaseModel):
    """聊天响应 - 包含所有 Agent 的处理结果"""
    # 场景信息
    scenario: str
    scenario_name: str
    difficulty: str
    turn: int

    # AI 回复
    ai_reply: str

    # 用户输入
    user_input: str

    # 纠错结果
    correction: Optional[dict] = None

    # 评分结果
    score: Optional[dict] = None

    # 对话历史
    conversation_history: list[dict]


# ========== 生命周期 ==========

@app.on_event("startup")
async def startup():
    """应用启动时自动注册所有核心 Agent"""
    # 确保所有 Agent 已注册（通过 _ensure_registered 自动初始化）
    print(f"[API] Starting AI English Tutor v2")
    print(f"[API] Agents will be auto-registered on first session start")


# ========== API 路由 ==========

@app.get("/")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "service": "AI English Tutor",
        "version": "2.0.0",
        "architecture": "Multi-Agent Pipeline",
        "agents": orchestrator.list_agents(),
    }


@app.post("/api/session/start")
async def start_session(request: StartSessionRequest):
    """
    开始新会话 - 指定练习场景

    调用后会话初始化，Scenario Agent 生成开场白，
    返回完整的场景信息和 AI 开场白。
    """
    try:
        ctx = await orchestrator.start_session(
            scenario=request.scenario,
            difficulty=request.difficulty,
            level=request.level,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    scenario_meta = ctx.metadata or {}
    return {
        "status": "started",
        "scenario": ctx.scenario,
        "scenario_name": scenario_meta.get("scenario_name", ctx.scenario),
        "difficulty": ctx.difficulty,
        "level": ctx.level,
        "turn": ctx.turn,
        "opening_line": scenario_meta.get("opening_line", ""),
        "scenario_goal": ctx.scenario_goal,
        "conversation_history": ctx.conversation_history,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    发送消息并获取完整回复

    流程：Conversation Agent → Correction Agent → Scoring Agent
    返回 AI 回复、纠错结果、评分结果。
    """
    # 如果没有活跃会话，自动开始默认会话
    if not orchestrator.get_context():
        await orchestrator.start_session()

    try:
        result = await orchestrator.chat(request.message)
        return ChatResponse(
            scenario=result.scenario,
            scenario_name=result.scenario_name,
            difficulty=result.difficulty,
            turn=result.turn,
            ai_reply=result.ai_reply,
            user_input=result.user_input,
            correction=result.correction,
            score=result.score,
            conversation_history=result.conversation_history,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/session")
async def get_session():
    """获取当前会话状态"""
    summary = orchestrator.get_session_summary()
    if "error" in summary:
        raise HTTPException(status_code=404, detail="No active session")
    return summary


@app.post("/api/session/end")
async def end_session():
    """结束当前会话"""
    orchestrator.end_session()
    return {"message": "Session ended"}
