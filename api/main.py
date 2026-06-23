"""
FastAPI 主应用入口

提供 RESTful API 供前端或外部系统调用。
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from agents.orchestrator import Orchestrator
from agents.conversation_agent import ConversationAgent

# 创建 FastAPI 应用
app = FastAPI(
    title="AI English Tutor",
    description="AI英语口语陪练系统 - 多Agent协作架构",
    version="1.0.0",
)

# 全局 Orchestrator 实例
orchestrator = Orchestrator()


# ========== 请求/响应模型 ==========

class ChatRequest(BaseModel):
    """聊天请求"""
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """聊天响应"""
    reply: str
    agent: str
    scenario: str
    turn: int
    metadata: dict


class ScenarioRequest(BaseModel):
    """场景切换请求"""
    scenario: str  # interview / restaurant / meeting / travel / daily


# ========== 生命周期 ==========

@app.on_event("startup")
async def startup():
    """应用启动时注册所有 Agent"""
    # 注册对话 Agent（Demo 阶段只注册这一个）
    conv_agent = ConversationAgent()
    orchestrator.register(conv_agent)
    print(f"[API] Registered agents: {orchestrator.list_agents()}")


# ========== API 路由 ==========

@app.get("/")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "service": "AI English Tutor",
        "version": "1.0.0",
        "agents": orchestrator.list_agents(),
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    发送消息并获取回复

    首次调用会自动开始新会话。
    """
    # 如果没有活跃会话，开始新会话
    if not orchestrator.get_session_state():
        orchestrator.start_session()

    try:
        response = await orchestrator.chat(request.message)
        return ChatResponse(
            reply=response.content,
            agent=response.agent_name,
            scenario=response.metadata.get("scenario", "daily"),
            turn=response.metadata.get("turn", 1),
            metadata=response.metadata,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/scenario")
async def change_scenario(request: ScenarioRequest):
    """切换练习场景"""
    valid_scenarios = ["interview", "restaurant", "meeting", "travel", "daily"]
    if request.scenario not in valid_scenarios:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid scenario. Choose from: {valid_scenarios}",
        )

    response = await orchestrator.set_scenario(request.scenario)
    return {"message": response.content, "scenario": request.scenario}


@app.get("/api/session")
async def get_session():
    """获取当前会话状态"""
    state = orchestrator.get_session_state()
    if not state:
        raise HTTPException(status_code=404, detail="No active session")
    return state


@app.post("/api/session/end")
async def end_session():
    """结束当前会话"""
    orchestrator.end_session()
    return {"message": "Session ended"}


# ========== 命令行运行入口 ==========

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
