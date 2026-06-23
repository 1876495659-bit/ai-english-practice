"""
Orchestrator - Agent 调度中心

核心职责：
1. 统一管理所有 Agent 的生命周期
2. 作为 Agent 之间唯一的通信桥梁（禁止 Agent 直接交互）
3. 编排对话流程：场景初始化 → 对话生成 → 纠错 → 评分
4. 维护会话状态

通信协议：
    用户 → Orchestrator → Scenario Agent → 设置场景
                         → Conversation Agent → 生成回复
                         → Correction Agent → 纠错
                         → Scoring Agent → 评分
                         → 用户
"""

from typing import Any, Optional
from agents.base_agent import BaseAgent, AgentResponse
from agents.conversation_agent import ConversationAgent


class Orchestrator:
    """
    Agent 调度器

    所有 Agent 的注册、调度和协调都在这里完成。
    新增 Agent 只需注册，不需要修改其他代码。
    """

    def __init__(self):
        # Agent 注册表 - 通过 name 查找 Agent
        self._agents: dict[str, BaseAgent] = {}
        # 当前会话状态
        self._session_state: dict[str, Any] = {}
        # 当前活跃的 Agent
        self._active_agent: Optional[str] = None

    def register(self, agent: BaseAgent) -> None:
        """注册一个 Agent"""
        if agent.name in self._agents:
            raise ValueError(f"Agent '{agent.name}' is already registered")
        self._agents[agent.name] = agent
        print(f"[Orchestrator] Registered: {agent}")

    def get_agent(self, name: str) -> BaseAgent:
        """通过名称获取 Agent"""
        agent = self._agents.get(name)
        if not agent:
            raise KeyError(f"Agent '{name}' not found. Registered agents: {list(self._agents.keys())}")
        return agent

    def list_agents(self) -> list[str]:
        """列出所有已注册的 Agent"""
        return list(self._agents.keys())

    def start_session(self, context: Optional[dict] = None) -> None:
        """开始一个新的会话"""
        self._session_state = {
            "scenario": "daily",
            "level": "intermediate",
            "turn": 0,
            "history": [],
            **(context or {}),
        }
        # 清空所有 Agent 的历史
        for agent in self._agents.values():
            agent.clear_history()
        print(f"[Orchestrator] Session started with context: {self._session_state}")

    def end_session(self) -> None:
        """结束当前会话"""
        self._session_state.clear()
        self._active_agent = None
        print("[Orchestrator] Session ended")

    async def chat(self, user_input: str) -> AgentResponse:
        """
        处理一轮对话 - 核心入口方法

        流程：
        1. 将用户输入转发给 Conversation Agent
        2. 附加会话上下文（场景、轮次等）

        Args:
            user_input: 用户输入的文本

        Returns:
            AgentResponse: 对话回复
        """
        # 更新轮次
        self._session_state["turn"] += 1

        # 获取对话 Agent
        conv_agent = self.get_agent("conversation")

        # 构建上下文
        context = {
            "scenario": self._session_state.get("scenario", "daily"),
            "level": self._session_state.get("level", "intermediate"),
            "turn": self._session_state["turn"],
        }

        # 调用对话 Agent
        response = await conv_agent.process(user_input, context=context)

        # 记录到会话历史
        self._session_state["history"].append({
            "user": user_input,
            "assistant": response.content,
            "turn": context["turn"],
        })

        return response

    async def set_scenario(self, scenario: str) -> AgentResponse:
        """切换练习场景（预留，后续由 Scenario Agent 实现）"""
        self._session_state["scenario"] = scenario
        return AgentResponse(
            agent_name="orchestrator",
            content=f"场景已切换到: {scenario}",
            metadata={"scenario": scenario},
        )

    def get_session_state(self) -> dict:
        """获取当前会话状态"""
        return dict(self._session_state)
