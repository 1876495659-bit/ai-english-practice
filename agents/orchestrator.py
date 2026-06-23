"""
Orchestrator - Agent 调度中心 (v2 完整版)

核心职责：
1. 统一管理所有 Agent 的生命周期
2. 作为 Agent 之间唯一的通信桥梁（禁止 Agent 直接交互）
3. 编排对话流程：场景初始化 → 对话生成 → 纠错 → 评分
4. 维护会话状态和 MessageContext 数据流

通信协议：
    用户输入
        → Orchestrator.start_session(scenario)
            → ScenarioAgent.initialize_scenario()
                → 返回开场白
        → Orchestrator.chat(user_input)
            → ConversationAgent.process(ctx)   # 生成AI回复
            → CorrectionAgent.process(ctx)     # 纠错用户输入
            → ScoringAgent.process(ctx)        # 评分用户输入
            → 返回完整结果

架构约束：
    - Agent 之间禁止直接调用
    - 所有数据通过 MessageContext 传递
    - 所有 Agent 输出必须结构化（JSON优先）
"""

import asyncio
from typing import Any, Optional
from pydantic import BaseModel
from agents.base_agent import BaseAgent, AgentResponse, MessageContext, AgentMessage
from agents.scenario_agent import ScenarioAgent
from agents.conversation_agent import ConversationAgent
from agents.correction_agent import CorrectionAgent
from agents.scoring_agent import ScoringAgent


class ChatResult(BaseModel):
    """
    完整聊天结果 - 一次 chat() 调用返回的全部信息

    包含本轮对话中所有 Agent 的处理结果。
    """
    # 场景信息
    scenario: str = ""
    scenario_name: str = ""
    difficulty: str = ""
    difficulty_description: str = ""

    # 对话轮次
    turn: int = 0

    # AI 回复（Conversation Agent 产出）
    ai_reply: str = ""

    # 用户输入
    user_input: str = ""

    # 纠错结果（Correction Agent 产出）
    correction: Optional[dict] = None

    # 评分结果（Scoring Agent 产出）
    score: Optional[dict] = None

    # 原始对话历史
    conversation_history: list[dict] = []


class Orchestrator:
    """
    Agent 调度器 v2

    所有 Agent 的注册、调度和协调都在这里完成。
    新增 Agent 只需注册，不需要修改其他代码。

    使用示例：
        orch = Orchestrator()
        orch.start_session(scenario="interview", difficulty="medium")
        result = await orch.chat("Hello, I am ready for the interview")
        print(result.ai_reply)
        print(result.correction)
        print(result.score)
    """

    def __init__(self):
        # Agent 注册表
        self._agents: dict[str, BaseAgent] = {}

        # 当前会话的 MessageContext
        self._context: Optional[MessageContext] = None

        # 完整对话历史（持久化）
        self._full_history: list[dict] = []

    # ------------------------------------------------------------------
    # Agent 注册与管理
    # ------------------------------------------------------------------

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
            raise KeyError(
                f"Agent '{name}' not found. "
                f"Registered: {list(self._agents.keys())}"
            )
        return agent

    def list_agents(self) -> list[str]:
        """列出所有已注册的 Agent"""
        return list(self._agents.keys())

    def _ensure_registered(self) -> None:
        """确保所有核心 Agent 都已注册（自动注册未注册的）"""
        if "scenario" not in self._agents:
            self.register(ScenarioAgent())
        if "conversation" not in self._agents:
            self.register(ConversationAgent())
        if "correction" not in self._agents:
            self.register(CorrectionAgent())
        if "scoring" not in self._agents:
            self.register(ScoringAgent())

    # ------------------------------------------------------------------
    # 会话管理
    # ------------------------------------------------------------------

    async def start_session(
        self,
        scenario: str = "daily",
        difficulty: str = "medium",
        level: str = "intermediate",
    ) -> MessageContext:
        """
        开始一个新的会话

        流程：
        1. 确保所有 Agent 已注册
        2. 通过 Scenario Agent 初始化场景
        3. 返回完整的 MessageContext

        Args:
            scenario: 场景标识 (interview/restaurant/travel/meeting/daily)
            difficulty: 难度 (easy/medium/hard)
            level: 用户水平 (beginner/intermediate/advanced)

        Returns:
            初始化后的 MessageContext，包含开场白
        """
        self._ensure_registered()
        scenario_agent = self.get_agent("scenario")

        # 通过 Scenario Agent 初始化场景
        self._context = await scenario_agent.initialize_scenario(
            scenario_id=scenario,
            difficulty=difficulty,
            level=level,
        )
        self._context.turn = 1

        # 保存完整历史
        self._full_history = list(self._context.conversation_history)

        print(f"[Orchestrator] Session started: "
              f"scenario={scenario}, difficulty={difficulty}, level={level}")

        return self._context

    def end_session(self) -> None:
        """结束当前会话"""
        self._context = None
        self._full_history.clear()
        print("[Orchestrator] Session ended")

    def get_context(self) -> Optional[MessageContext]:
        """获取当前会话的 MessageContext"""
        return self._context

    # ------------------------------------------------------------------
    # 核心对话流程
    # ------------------------------------------------------------------

    async def chat(self, user_input: str) -> ChatResult:
        """
        处理一轮对话 - 核心入口方法

        完整流水线：
        1. 将用户输入追加到对话历史
        2. ConversationAgent → 生成 AI 回复
        3. CorrectionAgent → 分析用户表达（纠错+优化）
        4. ScoringAgent → 评估用户口语（四维评分）
        5. 组装 ChatResult 返回

        Args:
            user_input: 用户输入的文本

        Returns:
            ChatResult 包含所有 Agent 的处理结果
        """
        if not self._context:
            raise RuntimeError(
                "No active session. Call start_session() first."
            )

        # 更新轮次
        self._context.turn += 1

        # 将用户输入加入对话历史
        self._context.conversation_history.append(
            {"role": "user", "content": user_input}
        )
        self._full_history.append(
            {"role": "user", "content": user_input}
        )

        # ===== 第1步：Conversation Agent 生成 AI 回复 =====
        print(f"\n  [Pipeline] Turn {self._context.turn}: Conversation Agent")
        conv_agent = self.get_agent("conversation")
        await conv_agent.process(self._context)
        self._context.conversation_history.append(
            {"role": "assistant", "content": conv_agent._conversation_history[-1].content}
        )
        self._full_history.append(
            {"role": "assistant", "content": conv_agent._conversation_history[-1].content}
        )
        self._context.conversation_reply = conv_agent._conversation_history[-1].content

        # ===== 第2步：Correction Agent 纠错 =====
        print(f"  [Pipeline] Turn {self._context.turn}: Correction Agent")
        correction_agent = self.get_agent("correction")
        await correction_agent.process(self._context)

        # ===== 第3步：Scoring Agent 评分 =====
        print(f"  [Pipeline] Turn {self._context.turn}: Scoring Agent")
        scoring_agent = self.get_agent("scoring")
        await scoring_agent.process(self._context)

        # ===== 组装结果 =====
        scenario_meta = self._context.metadata or {}
        return ChatResult(
            scenario=self._context.scenario,
            scenario_name=scenario_meta.get("scenario_name", self._context.scenario),
            difficulty=self._context.difficulty,
            difficulty_description=scenario_meta.get("difficulty_description", ""),
            turn=self._context.turn,
            ai_reply=self._context.conversation_reply,
            user_input=user_input,
            correction=self._context.correction_result,
            score=self._context.score_result,
            conversation_history=list(self._full_history),
        )

    # ------------------------------------------------------------------
    # 便捷方法
    # ------------------------------------------------------------------

    async def set_scenario(self, scenario: str, difficulty: str = "medium") -> MessageContext:
        """切换练习场景"""
        if not self._context:
            raise RuntimeError("No active session.")

        self._context.scenario = scenario
        self._context.difficulty = difficulty

        scenario_agent = self.get_agent("scenario")
        self._context = await scenario_agent.initialize_scenario(
            scenario_id=scenario,
            difficulty=difficulty,
            level=self._context.level,
        )

        return self._context

    def get_session_summary(self) -> dict:
        """获取会话摘要"""
        if not self._context:
            return {"error": "No active session"}

        scenario_meta = self._context.metadata or {}
        return {
            "scenario": self._context.scenario,
            "scenario_name": scenario_meta.get("scenario_name", ""),
            "difficulty": self._context.difficulty,
            "level": self._context.level,
            "turn": self._context.turn,
            "history_length": len(self._full_history),
            "last_ai_reply": self._context.conversation_reply[-80:]
                if self._context.conversation_reply else "",
        }
