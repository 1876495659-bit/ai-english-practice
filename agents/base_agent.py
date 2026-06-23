"""
Base Agent - 所有 Agent 的抽象基类

设计原则：
1. 每个 Agent 必须继承此基类，保证统一的接口
2. Agent 之间不直接通信，全部通过 Orchestrator 调度
3. 每个 Agent 有唯一的 name 和 version，便于追踪和调试
4. 所有 Agent 的输出遵循统一的数据结构
5. 所有 Agent 间数据通过 MessageContext 传递
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
from pydantic import BaseModel, Field


# ============================================================================
# 统一消息上下文 - Agent 间数据流转的唯一载体
# ============================================================================

class MessageContext(BaseModel):
    """
    统一消息上下文 - 所有 Agent 间数据流转的唯一载体

    任何 Agent 之间的数据交换都必须通过此结构体，
    禁止 Agent 直接调用其他 Agent。

    典型流转路径：
        user_input → ScenarioAgent → ConversationAgent → CorrectionAgent → ScoringAgent
    """
    # --- 用户输入 ---
    user_input: str = Field(default="", description="用户原始输入（文本）")

    # --- 场景信息 ---
    scenario: str = Field(default="daily", description="当前场景标识")
    difficulty: str = Field(default="medium", description="难度等级 easy/medium/hard")
    scenario_goal: str = Field(default="", description="当前场景的对话目标")

    # --- 对话历史 ---
    conversation_history: list[dict] = Field(
        default_factory=list,
        description="完整对话历史 [{'role': 'user'|'assistant', 'content': '...'}]"
    )

    # --- 各 Agent 的处理结果 ---
    conversation_reply: str = Field(default="", description="Conversation Agent 的回复")
    correction_result: Optional[dict] = Field(
        default=None,
        description="Correction Agent 的结构化纠错结果"
    )
    score_result: Optional[dict] = Field(
        default=None,
        description="Scoring Agent 的结构化评分结果"
    )

    # --- 元数据 ---
    turn: int = Field(default=1, description="当前对话轮次")
    level: str = Field(default="intermediate", description="用户英语水平")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="额外扩展字段"
    )

    def to_dict(self) -> dict:
        """转换为字典（用于 LLM 调用）"""
        return self.model_dump(exclude_none=True)


# ============================================================================
# Agent 内部消息与响应
# ============================================================================

class AgentMessage(BaseModel):
    """Agent 内部消息数据结构"""
    role: str = Field(description="角色: user / assistant / system")
    content: str = Field(description="消息内容")
    metadata: dict[str, Any] = Field(default_factory=dict, description="附加元数据")


class AgentResponse(BaseModel):
    """Agent 标准响应结构"""
    agent_name: str = Field(description="发出响应的 Agent 名称")
    content: str = Field(description="回复内容")
    messages: list[AgentMessage] = Field(default_factory=list, description="完整对话历史")
    metadata: dict[str, Any] = Field(default_factory=dict, description="额外信息(评分/纠错等)")
    done: bool = Field(default=True, description="是否结束本轮对话")


# ============================================================================
# Agent 基类
# ============================================================================

class BaseAgent(ABC):
    """
    Agent 抽象基类

    所有具体 Agent 必须实现:
    - name: Agent 唯一标识
    - _build_system_prompt(ctx): 根据上下文构建系统提示词
    - _call_llm(messages, context): 调用 LLM 并返回结果
    - process(ctx): 处理 MessageContext 并返回新的 MessageContext
    """

    def __init__(self):
        self._conversation_history: list[AgentMessage] = []

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent 的唯一名称，用于标识和日志追踪"""
        ...

    @property
    def version(self) -> str:
        """Agent 版本号"""
        return "1.0.0"

    @abstractmethod
    def _build_system_prompt(self, ctx: MessageContext) -> str:
        """
        构建系统提示词

        参数 ctx 可以让 Agent 根据当前场景/上下文动态生成 prompt。
        优先从 prompts/ 目录加载模板文件。
        """
        ...

    def _load_prompt_template(self, template_name: str) -> str:
        """从 prompts/ 目录加载提示词模板"""
        import os
        prompt_path = os.path.join(
            os.path.dirname(__file__), "..", "prompts", f"{template_name}.txt"
        )
        if os.path.exists(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        return ""

    @abstractmethod
    async def _call_llm(self, messages: list[dict], ctx: MessageContext) -> str:
        """
        调用 LLM

        由子类实现具体的 LLM 调用逻辑，
        支持不同的 Provider (OpenAI / Anthropic / Groq)。
        """
        ...

    async def process(self, ctx: MessageContext) -> MessageContext:
        """
        处理消息上下文的主入口

        标准流程：
        1. 从 ctx 提取用户输入
        2. 构建系统提示词 + 消息列表
        3. 调用 LLM
        4. 更新 ctx 并返回

        Args:
            ctx: 输入的消息上下文

        Returns:
            更新后的消息上下文
        """
        # 添加用户消息到内部历史
        self._conversation_history.append(
            AgentMessage(role="user", content=ctx.user_input)
        )

        # 构建消息列表
        system_prompt = self._build_system_prompt(ctx)
        messages = [
            {"role": "system", "content": system_prompt},
            *[
                {"role": m.role, "content": m.content}
                for m in self._conversation_history
            ],
        ]

        # 调用 LLM
        response_content = await self._call_llm(messages, ctx)

        # 添加助手回复到内部历史
        self._conversation_history.append(
            AgentMessage(role="assistant", content=response_content)
        )

        # 返回更新后的上下文（由子类决定是否填充额外字段）
        return ctx

    def clear_history(self):
        """清空对话历史（新场景开始时调用）"""
        self._conversation_history.clear()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name} v{self.version}>"
