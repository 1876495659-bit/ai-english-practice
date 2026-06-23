"""
Base Agent - 所有 Agent 的抽象基类

设计原则：
1. 每个 Agent 必须继承此基类，保证统一的接口
2. Agent 之间不直接通信，全部通过 Orchestrator 调度
3. 每个 Agent 有唯一的 name 和 version，便于追踪和调试
4. 所有 Agent 的输出遵循统一的数据结构
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
from pydantic import BaseModel, Field


class AgentMessage(BaseModel):
    """Agent 消息数据结构 - 所有 Agent 通信的统一格式"""
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


class BaseAgent(ABC):
    """
    Agent 抽象基类

    所有具体 Agent 必须实现:
    - name: Agent 唯一标识
    - version: 版本号
    - _build_system_prompt(): 构建系统提示词
    - _call_llm(): 调用 LLM 并返回结果
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
    def _build_system_prompt(self) -> str:
        """
        构建系统提示词

        优先从 prompts/ 目录加载模板文件，
        如果没有则返回默认提示词。
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
    async def _call_llm(self, messages: list[dict]) -> str:
        """
        调用 LLM

        由子类实现具体的 LLM 调用逻辑，
        支持不同的 Provider (OpenAI / Anthropic / Groq)。
        """
        ...

    async def process(self, user_input: str, context: Optional[dict] = None) -> AgentResponse:
        """
        处理用户输入的主入口

        Args:
            user_input: 用户输入文本
            context: 可选上下文信息（如场景、评分等）

        Returns:
            AgentResponse 标准化响应
        """
        # 添加用户消息到历史
        self._conversation_history.append(AgentMessage(role="user", content=user_input))

        # 构建消息列表
        system_prompt = self._build_system_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            *[
                {"role": m.role, "content": m.content}
                for m in self._conversation_history
            ],
        ]

        # 调用 LLM
        response_content = await self._call_llm(messages)

        # 添加助手回复到历史
        self._conversation_history.append(
            AgentMessage(role="assistant", content=response_content)
        )

        # 构建响应
        return AgentResponse(
            agent_name=self.name,
            content=response_content,
            messages=list(self._conversation_history),
            metadata=context or {},
        )

    def clear_history(self):
        """清空对话历史（新场景开始时调用）"""
        self._conversation_history.clear()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name} v{self.version}>"
