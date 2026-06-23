"""
Conversation Agent - 对话生成Agent

负责：
1. 根据场景和上下文生成英语对话回复
2. 保持对话连贯性
3. 适配用户英语水平

注意：此类为 Demo 实现，使用 mock LLM 调用以便快速验证流程。
实际使用时替换 _call_llm 方法即可接入真实 LLM。
"""

import json
import asyncio
from typing import Any, Optional

from agents.base_agent import BaseAgent, AgentResponse


class ConversationAgent(BaseAgent):
    """对话生成 Agent - Demo 版本（Mock LLM）"""

    @property
    def name(self) -> str:
        return "conversation"

    def _build_system_prompt(self) -> str:
        """加载对话提示词模板"""
        template = self._load_prompt_template("conversation")
        if template:
            return template
        return (
            "你是一个专业的英语口语陪练助手。"
            "请用英语进行对话，并在括号中提供中文翻译。"
        )

    async def _call_llm(self, messages: list[dict]) -> str:
        """
        Mock LLM 调用 - 用于验证流程

        实际项目中应替换为真实的 LLM 调用，例如：
        ```python
        client = get_llm_client()
        response = await client.chat.completions.create(
            model=settings.openai_model,
            messages=messages,
        )
        return response.choices[0].message.content
        ```
        """
        # 模拟 LLM 延迟
        await asyncio.sleep(0.5)

        # 从最后一条 user 消息中提取用户输入
        user_message = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                user_message = msg["content"]
                break

        # 简单的 mock 回复逻辑
        scenario = "daily"
        turn = 1
        for msg in messages:
            if msg["role"] == "system":
                # 从系统提示词中提取场景信息
                if "interview" in msg["content"]:
                    scenario = "interview"
                elif "restaurant" in msg["content"]:
                    scenario = "restaurant"

        # 根据轮次生成不同的 mock 回复
        replies = {
            1: "Hello! Welcome to our English practice session. Let's start with a simple greeting. How are you today? （你好！欢迎来到英语口语练习。我们先从一个简单的问候开始。你今天怎么样？）",
            2: "That's great to hear! Can you tell me more about your hobbies or interests? （很高兴听到这个！你能告诉我更多关于你的兴趣爱好吗？）",
            3: "Interesting! What kind of movies or books do you enjoy? （真有趣！你喜欢什么样的电影或书籍呢？）",
        }

        # 简单计数
        user_msg_count = sum(1 for m in messages if m["role"] == "user")
        reply_index = min(user_msg_count, len(replies))
        return replies.get(reply_index, "That's wonderful! Could you elaborate on that? Please try to use more descriptive words. （太好了！你能详细说说吗？请尝试使用更多描述性的词汇。）")

    async def process(self, user_input: str, context: Optional[dict] = None) -> AgentResponse:
        """
        覆写父类的 process 方法，注入场景上下文到对话中
        """
        scenario = context.get("scenario", "daily") if context else "daily"
        level = context.get("level", "intermediate") if context else "intermediate"
        turn = context.get("turn", 1) if context else 1

        # 将上下文信息添加到对话历史中（作为 system 消息）
        context_msg = AgentMessage(
            role="system",
            content=f"[SCENARIO: {scenario}] [LEVEL: {level}] [TURN: {turn}]",
        )
        self._conversation_history.append(context_msg)
        self._conversation_history.append(AgentMessage(role="user", content=user_input))

        system_prompt = self._build_system_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            *[
                {"role": m.role, "content": m.content}
                for m in self._conversation_history
                if m.role != "system" or not m.content.startswith("[SCENARIO:")
            ],
        ]

        response_content = await self._call_llm(messages)

        self._conversation_history.append(
            AgentMessage(role="assistant", content=response_content)
        )

        return AgentResponse(
            agent_name=self.name,
            content=response_content,
            messages=[m for m in self._conversation_history if m.role != "system" or not m.content.startswith("[SCENARIO:")],
            metadata={
                "scenario": scenario,
                "level": level,
                "turn": turn,
            },
        )
