"""
Correction Agent - 语法与表达纠错Agent

职责：
1. 分析用户英语表达中的语法错误
2. 提供修正版本
3. 给出更地道/自然的表达建议
4. 输出结构化 JSON 数据

输出格式：
{
    "original": "用户原始表达",
    "errors": [
        {"type": "grammar|vocabulary|pronunciation", "issue": "问题描述", "position": "错误位置"}
    ],
    "corrected": "修正后的表达",
    "suggestion": "更地道的表达建议",
    "explanation": "错误解释（中文）"
}
"""

import json
import asyncio
from typing import Any, Optional

from agents.base_agent import BaseAgent, AgentResponse, MessageContext, AgentMessage


class CorrectionAgent(BaseAgent):
    """语法与表达纠错 Agent"""

    @property
    def name(self) -> str:
        return "correction"

    def _build_system_prompt(self, ctx: MessageContext) -> str:
        """构建纠错系统提示词"""
        template = self._load_prompt_template("correction")
        return (
            f"{template}\n\n"
            f"用户当前水平: {ctx.level}\n"
            f"当前场景: {ctx.scenario}"
        )

    async def _call_llm(self, messages: list[dict], ctx: MessageContext) -> str:
        """
        Mock LLM 调用 - 生成纠错结果

        实际使用时替换为真实 LLM 调用。
        """
        await asyncio.sleep(0.3)

        # 从消息中提取用户最后一条输入
        user_input = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                user_input = msg["content"]
                break

        # 简单的 mock 纠错逻辑
        # 检测常见错误模式
        errors_found = []
        corrected = user_input
        suggestion = user_input

        # 检测常见语法错误
        if user_input.lower().startswith("i am"):
            errors_found.append({
                "type": "grammar",
                "issue": "可以使用更简洁的表达",
            })

        if "good" in user_input.lower() and "very" in user_input.lower():
            errors_found.append({
                "type": "vocabulary",
                "issue": "'very good' 过于简单，可以尝试更丰富的词汇",
            })
            suggestion = user_input.replace("very good", "fantastic/outstanding")

        # 如果没有检测到错误，给出中性反馈
        if not errors_found:
            return json.dumps({
                "original": user_input,
                "errors": [],
                "corrected": user_input,
                "suggestion": user_input,
                "explanation": "无明显语法错误，表达良好！",
                "has_errors": False,
            }, ensure_ascii=False, indent=2)

        # 构造 mock 纠错结果
        return json.dumps({
            "original": user_input,
            "errors": errors_found,
            "corrected": corrected,
            "suggestion": suggestion,
            "explanation": f"发现 {len(errors_found)} 个问题：",
            "has_errors": True,
        }, ensure_ascii=False, indent=2)

    async def analyze(self, user_text: str, ctx: MessageContext) -> dict:
        """
        分析用户文本并返回结构化纠错结果

        Args:
            user_text: 用户输入的文本
            ctx: 消息上下文

        Returns:
            结构化纠错结果 dict
        """
        # 构建消息列表
        system_prompt = self._build_system_prompt(ctx)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]

        # 调用 LLM
        raw_response = await self._call_llm(messages, ctx)

        # 解析 JSON 结果
        try:
            result = json.loads(raw_response)
        except json.JSONDecodeError:
            # 如果不是 JSON，包装成标准格式
            result = {
                "original": user_text,
                "errors": [],
                "corrected": raw_response,
                "suggestion": raw_response,
                "explanation": "LLM 返回非结构化结果，无法解析纠错详情",
                "has_errors": False,
            }

        return result

    async def process(self, ctx: MessageContext) -> MessageContext:
        """
        处理用户输入，执行纠错分析

        从对话历史中提取用户最新输入，进行纠错，
        并将结果写入 ctx.correction_result。
        """
        # 从对话历史中提取用户最新输入
        user_input = ""
        for msg in reversed(ctx.conversation_history):
            if msg.get("role") == "user":
                user_input = msg.get("content", "")
                break

        if not user_input:
            ctx.correction_result = {
                "original": "",
                "errors": [],
                "corrected": "",
                "suggestion": "",
                "explanation": "没有检测到用户输入",
                "has_errors": False,
            }
            return ctx

        # 执行纠错分析
        correction = await self.analyze(user_input, ctx)
        ctx.correction_result = correction

        return ctx
