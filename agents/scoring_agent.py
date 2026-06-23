"""
Scoring Agent - 口语评分Agent

职责：
1. 从四个维度对用户口语进行评分（0-10分）
2. 给出综合反馈和建议

评分维度：
- fluency: 流利度（连贯性、节奏感）
- grammar: 语法准确性
- vocabulary: 词汇丰富度和准确性
- naturalness: 表达自然度

输出格式：
{
    "scores": {
        "fluency": 7.5,
        "grammar": 6.0,
        "vocabulary": 8.0,
        "naturalness": 7.0
    },
    "total": 7.1,
    "feedback_en": "英文总结反馈",
    "feedback_zh": "中文建议",
    "strengths": ["优点1", "优点2"],
    "improvements": ["改进建议1", "改进建议2"]
}
"""

import json
import asyncio
import math
from typing import Any, Optional

from agents.base_agent import BaseAgent, AgentResponse, MessageContext, AgentMessage


class ScoringAgent(BaseAgent):
    """口语评分 Agent"""

    @property
    def name(self) -> str:
        return "scoring"

    def _build_system_prompt(self, ctx: MessageContext) -> str:
        """构建评分系统提示词"""
        template = self._load_prompt_template("scoring")
        return (
            f"{template}\n\n"
            f"用户当前水平: {ctx.level}\n"
            f"当前场景: {ctx.scenario}"
        )

    async def _call_llm(self, messages: list[dict], ctx: MessageContext) -> str:
        """
        Mock LLM 调用 - 生成评分结果

        实际使用时替换为真实 LLM 调用。
        """
        await asyncio.sleep(0.3)

        # 从消息中提取用户最新输入
        user_input = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                user_input = msg["content"]
                break

        # 简单的 mock 评分逻辑
        # 根据输入长度和质量给分
        word_count = len(user_input.split()) if user_input else 0

        # 基础分：根据用词复杂度
        has_complex_words = any(
            w.lower() in {"however", "therefore", "although", "moreover",
                          "significant", "particular", "environment",
                          "opportunity", "experience", "interesting"}
            for w in user_input.split()
        )
        base_score = 5.0 + min(word_count / 5, 3.0)  # 字数越多分越高，上限+3
        if has_complex_words:
            base_score += 1.0

        # 添加随机波动（模拟不同维度的差异）
        import random
        random.seed(hash(user_input) % (2**32))
        fluency = round(min(max(base_score + random.uniform(-1, 1), 1.0), 10.0), 1)
        grammar = round(min(max(base_score + random.uniform(-1.5, 1.5), 1.0), 10.0), 1)
        vocabulary = round(min(max(base_score + random.uniform(-1, 2), 1.0), 10.0), 1)
        naturalness = round(min(max(base_score + random.uniform(-1.5, 1), 1.0), 10.0), 1)

        total = round((fluency + grammar + vocabulary + naturalness) / 4, 1)

        # 生成反馈
        strengths = []
        improvements = []

        if fluency >= 7:
            strengths.append("表达流畅，思路清晰")
        else:
            improvements.append("注意提高表达的连贯性，减少停顿")

        if grammar >= 7:
            strengths.append("语法使用较为准确")
        else:
            improvements.append("注意基本语法结构，特别是时态和主谓一致")

        if vocabulary >= 7:
            strengths.append("词汇运用较为丰富")
        else:
            improvements.append("尝试使用更多样化的词汇，避免重复")

        if naturalness >= 7:
            strengths.append("表达自然，接近母语者习惯")
        else:
            improvements.append("多听多模仿母语者的表达方式")

        if not strengths:
            strengths.append("继续保持练习，进步空间很大")
        if not improvements:
            improvements.append("当前表现优秀，挑战更高难度")

        return json.dumps({
            "scores": {
                "fluency": fluency,
                "grammar": grammar,
                "vocabulary": vocabulary,
                "naturalness": naturalness,
            },
            "total": total,
            "feedback_en": (
                f"Overall {total}/10. "
                f"You showed good "
                f"{', '.join(s.split('，')[0] for s in strengths)}. "
                f"To improve, focus on "
                f"{', '.join(i.split('，')[0] for i in improvements)}."
            ),
            "feedback_zh": f"综合评分 {total}/10。{'; '.join(strengths + improvements)}",
            "strengths": strengths,
            "improvements": improvements,
        }, ensure_ascii=False, indent=2)

    async def evaluate(self, user_text: str, ctx: MessageContext) -> dict:
        """
        评估用户文本并返回结构化评分结果

        Args:
            user_text: 用户输入的文本
            ctx: 消息上下文

        Returns:
            结构化评分结果 dict
        """
        system_prompt = self._build_system_prompt(ctx)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]

        raw_response = await self._call_llm(messages, ctx)

        try:
            result = json.loads(raw_response)
        except json.JSONDecodeError:
            result = {
                "scores": {
                    "fluency": 5.0,
                    "grammar": 5.0,
                    "vocabulary": 5.0,
                    "naturalness": 5.0,
                },
                "total": 5.0,
                "feedback_en": "Could not generate detailed evaluation.",
                "feedback_zh": "无法生成详细评估",
                "strengths": [],
                "improvements": ["继续练习"],
            }

        return result

    async def process(self, ctx: MessageContext) -> MessageContext:
        """
        处理用户输入，执行评分

        从对话历史中提取用户最新输入，进行评分，
        并将结果写入 ctx.score_result。
        """
        user_input = ""
        for msg in reversed(ctx.conversation_history):
            if msg.get("role") == "user":
                user_input = msg.get("content", "")
                break

        if not user_input:
            ctx.score_result = {
                "scores": {"fluency": 0, "grammar": 0, "vocabulary": 0, "naturalness": 0},
                "total": 0,
                "feedback_en": "No input to evaluate.",
                "feedback_zh": "没有检测到用户输入",
                "strengths": [],
                "improvements": [],
            }
            return ctx

        score = await self.evaluate(user_input, ctx)
        ctx.score_result = score

        return ctx
