"""
Conversation Agent - 对话生成Agent (v2 完整版)

负责：
1. 根据场景和上下文生成英语对话回复
2. 保持对话连贯性
3. 适配用户水平和难度等级
4. 输出结构化结果

注意：当前使用 Mock LLM 以便快速验证流程。
替换 _call_llm 方法即可接入真实 LLM。
"""

import json
import asyncio
from typing import Any, Optional

from agents.base_agent import BaseAgent, AgentResponse, MessageContext, AgentMessage


class ConversationAgent(BaseAgent):
    """对话生成 Agent - v2 完整版"""

    @property
    def name(self) -> str:
        return "conversation"

    def _build_system_prompt(self, ctx: MessageContext) -> str:
        """根据场景上下文构建系统提示词"""
        template = self._load_prompt_template("conversation")

        scenario_name = ctx.metadata.get("scenario_name", ctx.scenario)
        diff_desc = ctx.metadata.get("difficulty_description", "")

        return (
            f"{template}\n\n"
            f"=== 当前场景 ===\n"
            f"场景: {scenario_name} ({ctx.scenario})\n"
            f"难度: {diff_desc} ({ctx.difficulty})\n"
            f"用户水平: {ctx.level}\n"
            f"对话轮次: {ctx.turn}\n"
            f"场景目标: {ctx.scenario_goal}"
        )

    async def _call_llm(self, messages: list[dict], ctx: MessageContext) -> str:
        """
        Mock LLM 调用 - 生成对话回复

        实际使用时替换为：
        ```python
        from config.providers import get_llm_client
        from config.settings import settings

        client = get_llm_client()
        response = await client.chat.completions.create(
            model=settings.openai_model,
            messages=messages,
            temperature=0.8,
        )
        return response.choices[0].message.content
        ```
        """
        await asyncio.sleep(0.5)

        # 从最后一条 user 消息提取用户输入
        user_input = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                user_input = msg["content"]
                break

        # 根据场景和轮次生成 mock 回复
        scenario_replies = {
            "interview": {
                1: "Good morning! Thank you for coming today. Could you start by telling me a little about yourself? （早上好！感谢你今天前来。能否先介绍一下你自己？）",
                2: "That's impressive background! Can you tell me about a specific project you're particularly proud of? （很棒的背景！能告诉我一个你特别自豪的具体项目吗？）",
                3: "Great answer. What would you say is your greatest professional strength, and how has it helped you? （回答得很好。你认为你最大的职业优势是什么，它如何帮助了你？）",
                4: "Interesting perspective. How do you handle pressure and tight deadlines in your work? （有趣的观点。你如何处理工作中的压力和紧迫的截止日期？）",
                5: "Thank you for sharing that. Do you have any questions for us before we conclude? （感谢分享。在我们结束之前，你有什么问题想问我们的吗？）",
            },
            "restaurant": {
                1: "Good evening! Welcome to our restaurant. Table for how many? （晚上好！欢迎来到我们餐厅。几位用餐？）",
                2: "Excellent choice! Would you like to start with an appetizer or perhaps a salad? （很好的选择！想要先来份开胃菜或沙拉吗？）",
                3: "Sure thing! Anything to drink? We have a special fresh lemonade today. （当然！需要喝点什么吗？我们今天有特制鲜柠檬汁。）",
                4: "Perfect! I'll put that order in for you right away. （太好了！我马上为您下单。）",
                5: "How was everything? Is there anything else I can get for you? （一切都好吗？还需要什么吗？）",
            },
            "travel": {
                1: "Hi! Are you visiting our city? Can I help you find somewhere? （你好！你是来我们城市旅游的吗？需要帮忙找地方吗？）",
                2: "Oh, that's a great area to explore! Would you like a map or directions? （哦，那是个很棒的地方！需要地图或路线指引吗？）",
                3: "Absolutely, it's just a short walk from here. Turn left at the traffic light. （当然，从这里步行很短距离就到。在红绿灯处左转。）",
                4: "That's one of our most popular spots! You won't regret it. （这是最受欢迎的景点之一！你不会后悔的。）",
                5: "Enjoy your visit! Let me know if you need any more help. （玩得开心！如果需要更多帮助随时告诉我。）",
            },
            "meeting": {
                1: "Good morning everyone. Let's get started. Does anyone have updates on the Q3 project? （大家早上好。我们开始吧。有人汇报Q3项目的进展吗？）",
                2: "Thanks for the update. Does anyone have additional thoughts on this approach? （谢谢汇报。大家对这个方法还有什么想法吗？）",
                3: "Good point. Let's make sure we align on the timeline before moving forward. （好观点。在继续之前我们要确保时间表一致。）",
                4: "Agreed. Let's schedule a follow-up to discuss the details further. （同意。我们安排后续会议深入讨论细节。）",
                5: "Great discussion, everyone. I'll send out the meeting notes by end of day. （讨论得很好。我会在今天结束前发出会议纪要。）",
            },
            "daily": {
                1: "Hey! How's your day going? Anything interesting happening? （嘿！今天过得怎么样？有什么有趣的事吗？）",
                2: "That sounds fun! What else do you like to do in your free time? （听起来很有趣！你空闲时间还喜欢做什么？）",
                3: "Really? I've always wanted to try that. How did you get into it? （真的？我一直想试试。你是怎么开始接触这个的？）",
                4: "Haha, that's a great story! What happened next? （哈哈，好故事！后来发生了什么？）",
                5: "That's awesome! We should definitely do that sometime. （太棒了！我们找个时间一定要去做。）",
            },
        }

        replies = scenario_replies.get(ctx.scenario, scenario_replies["daily"])
        reply_index = min(ctx.turn, len(replies))
        return replies.get(reply_index, replies[max(replies.keys())])

    async def process(self, ctx: MessageContext) -> MessageContext:
        """
        处理消息上下文，生成对话回复

        覆写基类 process 方法以适配 MessageContext 架构。
        """
        # 构建消息列表
        system_prompt = self._build_system_prompt(ctx)
        messages = [
            {"role": "system", "content": system_prompt},
            *[
                {"role": msg["role"], "content": msg["content"]}
                for msg in ctx.conversation_history
            ],
        ]

        # 调用 LLM
        response_content = await self._call_llm(messages, ctx)

        # 保存到内部历史
        self._conversation_history.append(
            AgentMessage(role="assistant", content=response_content)
        )

        # 更新上下文中的 AI 回复
        ctx.conversation_reply = response_content

        return ctx
