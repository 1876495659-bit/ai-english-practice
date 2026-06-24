"""
Conversation Node - 对话生成节点

作为 LangGraph StateGraph 的一个 Node 运行。
职责：
1. 根据场景和上下文生成英语对话回复
2. 保持对话连贯性
3. 适配用户水平和难度等级
4. 支持真实 LLM 调用（OpenAI/Anthropic/Groq）或 mock 回退

注意：此文件作为独立 Node 函数运行，不继承 BaseAgent。
"""

from __future__ import annotations

import logging
from typing import Any

from agents.llm_client import call_llm
from agents.scenarios import get_scenario_config

logger = logging.getLogger(__name__)

# ============================================================================
# Node 入口
# ============================================================================


async def conversation_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    对话生成 Node

    优先使用真实 LLM 生成对话回复，LLM 失败时回退到 mock。

    Args:
        state: 当前图状态

    Returns:
        State 增量更新 dict
    """
    scenario: str = state.get("scenario", "daily")
    turn: int = state.get("turn", 1)
    level: str = state.get("level", "intermediate")
    difficulty: str = state.get("difficulty", "medium")
    metadata: dict[str, Any] = state.get("metadata", {})
    messages: list = state.get("messages", [])

    scenario_name: str = metadata.get("scenario_name", scenario)
    diff_desc: str = metadata.get("difficulty_description", "")
    scenario_goal: str = state.get("scenario_goal", "")

    # 构建用户消息历史（排除 system 角色）
    user_history = _build_conversation_history(messages)

    # 构建系统提示
    system_prompt = _build_system_prompt(
        scenario_name=scenario_name,
        scenario_id=scenario,
        difficulty=diff_desc,
        level=level,
        turn=turn,
        scenario_goal=scenario_goal,
    )

    # 尝试 LLM 调用（带 fallback）
    ai_reply = await _generate_reply(
        system_prompt=system_prompt,
        user_history=user_history,
        scenario=scenario,
        turn=turn,
    )

    return {
        "ai_reply": ai_reply,
        "messages": [{"role": "assistant", "content": ai_reply}],
    }


# ============================================================================
# 辅助函数
# ============================================================================


def _build_conversation_history(messages: list) -> list[dict[str, str]]:
    """
    从 LangGraph messages 构建对话历史

    提取最近的 6 轮对话（12 条消息）作为上下文。
    兼容 dict 和 BaseMessage 格式。
    """
    history: list[dict[str, str]] = []
    # 取最近 12 条消息（约 6 轮对话）
    recent = messages[-12:] if len(messages) > 12 else messages

    for msg in recent:
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "type", None) or getattr(msg, "_getType", lambda: "")()
            content = getattr(msg, "content", "")

        # Map LangGraph role names to standard names
        if role == "human":
            role = "user"
        elif role == "ai":
            role = "assistant"

        if content and role in ("user", "assistant"):
            history.append({"role": role, "content": content})

    return history


def _build_system_prompt(
    scenario_name: str,
    scenario_id: str,
    difficulty: str,
    level: str,
    turn: int,
    scenario_goal: str,
) -> str:
    """构建对话生成的系统提示词"""
    return (
        f"You are an AI English conversation partner in the {scenario_name} ({scenario_id}) scenario.\n\n"
        f"## Your Role\n"
        f"Professional English speaking practice assistant helping Chinese learners improve their oral English.\n\n"
        f"## Context\n"
        f"- Difficulty: {difficulty}\n"
        f"- User Level: {level}\n"
        f"- Turn: {turn}\n"
        f"- Goal: {scenario_goal}\n\n"
        f"## Rules\n"
        f"1. Each reply should be 2-4 sentences, giving the user room to respond\n"
        f"2. Use vocabulary and sentence structures appropriate for the user's level\n"
        f"3. Keep the conversation coherent and natural\n"
        f"4. Always provide a Chinese translation in parentheses after your English response\n"
        f"5. Stay in character for the {scenario_name} scenario\n"
        f"6. Ask follow-up questions to encourage the user to speak more\n\n"
        f"## Output Format\n"
        f"[English response] （中文翻译）\n\n"
        f"Remember: Be encouraging, patient, and help the learner build confidence."
    )


async def _generate_reply(
    system_prompt: str,
    user_history: list[dict[str, str]],
    scenario: str,
    turn: int,
) -> str:
    """
    生成 AI 回复

    优先使用真实 LLM（需 llm_enabled=True），失败时回退到 mock。
    """
    # 检查是否启用了 LLM
    try:
        from config.settings import settings
        if not getattr(settings, "llm_enabled", False):
            return _mock_reply(scenario, turn)
    except Exception:
        pass

    # 尝试 LLM
    try:
        llm_messages = [
            {"role": "system", "content": system_prompt},
            *user_history,
        ]
        reply = await call_llm(
            llm_messages,
            temperature=0.8,
            max_tokens=256,
        )
        logger.info(f"[ConversationNode] LLM reply generated (turn={turn})")
        return reply.strip()
    except Exception as e:
        logger.warning(f"[ConversationNode] LLM call failed, falling back to mock: {e}")

    # Mock fallback
    return _mock_reply(scenario, turn)


def _mock_reply(scenario: str, turn: int) -> str:
    """Mock 回复 - 当 LLM 不可用时使用"""
    scenario_replies: dict[str, dict[int, str]] = {
        "interview": {
            1: "Good morning! Thank you for coming today. Could you start by telling me a little about yourself? （早上好！感谢你今天前来。能否先介绍一下你自己？）",
            2: "That's impressive background! Can you tell me about a specific project you're particularly proud of? （很棒的背景！能告诉我一个你特别自豪的具体项目吗？）",
            3: "Great answer. What would you say is your greatest professional strength, and how has it helped you? （回答得很好。你认为你最大的职业优势是什么，它如何帮助了你？）",
        },
        "restaurant": {
            1: "Good evening! Welcome to our restaurant. Table for how many? （晚上好！欢迎来到我们餐厅。几位用餐？）",
            2: "Excellent choice! Would you like to start with an appetizer or perhaps a salad? （很好的选择！想要先来份开胃菜或沙拉吗？）",
            3: "Sure thing! Anything to drink? We have a special fresh lemonade today. （当然！需要喝点什么吗？我们今天有特制鲜柠檬汁。）",
        },
        "travel": {
            1: "Hi! Are you visiting our city? Can I help you find somewhere? （你好！你是来我们城市旅游的吗？需要帮忙找地方吗？）",
            2: "Oh, that's a great area to explore! Would you like a map or directions? （哦，那是个很棒的地方！需要地图或路线指引吗？）",
            3: "Absolutely, it's just a short walk from here. Turn left at the traffic light. （当然，从这里步行很短距离就到。在红绿灯处左转。）",
        },
        "meeting": {
            1: "Good morning everyone. Let's get started. Does anyone have updates on the Q3 project? （大家早上好。我们开始吧。有人汇报Q3项目的进展吗？）",
            2: "Thanks for the update. Does anyone have additional thoughts on this approach? （谢谢汇报。大家对这个方法还有什么想法吗？）",
            3: "Good point. Let's make sure we align on the timeline before moving forward. （好观点。在继续之前我们要确保时间表一致。）",
        },
        "daily": {
            1: "Hey! How's your day going? Anything interesting happening? （嘿！今天过得怎么样？有什么有趣的事吗？）",
            2: "That sounds fun! What else do you like to do in your free time? （听起来很有趣！你空闲时间还喜欢做什么？）",
            3: "Really? I've always wanted to try that. How did you get into it? （真的？我一直想试试。你是怎么开始接触这个的？）",
        },
    }

    replies = scenario_replies.get(scenario, scenario_replies["daily"])
    reply_index = min(turn, max(replies.keys()))
    return replies.get(reply_index, list(replies.values())[-1])
