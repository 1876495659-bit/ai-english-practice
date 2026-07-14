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
from agents.prompts_loader import load_prompt

logger = logging.getLogger(__name__)


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

    # 从模板文件加载 prompt
    system_prompt = load_prompt(
        "conversation",
        scenario_name=scenario_name,
        scenario_id=scenario,
        difficulty=diff_desc,
        level=level,
        turn=str(turn),
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
    recent = messages[-12:] if len(messages) > 12 else messages

    for msg in recent:
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "type", None) or getattr(msg, "_getType", lambda: "")()
            content = getattr(msg, "content", "")

        if role == "human":
            role = "user"
        elif role == "ai":
            role = "assistant"

        if content and role in ("user", "assistant"):
            history.append({"role": role, "content": content})

    return history


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
    try:
        from config.settings import settings
        if not getattr(settings, "llm_enabled", False):
            return _mock_reply(scenario, turn)
    except Exception:
        pass

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

    return _mock_reply(scenario, turn)


def _mock_reply(scenario: str, turn: int) -> str:
    """Mock 回复 - 当 LLM 不可用时使用，覆盖 5 个场景 × 10 轮对话"""
    scenario_replies: dict[str, dict[int, str]] = {
        "interview": {
            1: "Good morning! Thank you for coming today. Could you start by telling me a little about yourself? （早上好！感谢你今天前来。能否先介绍一下你自己？）",
            2: "That's impressive background! Can you tell me about a specific project you're particularly proud of? What was your role? （背景很出色！能告诉我一个你特别自豪的具体项目吗？你的角色是什么？）",
            3: "Great answer. What would you say is your greatest professional strength, and how has it helped you succeed? （回答得很好。你认为你最大的职业优势是什么，它如何帮助你取得成功？）",
            4: "Interesting. Now let's talk about challenges. Can you describe a time when things didn't go as planned at work? （有意思。让我们谈谈挑战。能描述一次工作中计划没有按计划进行的情况吗？）",
            5: "How did you handle that situation? What did you learn from it? （你是如何处理那种情况的？你从中学到了什么？）",
            6: "That shows good resilience. Where do you see yourself in five years? （这显示了很好的韧性。你认为五年后你在哪里？）",
            7: "That's a thoughtful answer. Why are you interested in this particular position at our company? （很有深度的回答。为什么对这个职位感兴趣？）",
            8: "What motivates you to apply for this role? （是什么激励你申请这个职位？）",
            9: "Do you have any questions for us before we wrap up? （在我们结束之前，你有什么问题要问我们吗？）",
            10: "Thank you for your time today. We'll be in touch within the week regarding next steps. （感谢你今天的时间。我们将在本周内联系你关于下一步的安排。）",
        },
        "restaurant": {
            1: "Good evening! Welcome to our restaurant. Table for how many? （晚上好！欢迎来到我们餐厅。几位用餐？）",
            2: "Excellent choice! Would you like to start with an appetizer or perhaps a salad? （很好的选择！想要先来份开胃菜或沙拉吗？）",
            3: "Sure thing! Anything to drink? We have a special fresh lemonade today. （当然！需要喝点什么吗？我们今天有特制鲜柠檬汁。）",
            4: "Have you decided on your main course? Our chef's special today is grilled salmon with herbs. （决定好主菜了吗？今天厨师特选是香草烤三文鱼。）",
            5: "That sounds delicious! Would you like any sides with that? French fries, mashed potatoes, or steamed vegetables? （听起来很好吃！需要配菜吗？薯条、土豆泥或蒸蔬菜？）",
            6: "Your meal will be ready shortly. Is there anything else I can get you right now? （您的餐点马上就好。现在还需要什么吗？）",
            7: "How is everything tasting so far? （到目前为止味道怎么样？）",
            8: "Would you like to try our dessert menu? We have chocolate lava cake and tiramisu today. （想看看我们的甜点菜单吗？今天有巧克力熔岩蛋糕和提拉米苏。）",
            9: "Can I take your order for dessert? （可以点甜点了吗？）",
            10: "Will you be sharing or would you like separate portions? And may I bring the check? （要分享还是各点各的？我可以拿账单了吗？）",
        },
        "travel": {
            1: "Hi! Are you visiting our city? Can I help you find somewhere? （你好！你是来我们城市旅游的吗？需要帮忙找地方吗？）",
            2: "Oh, that's a great area to explore! Would you like a map or directions? （哦，那是个很棒的地方！需要地图或路线指引吗？）",
            3: "Absolutely, it's just a short walk from here. Turn left at the traffic light. （当然，从这里步行很短距离就到。在红绿灯处左转。）",
            4: "For the hotel, I'd recommend taking the subway Line 2 — it's the fastest way. （去酒店的话，我建议坐地铁2号线——最快的方式。）",
            5: "The nearest station is about 5 minutes on foot. You'll see signs in English. （最近的车站大约步行5分钟。你会看到英文指示牌。）",
            6: "If you're looking for local restaurants, there's a great food street two blocks away. （如果你想找当地餐馆，两条街外有一条美食街。）",
            7: "Be careful with your belongings in crowded areas. （在拥挤的地方要注意随身物品。）",
            8: "Would you like me to write down the address in Chinese so you can show a taxi driver? （要我帮你用中文写下地址以便给出租车司机看吗？）",
            9: "The best time to visit the museum is early morning — fewer crowds. （参观博物馆最好的时间是清晨——人少。）",
            10: "Enjoy your stay! Don't forget to check the weather forecast before heading out. （祝你们玩得开心！出门前别忘了查看天气预报。）",
        },
        "meeting": {
            1: "Good morning everyone. Let's get started. Does anyone have updates on the Q3 project? （大家早上好。我们开始吧。有人汇报Q3项目的进展吗？）",
            2: "Thanks for the update. Does anyone have additional thoughts on this approach? （谢谢汇报。大家对这个方法还有什么想法吗？）",
            3: "Good point. Let's make sure we align on the timeline before moving forward. （好观点。在继续之前我们要确保时间表一致。）",
            4: "I think we should assign clear ownership for each deliverable. （我认为我们应该为每个交付物分配明确的所有者。）",
            5: "What's the budget situation for this quarter? （这个季度的预算情况如何？）",
            6: "Let's schedule a follow-up meeting to review progress next week. （我们安排一个下周的跟进会议来审查进度。）",
            7: "Can someone summarize the action items we discussed today? （有人能总结一下我们今天讨论的行动事项吗？）",
            8: "I'll send out the meeting notes by end of day. （我会在今天下班前发出会议纪要。）",
            9: "Are there any blockers preventing us from meeting the deadline? （有什么阻碍我们按时完成任务的吗？）",
            10: "Great discussion today. Let's reconvene on Friday to finalize the plan. （今天的讨论很好。我们周五再开会敲定计划。）",
        },
        "daily": {
            1: "Hey! How's your day going? Anything interesting happening? （嘿！今天过得怎么样？有什么有趣的事吗？）",
            2: "That sounds fun! What else do you like to do in your free time? （听起来很有趣！你空闲时间还喜欢做什么？）",
            3: "Really? I've always wanted to try that. How did you get into it? （真的？我一直想试试。你是怎么开始接触这个的？）",
            4: "What kind of music are you listening to these days? （你最近在听什么类型的音乐？）",
            5: "Do you prefer reading books or watching movies? （你喜欢看书还是看电影？）",
            6: "That's a great hobby! How long have you been doing it? （很好的爱好！你做这个多久了？）",
            7: "Have you traveled anywhere interesting recently? （你最近去过什么有趣的地方旅行吗？）",
            8: "What's your favorite season and why? （你最喜欢的季节是什么，为什么？）",
            9: "If you could live anywhere in the world, where would you choose? （如果你能在世界上任何地方生活，你会选择哪里？）",
            10: "It was great chatting with you! What are your plans for the weekend? （和你聊天很开心！周末有什么计划？）",
        },
    }

    replies = scenario_replies.get(scenario, scenario_replies["daily"])
    reply_index = min(turn, max(replies.keys()))
    return replies.get(reply_index, list(replies.values())[-1])
