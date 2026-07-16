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
import re
from typing import Any

from agents.llm_client import call_llm
from agents.prompts_loader import load_prompt

logger = logging.getLogger(__name__)


async def conversation_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    对话生成 Node

    优先使用真实 LLM 生成对话回复，LLM 失败时回退到智能 mock。
    智能 mock 会根据用户输入内容生成上下文相关的回复，而非纯静态轮次字典。

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

    # 提取最近的用户输入用于生成上下文相关回复
    user_input = _extract_last_user_input(messages)

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
        user_input=user_input,
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
    user_input: str = "",
) -> str:
    """
    生成 AI 回复

    优先使用真实 LLM（需 llm_enabled=True），失败时回退到智能 mock。
    智能 mock 会根据用户输入内容生成上下文相关的回复，而非纯静态轮次字典。
    """
    try:
        from config.settings import settings
        if not getattr(settings, "llm_enabled", False):
            return _smart_mock_reply(scenario, turn, user_input)
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
        logger.warning(f"[ConversationNode] LLM call failed, falling back to smart mock: {e}")

    return _smart_mock_reply(scenario, turn, user_input)


def _extract_last_user_input(messages: list) -> str:
    """从消息历史中提取最近一条用户输入"""
    for msg in reversed(messages):
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "type", None) or getattr(msg, "_getType", lambda: "")()
            content = getattr(msg, "content", "")
        if role in ("user", "human") and content:
            return str(content)
    return ""


# ============================================================================
# 智能 Mock 回复 — 基于关键词和上下文的动态回复
# ============================================================================

# 通用回复模板：根据用户输入内容生成上下文相关的回复
_GENERIC_RESPONSES = {
    # 问候类
    r"\b(hi|hello|hey|good\s+(morning|afternoon|evening))\b": [
        "Hello! {greeting}! How are you doing today? （你好！{greeting}！你今天怎么样？）",
        "Hey there! Great to see you! How has your day been so far? （嘿！很高兴见到你！你今天过得怎么样？）",
        "Hi! Welcome! What would you like to talk about today? （嗨！欢迎！今天想聊些什么？）",
    ],
    # 近况类
    r"\b(how\s+are\s+you|how('re| is)\s+it\s+going|what('s| is)\s+up)\b": [
        "I'm doing great, thank you for asking! How about you? Anything new? （我很好，谢谢关心！你呢？有什么新鲜事吗？）",
        "Pretty good! Thanks for checking in. What's new with you? （挺好的！谢谢关心。你有什么新动态吗？）",
    ],
    # 喜好/兴趣类
    r"\b(like\s+(to|doing)|love\s+to|hobby|interest|enjoy)\b": [
        "That's wonderful! Sharing hobbies is a great way to practice English. Tell me more about it! （太棒了！分享爱好是练习英语的好方法。多告诉我一些吧！）",
        "Interesting! What made you interested in that? （很有意思！是什么让你对它感兴趣的？）",
    ],
    # 活动/日常类
    r"\b(play|watch|read|listen|cook|travel|exercise|sport|game)\b": [
        "That sounds fun! How often do you get to do that? （听起来很有趣！你多久做一次这个？）",
        "Nice! I love hearing about people's activities. What's your favorite part about it? （不错！我喜欢听大家分享活动。你最喜欢哪部分？）",
        "That's a great activity! Do you do it alone or with friends? （很棒的活动！你是一个人做还是和朋友一起？）",
    ],
    # 工作/学习类
    r"\b(work|study|school|job|class|learn|teacher|student|office)\b": [
        "That's important! How long have you been doing that? （这很重要！你做这个多久了？）",
        "Interesting! What do you enjoy most about your work/studies? （有意思！你最喜欢工作/学习中的什么？）",
    ],
    # 食物/餐饮类
    r"\b(eat|food|cook|restaurant|hungry|dinner|lunch|breakfast|like\s+to\s+eat)\b": [
        "Food is always a great topic! What's your favorite cuisine? （美食总是个好话题！你最喜欢哪种菜系？）",
        "Yum! That sounds delicious. Have you ever tried cooking it yourself? （好吃！听起来很美味。你试过自己做吗？）",
    ],
    # 天气/季节类
    r"\b(weather|rain|sun|cold|hot|warm|snow|season|spring|summer|fall|autumn|winter)\b": [
        "Weather can really affect our mood, can't it? What's the weather like where you are? （天气确实会影响心情。你那里的天气怎么样？）",
        "That's a classic conversation starter! Do you prefer sunny days or rainy days? （这是经典的话题开头！你喜欢晴天还是雨天？）",
    ],
    # 感谢类
    r"\b(thank|thanks|appreciate)\b": [
        "You're very welcome! Is there anything else you'd like to practice? （不客气！还有什么想练习的吗？）",
        "My pleasure! Keep up the great work with your English! （我的荣幸！继续好好练习英语！）",
    ],
    # 道歉类
    r"\b(sorry|my\s+apologies|excuse\s+me)\b": [
        "No worries at all! What were you trying to say? （完全没关系！你想说什么来着？）",
        "That's okay! Don't worry about small mistakes — that's how we learn! （没关系！别担心小错误——我们就是这样学习的！）",
    ],
    # 告别类
    r"\b(goodbye|see\s+you|bye|have\s+a\s+(good|great|nice))\b": [
        "Goodbye! It was great chatting with you! Come back soon! （再见！和你聊天很开心！常回来！）",
        "See you later! Keep practicing your English every day! （回头见！每天坚持练习英语！）",
    ],
    # 疑问类
    r"\b(what|how|where|when|why|who|can\s+you|do\s+you)\b": [
        "That's a great question! Let me think about that... What do you think first? （好问题！让我想想……你第一反应是什么？）",
        "I like how you're thinking critically! Here's my take on that... （我喜欢你批判性思考！我是这么看的……）",
    ],
}

# 场景特定前缀回复（当用户输入与场景无关时提供引导）
_SCENARIO_GUIDES = {
    "interview": [
        "That's an interesting point! In an interview context, how would you express that professionally? （有意思的观点！在面试中，你会如何专业地表达这个观点？）",
        "Good observation! Now imagine you're in an interview — how would you phrase that answer? （好观察！想象你在面试中——你会怎么组织这个回答？）",
    ],
    "restaurant": [
        "Speaking of food, have you ever ordered something in English at a restaurant? （说到美食，你有没有在餐厅用英语点过餐？）",
        "That reminds me of dining out! Would you like to practice ordering food in English? （这让我想到外出就餐！想练习用英语点餐吗？）",
    ],
    "travel": [
        "Travel is such a great topic! Have you been anywhere interesting lately? （旅行是个好话题！你最近去过什么有趣的地方吗？）",
        "That's perfect for travel conversations! Where would you like to go? （这很适合旅行对话！你想去哪里？）",
    ],
    "meeting": [
        "In a business meeting, you might say: 'I agree with that point, and I'd like to add...' （商务会议中，你可以说：'我同意那个观点，我还想补充……'）",
        "That's a good point to bring up in a meeting! How would you introduce it formally? （这是在会议上提出的好观点！你会如何正式地引入它？）",
    ],
    "daily": [
        "That's a nice topic for everyday conversation! Tell me more about it. （这是日常对话的好话题！多告诉我一些。）",
        "Great! Everyday conversations are the best way to practice. What else is on your mind? （太好了！日常对话是最好的练习方式。你还有什么想法？）",
    ],
}


def _classify_input(text: str) -> tuple[str, list[str]]:
    """
    对用户输入进行分类，返回 (类别, 匹配到的关键词列表)。

    Returns:
        (category, matched_keywords)
    """
    if not text:
        return ("unknown", [])

    lower = text.lower()
    matched_keywords = []

    for pattern, _responses in _GENERIC_RESPONSES.items():
        if re.search(pattern, lower):
            matched_keywords.append(re.search(pattern, lower).group(0))

    return ("known" if matched_keywords else "unknown", matched_keywords)


def _pick_response(responses: list[str], turn: int, seed: str = "") -> str:
    """从响应列表中选择一个（考虑 turn 和 seed 以实现多样性）"""
    if not responses:
        return "That's interesting! Could you tell me more about that? （很有意思！能多告诉我一些吗？）"

    # 使用 turn + seed 的确定性哈希来选择，保证同一输入在不同轮次可能得到不同回复
    # 注意：不使用内置 hash()，因为 Python 3.3+ 默认随机化 PYTHONHASHSEED
    # 使用简单的字符串哈希算法保证跨平台一致
    h = 0
    for ch in f"{seed}{turn}":
        h = (h * 31 + ord(ch)) % (2**32)
    return responses[h % len(responses)]


def _smart_mock_reply(
    scenario: str,
    turn: int,
    user_input: str = "",
) -> str:
    """
    智能 mock 回复 — 根据用户输入内容生成上下文相关的回复。

    不再是纯静态轮次字典，而是：
    1. 分析用户输入的关键词和意图
    2. 选择匹配的回复模板
    3. 填充变量（如 greeting）
    4. 结合场景给出引导性回复

    Args:
        scenario: 场景标识
        turn: 当前轮次
        user_input: 用户实际输入

    Returns:
        上下文相关的 AI 回复
    """
    if not user_input:
        # 没有用户输入时使用原始预设逻辑
        return _fallback_static_reply(scenario, turn)

    category, keywords = _classify_input(user_input)

    if category == "known":
        # 找到匹配的回复模板
        for pattern, responses in _GENERIC_RESPONSES.items():
            if any(re.search(pattern, user_input.lower()) for _ in keywords):
                response_template = _pick_response(responses, turn, user_input)
                # 替换 {greeting} 等变量
                greeting = "there"
                lower = user_input.lower()
                if "morning" in lower:
                    greeting = "good morning"
                elif "afternoon" in lower:
                    greeting = "good afternoon"
                elif "evening" in lower:
                    greeting = "good evening"
                return response_template.format(greeting=greeting)

    # 未匹配到具体类别 → 使用场景引导 + 通用回应
    guide = _pick_response(_SCENARIO_GUIDES.get(scenario, _SCENARIO_GUIDES["daily"]), turn, user_input)

    # 如果用户说了很长的话，给予更具体的回应
    words = user_input.split()
    if len(words) >= 5:
        # 用户说了很多 → 鼓励 + 追问
        follow_up = _pick_response([
            "That's a detailed explanation! I appreciate you sharing that. "
            "Could you elaborate on one specific aspect? "
            f"（这是一个详细的解释！感谢你分享。你能详细说说某个具体方面吗？）",
            "Great job expressing yourself in English! "
            "Let me ask you a follow-up question to keep the conversation going. "
            f"（你用英语表达得很好！让我问一个后续问题来继续对话。）",
        ], turn, user_input)
        return f"{guide}\n\n{follow_up}"
    elif len(words) <= 2:
        # 用户说得很少 → 鼓励多说
        encourage = _pick_response([
            "Good start! Try to add a bit more detail. "
            "What made you think of that? "
            f"（很好的开始！试着多加一些细节。是什么让你想到这个的？）",
            "Nice! I'd love to hear more about that. "
            "Can you tell me why you think so? "
            f"（不错！我想多听听你的想法。能告诉我你为什么这么想吗？）",
        ], turn, user_input)
        return f"{guide}\n\n{encourage}"

    # 默认：场景引导 + 友好回应
    return guide


def _fallback_static_reply(scenario: str, turn: int) -> str:
    """
    当没有用户输入时的回退静态回复。
    使用原始预设字典作为兜底。
    """
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
