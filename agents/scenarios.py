"""
场景配置数据 - 所有练习场景的 JSON 驱动配置

此模块是纯数据层，不包含任何业务逻辑。
新增场景只需在此添加配置字典即可。
"""

from typing import Any

SCENARIOS: dict[str, dict[str, Any]] = {
    "interview": {
        "id": "interview",
        "name": "英语面试",
        "description": "模拟英文面试场景，练习自我介绍、常见问题回答",
        "goal": "成功完成一轮英文面试，展示个人能力和经验",
        "roles": ["面试官", "求职者"],
        "opening_lines": [
            "Good morning! Thank you for coming today. Could you start by telling me a little about yourself?",
            "Hi, welcome! Let's begin. Can you introduce yourself and share your background?",
            "Hello! Thanks for joining us. To get started, please tell me about yourself.",
        ],
        "difficulty_levels": {
            "easy": {
                "description": "基础面试问题",
                "focus": "自我介绍、教育背景、基本工作经验",
                "vocabulary_level": "basic",
                "prompt_suffix": "使用简单直接的英语，句子不超过20个单词。",
            },
            "medium": {
                "description": "标准面试",
                "focus": "行为面试题、项目经验、优缺点分析",
                "vocabulary_level": "intermediate",
                "prompt_suffix": "使用中等难度的英语，适当引入专业术语。",
            },
            "hard": {
                "description": "高阶面试",
                "focus": "压力面试、情景分析、战略思维",
                "vocabulary_level": "advanced",
                "prompt_suffix": "使用高级英语，包含复杂句型和行业术语。",
            },
        },
    },
    "restaurant": {
        "id": "restaurant",
        "name": "餐厅点餐",
        "description": "模拟在外国餐厅点餐的场景",
        "goal": "能够独立完成点餐、询问菜品、处理问题的全流程",
        "roles": ["服务员", "顾客"],
        "opening_lines": [
            "Good evening! Welcome to our restaurant. Table for how many?",
            "Hi there! Do you have a reservation, or would you like to walk in?",
            "Welcome! Right this way. Here's the menu. Can I start you off with something to drink?",
        ],
        "difficulty_levels": {
            "easy": {
                "description": "简单点餐",
                "focus": "基础菜品名称、数量表达、简单付款",
                "vocabulary_level": "basic",
                "prompt_suffix": "使用简单日常用语，避免复杂句型。",
            },
            "medium": {
                "description": "完整用餐流程",
                "focus": "询问推荐菜、特殊饮食需求、抱怨处理",
                "vocabulary_level": "intermediate",
                "prompt_suffix": "使用自然口语，可适当加入礼貌用语。",
            },
            "hard": {
                "description": "复杂场景",
                "focus": "处理投诉、退菜、特殊饮食要求、小费文化",
                "vocabulary_level": "advanced",
                "prompt_suffix": "使用地道英语，包含习语和文化相关的内容。",
            },
        },
    },
    "travel": {
        "id": "travel",
        "name": "旅行出行",
        "description": "模拟在国外旅行中遇到的各种场景",
        "goal": "能够在旅行中独立解决问路、住宿、交通等问题",
        "roles": ["当地人/工作人员", "旅行者"],
        "opening_lines": [
            "Hi! Are you visiting our city? Can I help you find somewhere?",
            "Welcome! It's a beautiful day for sightseeing. Where are you headed?",
            "Hello! Looks like you're exploring. Need any directions or recommendations?",
        ],
        "difficulty_levels": {
            "easy": {
                "description": "基础问路与交通",
                "focus": "问路、打车、买票",
                "vocabulary_level": "basic",
                "prompt_suffix": "使用简单清晰的英语，短句为主。",
            },
            "medium": {
                "description": "住宿与游玩",
                "focus": "酒店入住、景点介绍、活动安排",
                "vocabulary_level": "intermediate",
                "prompt_suffix": "使用自然口语，适当描述细节。",
            },
            "hard": {
                "description": "复杂旅行场景",
                "focus": "航班延误、行李丢失、紧急求助",
                "vocabulary_level": "advanced",
                "prompt_suffix": "使用高级英语，包含正式场合用语。",
            },
        },
    },
    "meeting": {
        "id": "meeting",
        "name": "商务会议",
        "description": "模拟国际商务会议场景",
        "goal": "能够参与英文商务会议，发表观点，进行讨论",
        "roles": ["会议主持人/参会者"],
        "opening_lines": [
            "Good morning everyone. Let's get started. Does anyone have updates on the Q3 project?",
            "Welcome to the team meeting. Today we'll discuss the new product roadmap.",
            "Hi all, thanks for joining. Let's go around the table and share our progress.",
        ],
        "difficulty_levels": {
            "easy": {
                "description": "简单会议交流",
                "focus": "打招呼、简单进度汇报",
                "vocabulary_level": "basic",
                "prompt_suffix": "使用简洁的商务英语，避免复杂从句。",
            },
            "medium": {
                "description": "常规会议",
                "focus": "议题讨论、提出建议、表达同意/反对",
                "vocabulary_level": "intermediate",
                "prompt_suffix": "使用标准商务英语，包含会议常用表达。",
            },
            "hard": {
                "description": "高级谈判",
                "focus": "商务谈判、危机处理、战略决策",
                "vocabulary_level": "advanced",
                "prompt_suffix": "使用高级商务英语，包含正式谈判用语。",
            },
        },
    },
    "daily": {
        "id": "daily",
        "name": "日常对话",
        "description": "自由日常对话，无特定场景限制",
        "goal": "提升日常英语交流的流利度和自然度",
        "roles": ["朋友/对话伙伴"],
        "opening_lines": [
            "Hey! How's your day going? Anything interesting happening?",
            "Hi there! What have you been up to lately?",
            "Hello! Good to see you. Want to chat about something fun?",
        ],
        "difficulty_levels": {
            "easy": {
                "description": "基础闲聊",
                "focus": "天气、爱好、日常生活",
                "vocabulary_level": "basic",
                "prompt_suffix": "使用最基础的日常英语，像和朋友聊天一样自然。",
            },
            "medium": {
                "description": "深入交流",
                "focus": "观点分享、经历讲述、情感表达",
                "vocabulary_level": "intermediate",
                "prompt_suffix": "使用自然的日常英语，可以适当表达个人观点。",
            },
            "hard": {
                "description": "深度讨论",
                "focus": "社会话题、文化差异、抽象概念",
                "vocabulary_level": "advanced",
                "prompt_suffix": "使用高级日常英语，涉及更深层次的讨论。",
            },
        },
    },
}


def get_scenario_config(scenario_id: str) -> dict[str, Any]:
    """
    获取场景配置（带默认回退）

    Args:
        scenario_id: 场景标识

    Returns:
        场景配置字典
    """
    return SCENARIOS.get(scenario_id, SCENARIOS["daily"])


def get_difficulty_config(scenario_id: str, difficulty: str) -> dict[str, Any]:
    """
    获取场景+难度的配置

    Args:
        scenario_id: 场景标识
        difficulty: 难度等级

    Returns:
        难度配置字典
    """
    scenario_config = get_scenario_config(scenario_id)
    return scenario_config["difficulty_levels"].get(
        difficulty, scenario_config["difficulty_levels"]["medium"]
    )


def list_available_scenarios() -> list[str]:
    """
    列出所有可用场景标识

    Returns:
        场景标识列表
    """
    return list(SCENARIOS.keys())
