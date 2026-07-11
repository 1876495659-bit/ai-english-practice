"""
单元测试 - Conversation Node

验证 conversation_node 的对话历史构建、mock reply 逻辑。
"""

import pytest
from agents.conversation_node import (
    _build_conversation_history,
    _mock_reply,
)


class TestBuildConversationHistory:
    """测试 _build_conversation_history 函数"""

    def test_empty_messages(self):
        """空消息列表返回空历史"""
        assert _build_conversation_history([]) == []

    def test_dict_format_extract_user_and_ai(self):
        """dict 格式：正确提取 user/assistant 消息"""
        messages = [
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "Hi there"},
        ]
        result = _build_conversation_history(messages)
        assert len(result) == 2
        assert result[0] == {"role": "assistant", "content": "Hello!"}
        assert result[1] == {"role": "user", "content": "Hi there"}

    def test_dict_format_filters_system_messages(self):
        """dict 格式：过滤 system 角色"""
        messages = [
            {"role": "system", "content": "You are a tutor"},
            {"role": "user", "content": "Hello"},
        ]
        result = _build_conversation_history(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_dict_format_filters_empty_content(self):
        """dict 格式：过滤空内容消息"""
        messages = [
            {"role": "user", "content": ""},
            {"role": "user", "content": "valid message"},
        ]
        result = _build_conversation_history(messages)
        assert len(result) == 1
        assert result[0]["content"] == "valid message"

    def test_human_ai_role_mapping(self):
        """LangGraph 格式：human/ai 映射到 user/assistant"""
        messages = [
            {"role": "human", "content": "test input"},
            {"role": "ai", "content": "test output"},
        ]
        result = _build_conversation_history(messages)
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

    def test_truncate_to_12_messages(self):
        """超过 12 条消息时只取最近 12 条（6 轮）"""
        messages = [{"role": "user", "content": f"msg_{i}"} for i in range(20)]
        result = _build_conversation_history(messages)
        assert len(result) == 12
        # 应该保留最后 12 条
        assert result[0]["content"] == "msg_8"
        assert result[-1]["content"] == "msg_19"

    def test_recent_context_preserved(self):
        """消息少于 12 条时全部保留"""
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "third"},
        ]
        result = _build_conversation_history(messages)
        assert len(result) == 3


class TestMockReply:
    """测试 _mock_reply 函数"""

    def test_interview_turn_1(self):
        """面试场景第 1 轮回复"""
        result = _mock_reply("interview", 1)
        assert "Good morning" in result
        assert "yourself" in result.lower() or "自己" in result

    def test_interview_turn_2(self):
        """面试场景第 2 轮回复"""
        result = _mock_reply("interview", 2)
        assert "impressive" in result or "自豪" in result

    def test_restaurant_turn_1(self):
        """餐厅场景第 1 轮回复"""
        result = _mock_reply("restaurant", 1)
        assert "Welcome" in result or "欢迎来到" in result

    def test_travel_turn_1(self):
        """旅行场景第 1 轮回复"""
        result = _mock_reply("travel", 1)
        assert "visiting" in result.lower() or "旅游" in result

    def test_meeting_turn_1(self):
        """会议场景第 1 轮回复"""
        result = _mock_reply("meeting", 1)
        assert "morning" in result.lower() or "开始" in result

    def test_daily_turn_1(self):
        """日常场景第 1 轮回复"""
        result = _mock_reply("daily", 1)
        assert "day" in result.lower() or "今天" in result

    def test_unknown_scenario_fallback_to_daily(self):
        """未知场景回退到 daily"""
        result = _mock_reply("nonexistent", 1)
        assert result is not None
        assert len(result) > 0

    def test_turn_exceeds_max_index(self):
        """turn 超出最大索引时使用最后一个回复"""
        result = _mock_reply("daily", 100)
        assert result is not None
        assert len(result) > 0

    def test_all_scenarios_have_replies(self):
        """所有场景都有对应的 mock 回复"""
        scenarios = ["interview", "restaurant", "travel", "meeting", "daily"]
        for scenario in scenarios:
            for turn in range(1, 4):
                result = _mock_reply(scenario, turn)
                assert isinstance(result, str)
                assert len(result) > 0
