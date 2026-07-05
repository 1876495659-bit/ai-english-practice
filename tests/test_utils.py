"""
单元测试 - 公共工具函数（extract_latest_user_input）

验证 agents/utils.py 中的消息提取函数兼容 dict 和 BaseMessage 两种格式。
"""

import pytest
from agents.utils import extract_latest_user_input


class TestExtractLatestUserInput:
    """测试 extract_latest_user_input 函数"""

    def test_dict_format_user_message(self):
        """dict 格式：正常提取 user 消息"""
        messages = [
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "I like pizza"},
        ]
        assert extract_latest_user_input(messages) == "I like pizza"

    def test_dict_format_multiple_users(self):
        """dict 格式：提取最新的一条 user 消息"""
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]
        assert extract_latest_user_input(messages) == "second"

    def test_dict_format_no_user(self):
        """dict 格式：没有 user 消息时返回空字符串"""
        messages = [
            {"role": "assistant", "content": "Hello!"},
        ]
        assert extract_latest_user_input(messages) == ""

    def test_dict_format_empty_list(self):
        """dict 格式：空列表返回空字符串"""
        assert extract_latest_user_input([]) == ""

    def test_dict_format_whitespace_only(self):
        """dict 格式：空白内容会被 strip 后返回空字符串"""
        messages = [
            {"role": "user", "content": "   "},
        ]
        assert extract_latest_user_input(messages) == ""

    def test_dict_format_with_leading_trailing_spaces(self):
        """dict 格式：自动 strip 首尾空格"""
        messages = [
            {"role": "user", "content": "  hello world  "},
        ]
        assert extract_latest_user_input(messages) == "hello world"

    def test_base_message_format_human_role(self):
        """BaseMessage 格式：.type='human' 的角色"""
        msg = pytest.importorskip("langchain_core.messages", reason="langchain_core not installed")
        human_msg = msg.HumanMessage(content="Speak English with me")
        messages = [
            msg.AIMessage(content="Sure!"),
            human_msg,
        ]
        assert extract_latest_user_input(messages) == "Speak English with me"

    def test_base_message_format_mixed(self):
        """混合格式：dict 和 BaseMessage 混用时能正确识别"""
        msg = pytest.importorskip("langchain_core.messages", reason="langchain_core not installed")
        human_msg = msg.HumanMessage(content="Bonjour!")
        messages = [
            {"role": "assistant", "content": "Hi there!"},
            human_msg,
        ]
        assert extract_latest_user_input(messages) == "Bonjour!"

    def test_base_message_format_no_human(self):
        """BaseMessage 格式：没有 human 消息时返回空字符串"""
        msg = pytest.importorskip("langchain_core.messages", reason="langchain_core not installed")
        ai_msg = msg.AIMessage(content="How can I help?")
        messages = [ai_msg]
        assert extract_latest_user_input(messages) == ""

    def test_prefer_latest_over_earlier(self):
        """验证总是取最新的 user/human 消息，忽略更早的"""
        messages = [
            {"role": "user", "content": "old message"},
            {"role": "assistant", "content": "intermediate"},
            {"role": "user", "content": "latest message"},
        ]
        assert extract_latest_user_input(messages) == "latest message"
