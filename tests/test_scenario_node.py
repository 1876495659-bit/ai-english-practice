"""
单元测试 - Scenario Node

验证 scenario_node 的场景初始化、开场白生成、metadata 更新。
"""

import pytest
from agents.scenario_node import scenario_node


class TestScenarioNode:
    """测试 scenario_node 函数"""

    def _make_state(self, **overrides):
        """构建测试用 state 字典"""
        base = {
            "scenario": "daily",
            "difficulty": "medium",
            "level": "intermediate",
            "turn": 0,
            "retry_count": 0,
            "max_retries": 3,
            "session_active": True,
            "messages": [],
            "correction": {},
            "score": {},
            "ai_reply": "",
            "metadata": {},
            "skill_progress": {
                "total_turns": 0,
                "avg_score": 0.0,
                "error_frequency": {},
                "weakest_dimension": "",
                "strongest_dimension": "",
                "improvement_trajectory": [],
            },
        }
        base.update(overrides)
        return base

    @pytest.mark.asyncio
    async def test_scenario_node_initializes_daily(self):
        """日常场景：turn=0 时生成开场白和 metadata"""
        result = await scenario_node(self._make_state(scenario="daily"))
        assert "messages" in result
        assert "metadata" in result
        assert "scenario_name" in result["metadata"]
        assert result["metadata"]["scenario_name"] == "日常对话"

    @pytest.mark.asyncio
    async def test_scenario_node_initializes_interview(self):
        """面试场景：正确设置场景名称和目标"""
        result = await scenario_node(self._make_state(scenario="interview"))
        assert result["metadata"]["scenario_name"] == "英语面试"
        assert "opening_line" in result["metadata"]
        assert len(result["messages"]) > 0

    @pytest.mark.asyncio
    async def test_scenario_node_initializes_restaurant(self):
        """餐厅场景：正确设置"""
        result = await scenario_node(self._make_state(scenario="restaurant"))
        assert result["metadata"]["scenario_name"] == "餐厅点餐"

    @pytest.mark.asyncio
    async def test_scenario_node_initializes_travel(self):
        """旅行场景：正确设置"""
        result = await scenario_node(self._make_state(scenario="travel"))
        assert result["metadata"]["scenario_name"] == "旅行出行"

    @pytest.mark.asyncio
    async def test_scenario_node_initializes_meeting(self):
        """会议场景：正确设置"""
        result = await scenario_node(self._make_state(scenario="meeting"))
        assert result["metadata"]["scenario_name"] == "商务会议"

    @pytest.mark.asyncio
    async def test_scenario_node_opening_line_in_messages(self):
        """开场白应作为 assistant 消息注入 messages"""
        result = await scenario_node(self._make_state(scenario="daily"))
        messages = result.get("messages", [])
        assert len(messages) >= 1
        assert messages[0]["role"] == "assistant"
        assert len(messages[0]["content"]) > 0

    @pytest.mark.asyncio
    async def test_scenario_node_metadata_contains_difficulty_description(self):
        """metadata 应包含难度描述"""
        result = await scenario_node(self._make_state(scenario="daily", difficulty="hard"))
        assert "difficulty_description" in result["metadata"]
        assert result["metadata"]["difficulty_description"] == "深度讨论"

    @pytest.mark.asyncio
    async def test_scenario_node_metadata_contains_focus(self):
        """metadata 应包含 focus 信息"""
        result = await scenario_node(self._make_state(scenario="interview", difficulty="easy"))
        assert "focus" in result["metadata"]
        assert "自我介绍" in result["metadata"]["focus"]

    @pytest.mark.asyncio
    async def test_scenario_node_sets_scenario_goal(self):
        """应设置 scenario_goal 字段"""
        result = await scenario_node(self._make_state(scenario="restaurant"))
        assert "scenario_goal" in result
        assert len(result["scenario_goal"]) > 0

    @pytest.mark.asyncio
    async def test_scenario_node_default_values(self):
        """不传 scenario/difficulty 时使用默认值"""
        result = await scenario_node(self._make_state())
        assert result["metadata"]["scenario_name"] == "日常对话"

    @pytest.mark.asyncio
    async def test_scenario_node_unknown_scenario_fallback(self):
        """未知场景回退到 daily"""
        result = await scenario_node(self._make_state(scenario="nonexistent"))
        assert result["metadata"]["scenario_name"] == "日常对话"

    @pytest.mark.asyncio
    async def test_scenario_node_different_difficulties(self):
        """不同难度应有不同的 difficulty_description"""
        r_easy = await scenario_node(self._make_state(scenario="daily", difficulty="easy"))
        r_medium = await scenario_node(self._make_state(scenario="daily", difficulty="medium"))
        r_hard = await scenario_node(self._make_state(scenario="daily", difficulty="hard"))
        assert r_easy["metadata"]["difficulty_description"] != r_hard["metadata"]["difficulty_description"]

    @pytest.mark.asyncio
    async def test_scenario_node_no_turn_zero_skip(self):
        """turn != 0 时不应注入开场白（仅返回 metadata）"""
        state = self._make_state(scenario="daily", turn=5)
        result = await scenario_node(state)
        # turn != 0 时不应覆盖 messages
        assert "messages" not in result or len(result.get("messages", [])) == 0
