"""
单元测试 - LLM Client

验证 llm_client 的 JSON 解析、safe_llm_call 回退逻辑。
注意：不测试真实 LLM 调用（需要 API key），只测试 mock/fallback 路径。
"""

import json
import pytest
from unittest.mock import AsyncMock, patch


class TestCallLlmJson:
    """测试 call_llm_json 函数"""

    @pytest.mark.asyncio
    async def test_call_llm_json_parses_valid_json(self):
        """有效的 JSON 响应应被正确解析"""
        sample_json = {
            "scores": {"fluency": 7.5, "grammar": 8.0},
            "total": 7.8,
        }
        with patch("agents.llm_client.call_llm", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = json.dumps(sample_json)
            from agents.llm_client import call_llm_json
            result = await call_llm_json([{"role": "user", "content": "test"}])
            assert result == sample_json
            mock_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_llm_json_rejects_invalid_json(self):
        """无效的 JSON 应抛出 JSONDecodeError"""
        with patch("agents.llm_client.call_llm", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = "this is not json {{{"
            from agents.llm_client import call_llm_json
            with pytest.raises(json.JSONDecodeError):
                await call_llm_json([{"role": "user", "content": "test"}])

    @pytest.mark.asyncio
    async def test_call_llm_json_sets_response_format(self):
        """call_llm_json 应自动设置 response_format={'type': 'json_object'}"""
        with patch("agents.llm_client.call_llm", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = '{"test": 1}'
            from agents.llm_client import call_llm_json
            await call_llm_json([{"role": "user", "content": "test"}])
            # 检查是否传入了 response_format
            kwargs = mock_call.call_args
            assert kwargs[1]["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_call_llm_json_uses_lower_temperature(self):
        """JSON 模式应使用较低温度以获得稳定输出"""
        with patch("agents.llm_client.call_llm", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = '{"test": 1}'
            from agents.llm_client import call_llm_json
            await call_llm_json([{"role": "user", "content": "test"}])
            kwargs = mock_call.call_args
            assert kwargs[1]["temperature"] == 0.3

    @pytest.mark.asyncio
    async def test_call_llm_json_increases_max_tokens(self):
        """JSON 模式应使用更大的 max_tokens"""
        with patch("agents.llm_client.call_llm", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = '{"test": 1}'
            from agents.llm_client import call_llm_json
            await call_llm_json([{"role": "user", "content": "test"}])
            kwargs = mock_call.call_args
            assert kwargs[1]["max_tokens"] == 1024

    @pytest.mark.asyncio
    async def test_call_llm_json_custom_model(self):
        """支持自定义 model 参数"""
        with patch("agents.llm_client.call_llm", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = '{"test": 1}'
            from agents.llm_client import call_llm_json
            await call_llm_json(
                [{"role": "user", "content": "test"}],
                model="gpt-4",
            )
            kwargs = mock_call.call_args
            assert kwargs[1]["model"] == "gpt-4"


class TestSafeLlmCall:
    """测试 safe_llm_call 函数的 fallback 逻辑"""

    @pytest.mark.asyncio
    async def test_safe_llm_call_success(self):
        """LLM 调用成功时返回正常结果"""
        with patch("agents.llm_client.call_llm", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = "Hello world"
            from agents.llm_client import safe_llm_call
            result = await safe_llm_call([{"role": "user", "content": "hi"}])
            assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_safe_llm_call_fallback_provided(self):
        """LLM 失败时使用 fallback 函数"""
        with patch("agents.llm_client.call_llm", new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = Exception("API error")
            from agents.llm_client import safe_llm_call
            fallback = lambda: "Fallback reply"
            result = await safe_llm_call(
                [{"role": "user", "content": "hi"}],
                fallback_fn=fallback,
            )
            assert result == "Fallback reply"

    @pytest.mark.asyncio
    async def test_safe_llm_call_fallback_none_returns_empty(self):
        """LLM 失败且无 fallback 时返回空字符串"""
        with patch("agents.llm_client.call_llm", new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = Exception("API error")
            from agents.llm_client import safe_llm_call
            result = await safe_llm_call([{"role": "user", "content": "hi"}])
            assert result == ""

    @pytest.mark.asyncio
    async def test_safe_llm_call_passes_parameters(self):
        """safe_llm_call 应正确传递 temperature/max_tokens 等参数"""
        with patch("agents.llm_client.call_llm", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = "OK"
            from agents.llm_client import safe_llm_call
            await safe_llm_call(
                [{"role": "user", "content": "hi"}],
                temperature=0.9,
                max_tokens=256,
                response_format={"type": "json_object"},
                model="custom-model",
            )
            kwargs = mock_call.call_args
            assert kwargs[1]["temperature"] == 0.9
            assert kwargs[1]["max_tokens"] == 256
            assert kwargs[1]["response_format"] == {"type": "json_object"}
            assert kwargs[1]["model"] == "custom-model"


class TestCallLlm:
    """测试 call_llm 基础功能"""

    @pytest.mark.asyncio
    async def test_call_llm_adds_system_prefix(self):
        """call_llm 应在消息前自动添加系统提示"""
        from unittest.mock import MagicMock, AsyncMock

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("agents.llm_client.get_llm_client", return_value=mock_client):
            from agents.llm_client import call_llm
            await call_llm([{"role": "user", "content": "test"}])

            call_args = mock_client.chat.completions.create.call_args
            msgs = call_args[1]["messages"]
            assert msgs[0]["role"] == "system"
            assert "English language teaching assistant" in msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_call_llm_passes_user_messages(self):
        """用户消息应被包含在消息链中"""
        from unittest.mock import MagicMock, AsyncMock

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("agents.llm_client.get_llm_client", return_value=mock_client):
            from agents.llm_client import call_llm
            await call_llm([
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ])

            call_args = mock_client.chat.completions.create.call_args
            msgs = call_args[1]["messages"]
            assert len(msgs) == 3  # system + 2 user messages
            assert msgs[1]["role"] == "user"
            assert msgs[1]["content"] == "Hello"
