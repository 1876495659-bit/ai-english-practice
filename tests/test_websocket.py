"""
单元测试 — WebSocket 实时对话功能

验证：
1. api/sessions.py — 会话管理模块
2. api/websocket.py — WebSocket 处理器
3. ui/client.py — WSClient 客户端类
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ============================================================================
# 测试辅助函数
# ============================================================================


def _sync_run(coro):
    """同步调用协程"""
    return asyncio.run(coro)


# ============================================================================
# 测试 api/sessions.py
# ============================================================================


class TestSessionManagement:
    """测试会话管理公共模块"""

    def setup_method(self):
        """每个测试前重置会话状态"""
        # 通过导入模块并重置全局变量
        import sys
        from api import sessions
        # 清除已有会话
        sessions._sessions.clear()
        sessions._next_session_id = 0

    def test_get_or_create_session_creates_new(self):
        """创建新会话应返回 session_id 和 thread_id"""
        from api.sessions import get_or_create_session
        sid, tid = get_or_create_session()
        assert sid.startswith("session_")
        assert tid.startswith("thread_")
        assert "session_1" in sid or sid == "session_1"

    def test_get_or_create_session_reuses_active(self):
        """活跃会话应被复用"""
        from api.sessions import get_or_create_session
        sid1, tid1 = get_or_create_session()
        sid2, tid2 = get_or_create_session()
        assert sid1 == sid2
        assert tid1 == tid2

    def test_make_initial_state_structure(self):
        """初始状态应包含所有必要字段"""
        from api.sessions import make_initial_state
        state = make_initial_state()
        assert state["scenario"] == "daily"
        assert state["difficulty"] == "medium"
        assert state["level"] == "intermediate"
        assert state["turn"] == 0
        assert state["messages"] == []
        assert state["skill_progress"]["total_turns"] == 0

    def test_remove_session(self):
        """删除会话应返回 thread_id"""
        from api.sessions import get_or_create_session, remove_session
        sid, tid = get_or_create_session()
        removed_tid = remove_session(sid)
        assert removed_tid == tid
        assert sid not in [k for k in remove_session.__globals__.get("_sessions", {}).keys()]

    def test_list_sessions_empty(self):
        """无活跃会话时应返回空字典"""
        from api.sessions import list_sessions
        assert list_sessions() == {}

    def test_list_sessions_after_create(self):
        """创建会话后应列出"""
        from api.sessions import get_or_create_session, list_sessions
        get_or_create_session()
        sessions_dict = list_sessions()
        assert len(sessions_dict) == 1

    def test_get_active_session_ids(self):
        """获取活跃会话 ID 列表"""
        from api.sessions import get_active_session_ids, get_or_create_session
        assert get_active_session_ids() == []
        get_or_create_session()
        ids = get_active_session_ids()
        assert len(ids) == 1

    def test_multiple_sessions_not_possible(self):
        """get_or_create_session 应复用同一会话"""
        from api.sessions import get_or_create_session
        s1, _ = get_or_create_session()
        s2, _ = get_or_create_session()
        s3, _ = get_or_create_session()
        assert s1 == s2 == s3


# ============================================================================
# 测试 api/websocket.py
# ============================================================================


class TestWebSocketHandler:
    """测试 WebSocket 处理器"""

    @pytest.mark.asyncio
    async def test_handle_empty_message(self):
        """空消息应返回错误事件"""
        from api.websocket import handle_chat_websocket

        mock_ws = AsyncMock()
        mock_ws.send_json = AsyncMock()

        await handle_chat_websocket(mock_ws, {"type": "chat", "message": ""})

        # 应收到错误事件
        calls = [c[0][0] for c in mock_ws.send_json.call_args_list]
        error_events = [c for c in calls if c.get("type") == "error"]
        assert len(error_events) > 0
        assert "消息不能为空" in error_events[0].get("detail", "")

    @pytest.mark.asyncio
    async def test_handle_invalid_payload(self):
        """缺少 message 字段应返回错误"""
        from api.websocket import handle_chat_websocket

        mock_ws = AsyncMock()
        mock_ws.send_json = AsyncMock()

        await handle_chat_websocket(mock_ws, {"type": "chat"})

        calls = [c[0][0] for c in mock_ws.send_json.call_args_list]
        error_events = [c for c in calls if c.get("type") == "error"]
        assert len(error_events) > 0

    @pytest.mark.asyncio
    async def test_event_sequence_structure(self):
        """正常流程应推送 node_complete → correction → score → chat_complete"""
        from agents.graph_builder import reset_checkpointer
        reset_checkpointer()

        # 用 mock 的 graph.nodes 来验证事件序列
        with patch("api.websocket.get_graph") as mock_get_graph, \
             patch("api.websocket.get_or_create_session") as mock_get_session:

            mock_get_session.return_value = ("session_test", "thread_test")

            # 模拟图
            mock_graph = MagicMock()
            mock_graph.nodes = {
                "conversation": MagicMock(),
                "correction": MagicMock(),
                "scoring": MagicMock(),
            }

            def mock_invoke(state, config):
                # 模拟 conversation 节点
                if "conversation" in str(config):
                    pass
                return state

            mock_graph.nodes["conversation"].ainvoke = AsyncMock(
                side_effect=lambda state, **kw: {
                    **state,
                    "ai_reply": "That sounds great! Tell me more.",
                    "messages": state.get("messages", []) + [{"role": "assistant", "content": "That sounds great!"}],
                }
            )
            mock_graph.nodes["correction"].ainvoke = AsyncMock(
                side_effect=lambda state, **kw: {
                    **state,
                    "correction": {
                        "original": state.get("messages", [])[-1]["content"],
                        "has_errors": False,
                        "errors": [],
                        "corrected": "",
                        "explanation": "无明显语法错误",
                    },
                }
            )
            mock_graph.nodes["scoring"].ainvoke = AsyncMock(
                side_effect=lambda state, **kw: {
                    **state,
                    "score": {
                        "scores": {"fluency": 7.5, "grammar": 8.0, "vocabulary": 7.0, "naturalness": 7.5},
                        "total": 7.5,
                        "feedback_en": "Good job!",
                        "feedback_zh": "做得好！",
                        "strengths": ["表达流畅"],
                        "improvements": ["可以尝试更多样化的词汇"],
                    },
                    "skill_progress": {
                        "total_turns": 1,
                        "avg_score": 7.5,
                        "error_frequency": {},
                        "weakest_dimension": "",
                        "strongest_dimension": "",
                        "improvement_trajectory": [7.5],
                    },
                    "retry_count": 0,
                }
            )
            mock_graph.ainvoke = AsyncMock(return_value={})
            mock_graph.aget_state = AsyncMock(return_value=None)
            mock_graph.checkpointer = None

            mock_get_graph.return_value = mock_graph

            from api.websocket import handle_chat_websocket

            mock_ws = AsyncMock()
            mock_ws.send_json = AsyncMock()

            await handle_chat_websocket(mock_ws, {"type": "chat", "message": "Hello, how are you?"})

            # 检查推送的事件类型
            calls = [c[0][0] for c in mock_ws.send_json.call_args_list]
            event_types = [c.get("type") for c in calls]

            assert "node_complete" in event_types
            assert "correction" in event_types
            assert "score" in event_types
            assert "chat_complete" in event_types

    @pytest.mark.asyncio
    async def test_retry_command_handling(self):
        """Scoring 返回 Command 时应触发 retry 事件"""
        from langgraph.types import Command
        from agents.graph_builder import reset_checkpointer
        reset_checkpointer()

        with patch("api.websocket.get_graph") as mock_get_graph, \
             patch("api.websocket.get_or_create_session") as mock_get_session:

            mock_get_session.return_value = ("session_retry", "thread_retry")

            mock_graph = MagicMock()
            mock_graph.nodes = {
                "conversation": MagicMock(),
                "correction": MagicMock(),
                "scoring": MagicMock(),
            }

            call_count = {"conv": 0, "corr": 0, "score": 0}

            async def mock_conv_invoke(state, **kw):
                call_count["conv"] += 1
                return {
                    **state,
                    "ai_reply": "Try again! You can do it.",
                    "messages": state.get("messages", []) + [{"role": "assistant", "content": "Try again!"}],
                }

            async def mock_corr_invoke(state, **kw):
                call_count["corr"] += 1
                return {
                    **state,
                    "correction": {"original": "", "has_errors": True, "errors": [], "explanation": "请重试"},
                }

            async def mock_score_invoke(state, **kw):
                call_count["score"] += 1
                if call_count["score"] <= 1:
                    # 第一次返回 Command（低分）
                    from langgraph.types import Command
                    return Command(
                        update={"retry_count": 1},
                        goto="conversation",
                    )
                # 第二次返回正常结果
                return {
                    **state,
                    "score": {"scores": {"fluency": 6.0, "grammar": 5.5, "vocabulary": 5.0, "naturalness": 5.5}, "total": 5.5},
                    "skill_progress": {"total_turns": 2, "avg_score": 5.5, "error_frequency": {}, "weakest_dimension": "vocabulary", "strongest_dimension": "", "improvement_trajectory": [5.5]},
                    "retry_count": 1,
                }

            mock_graph.nodes["conversation"].ainvoke = mock_conv_invoke
            mock_graph.nodes["correction"].ainvoke = mock_corr_invoke
            mock_graph.nodes["scoring"].ainvoke = mock_score_invoke
            mock_graph.ainvoke = AsyncMock(return_value={})
            mock_graph.aget_state = AsyncMock(return_value=None)
            mock_graph.checkpointer = None

            mock_get_graph.return_value = mock_graph

            from api.websocket import handle_chat_websocket

            mock_ws = AsyncMock()
            mock_ws.send_json = AsyncMock()

            await handle_chat_websocket(mock_ws, {"type": "chat", "message": "bad english input"})

            calls = [c[0][0] for c in mock_ws.send_json.call_args_list]
            event_types = [c.get("type") for c in calls]

            assert "retry" in event_types
            # Conversation 应被调用多次（初始 + retry）
            assert call_count["conv"] >= 2


# ============================================================================
# 测试 ui/client.py — WSClient
# ============================================================================


class TestWSClient:
    """测试 WebSocket 客户端类"""

    def test_url_construction_http(self):
        """HTTP URL 应转换为 WS URL"""
        from ui.client import WSClient
        client = WSClient("http://localhost:8000")
        assert client.ws_url == "ws://localhost:8000/ws/chat"

    def test_url_construction_https(self):
        """HTTPS URL 应转换为 WSS URL"""
        from ui.client import WSClient
        client = WSClient("https://example.com/api")
        assert client.ws_url == "wss://example.com/api/ws/chat"

    def test_url_trailing_slash_removed(self):
        """URL 尾部的斜杠应被移除"""
        from ui.client import WSClient
        client = WSClient("http://localhost:8000/")
        assert client.ws_url == "ws://localhost:8000/ws/chat"

    def test_is_connected_default(self):
        """未连接时 is_connected 应为 False"""
        from ui.client import WSClient
        client = WSClient()
        assert client.is_connected is False

    @pytest.mark.asyncio
    async def test_send_before_connect_raises(self):
        """未连接时发送消息应抛出 RuntimeError"""
        from ui.client import WSClient
        client = WSClient()
        with pytest.raises(RuntimeError, match="Not connected"):
            await client.send_message("test")

    @pytest.mark.asyncio
    async def test_close_without_connect(self):
        """未连接时关闭不应报错"""
        from ui.client import WSClient
        client = WSClient()
        await client.close()
        assert client.is_connected is False

    @pytest.mark.asyncio
    async def test_import_error_websockets(self):
        """未安装 websockets 包时应抛出 ImportError"""
        from ui.client import WSClient
        client = WSClient()
        with patch.dict("sys.modules", {"websockets": None}):
            # 重新导入以清除缓存
            import importlib
            import ui.client as client_module
            importlib.reload(client_module)
            ws_client = client_module.WSClient()
            with pytest.raises(ImportError, match="websockets"):
                await ws_client.connect()


# ============================================================================
# 测试集成：REST API 与 WebSocket 共存
# ============================================================================


class TestAPICompatibility:
    """测试 REST API 与 WebSocket 端点共存"""

    def test_rest_api_still_importable(self):
        """REST API 模块应可正常导入"""
        from api.main import app
        assert app is not None
        assert app.title == "AI English Tutor"

    def test_websocket_route_exists(self):
        """WebSocket 路由应已注册"""
        from api.main import app
        routes = [route.path for route in app.routes]
        assert "/ws/chat" in routes

    def test_health_check_endpoint(self):
        """健康检查端点应正常工作"""
        from api.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "websocket" not in data["architecture"].lower() or "realtime" in data["features"]

    def test_session_start_endpoint(self):
        """会话启动端点应正常工作"""
        from api.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        resp = client.post("/api/session/start", json={
            "scenario": "daily",
            "difficulty": "medium",
            "level": "intermediate",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "started"
        assert "session_id" in data
        assert "opening_line" in data

    def test_chat_endpoint(self):
        """聊天端点应正常工作"""
        from api.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        resp = client.post("/api/chat", json={"message": "Hello, how are you?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "ai_reply" in data
        assert "correction" in data
        assert "score" in data
        assert "session_id" in data
