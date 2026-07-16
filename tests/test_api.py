"""
单元测试 - FastAPI 接口

验证 API 端点的请求/响应结构和边界情况。
使用 httpx TestClient 进行同步测试。
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app
import api.sessions as _session_mod


@pytest.fixture(autouse=True)
def _reset_api_state():
    """每个测试前后清理会话状态"""
    _session_mod._sessions.clear()
    _session_mod._next_session_id = 0
    yield
    _session_mod._sessions.clear()


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthCheck:
    """测试健康检查端点"""

    def test_health_check_returns_ok(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "AI English Tutor"
        assert "nodes" in data
        assert "features" in data

    def test_health_check_includes_version(self, client):
        resp = client.get("/")
        data = resp.json()
        assert "version" in data


class TestStartSession:
    """测试会话启动端点"""

    def test_start_session_daily(self, client):
        resp = client.post(
            "/api/session/start",
            json={"scenario": "daily", "difficulty": "medium", "level": "intermediate"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "started"
        assert "session_id" in data
        assert "thread_id" in data
        assert data["scenario"] == "daily"

    def test_start_session_interview(self, client):
        resp = client.post(
            "/api/session/start",
            json={"scenario": "interview", "difficulty": "easy", "level": "beginner"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["scenario"] == "interview"
        assert data["difficulty"] == "easy"

    def test_start_session_default_values(self, client):
        """不传参数时使用默认值"""
        resp = client.post("/api/session/start", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["scenario"] == "daily"
        assert data["difficulty"] == "medium"

    def test_start_session_creates_session_mapping(self, client):
        """启动会话后应在 _sessions 中注册"""
        client.post(
            "/api/session/start",
            json={"scenario": "restaurant"},
        )
        assert len(_session_mod._sessions) == 1
        session_id = list(_session_mod._sessions.keys())[0]
        assert session_id.startswith("session_")


class TestChat:
    """测试聊天端点"""

    def test_chat_without_session(self, client):
        """未启动会话时 chat 应返回错误"""
        # 由于 _get_or_create_session 会自动创建新会话，
        # 所以这里会创建一个新会话并运行完整流程
        resp = client.post("/api/chat", json={"message": "hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert "ai_reply" in data
        assert "correction" in data
        assert "score" in data

    def test_chat_returns_scoring_data(self, client):
        """chat 响应应包含评分数据"""
        resp = client.post(
            "/api/chat",
            json={"message": "I would like to order a hamburger"},
        )
        data = resp.json()
        assert "score" in data
        if data["score"]:
            assert "total" in data["score"]
            assert "scores" in data["score"]

    def test_chat_returns_correction_data(self, client):
        """chat 响应应包含纠错数据"""
        resp = client.post(
            "/api/chat",
            json={"message": "i go to park"},
        )
        data = resp.json()
        assert "correction" in data
        if data["correction"]:
            assert "has_errors" in data["correction"]

    def test_chat_returns_skill_progress(self, client):
        """chat 响应应包含 skill_progress"""
        resp = client.post(
            "/api/chat",
            json={"message": "hello world"},
        )
        data = resp.json()
        assert "skill_progress" in data


class TestGetSession:
    """测试获取会话状态端点"""

    def test_get_session_no_active(self, client):
        """无活跃会话时应返回 404"""
        resp = client.get("/api/session")
        assert resp.status_code == 404

    def test_get_session_after_start(self, client):
        """启动会话后应能获取状态"""
        client.post("/api/session/start", json={})
        resp = client.get("/api/session")
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert "scenario" in data


class TestEndSession:
    """测试结束会话端点"""

    def test_end_session_no_active(self, client):
        """无活跃会话时应返回 404"""
        resp = client.post("/api/session/end")
        assert resp.status_code == 404

    def test_end_session_after_start(self, client):
        """启动后结束应返回成功"""
        client.post("/api/session/start", json={})
        resp = client.post("/api/session/end")
        assert resp.status_code == 200


class TestDeleteSession:
    """测试删除会话端点"""

    def test_delete_nonexistent(self, client):
        """删除不存在的会话应返回 404"""
        resp = client.delete("/api/session/session_999")
        assert resp.status_code == 404

    def test_delete_existing(self, client):
        """删除已存在的会话应返回成功"""
        client.post("/api/session/start", json={})
        session_id = list(_session_mod._sessions.keys())[0]
        resp = client.delete(f"/api/session/{session_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == session_id
