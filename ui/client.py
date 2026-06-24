"""
API Client - 前端与后端 FastAPI 的通信层

封装所有后端 API 调用，提供类型安全的接口。
"""

from __future__ import annotations

import httpx
from typing import Any, Optional


class APIClient:
    """AI English Tutor FastAPI 客户端"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    def _get(self, path: str) -> dict[str, Any]:
        with httpx.Client(timeout=30) as client:
            resp = client.get(f"{self.base_url}{path}")
            resp.raise_for_status()
            return resp.json()

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                f"{self.base_url}{path}",
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()

    def health_check(self) -> dict[str, Any]:
        """健康检查"""
        return self._get("/")

    def start_session(
        self,
        scenario: str = "daily",
        difficulty: str = "medium",
        level: str = "intermediate",
    ) -> dict[str, Any]:
        """
        开始新会话

        Returns:
            {
                "session_id": "session_1",
                "thread_id": "thread_session_1",
                "scenario": "daily",
                "scenario_name": "日常对话",
                "difficulty": "medium",
                "opening_line": "...",
                "has_checkpoint": True,
                ...
            }
        """
        return self._post("/api/session/start", {
            "scenario": scenario,
            "difficulty": difficulty,
            "level": level,
        })

    def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        发送消息

        Args:
            message: 用户输入的英语
            session_id: 可选，用于关联特定会话

        Returns:
            {
                "session_id": "session_1",
                "scenario": "daily",
                "scenario_name": "日常对话",
                "difficulty": "medium",
                "turn": 1,
                "ai_reply": "...",
                "user_input": "...",
                "correction": {...},
                "score": {...},
                "skill_progress": {...},
                "retry_count": 0,
                "messages": [...],
                "has_checkpoint": True,
            }
        """
        return self._post("/api/chat", {"message": message})

    def get_session(self) -> dict[str, Any]:
        """获取当前会话状态"""
        return self._get("/api/session")

    def delete_session(self, session_id: str) -> dict[str, Any]:
        """删除会话"""
        return self._post(f"/api/session/{session_id}", {})
