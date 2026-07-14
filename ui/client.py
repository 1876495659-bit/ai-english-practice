"""
API Client - 前端与后端 FastAPI 的通信层

封装所有后端 API 调用，提供类型安全的接口。
支持：聊天、会话管理、ASR 语音转文本、TTS 文本转语音。
"""

from __future__ import annotations

import base64
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

    # ========================================================================
    # ASR — 语音转文本
    # ========================================================================

    def transcribe(self, audio_data: bytes, language: str = "en") -> dict[str, Any]:
        """
        上传音频数据，返回识别出的文本。

        Args:
            audio_data: 音频字节数据（WebM/WAV/MP3）
            language: 语言代码，默认 "en"

        Returns:
            {"status": "success", "text": "...", "language": "en", "audio_size": 1234}
        """
        with httpx.Client(timeout=30) as client:
            files = {"file": ("audio.webm", audio_data, "audio/webm")}
            data = {"language": language}
            resp = client.post(
                f"{self.base_url}/api/asr/transcribe",
                files=files,
                data=data,
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()

    # ========================================================================
    # TTS — 文本转语音
    # ========================================================================

    def synthesize(self, text: str, voice: Optional[str] = None, speed: float = 1.0) -> dict[str, Any]:
        """
        将文本转换为音频，返回 base64 编码的音频数据。

        Args:
            text: 要合成的英文文本
            voice: 声音选择（alloy/echo/fable/onyx/nova/shimmer）
            speed: 语速（0.25~4.0）

        Returns:
            {"status": "success", "audio_base64": "...", "voice": "alloy", ...}
        """
        payload = {"text": text, "speed": speed}
        if voice:
            payload["voice"] = voice
        return self._post("/api/tts/synthesize", payload)

    def get_voices(self) -> dict[str, Any]:
        """列出可用的 TTS 声音"""
        with httpx.Client(timeout=10) as client:
            resp = client.get(f"{self.base_url}/api/tts/voices", timeout=10)
            resp.raise_for_status()
            return resp.json()
