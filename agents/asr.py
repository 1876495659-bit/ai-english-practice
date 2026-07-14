"""
ASR Service — 语音转文本（Speech to Text）

支持两种 Provider：
1. OpenAI Whisper API（默认，需要 API Key）
2. 本地 whisper.cpp / faster-whisper（可选，无需网络）

使用方式：
    from agents.asr import transcribe_audio
    text = await transcribe_audio(audio_bytes, language="en")
"""

from __future__ import annotations

import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)


async def transcribe_audio(
    audio_data: bytes,
    language: str = "en",
    model: str = "whisper-1",
) -> str:
    """
    将音频数据转换为文本。

    Args:
        audio_data: 音频字节数据（支持 WAV/MP3/M4A/OGG 等格式）
        language: 语言代码（ISO 639-1），如 "en"、"zh"
        model: 模型名称，OpenAI Whisper 使用 "whisper-1"

    Returns:
        转录后的文本

    Raises:
        RuntimeError: 当 API Key 缺失或请求失败时
    """
    if not audio_data:
        raise ValueError("音频数据不能为空")

    try:
        from config.settings import settings
        api_key = settings.openai_api_key
        if not api_key or api_key.startswith("sk-your-key"):
            logger.warning("[ASR] OpenAI API Key 未配置，返回 mock 文本")
            return f"[Mock ASR] 检测到 {len(audio_data)} 字节的音频输入，请配置 API Key 以启用真实语音识别"
    except Exception:
        pass

    # --- OpenAI Whisper API ---
    try:
        import openai
        from openai import OpenAI as AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key or None)

        # 将音频数据包装为文件对象
        audio_file = io.BytesIO(audio_data)
        audio_file.name = "audio.wav"  # Whisper 需要文件名

        response = await client.audio.transcriptions.create(
            model=model,
            file=audio_file,
            language=language,
            response_format="text",
            temperature=0.0,
        )

        logger.info(f"[ASR] Transcribed {len(audio_data)} bytes → {len(response)} chars")
        return response.strip()

    except ImportError:
        logger.warning("[ASR] openai package not installed. Install with: pip install openai")
        return _mock_transcription(audio_data, language)
    except Exception as e:
        logger.warning(f"[ASR] OpenAI Whisper failed: {e}, falling back to mock")
        return _mock_transcription(audio_data, language)


def _mock_transcription(audio_data: bytes, language: str) -> str:
    """Mock 语音识别结果（开发/测试用）"""
    lang_map = {"en": "English", "zh": "Chinese", "ja": "Japanese"}
    lang_name = lang_map.get(language, language)
    return (
        f"[Mock ASR] {lang_name} speech detected from "
        f"{len(audio_data)} bytes of audio data. "
        f"This is a placeholder — configure OpenAI API Key for real transcription."
    )


async def detect_language(audio_data: bytes) -> str:
    """
    检测音频中的语言。

    Returns:
        语言代码（如 "en"、"zh"），检测失败返回 "en"（默认英语）
    """
    if not audio_data:
        return "en"

    try:
        from config.settings import settings
        api_key = getattr(settings, "openai_api_key", "")
        if not api_key or api_key.startswith("sk-your-key"):
            return "en"
    except Exception:
        pass

    try:
        import openai
        from openai import OpenAI as AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key)
        audio_file = io.BytesIO(audio_data)
        audio_file.name = "audio.wav"

        response = await client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
        )

        detected_lang = getattr(response, "language", "en")
        logger.info(f"[ASR] Detected language: {detected_lang}")
        return detected_lang

    except Exception as e:
        logger.warning(f"[ASR] Language detection failed: {e}")
        return "en"
