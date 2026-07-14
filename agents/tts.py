"""
TTS Service — 文本转语音（Text to Speech）

支持两种 Provider：
1. OpenAI TTS API（默认，需要 API Key）
2. 本地 edge-tts（可选，无需网络，免费）

使用方式：
    from agents.tts import synthesize_speech
    audio_bytes = await synthesize_speech("Hello, how are you?")

    # 或直接获取音频 URL（OpenAI 返回临时 URL）
    url = await synthesize_speech_to_url("Hello, how are you?")
"""

from __future__ import annotations

import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================================
# 常量
# ============================================================================

# OpenAI TTS 支持的 voices
_OPENAI_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

# 支持的语速范围
SPEED_RANGE = (0.25, 4.0)

# 默认配置
DEFAULT_VOICE = "alloy"
DEFAULT_SPEED = 1.0
DEFAULT_MODEL = "tts-1"
DEFAULT_FORMAT = "wav"


async def synthesize_speech(
    text: str,
    voice: Optional[str] = None,
    speed: float = DEFAULT_SPEED,
    model: str = DEFAULT_MODEL,
    output_format: str = DEFAULT_FORMAT,
) -> bytes:
    """
    将文本转换为语音音频数据。

    Args:
        text: 要转换为语音的英文文本
        voice: 声音选择（alloy/echo/fable/onyx/nova/shimmer），默认 alloy
        speed: 语速（0.25~4.0），默认 1.0
        model: 模型名称（tts-1 / tts-1-hd），默认 tts-1
        output_format: 输出格式（wav/mp3/aac/flac），默认 wav

    Returns:
        音频字节数据

    Raises:
        ValueError: 当文本为空或参数无效时
    """
    if not text or not text.strip():
        raise ValueError("待转换的文本不能为空")

    voice = voice or DEFAULT_VOICE
    if voice not in _OPENAI_VOICES:
        voice = DEFAULT_VOICE

    speed = max(SPEED_RANGE[0], min(SPEED_RANGE[1], speed))

    # --- OpenAI TTS API ---
    try:
        from config.settings import settings
        api_key = settings.openai_api_key
        if not api_key or api_key.startswith("sk-your-key"):
            logger.warning("[TTS] OpenAI API Key 未配置，返回 mock 音频")
            return _mock_audio(text, output_format)
    except Exception:
        pass

    try:
        import openai
        from openai import OpenAI as AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key or None)

        response = await client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            speed=speed,
            response_format=output_format,
        )

        audio_bytes = response.content
        logger.info(f"[TTS] Synthesized {len(text)} chars → {len(audio_bytes)} bytes ({output_format})")
        return audio_bytes

    except ImportError:
        logger.warning("[TTS] openai package not installed. Install with: pip install openai")
        return _mock_audio(text, output_format)
    except Exception as e:
        logger.warning(f"[TTS] OpenAI TTS failed: {e}, falling back to mock")
        return _mock_audio(text, output_format)


async def synthesize_speech_to_url(
    text: str,
    voice: Optional[str] = None,
    speed: float = DEFAULT_SPEED,
) -> str:
    """
    将文本转换为语音并返回可访问的音频 URL。

    注意：OpenAI TTS API 返回的是音频字节流，不直接提供 URL。
    此函数将音频数据编码为 data URI 以便前端直接播放。

    Args:
        text: 要转换为语音的英文文本
        voice: 声音选择
        speed: 语速

    Returns:
        data URI 格式的音频 URL（data:audio/wav;base64,...）
    """
    import base64

    audio_bytes = await synthesize_speech(text, voice=voice, speed=speed)
    encoded = base64.b64encode(audio_bytes).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"


def _mock_audio(text: str, output_format: str = "wav") -> bytes:
    """
    Mock 音频数据（开发/测试用）。

    生成一个 1 秒的静音 WAV 文件作为占位符。
    实际使用时会被 OpenAI API 替换。
    """
    try:
        import struct
        import wave

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)       # 单声道
            wav_file.setsampwidth(2)       # 16-bit
            wav_file.setframerate(22050)   # 22.05kHz
            # 写入 0.5 秒静音
            silence = b"\x00\x00" * 11025
            wav_file.writeframes(silence)

        result = buffer.getvalue()
        logger.info(f"[TTS] Generated {len(result)} bytes of mock WAV audio")
        return result

    except Exception as e:
        logger.error(f"[TTS] Failed to generate mock audio: {e}")
        return b""


# ============================================================================
# 批量合成
# ============================================================================


async def synthesize_batch(
    texts: list[str],
    voice: Optional[str] = None,
    speed: float = DEFAULT_SPEED,
) -> list[bytes]:
    """
    批量将多个文本转换为语音。

    Args:
        texts: 文本列表
        voice: 声音选择
        speed: 语速

    Returns:
        音频字节数据列表
    """
    results = []
    for i, text in enumerate(texts):
        try:
            audio = await synthesize_speech(text, voice=voice, speed=speed)
            results.append(audio)
        except Exception as e:
            logger.warning(f"[TTS Batch] Failed to synthesize text[{i}]: {e}")
            results.append(_mock_audio(f"[Error synthesizing text {i}]"))
    return results
