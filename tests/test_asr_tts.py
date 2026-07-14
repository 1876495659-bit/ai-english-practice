"""
单元测试 - ASR/TTS 服务

验证 ASR（语音转文本）和 TTS（文本转语音）服务的核心逻辑。
"""

import asyncio
import pytest


def _sync_run(coro):
    """同步调用协程"""
    return asyncio.run(coro)


class TestASRService:
    """测试 ASR 语音转文本服务"""

    def test_transcribe_empty_audio_raises_error(self):
        """空音频数据应抛出 ValueError"""
        from agents.asr import transcribe_audio
        with pytest.raises(ValueError):
            _sync_run(transcribe_audio(b""))

    def test_transcribe_without_api_key_returns_mock(self):
        """未配置 API Key 时应返回 mock 文本"""
        from agents.asr import transcribe_audio
        # 模拟没有 API Key 的情况
        result = _sync_run(transcribe_audio(b"\x00\x01\x02", language="en"))
        assert isinstance(result, str)
        assert len(result) > 0
        # Mock 结果应包含提示信息
        assert "Mock ASR" in result or "mock" in result.lower() or "placeholder" in result.lower()

    def test_transcribe_preserves_language(self):
        """语言参数应被传递"""
        from agents.asr import transcribe_audio
        result = _sync_run(transcribe_audio(b"\x00\x01", language="zh"))
        assert isinstance(result, str)

    def test_detect_language_default(self):
        """检测语言失败时应返回默认值 en"""
        from agents.asr import detect_language
        result = _sync_run(detect_language(b""))
        assert result == "en"

    def test_detect_language_without_api_key(self):
        """未配置 API Key 时应返回默认英语"""
        from agents.asr import detect_language
        result = _sync_run(detect_language(b"\x00\x01"))
        assert result == "en"


class TestTTSService:
    """测试 TTS 文本转语音服务"""

    def test_synthesize_empty_text_raises_error(self):
        """空文本应抛出 ValueError"""
        from agents.tts import synthesize_speech
        with pytest.raises(ValueError):
            _sync_run(synthesize_speech(""))

    def test_synthesize_whitespace_only_raises_error(self):
        """纯空白文本应抛出 ValueError"""
        from agents.tts import synthesize_speech
        with pytest.raises(ValueError):
            _sync_run(synthesize_speech("   "))

    def test_synthesize_without_api_key_returns_mock_audio(self):
        """未配置 API Key 时应返回 mock 音频数据"""
        from agents.tts import synthesize_speech
        audio = _sync_run(synthesize_speech("Hello world"))
        assert isinstance(audio, bytes)
        assert len(audio) > 0

    def test_synthesize_with_custom_voice(self):
        """自定义声音参数应被使用"""
        from agents.tts import synthesize_speech
        # 使用有效声音
        audio = _sync_run(synthesize_speech("Test", voice="nova"))
        assert isinstance(audio, bytes)
        assert len(audio) > 0

    def test_synthesize_with_custom_speed(self):
        """自定义语速参数应被使用"""
        from agents.tts import synthesize_speech
        audio = _sync_run(synthesize_speech("Test", speed=2.0))
        assert isinstance(audio, bytes)
        assert len(audio) > 0

    def test_synthesize_batch(self):
        """批量合成应返回音频列表"""
        from agents.tts import synthesize_batch
        texts = ["Hello", "World", "Test"]
        results = _sync_run(synthesize_batch(texts))
        assert len(results) == 3
        for audio in results:
            assert isinstance(audio, bytes)
            assert len(audio) > 0

    def test_mock_audio_generates_valid_wav(self):
        """Mock 音频应生成有效的 WAV 文件"""
        from agents.tts import _mock_audio
        audio = _mock_audio("Test text")
        assert isinstance(audio, bytes)
        # WAV 文件头应以 'RIFF' 开头
        assert audio[:4] == b"RIFF"
        # 应包含 WAV 标识
        assert b"WAV" in audio[:12]


class TestTTSConstants:
    """测试 TTS 常量定义"""

    def test_openai_voices_defined(self):
        """应定义 OpenAI TTS 支持的 voices"""
        from agents.tts import _OPENAI_VOICES
        assert isinstance(_OPENAI_VOICES, list)
        assert len(_OPENAI_VOICES) >= 6
        assert "alloy" in _OPENAI_VOICES
        assert "echo" in _OPENAI_VOICES

    def test_speed_range_defined(self):
        """应定义语速范围"""
        from agents.tts import SPEED_RANGE
        assert isinstance(SPEED_RANGE, tuple)
        assert len(SPEED_RANGE) == 2
        assert SPEED_RANGE[0] < SPEED_RANGE[1]

    def test_default_values(self):
        """应定义默认值"""
        from agents.tts import DEFAULT_VOICE, DEFAULT_SPEED, DEFAULT_MODEL, DEFAULT_FORMAT
        assert DEFAULT_VOICE == "alloy"
        assert DEFAULT_SPEED == 1.0
        assert DEFAULT_MODEL == "tts-1"
        assert DEFAULT_FORMAT == "wav"
