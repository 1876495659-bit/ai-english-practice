"""
全局配置管理

使用 pydantic v2 settings 提供类型安全的配置加载，
支持环境变量和 .env 文件两种方式。
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
    """系统全局配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # === LLM 配置 ===
    # 当前使用的 Provider: openai / anthropic / groq(qwen)
    llm_provider: Literal["openai", "anthropic", "groq"] = "openai"

    # OpenAI
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"

    # Anthropic
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"

    # Groq (支持 Qwen)
    groq_api_key: str = ""
    groq_model: str = "qwen-2.5-7b"

    # === 服务配置 ===
    # ASR (语音转文本) - 预留
    asr_provider: str = "openai_whisper"

    # TTS (文本转语音) - 预留
    tts_provider: str = "openai_tts"

    # === 应用配置 ===
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = False

    # === 评分配置 ===
    scoring_max_score: float = 100.0

    # === LLM 模式开关 ===
    # 是否启用真实 LLM 调用（默认 False 使用 mock）
    llm_enabled: bool = False
    # 各 Node 的 LLM 开关（细粒度控制）
    llm_mode_conversation: bool = False  # conversation node 使用 LLM
    llm_mode_correction: bool = False    # correction node 使用 LLM 纠错
    llm_mode_scoring: bool = False       # scoring node 使用 LLM 评分


# 全局单例配置
settings = Settings()
