"""
LLM Provider 配置

统一管理不同 LLM 厂商的客户端初始化逻辑。
新增 Provider 时只需在此扩展。

注意：SDK 导入采用惰性加载，避免因缺失未使用的 SDK 而启动失败。
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

from openai import AsyncOpenAI

from config.settings import settings


def get_openai_client() -> AsyncOpenAI:
    """获取 OpenAI 客户端"""
    return AsyncOpenAI(
        api_key=settings.openai_api_key or os.getenv("OPENAI_API_KEY", ""),
        base_url=settings.openai_base_url,
    )


def get_anthropic_client():
    """获取 Anthropic 客户端（惰性加载）"""
    try:
        from anthropic import AsyncAnthropic  # noqa: F811
    except ImportError:
        logger.warning("anthropic SDK not installed. Install with: pip install anthropic")
        raise RuntimeError(
            "Anthropic SDK not found. Please install it: pip install anthropic"
        )
    return AsyncAnthropic(
        api_key=settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY", ""),
    )


def get_groq_client():
    """获取 Groq 客户端（惰性加载）"""
    try:
        import groq  # noqa: F401
    except ImportError:
        logger.warning("groq SDK not installed. Install with: pip install groq")
        raise RuntimeError(
            "Groq SDK not found. Please install it: pip install groq"
        )
    from groq import Groq  # noqa: F811
    return Groq(
        api_key=settings.groq_api_key or os.getenv("GROQ_API_KEY", ""),
    )


def get_llm_client():
    """根据配置获取对应的 LLM 客户端"""
    clients = {
        "openai": get_openai_client,
        "anthropic": get_anthropic_client,
        "groq": get_groq_client,
    }
    factory = clients.get(settings.llm_provider)
    if not factory:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")
    return factory()
