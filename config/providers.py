"""
LLM Provider 配置

统一管理不同 LLM 厂商的客户端初始化逻辑。
新增 Provider 时只需在此扩展。
"""

import os
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from groq import Groq

from config.settings import settings


def get_openai_client() -> AsyncOpenAI:
    """获取 OpenAI 客户端"""
    return AsyncOpenAI(
        api_key=settings.openai_api_key or os.getenv("OPENAI_API_KEY", ""),
        base_url=settings.openai_base_url,
    )


def get_anthropic_client() -> AsyncAnthropic:
    """获取 Anthropic 客户端"""
    return AsyncAnthropic(
        api_key=settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY", ""),
    )


def get_groq_client() -> Groq:
    """获取 Groq 客户端 (支持 Qwen 等开源模型)"""
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
