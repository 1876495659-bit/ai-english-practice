"""
LLM Client - 统一 LLM 调用层

作为 LangGraph Node 的 LLM 基础设施。
支持 OpenAI / Anthropic / Groq 三种 provider。
所有 Node 通过此模块调用 LLM，不直接依赖具体 SDK。

设计原则：
- 统一的 async 接口
- 结构化输出（JSON）
- 降级策略：LLM 失败时回退到 mock
- 错误隔离：Node 不因 LLM 故障崩溃
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from config.providers import get_llm_client
from config.settings import settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)  # 默认 WARNING 级别，抑制调试日志噪音


# ============================================================================
# 常量
# ============================================================================

_DEFAULT_MODEL_MAP: dict[str, str] = {
    "openai": settings.openai_model,
    "anthropic": settings.anthropic_model,
    "groq": settings.groq_model,
}

_SYSTEM_PROMPT_PREFIX = (
    "You are a professional English language teaching assistant. "
    "You help Chinese learners improve their spoken English. "
    "Always respond in a helpful, encouraging manner."
)


# ============================================================================
# 核心调用函数
# ============================================================================


async def call_llm(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.7,
    max_tokens: int = 512,
    response_format: Optional[dict] = None,
    model: Optional[str] = None,
) -> str:
    """
    统一 LLM 调用接口

    Args:
        messages: [{"role": "user/system/assistant", "content": "..."}]
        temperature: 采样温度
        max_tokens: 最大生成长度
        response_format: {"type": "json_object"} 强制 JSON 输出
        model: 覆盖默认模型

    Returns:
        LLM 生成的文本响应

    Raises:
        Exception: 当所有 provider 都失败时抛出
    """
    provider = settings.llm_provider
    model_name = model or _DEFAULT_MODEL_MAP.get(provider, "gpt-4o-mini")

    # 构建带系统提示的消息链
    full_messages: list[dict[str, str]] = [
        {"role": "system", "content": _SYSTEM_PROMPT_PREFIX},
        *messages,
    ]

    # 根据 provider 分发
    if provider == "openai":
        return await _call_openai(full_messages, model_name, temperature, max_tokens, response_format)
    elif provider == "anthropic":
        return await _call_anthropic(full_messages, model_name, temperature, max_tokens)
    elif provider == "groq":
        return await _call_groq(full_messages, model_name, temperature, max_tokens, response_format)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


async def _call_openai(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    response_format: Optional[dict],
) -> str:
    """调用 OpenAI API"""
    try:
        client = get_llm_client()
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        # Ollama 本地模型响应较慢，增加超时时间
        import httpx
        client.timeout = httpx.Timeout(120.0)
        response = await client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response from OpenAI")
        return content
    except Exception as e:
        logger.error(f"OpenAI LLM call failed: {e}")
        raise


async def _call_anthropic(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """调用 Anthropic Claude API"""
    try:
        client = get_llm_client()
        # 分离 system message
        system_content = ""
        chat_messages: list[dict[str, str]] = []
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                chat_messages.append({"role": msg["role"], "content": msg["content"]})

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": chat_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system_content:
            kwargs["system"] = system_content

        response = await client.messages.create(**kwargs)
        content = response.content[0].text if response.content else ""
        if not content:
            raise ValueError("Empty response from Anthropic")
        return content
    except Exception as e:
        logger.error(f"Anthropic LLM call failed: {e}")
        raise


async def _call_groq(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    response_format: Optional[dict],
) -> str:
    """调用 Groq API (支持 Qwen 等开源模型)"""
    try:
        client = get_llm_client()
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        response = await client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response from Groq")
        return content
    except Exception as e:
        logger.error(f"Groq LLM call failed: {e}")
        raise


# ============================================================================
# 便捷函数
# ============================================================================


async def call_llm_json(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    model: Optional[str] = None,
) -> dict[str, Any]:
    """
    调用 LLM 并解析 JSON 响应

    自动设置 response_format 强制 JSON 输出，并将响应文本解析为 dict。

    Args:
        messages: 消息列表
        temperature: 降低温度以获得更稳定的结构化输出
        max_tokens: 最大 token 数（JSON 需要更多空间）
        model: 覆盖默认模型

    Returns:
        解析后的 dict

    Raises:
        json.JSONDecodeError: 当 LLM 返回无效 JSON 时
    """
    raw = await call_llm(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        model=model,
    )
    return json.loads(raw)


async def safe_llm_call(
    messages: list[dict[str, str]],
    fallback_fn=None,
    *,
    temperature: float = 0.7,
    max_tokens: int = 512,
    response_format: Optional[dict] = None,
    model: Optional[str] = None,
) -> str:
    """
    安全的 LLM 调用：失败时回退到 fallback_fn

    Args:
        messages: 消息列表
        fallback_fn: 可选的降级函数，返回 str
        temperature/max_tokens/response_format/model: 同 call_llm

    Returns:
        LLM 响应或 fallback 结果
    """
    try:
        return await call_llm(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            model=model,
        )
    except Exception as e:
        logger.warning(f"LLM call failed, using fallback: {e}")
        if fallback_fn:
            return fallback_fn()
        return ""
