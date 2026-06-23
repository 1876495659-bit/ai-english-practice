"""
Correction Node - 语法纠错节点

作为 LangGraph StateGraph 的一个 Node 运行。
职责：
1. 分析用户英语表达中的语法错误
2. 提供修正版本
3. 给出更地道/自然的表达建议
4. 输出结构化 JSON 数据

注意：此文件作为独立 Node 函数运行，不继承 BaseAgent。
"""

from __future__ import annotations

from typing import Any


async def correction_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    纠错 Node

    从 state 中提取用户最新输入，进行纠错分析，
    将结构化结果写入 state["correction"]。

    Args:
        state: 当前图状态

    Returns:
        State 增量更新 dict
    """
    # 从 messages 中提取用户最新输入
    messages: list[dict] = state.get("messages", [])
    user_input = ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            user_input = msg.get("content", "")
            break

    if not user_input:
        return {
            "correction": {
                "original": "",
                "errors": [],
                "corrected": "",
                "suggestion": "",
                "explanation": "没有检测到用户输入",
                "has_errors": False,
            }
        }

    correction = await _mock_correction(user_input)
    return {"correction": correction}


async def _mock_correction(user_input: str) -> dict[str, Any]:
    """Mock 纠错逻辑 - 检测常见错误模式"""
    errors_found: list[dict[str, str]] = []
    corrected = user_input
    suggestion = user_input

    lower = user_input.lower()

    # 检测常见语法错误
    if user_input and user_input[0:1].islower() and user_input[0:1].isalpha():
        errors_found.append({
            "type": "grammar",
            "issue": "首字母应大写",
        })

    if "i want" in lower:
        errors_found.append({
            "type": "style",
            "issue": "'I want' 语气较直接，建议用 'I would like' 更礼貌",
        })
        suggestion = user_input.replace("I want", "I would like").replace(
            "i want", "I would like"
        )

    if "very good" in lower:
        errors_found.append({
            "type": "vocabulary",
            "issue": "'very good' 过于简单，可以尝试更丰富的词汇",
        })
        suggestion = suggestion.replace("very good", "fantastic", 1)

    # 如果没有检测到错误
    if not errors_found:
        return {
            "original": user_input,
            "errors": [],
            "corrected": user_input,
            "suggestion": user_input,
            "explanation": "无明显语法错误，表达良好！",
            "has_errors": False,
        }

    return {
        "original": user_input,
        "errors": errors_found,
        "corrected": corrected,
        "suggestion": suggestion,
        "explanation": f"发现 {len(errors_found)} 个问题：",
        "has_errors": True,
    }
