"""
Scoring Node - 口语评分节点

作为 LangGraph StateGraph 的一个 Node 运行。
职责：
1. 从四个维度对用户口语进行评分（0-10分）
2. 给出综合反馈和建议

评分维度：
- fluency: 流利度
- grammar: 语法准确性
- vocabulary: 词汇丰富度
- naturalness: 表达自然度

注意：此文件作为独立 Node 函数运行，不继承 BaseAgent。
"""

from __future__ import annotations

import random
from typing import Any


async def scoring_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    评分 Node

    从 state 中提取用户最新输入，进行四维评分，
    将结构化结果写入 state["score"]。

    Args:
        state: 当前图状态

    Returns:
        State 增量更新 dict
    """
    # 从 messages 中提取用户最新输入（兼容 dict 和 LangGraph BaseMessage）
    messages: list = state.get("messages", [])
    user_input = ""
    for msg in reversed(messages):
        if isinstance(msg, dict):
            if msg.get("role") == "user":
                user_input = msg.get("content", "").strip()
                break
        else:
            role = getattr(msg, "type", None) or getattr(msg, "_getType", lambda: "")()
            if role == "human":
                user_input = getattr(msg, "content", "").strip()
                break

    if not user_input:
        return {
            "score": {
                "scores": {"fluency": 0, "grammar": 0, "vocabulary": 0, "naturalness": 0},
                "total": 0,
                "feedback_en": "No input to evaluate.",
                "feedback_zh": "没有检测到用户输入",
                "strengths": [],
                "improvements": [],
            }
        }

    score = await _mock_evaluate(user_input)
    return {"score": score}


async def _mock_evaluate(user_input: str) -> dict[str, Any]:
    """Mock 评分逻辑"""
    word_count = len(user_input.split()) if user_input else 0

    has_complex_words = any(
        w.lower() in {"however", "therefore", "although", "moreover",
                      "significant", "particular", "environment",
                      "opportunity", "experience", "interesting",
                      "delicious", "recommendation", "appointment"}
        for w in user_input.split()
    )

    base_score = 5.0 + min(word_count / 5, 3.0)
    if has_complex_words:
        base_score += 1.0

    seed_val = hash(user_input) % (2**32)
    rng = random.Random(seed_val)

    fluency = round(min(max(base_score + rng.uniform(-1, 1), 1.0), 10.0), 1)
    grammar = round(min(max(base_score + rng.uniform(-1.5, 1.5), 1.0), 10.0), 1)
    vocabulary = round(min(max(base_score + rng.uniform(-1, 2), 1.0), 10.0), 1)
    naturalness = round(min(max(base_score + rng.uniform(-1.5, 1), 1.0), 10.0), 1)

    total = round((fluency + grammar + vocabulary + naturalness) / 4, 1)

    strengths: list[str] = []
    improvements: list[str] = []

    if fluency >= 7:
        strengths.append("表达流畅，思路清晰")
    else:
        improvements.append("注意提高表达的连贯性，减少停顿")

    if grammar >= 7:
        strengths.append("语法使用较为准确")
    else:
        improvements.append("注意基本语法结构，特别是时态和主谓一致")

    if vocabulary >= 7:
        strengths.append("词汇运用较为丰富")
    else:
        improvements.append("尝试使用更多样化的词汇，避免重复")

    if naturalness >= 7:
        strengths.append("表达自然，接近母语者习惯")
    else:
        improvements.append("多听多模仿母语者的表达方式")

    if not strengths:
        strengths.append("继续保持练习，进步空间很大")
    if not improvements:
        improvements.append("当前表现优秀，挑战更高难度")

    return {
        "scores": {
            "fluency": fluency,
            "grammar": grammar,
            "vocabulary": vocabulary,
            "naturalness": naturalness,
        },
        "total": total,
        "feedback_en": (
            f"Overall {total}/10. "
            f"You showed good "
            f"{', '.join(s.split('，')[0] for s in strengths)}. "
            f"To improve, focus on "
            f"{', '.join(i.split('，')[0] for i in improvements)}."
        ),
        "feedback_zh": f"综合评分 {total}/10。{'; '.join(strengths + improvements)}",
        "strengths": strengths,
        "improvements": improvements,
    }
