"""
Scoring Node - 口语评分节点

作为 LangGraph StateGraph 的一个 Node 运行。
职责：
1. 从四个维度对用户口语进行评分（0-10分）
2. 给出综合反馈和建议
3. 更新用户能力追踪数据（Skill Progress）
4. 自适应调整难度（根据历史表现）
5. 通过 Command 控制条件路由（Loop Training）

双通道设计（Stage 4）：
- 规则引擎通道：快速、确定性评分（基于词数/复杂度启发式）
- LLM 通道：智能评分，通过 LLM_MODE_SCORING 配置开关

自适应学习（Stage 5）：
- 维护 skill_progress 追踪用户能力轨迹
- 连续高分 → 自动提升难度
- 连续低分 → 自动降低难度
- 评分低 → 通过 Command 路由回 conversation 重新练习
- 超过 max_retries → 强制 END

注意：此文件作为独立 Node 函数运行，不继承 BaseAgent。
"""

from __future__ import annotations

import json
import logging
import random
from typing import Any

from agents.llm_client import call_llm_json
from agents.prompts_loader import load_prompt
from agents.utils import extract_latest_user_input

logger = logging.getLogger(__name__)

# ============================================================================
# 评分常量
# ============================================================================

_ADVANCED_WORDS: set[str] = {
    "however", "therefore", "although", "moreover", "nevertheless",
    "significant", "particular", "environment", "opportunity", "experience",
    "interesting", "delicious", "recommendation", "appointment",
    "substantial", "considerable", "approximately", "particularly",
    "frequently", "occasionally", "sufficient", "appropriate",
    "demonstrate", "accomplish", "establish", "participate",
    "consequently", "notwithstanding", "whereas",
}

_COMPLEX_PHRASES: list[str] = [
    "in my opinion", "on the other hand", "as a matter of fact",
    "for instance", "in addition", "as well as", "due to",
    "in spite of", "rather than", "in terms of", "by the way",
    "as far as", "so far", "right now", "at least",
]

# 难度阈值：平均分 >= 7.5 提升，<= 4.5 降低
DIFFICULTY_UP_THRESHOLD = 7.5
DIFFICULTY_DOWN_THRESHOLD = 4.5
DIFFICULTY_LEVELS = ["easy", "medium", "hard"]


async def scoring_node(state: dict[str, Any]) -> dict[str, Any] | Any:
    """
    评分 Node

    双通道：优先 LLM 评分，失败回退到规则引擎。
    同时更新 skill_progress 和自适应难度。

    路由控制（Stage 5）：
    - 评分低（grammar < 6 或 fluency < 6）→ 递增 retry_count 并路由回 conversation
    - 超过 max_retries → 路由到 END
    - 评分良好 → 路由到 END

    Args:
        state: 当前图状态

    Returns:
        - 正常结束：返回 State 增量更新 dict
        - 需要循环：返回 langgraph.types.Command（带 goto="conversation"）
    """
    # LangGraph 1.x 消息管理策略：
    # 由于 checkpoint 不保留大部分字段（如 user_input），
    # 优先从 correction.original 获取（correction_node 已成功提取并持久化）
    correction = state.get("correction", {})
    if isinstance(correction, dict):
        user_input = correction.get("original", "").strip()

    # fallback 到 messages 字段（兼容直接 Node 调用场景）
    if not user_input:
        user_input = extract_latest_user_input(state.get("messages", []))

    if not user_input:
        return _empty_return(state)

    # 检查是否启用 LLM 评分
    try:
        from config.settings import settings
        use_llm = getattr(settings, "llm_mode_scoring", False)
    except Exception:
        use_llm = False

    score_data = None
    if use_llm:
        score_data = await _llm_evaluate(user_input)
        if score_data:
            logger.info("[ScoringNode] LLM score applied")
        else:
            logger.warning("[ScoringNode] LLM scoring failed, falling back to rule engine")

    # 规则引擎通道
    if not score_data:
        score_data = await _rule_engine_evaluate(user_input)

    # 更新 skill_progress
    progress_updates = _update_skill_progress(state, score_data)
    # 自适应难度调整
    difficulty_adjustment = _adjust_difficulty_adaptive(state, score_data)

    # 判断是否需要循环
    scores = score_data.get("scores", {})
    grammar = scores.get("grammar", 10)
    fluency = scores.get("fluency", 10)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    # 评分低且未达到最大重试次数 → 循环
    if (grammar < 6.0 or fluency < 6.0) and retry_count < max_retries:
        new_retry_count = retry_count + 1
        logger.info(
            f"[ScoringNode] Low scores (grammar={grammar}, fluency={fluency}), "
            f"routing back to conversation (retry {new_retry_count}/{max_retries})"
        )
        from langgraph.types import Command
        return Command(
            update={
                "score": score_data,
                "skill_progress": progress_updates,
                "retry_count": new_retry_count,
                **difficulty_adjustment,
            },
            goto="conversation",
        )

    # 评分良好或超过最大重试次数 → 结束
    logger.info(
        f"[ScoringNode] Scores OK (grammar={grammar}, fluency={fluency}), "
        f"ending session"
    )
    return {
        "score": score_data,
        "skill_progress": progress_updates,
        **difficulty_adjustment,
    }


def _empty_return(state: dict[str, Any]) -> dict[str, Any]:
    """当没有用户输入时的统一返回"""
    existing: dict[str, Any] = state.get("skill_progress", {})
    return {
        "score": {
            "scores": {"fluency": 0, "grammar": 0, "vocabulary": 0, "naturalness": 0},
            "total": 0,
            "feedback_en": "No input to evaluate.",
            "feedback_zh": "没有检测到用户输入",
            "strengths": [],
            "improvements": [],
        },
        "skill_progress": {
            "total_turns": existing.get("total_turns", 0) + 1,
            "avg_score": existing.get("avg_score", 0.0),
            "error_frequency": existing.get("error_frequency", {}),
            "weakest_dimension": existing.get("weakest_dimension", ""),
            "strongest_dimension": existing.get("strongest_dimension", ""),
            "improvement_trajectory": existing.get("improvement_trajectory", []),
        },
        "retry_count": 0,
    }


# ============================================================================
# LLM 评分通道
# ============================================================================


async def _llm_evaluate(user_input: str) -> dict[str, Any] | None:
    """使用 LLM 进行智能评分，从 prompts/scoring.txt 加载模板。"""
    try:
        prompt = load_prompt("scoring", user_input=user_input)
        result = await call_llm_json([
            {"role": "user", "content": prompt},
        ], temperature=0.3, max_tokens=512)
        return result
    except (json.JSONDecodeError, KeyError, Exception) as e:
        logger.warning(f"[ScoringNode] LLM JSON parse failed: {e}")
        return None


# ============================================================================
# 规则引擎评分通道
# ============================================================================


async def _rule_engine_evaluate(user_input: str) -> dict[str, Any]:
    """基于启发式的评分逻辑"""
    word_count = len(user_input.split()) if user_input else 0

    words = user_input.split()
    lower_words = [w.lower().strip(".,!?;:") for w in words]

    has_advanced = any(w in _ADVANCED_WORDS for w in lower_words)
    has_complex_phrase = any(phrase in user_input.lower() for phrase in _COMPLEX_PHRASES)
    sentences = [s.strip() for s in user_input.replace(".", ".").replace("!", "!").replace("?", "?").split(".") if s.strip()]
    avg_sentence_len = word_count / max(len(sentences), 1)

    base_score = 5.0 + min(word_count / 5, 3.0)
    if has_advanced:
        base_score += 1.0
    if has_complex_phrase:
        base_score += 0.5
    if 8 <= avg_sentence_len <= 20:
        base_score += 0.5

    seed_val = hash(user_input) % (2**32)
    rng = random.Random(seed_val)

    fluency = round(min(max(base_score + rng.uniform(-1, 1), 1.0), 10.0), 1)
    grammar = round(min(max(base_score + rng.uniform(-1.5, 1.5), 1.0), 10.0), 1)
    vocabulary = round(min(max(base_score + rng.uniform(-1, 2), 1.0), 10.0), 1)
    naturalness = round(min(max(base_score + rng.uniform(-1.5, 1), 1.0), 10.0), 1)

    total = round((fluency + grammar + vocabulary + naturalness) / 4, 1)

    strengths, improvements = _generate_feedback(
        fluency, grammar, vocabulary, naturalness, word_count
    )

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


def _generate_feedback(
    fluency: float, grammar: float, vocabulary: float, naturalness: float, word_count: int
) -> tuple[list[str], list[str]]:
    """根据各项分数生成优缺点反馈"""
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

    if word_count < 5:
        improvements.insert(0, "试着多说几句，练习更完整的表达")

    if not strengths:
        strengths.append("继续保持练习，进步空间很大")
    if not improvements:
        improvements.append("当前表现优秀，挑战更高难度")

    return strengths, improvements


# ============================================================================
# Stage 5: Skill Progress 追踪
# ============================================================================


def _update_skill_progress(
    state: dict[str, Any], score_data: dict[str, Any] | None
) -> dict[str, Any]:
    """更新用户能力追踪数据"""
    existing_progress: dict[str, Any] = state.get("skill_progress", {})
    scores = state.get("score", {}).get("scores", {}) if state.get("score") else {}
    correction = state.get("correction", {})

    total_turns = existing_progress.get("total_turns", 0) + 1

    error_freq: dict[str, int] = dict(existing_progress.get("error_frequency", {}))
    if correction and correction.get("errors"):
        for err in correction["errors"]:
            etype = err.get("type", "unknown")
            error_freq[etype] = error_freq.get(etype, 0) + 1

    trajectory: list[float] = list(existing_progress.get("improvement_trajectory", []))
    if score_data:
        trajectory.append(score_data.get("total", 0))

    if trajectory:
        avg_score = round(sum(trajectory) / len(trajectory), 1)
    else:
        avg_score = 0.0

    dim_scores = {k: v for k, v in scores.items()} if scores else {}
    strongest = max(dim_scores, key=dim_scores.get) if dim_scores else ""
    weakest = min(dim_scores, key=dim_scores.get) if dim_scores else ""

    return {
        "total_turns": total_turns,
        "avg_score": avg_score,
        "error_frequency": error_freq,
        "weakest_dimension": weakest,
        "strongest_dimension": strongest,
        "improvement_trajectory": trajectory,
    }


def _adjust_difficulty_adaptive(
    state: dict[str, Any], score_data: dict[str, Any] | None
) -> dict[str, str]:
    """
    自适应难度调整

    根据最近几次的评分趋势自动调整难度：
    - 连续高分 → 提升难度
    - 连续低分 → 降低难度
    - 波动大 → 保持当前难度
    """
    if not score_data:
        return {}

    current_difficulty = state.get("difficulty", "medium")
    trajectory = state.get("skill_progress", {}).get("improvement_trajectory", [])

    if len(trajectory) < 3:
        return {}

    recent = trajectory[-3:]
    avg_recent = sum(recent) / len(recent)

    increasing = recent[0] < recent[1] < recent[2]
    decreasing = recent[0] > recent[1] > recent[2]

    new_difficulty = current_difficulty

    if avg_recent >= DIFFICULTY_UP_THRESHOLD and increasing:
        idx = DIFFICULTY_LEVELS.index(current_difficulty)
        if idx < len(DIFFICULTY_LEVELS) - 1:
            new_difficulty = DIFFICULTY_LEVELS[idx + 1]
            logger.info(
                f"[Difficulty] Adaptive UP: {current_difficulty} → {new_difficulty} "
                f"(avg_recent={avg_recent:.1f})"
            )
    elif avg_recent <= DIFFICULTY_DOWN_THRESHOLD and decreasing:
        idx = DIFFICULTY_LEVELS.index(current_difficulty)
        if idx > 0:
            new_difficulty = DIFFICULTY_LEVELS[idx - 1]
            logger.info(
                f"[Difficulty] Adaptive DOWN: {current_difficulty} → {new_difficulty} "
                f"(avg_recent={avg_recent:.1f})"
            )

    if new_difficulty != current_difficulty:
        return {"difficulty": new_difficulty}

    return {}
