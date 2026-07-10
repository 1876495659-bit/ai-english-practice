"""
Correction Node - 语法纠错节点

作为 LangGraph StateGraph 的一个 Node 运行。
职责：
1. 分析用户英语表达中的语法错误
2. 提供修正版本
3. 给出更地道/自然的表达建议
4. 输出结构化 JSON 数据

双通道设计（Stage 4）：
- 规则引擎通道：快速、确定性、无需 API（4 层正则检测）
- LLM 通道：智能、上下文感知、需要 API 调用
- 默认使用规则引擎，LLM 失败时自动回退

注意：此文件作为独立 Node 函数运行，不继承 BaseAgent。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from agents.llm_client import call_llm_json
from agents.prompts_loader import load_prompt
from agents.utils import extract_latest_user_input
from config.settings import settings

logger = logging.getLogger(__name__)

# ============================================================================
# 预定义错误模式库（规则引擎）
# ============================================================================

_IRREGULAR_VERBS: dict[str, str] = {
    "go": "went", "goes": "went", "gone": "went",
    "eat": "ate", "eaten": "eaten",
    "see": "saw", "seen": "seen",
    "come": "came", "came": "came",
    "take": "took", "taken": "taken",
    "give": "gave", "given": "given",
    "make": "made", "made": "made",
    "know": "knew", "known": "known",
    "think": "thought", "thought": "thought",
    "tell": "told", "told": "told",
    "become": "became", "became": "became",
    "leave": "left", "left": "left",
    "feel": "felt", "felt": "felt",
    "put": "put", "put": "put",
    "bring": "brought", "brought": "brought",
    "begin": "began", "begun": "begun",
    "keep": "kept", "kept": "kept",
    "hold": "held", "held": "held",
    "write": "wrote", "written": "written",
    "stand": "stood", "stood": "stood",
    "lose": "lost", "lost": "lost",
    "pay": "paid", "paid": "paid",
    "meet": "met", "met": "met",
    "send": "sent", "sent": "sent",
    "build": "built", "built": "built",
    "spend": "spent", "spent": "spent",
    "run": "ran", "run": "run",
    "move": "moved", "moved": "moved",
    "live": "lived", "lived": "lived",
    "mean": "meant", "meant": "meant",
    "understand": "understood", "understood": "understood",
    "read": "read", "read": "read",
    "grow": "grew", "grown": "grown",
    "fall": "fell", "fallen": "fallen",
    "sit": "sat", "sat": "sat",
    "cut": "cut", "cut": "cut",
    "break": "broke", "broken": "broken",
    "choose": "chose", "chosen": "chosen",
    "drive": "drove", "driven": "driven",
    "sing": "sang", "sung": "sung",
    "swim": "swam", "swum": "swum",
    "throw": "threw", "thrown": "thrown",
    "wake": "woke", "woken": "woken",
    "wear": "wore", "worn": "worn",
    "win": "won", "won": "won",
    "forget": "forgot", "forgotten": "forgotten",
    "buy": "bought", "bought": "bought",
    "catch": "caught", "caught": "caught",
    "teach": "taught", "taught": "taught",
    "find": "found", "found": "found",
    "hide": "hid", "hidden": "hidden",
    "rise": "rose", "risen": "risen",
    "shake": "shook", "shaken": "shaken",
    "freeze": "froze", "frozen": "frozen",
}

_SUBJECT_VERB_PATTERNS: list[tuple[str, str, str]] = [
    (r"\b(he|she|it)\s+have\b", "主谓不一致：第三人称单数应使用 'has'", "has"),
    (r"\b(he|she|it)\s+don't\b", "主谓不一致：第三人称单数应使用 'doesn't'", "doesn't"),
    (r"\b(he|she|it)\s+didn't\b", "主谓不一致：第三人称单数应使用 'didn't'（正确）", None),
    (r"\b(I|you|we|they)\s+is\b", "主谓不一致：应使用 'am/are'", "are"),
    (r"\b(he|she|it)\s+am\b", "主谓不一致：应使用 'is'", "is"),
]

_CHINGLISH_PATTERNS: list[tuple[str, str, str]] = [
    ("I very like", "中式英语：'very' 不能修饰动词，应为 'I really like' 或 'I like ... very much'", "I really like"),
    ("open the computer", "搭配不当：打开电脑应为 'turn on the computer'", "turn on the computer"),
    ("close the computer", "搭配不当：关闭电脑应为 'turn off the computer'", "turn off the computer"),
    ("I am agree", "语法错误：'agree' 是动词，不应与 'am' 连用", "I agree"),
    ("It is my opinion", "冗余表达：可直接说 'I think/believe'", "I think"),
    ("learn knowledge", "搭配不当：应为 'study/ acquire knowledge'", "acquire knowledge"),
    ("make a photo", "搭配不当：拍照应为 'take a photo'", "take a photo"),
    ("give a talk", "搭配不当：演讲应为 'give a speech' 或 'make a presentation'", "give a speech"),
    ("have a breakfast", "冠词冗余：三餐前通常不加冠词", "have breakfast"),
    ("go school", "语法错误：应为 'go to school'", "go to school"),
    ("arrive the city", "介词缺失：应为 'arrive in the city' 或 'reach the city'", "arrive in the city"),
    ("discuss about", "冗余：'discuss' 直接接宾语，不需要 'about'", "discuss"),
    ("suggest to do", "语法错误：应为 'suggest doing' 或 'suggest that ...'", "suggest doing"),
    ("reply me", "介词缺失：应为 'reply to me'", "reply to me"),
    ("explain me", "介词缺失：应为 'explain to me'", "explain to me"),
    ("interested of", "介词搭配：应为 'interested in'", "in"),
    ("very good", "过于简单：可以尝试更丰富的词汇", "really great"),
    ("very bad", "过于简单：可以尝试更丰富的词汇", "not great"),
    ("very happy", "过于简单：可以尝试更丰富的词汇", "really pleased"),
    ("very tired", "过于简单：可以尝试更丰富的词汇", "pretty exhausted"),
    ("I want", "语气较直接：建议用 'I would like' 更礼貌", "I would like"),
]

_POLISH_UPGRADES: list[tuple[str, str, str]] = [
    ("very good", "really great", "outstanding"),
    ("very bad", "not great", "terrible"),
    ("very big", "quite large", "enormous"),
    ("very small", "pretty tiny", "minute"),
    ("very happy", "really pleased", "thrilled"),
    ("very sad", "quite upset", "devastated"),
    ("very tired", "pretty exhausted", "drained"),
    ("very hungry", "quite famished", "starving"),
    ("very cold", "pretty chilly", "freezing"),
    ("very hot", "quite sweltering", "boiling"),
    ("very interesting", "really fascinating", "captivating"),
    ("very important", "quite crucial", "paramount"),
    ("very difficult", "pretty challenging", "formidable"),
    ("very easy", "quite simple", "effortless"),
    ("I think", "In my opinion", "From my perspective"),
    ("I want", "I would like", "I'd be interested in"),
    ("I like", "I'm fond of", "I'm passionate about"),
    ("a lot", "quite a bit", "immensely"),
    ("get better", "improve", "excel"),
    ("get worse", "deteriorate", "decline"),
    ("help me", "assist me", "lend me a hand"),
    ("need to", "have to", "must"),
    ("try to", "attempt to", "endeavor to"),
    ("use", "utilize", "leverage"),
    ("start", "begin", "commence"),
    ("end", "finish", "conclude"),
    ("buy", "purchase", "acquire"),
    ("help", "assist", "facilitate"),
    ("show", "demonstrate", "illustrate"),
    ("tell", "inform", "notify"),
    ("ask", "request", "inquire"),
    ("answer", "respond", "rejoin"),
    ("big", "large", "substantial"),
    ("small", "tiny", "modest"),
    ("nice", "pleasant", "delightful"),
    ("bad", "poor", "dreadful"),
    ("smart", "clever", "brilliant"),
    ("quick", "fast", "rapid"),
    ("slow", "sluggish", "gradual"),
    ("old", "elderly", "ancient"),
    ("young", "youthful", "juvenile"),
    ("new", "novel", "groundbreaking"),
    ("beautiful", "lovely", "stunning"),
    ("strong", "powerful", "robust"),
    ("weak", "frail", "feeble"),
    ("rich", "wealthy", "affluent"),
    ("poor", "impoverished", "needy"),
]


# ============================================================================
# Node 入口
# ============================================================================


def _empty_correction(reason: str = "") -> dict[str, Any]:
    """返回空的纠错结果"""
    return {
        "original": "",
        "errors": [],
        "error_details": [],
        "corrected": "",
        "suggestion": "",
        "polished": "",
        "explanation": reason,
        "has_errors": False,
        "polish_level": "basic",
    }


async def correction_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    纠错 Node

    双通道策略：
    1. 规则引擎（默认）：快速、确定性纠错
    2. LLM 通道（可选）：智能上下文纠错，通过 LLM_MODE_CORRECTION 配置开关

    Args:
        state: 当前图状态

    Returns:
        State 增量更新 dict
    """
    messages: list = state.get("messages", [])
    # LangGraph 1.x 的 add_messages reducer 在图管道中可能导致 messages 丢失，
    # 优先从 state.user_input 获取（由 API 层或上游节点注入）
    user_input = state.get("user_input", "").strip()
    if not user_input:
        user_input = extract_latest_user_input(messages)

    if not user_input:
        return {"correction": _empty_correction("没有检测到用户输入")}

    scenario: str = state.get("scenario", "daily")
    difficulty: str = state.get("difficulty", "medium")
    level: str = state.get("level", "intermediate")
    turn: int = state.get("turn", 1)

    # 检查是否启用 LLM 纠错
    use_llm = getattr(settings, "llm_mode_correction", False)

    if use_llm:
        correction = await _llm_correction(
            user_input=user_input,
            scenario=scenario,
            level=level,
        )
        if correction:
            logger.info(f"[CorrectionNode] LLM correction applied (turn={turn})")
            return {"correction": correction}
        logger.warning("[CorrectionNode] LLM correction failed, falling back to rule engine")

    # 规则引擎通道（始终可用）
    correction = await _rule_engine_correction(
        user_input=user_input,
        scenario=scenario,
        difficulty=difficulty,
        level=level,
        turn=turn,
    )
    return {"correction": correction}


# ============================================================================
# LLM 纠错通道
# ============================================================================


async def _llm_correction(
    user_input: str,
    scenario: str,
    level: str,
) -> dict[str, Any] | None:
    """
    使用 LLM 进行智能纠错，从 prompts/correction.txt 加载模板。
    """
    try:
        prompt = load_prompt(
            "correction",
            user_input=user_input,
            scenario=scenario,
            level=level,
        )
        result = await call_llm_json([
            {"role": "user", "content": prompt},
        ], temperature=0.3, max_tokens=512)
        return result
    except (json.JSONDecodeError, KeyError, Exception) as e:
        logger.warning(f"[CorrectionNode] LLM JSON parse failed: {e}")
        return None


# ============================================================================
# 规则引擎通道
# ============================================================================


async def _rule_engine_correction(
    user_input: str,
    scenario: str,
    difficulty: str,
    level: str,
    turn: int,
) -> dict[str, Any]:
    """基于规则的纠错引擎"""
    errors: list[dict[str, Any]] = []
    error_details: list[dict[str, Any]] = []
    corrected = user_input
    suggestion = user_input
    polished = user_input

    for err in _check_basic_grammar(user_input):
        errors.append(err["error"])
        error_details.append(err["detail"])
        corrected = _apply_fix(corrected, err["pattern"], err["replacement"])
        suggestion = corrected

    for err in _check_grammar_structure(user_input):
        errors.append(err["error"])
        error_details.append(err["detail"])
        corrected = _apply_fix(corrected, err["pattern"], err["replacement"])
        suggestion = corrected

    for err in _check_chinglish(user_input):
        errors.append(err["error"])
        error_details.append(err["detail"])
        corrected = _apply_fix(corrected, err["pattern"], err["replacement"])
        suggestion = corrected

    polish_level = _determine_polish_level(level, difficulty, scenario)
    if polish_level != "basic":
        polished = _apply_polish(suggestion, polish_level)

    has_errors = len(errors) > 0
    explanation = _generate_explanation(errors, error_details, polish_level)
    if not has_errors and turn > 0:
        explanation = _generate_positive_feedback(user_input, level)

    return {
        "original": user_input,
        "errors": errors,
        "error_details": error_details,
        "corrected": corrected,
        "suggestion": suggestion,
        "polished": polished,
        "explanation": explanation,
        "has_errors": has_errors,
        "polish_level": polish_level,
    }


# ============================================================================
# 规则引擎实现
# ============================================================================


def _check_basic_grammar(text: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    if text and text[0].isalpha() and text[0].islower():
        results.append({
            "error": {"type": "grammar", "issue": "句子首字母应大写"},
            "detail": {"position": 0, "context": text[:20], "fix": "将首字母大写"},
            "pattern": re.compile(r"^([a-z])"),
            "replacement": lambda m: m.group(1).upper(),
        })

    if re.search(r"\bi\b", text):
        results.append({
            "error": {"type": "grammar", "issue": "代词 'I' 必须大写"},
            "detail": {"position": text.lower().find(" i "), "context": text[:20], "fix": "将 'i' 改为 'I'"},
            "pattern": re.compile(r"\bi\b"),
            "replacement": "I",
        })

    if re.search(r"[,.!?]\s{2,}", text):
        results.append({
            "error": {"type": "punctuation", "issue": "标点后不应有多个空格"},
            "detail": {"position": -1, "context": text[:20], "fix": "标点后保留一个空格"},
            "pattern": re.compile(r"([,.!?])\s{2,}"),
            "replacement": r"\1 ",
        })

    stripped = text.strip()
    if stripped and not stripped[-1] in ".!?," and re.search(r"\b\w+\s*$", stripped):
        results.append({
            "error": {"type": "punctuation", "issue": "句子末尾建议添加标点符号"},
            "detail": {"position": len(stripped), "context": stripped[:20], "fix": "添加句号 '.'"},
            "pattern": re.compile(r"(\w)(\s*)$"),
            "replacement": r"\1.",
        })

    return results


def _check_grammar_structure(text: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    lower = text.lower()

    for pattern_str, desc, replacement in _SUBJECT_VERB_PATTERNS:
        if replacement is None:
            continue
        match = re.search(pattern_str, lower)
        if match:
            results.append({
                "error": {"type": "grammar", "issue": desc},
                "detail": {"position": match.start(), "context": text[max(0, match.start()-10):match.end()+10], "fix": replacement},
                "pattern": re.compile(pattern_str, re.IGNORECASE),
                "replacement": lambda m, r=replacement: m.group(1) + " " + r,
            })

    an_pattern = re.compile(r"\ban\s+([bcdfghjklmnpqrstvwxyz]\w*)", re.IGNORECASE)
    match = an_pattern.search(lower)
    if match:
        results.append({
            "error": {"type": "grammar", "issue": f"冠词错误：'{match.group(1)}' 以辅音开头，应使用 'a'"},
            "detail": {"position": match.start(), "context": text[max(0, match.start()-2):match.end()+2], "fix": "an → a"},
            "pattern": an_pattern,
            "replacement": lambda m: f"a {m.group(1)}",
        })

    a_pattern = re.compile(r"\ba\s+([aeiou]\w*)", re.IGNORECASE)
    match = a_pattern.search(lower)
    if match:
        results.append({
            "error": {"type": "grammar", "issue": f"冠词错误：'{match.group(1)}' 以元音开头，应使用 'an'"},
            "detail": {"position": match.start(), "context": text[max(0, match.start()-2):match.end()+2], "fix": "a → an"},
            "pattern": a_pattern,
            "replacement": lambda m: f"an {m.group(1)}",
        })

    for base, past in _IRREGULAR_VERBS.items():
        if base == past:
            continue
        pattern = re.compile(r"\b(he|she|it|someone|everyone)\s+" + re.escape(base) + r"\b", re.IGNORECASE)
        match = pattern.search(lower)
        if match:
            has_past_marker = bool(re.search(r"\byesterday|last\s+\w+|ago|in\s+20\d\d|when\s+I", lower))
            if has_past_marker:
                results.append({
                    "error": {"type": "grammar", "issue": f"时态错误：过去语境中 '{base}' 的过去式应为 '{past}'"},
                    "detail": {"position": match.start(), "context": text[max(0, match.start()-5):match.end()+15], "fix": f"{base} → {past}"},
                    "pattern": pattern,
                    "replacement": lambda m, p=past: m.group(1) + " " + p,
                })

    return results


def _check_chinglish(text: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    lower = text.lower()

    for pattern, desc, replacement in _CHINGLISH_PATTERNS:
        if replacement is None:
            continue
        regex = re.compile(re.escape(pattern), re.IGNORECASE)
        match = regex.search(lower)
        if match:
            results.append({
                "error": {"type": "style", "issue": desc},
                "detail": {"position": match.start(), "context": text[max(0, match.start()-10):match.end()+10], "fix": replacement},
                "pattern": regex,
                "replacement": replacement,
            })

    return results


def _determine_polish_level(level: str, difficulty: str, scenario: str) -> str:
    if level == "beginner" and difficulty == "easy":
        return "basic"
    if level == "advanced" or difficulty == "hard":
        return "advanced"
    return "enhanced"


def _apply_polish(text: str, polish_level: str) -> str:
    if polish_level == "basic":
        return text
    result = text
    for low, mid, adv in _POLISH_UPGRADES:
        replacement = mid if polish_level == "enhanced" else adv
        regex = re.compile(re.escape(low), re.IGNORECASE)
        if regex.search(result):
            result = regex.sub(replacement, result)
    return result


def _apply_fix(text: str, pattern: Any, replacement: Any) -> str:
    try:
        if callable(replacement):
            return pattern.sub(lambda m: replacement(m), text)
        return pattern.sub(replacement, text)
    except Exception:
        return text


def _generate_explanation(errors: list, error_details: list, polish_level: str) -> str:
    if not errors:
        return "无明显语法错误，表达良好！"
    type_counts: dict[str, int] = {}
    for err in errors:
        etype = err.get("type", "unknown")
        type_counts[etype] = type_counts.get(etype, 0) + 1
    parts: list[str] = [f"共发现 {len(errors)} 个问题："]
    if "grammar" in type_counts:
        parts.append(f"- 语法 {type_counts['grammar']} 处：注意基本语法规则")
    if "punctuation" in type_counts:
        parts.append(f"- 标点 {type_counts['punctuation']} 处：注意标点符号使用")
    if "style" in type_counts:
        parts.append(f"- 表达 {type_counts['style']} 处：有更地道的说法")
    if polish_level != "basic":
        parts.append(f"- 表达升级（{polish_level}）：提供了更高级的表达建议")
    return "\n".join(parts)


def _generate_positive_feedback(text: str, level: str) -> str:
    word_count = len(text.split())
    if word_count >= 10:
        return "表达流畅，用词准确，继续保持！"
    elif word_count >= 5:
        return "表达不错，可以尝试使用更丰富的词汇和更复杂的句式。"
    else:
        return "简洁明了，试着多说几句，练习更完整的表达。"
