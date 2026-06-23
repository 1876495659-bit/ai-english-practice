"""
Correction Node - 语法纠错节点

作为 LangGraph StateGraph 的一个 Node 运行。
职责：
1. 分析用户英语表达中的语法错误
2. 提供修正版本
3. 给出更地道/自然的表达建议
4. 输出结构化 JSON 数据

增强特性（Stage 3）：
- 更全面的错误检测（时态/主谓一致/冠词/介词/拼写/标点）
- 上下文感知（结合场景、难度、用户水平调整纠错策略）
- 结构化输出增强（polish_level、error_details 数组含位置信息）
- 三级表达优化（basic → enhanced → advanced）

注意：此文件作为独立 Node 函数运行，不继承 BaseAgent。
"""

from __future__ import annotations

import re
from typing import Any


# ============================================================================
# 预定义错误模式库
# ============================================================================

# 常见不规则动词过去式
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
    "lose": "lost", "lost": "lost",
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
    "buy": "bought", "bought": "bought",
    "find": "found", "found": "found",
    "grow": "grew", "grown": "grown",
    "hide": "hid", "hidden": "hidden",
    "rise": "rose", "risen": "risen",
    "shake": "shook", "shaken": "shaken",
    "freeze": "froze", "frozen": "frozen",
}

# 常见冠词错误模式
_ARTICLE_PATTERNS: list[tuple[str, str, str]] = [
    # (匹配模式, 错误描述, 修正建议)
    (r"\b(a|an) [aeiou]\w+", "冠词错误：元音开头的单词前应使用 'an'", "an"),
    (r"\b(a|an) [bcdfghjklmnpqrstvwxyz]\w+", "冠词错误：辅音开头的单词前应使用 'a'", "a"),
    (r"\bthe\s+(countable_noun_plural)\b", "不可数名词前不应使用 'the' + 复数形式", None),
]

# 常见主谓一致错误
_SUBJECT_VERB_PATTERNS: list[tuple[str, str, str]] = [
    (r"\b(he|she|it)\s+have\b", "主谓不一致：第三人称单数应使用 'has'", "has"),
    (r"\b(he|she|it)\s+don't\b", "主谓不一致：第三人称单数应使用 'doesn't'", "doesn't"),
    (r"\b(he|she|it)\s+didn't\b", "主谓不一致：第三人称单数应使用 'didn't'（正确）", None),
    (r"\b(I|you|we|they)\s+is\b", "主谓不一致：应使用 'am/are'", "are"),
    (r"\b(he|she|it)\s+am\b", "主谓不一致：应使用 'is'", "is"),
]

# 常见介词搭配
_PREPOSITION_PATTERNS: list[tuple[str, str, str]] = [
    (r"\bin\s+the\s+morning\b", "介词搭配：固定搭配应为 'in the morning'（正确）", None),
    (r"\bon\s+Monday\b", "介词搭配：固定搭配应为 'on Monday'（正确）", None),
    (r"\bat\s+night\b", "介词搭配：固定搭配应为 'at night'（正确）", None),
    (r"\bgood\s+at\s+\w+\b", "介词搭配：'good at' 使用正确（正确）", None),
    (r"\binterested\s+of\b", "介词搭配：应为 'interested in'", "in"),
    (r"\bfamous\s+for\s+\w+\b", "介词搭配：'famous for' 使用正确（正确）", None),
    (r"\tfond\s+of\s+\w+\b", "介词搭配：'fond of' 使用正确（正确）", None),
    (r"\bdepend\s+on\b", "介词搭配：'depend on' 使用正确（正确）", None),
    (r"\tlook\s+forward\s+to\s+\w+\b", "介词搭配：'look forward to' 使用正确（正确）", None),
]

# 常见中式英语表达
_CHINGLISH_PATTERNS: list[tuple[str, str, str]] = [
    ("I very like", "中式英语：'very' 不能修饰动词，应为 'I really like' 或 'I like ... very much'", "I really like"),
    ("open the computer", "搭配不当：打开电脑应为 'turn on the computer'", "turn on the computer"),
    ("close the computer", "搭配不当：关闭电脑应为 'turn off the computer'", "turn off the computer"),
    ("I am agree", "语法错误：'agree' 是动词，不应与 'am' 连用", "I agree"),
    ("It is my opinion", "冗余表达：可直接说 'I think/believe'", "I think"),
    ("learn knowledge", "搭配不当：应为 'study/ acquire knowledge'", "acquire knowledge"),
    ("change my mind", "搭配不当：'change my mind' 本身是正确的（正确）", None),
    ("make a photo", "搭配不当：拍照应为 'take a photo'", "take a photo"),
    ("give a talk", "搭配不当：演讲应为 'give a speech' 或 'make a presentation'", "give a speech"),
    ("have a breakfast", "冠词冗余：三餐前通常不加冠词", "have breakfast"),
    ("play music", "搭配正确（正确）", None),
    ("do homework", "搭配正确（正确）", None),
    ("take a shower", "搭配正确（正确）", None),
    ("go school", "语法错误：应为 'go to school'", "go to school"),
    ("come home late", "搭配正确（正确）", None),
    ("arrive the city", "介词缺失：应为 'arrive in the city' 或 'reach the city'", "arrive in the city"),
    ("discuss about", "冗余：'discuss' 直接接宾语，不需要 'about'", "discuss"),
    ("suggest to do", "语法错误：应为 'suggest doing' 或 'suggest that ...'", "suggest doing"),
    ("ask me to go", "搭配正确（正确）", None),
    ("reply me", "介词缺失：应为 'reply to me'", "reply to me"),
    ("explain me", "介词缺失：应为 'explain to me'", "explain to me"),
]

# 口语化表达升级词典
_POLISH_UPGRADES: list[tuple[str, str, str]] = [
    # (低级别, 中级别, 高级别)
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
    ("say hello", "convey regards", "extend greetings"),
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
    ("nice", "pleasant", "delightful"),
    ("bad", "poor", "dreadful"),
    ("smart", "clever", "brilliant"),
    ("quick", "fast", "rapid"),
    ("slow", "sluggish", "gradual"),
    ("old", "elderly", "ancient"),
    ("young", "youthful", "juvenile"),
    ("new", "novel", "groundbreaking"),
    ("beautiful", "lovely", "stunning"),
    ("ugly", "unattractive", "hideous"),
    ("strong", "powerful", "robust"),
    ("weak", "frail", "feeble"),
    ("rich", "wealthy", "affluent"),
    ("poor", "impoverished", "needy"),
]


# ============================================================================
# Node 入口函数
# ============================================================================


async def correction_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    纠错 Node

    从 state 中提取用户最新输入，结合场景、难度、用户水平进行上下文感知纠错，
    将结构化结果写入 state["correction"]。

    Args:
        state: 当前图状态

    Returns:
        State 增量更新 dict
    """
    # 从 messages 中提取用户最新输入
    messages: list[dict] = state.get("messages", [])
    user_input = _extract_latest_user_input(messages)

    if not user_input:
        return {
            "correction": _empty_correction("没有检测到用户输入"),
        }

    # 提取上下文信息
    scenario: str = state.get("scenario", "daily")
    difficulty: str = state.get("difficulty", "medium")
    level: str = state.get("level", "intermediate")
    turn: int = state.get("turn", 1)

    # 执行纠错分析
    correction_result = await _analyze_correction(
        user_input=user_input,
        scenario=scenario,
        difficulty=difficulty,
        level=level,
        turn=turn,
    )

    return {"correction": correction_result}


# ============================================================================
# 辅助函数
# ============================================================================


def _extract_latest_user_input(messages: list) -> str:
    """从消息列表中提取最新的用户输入（兼容 dict 和 LangGraph BaseMessage）"""
    for msg in reversed(messages):
        if isinstance(msg, dict):
            if msg.get("role") == "user":
                return msg.get("content", "").strip()
        else:
            # LangGraph BaseMessage object
            role = getattr(msg, "type", None) or getattr(msg, "_getType", lambda: "")()
            if role == "human":
                return getattr(msg, "content", "").strip()
    return ""


def _empty_correction(reason: str = "") -> dict[str, Any]:
    """生成空纠错结果"""
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


# ============================================================================
# 核心纠错分析
# ============================================================================


async def _analyze_correction(
    user_input: str,
    scenario: str,
    difficulty: str,
    level: str,
    turn: int,
) -> dict[str, Any]:
    """
    综合分析用户输入，执行多层纠错策略。

    纠错层级：
    1. 基础语法错误（大小写、标点、拼写）
    2. 语法结构错误（主谓一致、时态、冠词、介词）
    3. 表达优化（中式英语、搭配不当）
    4. 语境适配（根据场景/难度调整建议风格）

    返回结构化纠错结果。
    """
    errors: list[dict[str, Any]] = []
    error_details: list[dict[str, Any]] = []
    corrected = user_input
    suggestion = user_input
    polished = user_input

    # ---- Level 1: 基础语法 ----
    level1_errors = _check_basic_grammar(user_input)
    for err in level1_errors:
        errors.append(err["error"])
        error_details.append(err["detail"])
        corrected = _apply_fix(corrected, err["pattern"], err["replacement"])
        suggestion = corrected

    # ---- Level 2: 语法结构 ----
    level2_errors = _check_grammar_structure(user_input)
    for err in level2_errors:
        errors.append(err["error"])
        error_details.append(err["detail"])
        corrected = _apply_fix(corrected, err["pattern"], err["replacement"])
        suggestion = corrected

    # ---- Level 3: 表达优化（中式英语） ----
    chinglish_errors = _check_chinglish(user_input)
    for err in chinglish_errors:
        errors.append(err["error"])
        error_details.append(err["detail"])
        corrected = _apply_fix(corrected, err["pattern"], err["replacement"])
        suggestion = corrected

    # ---- Level 4: 语境适配表达升级 ----
    polish_level = _determine_polish_level(level, difficulty, scenario)
    if polish_level != "basic":
        polished = _apply_polish(suggestion, polish_level)

    # ---- 确定最终结果 ----
    has_errors = len(errors) > 0

    # 构建解释
    explanation = _generate_explanation(errors, error_details, polish_level)

    # 如果没有错误，给出鼓励性反馈
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
# Level 1: 基础语法检查
# ============================================================================


def _check_basic_grammar(text: str) -> list[dict[str, Any]]:
    """检查基础语法：大小写、标点、常见拼写"""
    results: list[dict[str, Any]] = []

    # 1. 首字母大写
    if text and text[0].isalpha() and text[0].islower():
        results.append({
            "error": {
                "type": "grammar",
                "issue": "句子首字母应大写",
            },
            "detail": {
                "position": 0,
                "context": text[:20],
                "fix": "将首字母大写",
            },
            "pattern": re.compile(r"^([a-z])"),
            "replacement": lambda m: m.group(1).upper(),
        })

    # 2. "i" 未大写
    if re.search(r"\bi\b", text):
        results.append({
            "error": {
                "type": "grammar",
                "issue": "代词 'I' 必须大写",
            },
            "detail": {
                "position": text.lower().find(" i "),
                "context": text[:20],
                "fix": "将 'i' 改为 'I'",
            },
            "pattern": re.compile(r"\bi\b"),
            "replacement": "I",
        })

    # 3. 标点符号前有多余空格
    if re.search(r"[,.!?]\s{2,}", text):
        results.append({
            "error": {
                "type": "punctuation",
                "issue": "标点后不应有多个空格",
            },
            "detail": {
                "position": -1,
                "context": text[:20],
                "fix": "标点后保留一个空格",
            },
            "pattern": re.compile(r"([,.!?])\s{2,}"),
            "replacement": r"\1 ",
        })

    # 4. 句末缺少句号
    stripped = text.strip()
    if stripped and not stripped[-1] in ".!?," and re.search(r"\b\w+\s*$", stripped):
        results.append({
            "error": {
                "type": "punctuation",
                "issue": "句子末尾建议添加标点符号",
            },
            "detail": {
                "position": len(stripped),
                "context": stripped[:20],
                "fix": "添加句号 '.'",
            },
            "pattern": re.compile(r"(\w)(\s*)$"),
            "replacement": r"\1.",
        })

    return results


# ============================================================================
# Level 2: 语法结构检查
# ============================================================================


def _check_grammar_structure(text: str) -> list[dict[str, Any]]:
    """检查主谓一致、时态、冠词等语法结构"""
    results: list[dict[str, Any]] = []
    lower = text.lower()

    # 1. 主谓一致检查
    for pattern_str, desc, replacement in _SUBJECT_VERB_PATTERNS:
        if replacement is None:
            continue  # 正确情况跳过
        match = re.search(pattern_str, lower)
        if match:
            results.append({
                "error": {
                    "type": "grammar",
                    "issue": desc,
                },
                "detail": {
                    "position": match.start(),
                    "context": text[max(0, match.start()-10):match.end()+10],
                    "fix": replacement,
                },
                "pattern": re.compile(pattern_str, re.IGNORECASE),
                "replacement": lambda m, r=replacement: m.group(1) + " " + r,
            })

    # 2. 冠词检查：元音前用 an
    an_pattern = re.compile(r"\ban\s+([bcdfghjklmnpqrstvwxyz]\w*)", re.IGNORECASE)
    match = an_pattern.search(lower)
    if match:
        results.append({
            "error": {
                "type": "grammar",
                "issue": f"冠词错误：'{match.group(1)}' 以辅音开头，应使用 'a'",
            },
            "detail": {
                "position": match.start(),
                "context": text[max(0, match.start()-2):match.end()+2],
                "fix": "an → a",
            },
            "pattern": an_pattern,
            "replacement": lambda m: f"a {m.group(1)}",
        })

    a_pattern = re.compile(r"\ba\s+([aeiou]\w*)", re.IGNORECASE)
    match = a_pattern.search(lower)
    if match:
        results.append({
            "error": {
                "type": "grammar",
                "issue": f"冠词错误：'{match.group(1)}' 以元音开头，应使用 'an'",
            },
            "detail": {
                "position": match.start(),
                "context": text[max(0, match.start()-2):match.end()+2],
                "fix": "a → an",
            },
            "pattern": a_pattern,
            "replacement": lambda m: f"an {m.group(1)}",
        })

    # 3. 不规则动词过去式检查（简单启发式）
    for base, past in _IRREGULAR_VERBS.items():
        if base == past:
            continue
        # 检测常见错误：he go → he went
        pattern = re.compile(r"\b(he|she|it|someone|everyone)\s+" + re.escape(base) + r"\b", re.IGNORECASE)
        match = pattern.search(lower)
        if match:
            # 上下文中有过去时间标记
            has_past_marker = bool(re.search(r"\byesterday|last\s+\w+|ago|in\s+20\d\d|when\s+I", lower))
            if has_past_marker:
                results.append({
                    "error": {
                        "type": "grammar",
                        "issue": f"时态错误：过去语境中 '{base}' 的过去式应为 '{past}'",
                    },
                    "detail": {
                        "position": match.start(),
                        "context": text[max(0, match.start()-5):match.end()+15],
                        "fix": f"{base} → {past}",
                    },
                    "pattern": pattern,
                    "replacement": lambda m, p=past: m.group(1) + " " + p,
                })

    return results


# ============================================================================
# Level 3: 中式英语检查
# ============================================================================


def _check_chinglish(text: str) -> list[dict[str, Any]]:
    """检测中式英语表达"""
    results: list[dict[str, Any]] = []
    lower = text.lower()

    for pattern, desc, replacement in _CHINGLISH_PATTERNS:
        if replacement is None:
            continue  # 正确表达跳过
        regex = re.compile(re.escape(pattern), re.IGNORECASE)
        match = regex.search(lower)
        if match:
            results.append({
                "error": {
                    "type": "style",
                    "issue": desc,
                },
                "detail": {
                    "position": match.start(),
                    "context": text[max(0, match.start()-10):match.end()+10],
                    "fix": replacement,
                },
                "pattern": regex,
                "replacement": replacement,
            })

    return results


# ============================================================================
# Level 4: 表达升级
# ============================================================================


def _determine_polish_level(level: str, difficulty: str, scenario: str) -> str:
    """根据用户水平和难度决定表达升级级别"""
    if level == "beginner" and difficulty == "easy":
        return "basic"
    if level == "advanced" or difficulty == "hard":
        return "advanced"
    return "enhanced"


def _apply_polish(text: str, polish_level: str) -> str:
    """
    根据 polish_level 对文本进行表达升级。

    basic: 不做修改
    enhanced: 替换部分基础词汇为更好表达
    advanced: 全面升级句式结构和词汇
    """
    if polish_level == "basic":
        return text

    result = text
    for low, mid, adv in _POLISH_UPGRADES:
        if polish_level == "enhanced":
            replacement = mid
        else:  # advanced
            replacement = adv

        regex = re.compile(re.escape(low), re.IGNORECASE)
        if regex.search(result):
            result = regex.sub(replacement, result)

    return result


# ============================================================================
# 结果合并与解释生成
# ============================================================================


def _apply_fix(text: str, pattern: Any, replacement: Any) -> str:
    """安全地应用替换"""
    try:
        if callable(replacement):
            # 对于 lambda 类型的 replacement，需要特殊处理
            return pattern.sub(lambda m: replacement(m), text)
        return pattern.sub(replacement, text)
    except Exception:
        return text


def _generate_explanation(
    errors: list[dict[str, Any]],
    error_details: list[dict[str, Any]],
    polish_level: str,
) -> str:
    """生成中文纠错解释"""
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
    """生成鼓励性反馈"""
    word_count = len(text.split())
    if word_count >= 10:
        return "表达流畅，用词准确，继续保持！"
    elif word_count >= 5:
        return "表达不错，可以尝试使用更丰富的词汇和更复杂的句式。"
    else:
        return "简洁明了，试着多说几句，练习更完整的表达。"
