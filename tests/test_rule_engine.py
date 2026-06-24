"""
单元测试 - 规则引擎纠错

验证 correction_node 的规则引擎各层检测逻辑。
"""

import pytest
from agents.correction_node import (
    _check_basic_grammar,
    _check_grammar_structure,
    _check_chinglish,
    _determine_polish_level,
    _apply_polish,
    _generate_explanation,
    _generate_positive_feedback,
)


class TestBasicGrammar:
    """Level 1: 基础语法检测"""

    def test_lowercase_first_letter(self):
        results = _check_basic_grammar("hello world")
        assert len(results) >= 1
        assert any(r["error"]["type"] == "grammar" for r in results)

    def test_lowercase_i(self):
        results = _check_basic_grammar("i go to school")
        assert len(results) >= 1
        assert any("I" in r["error"]["issue"] for r in results)

    def test_missing_period(self):
        results = _check_basic_grammar("hello world")
        assert len(results) >= 1
        assert any(r["error"]["type"] == "punctuation" for r in results)

    def test_double_space(self):
        results = _check_basic_grammar("hello  world.")
        assert len(results) >= 1

    def test_correct_simple(self):
        results = _check_basic_grammar("Hello world.")
        # Should have minimal errors for a well-formed sentence
        assert len(results) == 0


class TestGrammarStructure:
    """Level 2: 语法结构检测"""

    def test_subject_verb_agreement(self):
        results = _check_grammar_structure("He have a car")
        assert len(results) >= 1
        assert any("主谓不一致" in r["error"]["issue"] for r in results)

    def test_article_a_vs_an(self):
        results = _check_grammar_structure("a apple")
        assert len(results) >= 1
        assert any("元音" in r["error"]["issue"] for r in results)

    def test_article_an_vs_a(self):
        results = _check_grammar_structure("an book")
        assert len(results) >= 1
        assert any("辅音" in r["error"]["issue"] for r in results)

    def test_irregular_verb_past(self):
        results = _check_grammar_structure("He go to school yesterday")
        assert len(results) >= 1
        assert any("时态" in r["error"]["issue"] for r in results)


class TestChinglish:
    """Level 3: 中式英语检测"""

    def test_very_like(self):
        results = _check_chinglish("I very like it")
        assert len(results) >= 1
        assert any("中式英语" in r["error"]["issue"] for r in results)

    def test_am_agree(self):
        results = _check_chinglish("I am agree with you")
        assert len(results) >= 1

    def test_go_school(self):
        results = _check_chinglish("I go school every day")
        assert len(results) >= 1


class TestPolishLevel:
    """Level 4: 表达升级"""

    def test_beginner_easy_is_basic(self):
        assert _determine_polish_level("beginner", "easy", "daily") == "basic"

    def test_advanced_is_advanced(self):
        assert _determine_polish_level("advanced", "medium", "daily") == "advanced"

    def test_hard_is_advanced(self):
        assert _determine_polish_level("intermediate", "hard", "daily") == "advanced"

    def test_default_is_enhanced(self):
        assert _determine_polish_level("intermediate", "medium", "daily") == "enhanced"

    def test_basic_polish_no_change(self):
        text = "I very like it"
        result = _apply_polish(text, "basic")
        assert result == text

    def test_enhanced_polish(self):
        text = "very good"
        result = _apply_polish(text, "enhanced")
        assert result != text  # Should be upgraded

    def test_advanced_polish(self):
        text = "very good"
        result = _apply_polish(text, "advanced")
        assert result != text  # Should be upgraded


class TestFeedback:
    """反馈生成测试"""

    def test_no_errors_positive(self):
        result = _generate_explanation([], [], "basic")
        assert "无明显语法错误" in result

    def test_with_errors(self):
        errors = [{"type": "grammar", "issue": "test"}]
        result = _generate_explanation(errors, [], "basic")
        assert "共发现 1 个问题" in result

    def test_positive_feedback_short(self):
        result = _generate_positive_feedback("hi", "beginner")
        assert "简洁明了" in result

    def test_positive_feedback_medium(self):
        result = _generate_positive_feedback("I like this very much", "intermediate")
        assert "表达不错" in result

    def test_positive_feedback_long(self):
        result = _generate_positive_feedback(
            "I would like to visit the museum this weekend", "advanced"
        )
        assert "表达不错" in result or "表达流畅" in result


class TestScenarios:
    """场景配置测试"""

    def test_list_scenarios(self):
        from agents.scenarios import list_available_scenarios
        scenarios = list_available_scenarios()
        assert len(scenarios) == 5
        assert "daily" in scenarios
        assert "interview" in scenarios

    def test_get_scenario_config(self):
        from agents.scenarios import get_scenario_config
        config = get_scenario_config("interview")
        assert config["name"] == "英语面试"
        assert "opening_lines" in config
        assert len(config["opening_lines"]) > 0

    def test_get_unknown_scenario_fallback(self):
        from agents.scenarios import get_scenario_config
        config = get_scenario_config("nonexistent")
        assert config["id"] == "daily"  # Falls back to daily

    def test_get_difficulty_config(self):
        from agents.scenarios import get_difficulty_config
        config = get_difficulty_config("restaurant", "easy")
        assert config["description"] == "简单点餐"
        assert "focus" in config

    def test_get_unknown_difficulty_fallback(self):
        from agents.scenarios import get_difficulty_config
        config = get_difficulty_config("daily", "nonexistent")
        assert config["description"] == "深入交流"  # Falls back to medium
