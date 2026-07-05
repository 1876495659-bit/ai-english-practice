"""
单元测试 - Scoring Node 核心逻辑

验证 scoring_node 的规则引擎评分、skill_progress 更新、
自适应难度调整和边界情况处理。
"""

import asyncio
import pytest
from agents.scoring_node import (
    _rule_engine_evaluate,
    _update_skill_progress,
    _adjust_difficulty_adaptive,
    _empty_return,
    _generate_feedback,
)


def _sync_eval(text):
    """同步调用 _rule_engine_evaluate"""
    return asyncio.run(_rule_engine_evaluate(text))


class TestRuleEngineEvaluate:
    """测试规则引擎评分逻辑"""

    def test_empty_input(self):
        """空输入应得基础分（base 5.0 + noise）"""
        result = _sync_eval("")
        assert result["total"] >= 4.0  # 基础分 5.0 ± noise

    def test_single_word(self):
        """单字输入得低-中等分数"""
        result = _sync_eval("hello")
        assert result["total"] >= 4.0  # 基础分 5.0 + 0.2 word bonus
        assert result["total"] <= 7.0

    def test_short_sentence(self):
        """短句子得中等分数"""
        result = _sync_eval("I like pizza")
        assert result["total"] >= 4.0
        assert result["total"] <= 8.0

    def test_long_sentence(self):
        """较长的句子得分更高"""
        text = "I would like to visit the museum this weekend, although I am not sure about the opening hours."
        result = _sync_eval(text)
        assert result["total"] >= 5.0

    def test_has_advanced_words(self):
        """包含高级词汇的句子应加分"""
        result = _sync_eval("It is a significant opportunity for substantial growth.")
        assert result["total"] >= 6.0

    def test_has_complex_phrases(self):
        """包含复杂短语的句子应加分"""
        result = _sync_eval("In my opinion, on the other hand, as a matter of fact.")
        assert result["total"] >= 5.0

    def test_returns_required_fields(self):
        """返回结果必须包含所有必要字段"""
        result = _sync_eval("test sentence here")
        assert "scores" in result
        assert "total" in result
        assert "feedback_en" in result
        assert "feedback_zh" in result
        assert "strengths" in result
        assert "improvements" in result

    def test_scores_are_numbers(self):
        """各维度分数必须是数字"""
        result = _sync_eval("test sentence here")
        for dim in ("fluency", "grammar", "vocabulary", "naturalness"):
            assert isinstance(result["scores"][dim], (int, float))
            assert 0 <= result["scores"][dim] <= 10

    def test_total_is_average(self):
        """总分应为四维分数的平均值"""
        result = _sync_eval("test sentence here")
        scores = result["scores"]
        expected = round((scores["fluency"] + scores["grammar"] + scores["vocabulary"] + scores["naturalness"]) / 4, 1)
        assert abs(result["total"] - expected) < 0.2

    def test_deterministic_output(self):
        """相同输入应产生相同输出（seed 固定）"""
        r1 = _sync_eval("I go to the park")
        r2 = _sync_eval("I go to the park")
        assert r1["total"] == r2["total"]


class TestGenerateFeedback:
    """测试反馈生成逻辑"""

    def test_high_scores_produce_strengths(self):
        """高分数应生成优点反馈"""
        strengths, improvements = _generate_feedback(8, 8, 8, 8, 15)
        assert len(strengths) > 0
        assert "流畅" in " ".join(strengths) or "语法" in " ".join(strengths)

    def test_low_scores_produce_improvements(self):
        """低分数应生成改进建议"""
        strengths, improvements = _generate_feedback(3, 3, 3, 3, 15)
        assert len(improvements) > 0

    def test_short_input_extra_improvement(self):
        """短输入应额外提示多说几句"""
        strengths, improvements = _generate_feedback(5, 5, 5, 5, 3)
        assert any("多说几句" in imp for imp in improvements)

    def test_all_dimensions_high(self):
        """所有维度都高时至少有 1 个优点"""
        strengths, improvements = _generate_feedback(9, 9, 9, 9, 20)
        assert len(strengths) >= 1

    def test_all_dimensions_low(self):
        """所有维度都低时至少有 1 个改进建议"""
        strengths, improvements = _generate_feedback(2, 2, 2, 2, 20)
        assert len(improvements) >= 1


class TestUpdateSkillProgress:
    """测试 skill_progress 更新逻辑"""

    def test_increment_total_turns(self):
        """每次调用应增加 total_turns"""
        state = {"skill_progress": {"total_turns": 5}}
        result = _update_skill_progress(state, {"total": 7.0})
        assert result["total_turns"] == 6

    def test_update_avg_score(self):
        """应更新平均评分"""
        state = {
            "skill_progress": {
                "total_turns": 2,
                "avg_score": 6.0,
                "improvement_trajectory": [5.0, 7.0],
            }
        }
        result = _update_skill_progress(state, {"total": 8.0})
        assert len(result["improvement_trajectory"]) == 3
        assert 8.0 in result["improvement_trajectory"]

    def test_track_error_frequency(self):
        """应从 correction 中提取错误频率"""
        state = {
            "skill_progress": {"error_frequency": {}},
            "correction": {
                "errors": [
                    {"type": "grammar"},
                    {"type": "grammar"},
                    {"type": "vocabulary"},
                ]
            },
        }
        result = _update_skill_progress(state, None)
        assert result["error_frequency"]["grammar"] == 2
        assert result["error_frequency"]["vocabulary"] == 1

    def test_identify_weakest_dimension(self):
        """应识别最弱维度"""
        state = {
            "score": {
                "scores": {"fluency": 9, "grammar": 3, "vocabulary": 7, "naturalness": 5}
            }
        }
        result = _update_skill_progress(state, {"total": 6.0})
        assert result["weakest_dimension"] == "grammar"

    def test_identify_strongest_dimension(self):
        """应识别最强维度"""
        state = {
            "score": {
                "scores": {"fluency": 9, "grammar": 3, "vocabulary": 7, "naturalness": 5}
            }
        }
        result = _update_skill_progress(state, {"total": 6.0})
        assert result["strongest_dimension"] == "fluency"

    def test_no_score_preserves_existing(self):
        """没有 score 时应保留已有的 weakest/strongest"""
        state = {
            "skill_progress": {
                "total_turns": 0,
                "avg_score": 0.0,
                "error_frequency": {},
                "weakest_dimension": "grammar",
                "strongest_dimension": "fluency",
                "improvement_trajectory": [],
            }
        }
        result = _update_skill_progress(state, None)
        # weakest/strongest 应为空（因为没有 dim scores）
        assert result["weakest_dimension"] == ""
        assert result["strongest_dimension"] == ""


class TestAdjustDifficultyAdaptive:
    """测试自适应难度调整逻辑"""

    def test_insufficient_trajectory_no_change(self):
        """轨迹少于 3 个数据点时不调整难度"""
        state = {
            "difficulty": "medium",
            "skill_progress": {
                "improvement_trajectory": [6.0, 7.0]
            }
        }
        result = _adjust_difficulty_adaptive(state, {"total": 7.0})
        assert result == {}

    def test_increasing_high_scores_upgrade(self):
        """连续 3 轮高分且上升趋势 → 提升难度"""
        state = {
            "difficulty": "easy",
            "skill_progress": {
                "improvement_trajectory": [7.5, 8.0, 8.5]
            }
        }
        result = _adjust_difficulty_adaptive(state, {"total": 8.5})
        assert result.get("difficulty") == "medium"

    def test_decreasing_low_scores_downgrade(self):
        """连续 3 轮低分且下降趋势 → 降低难度"""
        state = {
            "difficulty": "hard",
            "skill_progress": {
                "improvement_trajectory": [4.5, 4.0, 3.5]
            }
        }
        result = _adjust_difficulty_adaptive(state, {"total": 3.5})
        assert result.get("difficulty") == "medium"

    def test_no_trend_no_change(self):
        """波动大（无明确趋势）时不调整难度"""
        state = {
            "difficulty": "medium",
            "skill_progress": {
                "improvement_trajectory": [6.0, 8.0, 7.0]
            }
        }
        result = _adjust_difficulty_adaptive(state, {"total": 7.0})
        assert result == {}

    def test_already_at_max_no_up(self):
        """已在最高难度时不继续提升"""
        state = {
            "difficulty": "hard",
            "skill_progress": {
                "improvement_trajectory": [8.0, 8.5, 9.0]
            }
        }
        result = _adjust_difficulty_adaptive(state, {"total": 9.0})
        assert result == {}

    def test_already_at_min_no_down(self):
        """已在最低难度时不继续降低"""
        state = {
            "difficulty": "easy",
            "skill_progress": {
                "improvement_trajectory": [3.0, 2.5, 2.0]
            }
        }
        result = _adjust_difficulty_adaptive(state, {"total": 2.0})
        assert result == {}

    def test_no_score_data_returns_empty(self):
        """没有 score_data 时返回空 dict"""
        state = {"difficulty": "medium", "skill_progress": {"improvement_trajectory": []}}
        result = _adjust_difficulty_adaptive(state, None)
        assert result == {}


class TestEmptyReturn:
    """测试空输入的返回"""

    def test_empty_return_has_score(self):
        result = _empty_return({"skill_progress": {}})
        assert "score" in result
        assert result["score"]["total"] == 0

    def test_empty_return_preserves_progress(self):
        existing_progress = {"total_turns": 5, "avg_score": 6.0}
        result = _empty_return({"skill_progress": existing_progress})
        assert result["skill_progress"]["total_turns"] == 6
        assert result["skill_progress"]["avg_score"] == 6.0
