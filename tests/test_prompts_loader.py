"""
单元测试 - Prompts Loader

验证 prompts_loader.py 的模板加载和变量替换功能。
"""

import pytest
from agents.prompts_loader import load_prompt, _PROMPTS_DIR


class TestLoadPrompt:
    """测试 load_prompt 函数"""

    def test_load_conversation_prompt(self):
        """加载 conversation 模板成功"""
        result = load_prompt("conversation")
        assert isinstance(result, str)
        assert len(result) > 0
        assert "You are an AI English conversation partner" in result

    def test_load_correction_prompt(self):
        """加载 correction 模板成功"""
        result = load_prompt("correction")
        assert isinstance(result, str)
        assert "Analyze the following English sentence" in result

    def test_load_scoring_prompt(self):
        """加载 scoring 模板成功"""
        result = load_prompt("scoring")
        assert isinstance(result, str)
        assert "Return ONLY a valid JSON object" in result

    def test_load_scenario_prompt(self):
        """加载 scenario 模板成功"""
        result = load_prompt("scenario")
        assert isinstance(result, str)
        assert "你是AI英语口语陪练系统中的场景控制Node" in result

    def test_variable_substitution(self):
        """模板变量替换正确"""
        result = load_prompt(
            "conversation",
            scenario_name="面试",
            scenario_id="interview",
            difficulty="标准面试",
            level="intermediate",
            turn="3",
            scenario_goal="练习面试",
        )
        assert "面试" in result
        assert "interview" in result
        assert "标准面试" in result
        assert "3" in result

    def test_multiple_variables_substituted(self):
        """多个变量同时替换"""
        result = load_prompt(
            "correction",
            user_input="I go park",
            scenario="daily",
            level="beginner",
        )
        assert "I go park" in result
        assert "daily" in result
        assert "beginner" in result

    def test_empty_template_var(self):
        """空字符串变量替换后仍正常"""
        result = load_prompt(
            "conversation",
            scenario_name="",
            turn="",
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_nonexistent_template_raises(self):
        """不存在的模板文件应抛出 FileNotFoundError"""
        with pytest.raises(FileNotFoundError, match="template not found"):
            load_prompt("nonexistent_template_xyz")

    def test_prompts_dir_exists(self):
        """prompts 目录应存在"""
        assert _PROMPTS_DIR.exists()
        assert _PROMPTS_DIR.is_dir()

    def test_all_expected_templates_exist(self):
        """所有预期的模板文件都应存在"""
        expected = ["conversation", "correction", "scoring", "scenario"]
        for name in expected:
            path = _PROMPTS_DIR / f"{name}.txt"
            assert path.exists(), f"Missing template: {name}"
