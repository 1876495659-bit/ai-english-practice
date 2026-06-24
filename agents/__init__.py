"""
Agents 模块 - LangGraph 多Agent系统

导出所有 Node 函数和状态定义：
- scenario_node: 场景控制节点
- conversation_node: 对话生成节点（支持 LLM/mode 切换）
- correction_node: 语法纠错节点（支持 LLM/mode 切换）
- scoring_node: 评分节点（支持 LLM/mode 切换 + Command 路由）
- llm_client: 统一 LLM 调用层
- state: 统一状态定义（含 SkillProgress）
- graph_builder: 图构建器
- scenarios: 场景配置数据
"""

from agents.scenario_node import scenario_node
from agents.conversation_node import conversation_node
from agents.correction_node import correction_node
from agents.scoring_node import scoring_node
from agents.llm_client import call_llm, call_llm_json, safe_llm_call
from agents.state import (
    EnglishTutorState,
    CorrectionResult,
    ScoreResult,
    ErrorItem,
    SkillProgress,
)
from agents.graph_builder import build_graph, get_graph, reset_graph
from agents.scenarios import SCENARIOS, get_scenario_config, list_available_scenarios

__all__ = [
    "scenario_node",
    "conversation_node",
    "correction_node",
    "scoring_node",
    "call_llm",
    "call_llm_json",
    "safe_llm_call",
    "EnglishTutorState",
    "CorrectionResult",
    "ScoreResult",
    "ErrorItem",
    "SkillProgress",
    "build_graph",
    "get_graph",
    "reset_graph",
    "SCENARIOS",
    "get_scenario_config",
    "list_available_scenarios",
]
