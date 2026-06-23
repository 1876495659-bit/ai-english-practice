"""
Scenario Node - 场景控制节点

作为 LangGraph StateGraph 的一个 Node 运行。
职责：
1. 管理所有练习场景的配置（JSON 驱动）
2. 生成场景开场白
3. 动态调整对话难度

数据源：从 agents.scenarios 模块读取配置。
"""

from __future__ import annotations

from typing import Any

from agents.scenarios import get_difficulty_config, get_scenario_config


async def scenario_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    场景控制 Node

    负责：
    - 如果是首轮对话（turn == 0），生成场景开场白
    - 更新 metadata 中的场景信息
    - 将开场白注入到 messages 中

    Args:
        state: 当前图状态

    Returns:
        State 增量更新 dict
    """
    scenario_id: str = state.get("scenario", "daily")
    difficulty: str = state.get("difficulty", "medium")

    scenario_config = get_scenario_config(scenario_id)
    diff_config = get_difficulty_config(scenario_id, difficulty)

    updates: dict[str, Any] = {
        "metadata": {
            "scenario_name": scenario_config["name"],
            "scenario_description": scenario_config["description"],
            "roles": scenario_config["roles"],
            "difficulty_description": diff_config["description"],
            "focus": diff_config["focus"],
        },
        "scenario_goal": scenario_config["goal"],
    }

    # 首轮：生成开场白
    if state.get("turn", 0) == 0:
        opening_lines = scenario_config["opening_lines"]
        opening_line = opening_lines[0]

        updates["metadata"]["opening_line"] = opening_line

        # 将开场白作为 assistant 消息注入
        updates["messages"] = [{"role": "assistant", "content": opening_line}]

        print(
            f"[ScenarioNode] Initialized: {scenario_config['name']} "
            f"({diff_config['description']})"
        )
        print(f"[ScenarioNode] Opening: {opening_line[:80]}")

    return updates
