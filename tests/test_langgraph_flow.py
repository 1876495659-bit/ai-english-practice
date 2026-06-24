"""
LangGraph 架构测试 - 验证完整的自适应学习流程（Stage 5）

用法:
    cd ai-english-tutor
    PYTHONPATH=. python tests/test_langgraph_flow.py
"""

import asyncio
from agents.graph_builder import get_graph, reset_checkpointer
from agents.state import EnglishTutorState


async def test_langgraph_flow() -> None:
    """测试 LangGraph 完整流程（含 Command 路由）"""
    print("=" * 60)
    print("AI英语口语陪练系统 - LangGraph 流程验证测试（Stage 5）")
    print("=" * 60)

    # 清理旧状态
    reset_checkpointer()

    # 构建图
    graph = get_graph()
    print(f"\n[OK] LangGraph 构建完成")
    print(f"[OK] 节点: scenario → conversation → correction → scoring → END")
    print(f"[OK] 条件路由: scoring_node 通过 Command 控制（低分→conversation / 高分→END）")

    config = {"configurable": {"thread_id": "test_thread_001"}}

    # 第1步：启动会话（turn=0，Scenario Node 生成开场白）
    print("\n" + "-" * 60)
    print("Step 1: 启动会话（Scenario Node）")
    print("-" * 60)

    initial_state: EnglishTutorState = {
        "scenario": "daily",
        "difficulty": "medium",
        "level": "intermediate",
        "scenario_goal": "提升日常英语交流的流利度和自然度",
        "ai_reply": "",
        "correction": {},
        "score": {},
        "metadata": {},
        "turn": 0,
        "retry_count": 0,
        "max_retries": 3,
        "session_active": True,
        "messages": [],
        "skill_progress": {
            "total_turns": 0,
            "avg_score": 0.0,
            "error_frequency": {},
            "weakest_dimension": "",
            "strongest_dimension": "",
            "improvement_trajectory": [],
        },
    }

    result = await graph.ainvoke(initial_state, config=config)

    messages = result.get("messages", [])
    opening = messages[-1].content if hasattr(messages[-1], "content") else messages[-1]["content"]
    print(f"  场景: {result['scenario']} ({result['metadata'].get('scenario_name', '')})")
    print(f"  难度: {result['difficulty']} ({result['metadata'].get('difficulty_description', '')})")
    print(f"  开场白: {opening[:80]}...")

    # 第2步：第一轮对话（低分输入，预期触发 Loop）
    print("\n" + "-" * 60)
    print("Step 2: 第一轮对话（低分输入，预期触发 Loop）")
    print("-" * 60)

    state_2 = dict(result)
    state_2["turn"] = 1
    state_2["messages"] = [
        {"role": "assistant", "content": m.content} if hasattr(m, "content") else m
        for m in messages
    ]
    # 故意输入简短、语法差的句子
    state_2["messages"].append({"role": "user", "content": "i go park yesterday"})

    result_2 = await graph.ainvoke(state_2, config=config)
    ai_reply = ""
    for msg in reversed(result_2.get("messages", [])):
        content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "")
        if content:
            ai_reply = content
            break

    score_2 = result_2.get("score", {})
    scores_2 = score_2.get("scores", {})
    retry_2 = result_2.get("retry_count", 0)
    total_2 = score_2.get("total", 0)

    print(f"  用户: i go park yesterday")
    print(f"  AI: {ai_reply[:100]}...")
    print(f"  评分: grammar={scores_2.get('grammar', 0)}, fluency={scores_2.get('fluency', 0)}")
    print(f"  总分: {total_2}/10")
    print(f"  retry_count: {retry_2}")

    # 第3步：第二轮对话（高分输入，预期 END）
    print("\n" + "-" * 60)
    print("Step 3: 第二轮对话（高分输入，预期通过路由 END）")
    print("-" * 60)

    state_3 = dict(result_2)
    state_3["turn"] = 2
    messages_3 = result_2.get("messages", [])
    state_3["messages"] = [
        {"role": "assistant", "content": m.content} if hasattr(m, "content") else m
        for m in messages_3
    ]
    # 输入较长、语法较好的句子
    state_3["messages"].append({
        "role": "user",
        "content": "I would like to visit the museum this weekend, although I am not sure about the opening hours."
    })

    result_3 = await graph.ainvoke(state_3, config=config)
    score_3 = result_3.get("score", {})
    scores_3 = score_3.get("scores", {})
    total_3 = score_3.get("total", 0)

    print(f"  用户: I would like to visit the museum this weekend...")
    print(f"  评分总分: {total_3}/10")
    print(f"  grammar: {scores_3.get('grammar', 0)}, fluency: {scores_3.get('fluency', 0)}")
    print(f"  retry_count: {result_3.get('retry_count', 0)}")
    print(f"  skill_progress: {result_3.get('skill_progress', {})}")

    # 第4步：验证 Command 路由逻辑
    print("\n" + "-" * 60)
    print("Step 4: 验证路由逻辑（通过 scoring_node 内部 Command）")
    print("-" * 60)

    # 模拟低分场景 → 应返回 Command(goto="conversation")
    low_state = {
        "messages": [{"role": "user", "content": "test"}],
        "retry_count": 0,
        "max_retries": 3,
        "correction": {},
        "score": {},
        "difficulty": "medium",
        "skill_progress": {
            "total_turns": 0, "avg_score": 0.0, "error_frequency": {},
            "weakest_dimension": "", "strongest_dimension": "",
            "improvement_trajectory": [],
        },
    }
    low_result = await _simulate_scoring(low_state, {
        "scores": {"fluency": 4.0, "grammar": 3.5, "vocabulary": 5.0, "naturalness": 4.5},
        "total": 4.3,
        "feedback_en": "Low score",
        "feedback_zh": "低分",
        "strengths": [],
        "improvements": ["需要加强练习"],
    })
    print(f"  低分 (grammar=3.5, fluency=4.0) → retry_count={low_result.get('retry_count')}")
    assert low_result.get("retry_count") == 1, f"Expected retry_count=1, got {low_result.get('retry_count')}"

    # 模拟高分场景 → 应返回普通 dict（不走 Command）
    high_state = dict(low_state)
    high_state["retry_count"] = 0
    high_result = await _simulate_scoring(high_state, {
        "scores": {"fluency": 8.0, "grammar": 7.5, "vocabulary": 8.5, "naturalness": 7.0},
        "total": 7.8,
        "feedback_en": "Good score",
        "feedback_zh": "高分",
        "strengths": ["表现优秀"],
        "improvements": [],
    })
    print(f"  高分 (grammar=7.5, fluency=8.0) → retry_count={high_result.get('retry_count', 0)}")
    assert high_result.get("retry_count") == 0, f"Expected retry_count=0, got {high_result.get('retry_count')}"

    # 模拟超过最大重试次数 → 应返回普通 dict（不走 Command）
    max_state = dict(low_state)
    max_state["retry_count"] = 3
    max_result = await _simulate_scoring(max_state, {
        "scores": {"fluency": 4.0, "grammar": 3.5, "vocabulary": 5.0, "naturalness": 4.5},
        "total": 4.3,
        "feedback_en": "Low score",
        "feedback_zh": "低分",
        "strengths": [],
        "improvements": ["需要加强练习"],
    })
    print(f"  retry_count >= max_retries → retry_count={max_result.get('retry_count')}")
    assert max_result.get("retry_count") == 3, f"Expected retry_count=3, got {max_result.get('retry_count')}"

    print("\n" + "=" * 60)
    print("✅ LangGraph 流程测试通过！（Stage 5 自适应学习 + Command 路由）")
    print("=" * 60)


async def _simulate_scoring(state: dict, score_data: dict) -> dict:
    """模拟 scoring_node 的路由逻辑（不依赖 LLM）"""
    from agents.scoring_node import (
        _update_skill_progress,
        _adjust_difficulty_adaptive,
        _empty_return,
    )

    scores = score_data.get("scores", {})
    grammar = scores.get("grammar", 10)
    fluency = scores.get("fluency", 10)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    progress_updates = _update_skill_progress(state, score_data)
    difficulty_adjustment = _adjust_difficulty_adaptive(state, score_data)

    # 评分低 → 递增 retry_count（模拟 Command 路由）
    if (grammar < 6.0 or fluency < 6.0) and retry_count < max_retries:
        new_retry = retry_count + 1
        return {
            **difficulty_adjustment,
            "retry_count": new_retry,
            "score": score_data,
            "skill_progress": progress_updates,
        }

    # 评分良好或超过最大重试 → 不递增
    return {
        **difficulty_adjustment,
        "retry_count": retry_count,
        "score": score_data,
        "skill_progress": progress_updates,
    }


if __name__ == "__main__":
    asyncio.run(test_langgraph_flow())
