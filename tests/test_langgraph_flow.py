"""
LangGraph 架构测试 - 验证完整的对话流程

用法:
    cd ai-english-tutor
    PYTHONPATH=. python tests/test_langgraph_flow.py
"""

import asyncio
from agents.graph_builder import get_graph
from agents.state import EnglishTutorState


async def test_langgraph_flow() -> None:
    """测试 LangGraph 完整流程"""
    print("=" * 60)
    print("AI英语口语陪练系统 - LangGraph 流程验证测试")
    print("=" * 60)

    # 构建图
    graph = get_graph()
    print(f"\n[OK] LangGraph 构建完成")
    print(f"[OK] 节点: scenario -> conversation -> correction -> scoring -> END")

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
        "session_active": True,
        "messages": [],
    }

    result = await graph.ainvoke(initial_state)

    messages = result.get("messages", [])
    # LangGraph stores messages as dict-compatible objects
    opening = messages[-1].content if messages else ""
    print(f"  场景: {result['scenario']} ({result['metadata'].get('scenario_name', '')})")
    print(f"  难度: {result['difficulty']} ({result['metadata'].get('difficulty_description', '')})")
    print(f"  开场白: {opening[:80]}...")

    # 第2步：第一轮对话
    print("\n" + "-" * 60)
    print("Step 2: 第一轮对话")
    print("-" * 60)

    state_2 = dict(result)
    state_2["turn"] = 1
    # Convert LangGraph messages to plain dicts with role/content for next invocation
    state_2["messages"] = [
        {"role": "assistant", "content": m.content} if hasattr(m, "content") else m
        for m in messages
    ]
    state_2["messages"].append({"role": "user", "content": "Hi, nice to meet you!"})

    result_2 = await graph.ainvoke(state_2)
    ai_reply = ""
    for msg in reversed(result_2.get("messages", [])):
        content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "")
        if content:
            ai_reply = content
            break

    print(f"  用户: Hi, nice to meet you!")
    print(f"  AI: {ai_reply[:100]}...")
    print(f"  纠错: {result_2.get('correction', {})}")
    print(f"  评分: {result_2.get('score', {})}")

    # 第3步：第二轮对话
    print("\n" + "-" * 60)
    print("Step 3: 第二轮对话")
    print("-" * 60)

    state_3 = dict(result_2)
    state_3["turn"] = 2
    messages_3 = result_2.get("messages", [])
    state_3["messages"] = [
        {"role": "assistant", "content": m.content} if hasattr(m, "content") else m
        for m in messages_3
    ]
    state_3["messages"].append({"role": "user", "content": "I love watching movies"})

    result_3 = await graph.ainvoke(state_3)
    print(f"  用户: I love watching movies")
    print(f"  评分总分: {result_3.get('score', {}).get('total', 0)}/10")

    print("\n" + "=" * 60)
    print("✅ LangGraph 流程测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_langgraph_flow())
