"""
测试脚本 - 验证完整的对话流程

用法:
    python tests/test_flow.py
"""

import asyncio
from agents.orchestrator import Orchestrator
from agents.conversation_agent import ConversationAgent


async def test_chat_flow():
    """测试完整的对话流程"""
    print("=" * 60)
    print("AI英语口语陪练系统 - 流程验证测试")
    print("=" * 60)

    # 1. 创建 Orchestrator
    orch = Orchestrator()

    # 2. 注册 Conversation Agent
    conv_agent = ConversationAgent()
    orch.register(conv_agent)
    print(f"\n[OK] 已注册 Agent: {conv_agent}")
    print(f"[OK] 当前可用 Agent: {orch.list_agents()}")

    # 3. 开始会话
    orch.start_session({"scenario": "daily", "level": "beginner"})

    # 4. 模拟多轮对话
    test_messages = [
        "Hi, nice to meet you!",
        "I love watching movies in my free time.",
        "Can you recommend some good English movies?",
    ]

    print("\n" + "-" * 60)
    print("开始对话测试:")
    print("-" * 60)

    for i, msg in enumerate(test_messages, 1):
        print(f"\n>>> 用户 (第{i}轮): {msg}")
        response = await orch.chat(msg)
        print(f"<<< AI ({response.agent_name}): {response.content[:120]}...")
        print(f"    [场景: {response.metadata.get('scenario')}, "
              f"轮次: {response.metadata.get('turn')}]")

    # 5. 查看会话状态
    state = orch.get_session_state()
    print(f"\n{'=' * 60}")
    print(f"会话总结:")
    print(f"  场景: {state.get('scenario')}")
    print(f"  水平: {state.get('level')}")
    print(f"  总轮次: {state.get('turn')}")
    print(f"  对话历史条目: {len(state.get('history', []))}")
    print(f"{'=' * 60}")

    # 6. 结束会话
    orch.end_session()
    print("\n[OK] 测试完成！流程验证通过。")


if __name__ == "__main__":
    asyncio.run(test_chat_flow())
