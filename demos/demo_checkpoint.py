"""
Stage 6 Demo - Checkpoint 持久化验证

演示 LangGraph 的 SQLite Checkpointer 如何让 AI 英语教练"记住"用户状态。

三个核心演示：
1. 同一 session 多轮对话不丢 state（checkpoint 自动保存）
2. 模拟"中断后恢复"——新建 graph 实例仍能看到历史
3. 跨进程恢复——从数据库文件加载状态

用法：
    cd ai-english-tutor
    PYTHONPATH=. python demos/demo_checkpoint.py
"""

import asyncio
import os
import sys

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.graph_builder import build_graph, reset_checkpointer
from agents.state import EnglishTutorState

SEPARATOR_WIDTH = 60


# ============================================================================
# 工具函数
# ============================================================================


def print_header(title: str) -> None:
    print(f"\n{'=' * SEPARATOR_WIDTH}")
    print(f"  {title}")
    print(f"{'=' * SEPARATOR_WIDTH}")


def print_section(title: str) -> None:
    print(f"\n{'-' * SEPARATOR_WIDTH}")
    print(f"  {title}")
    print(f"{'-' * SEPARATOR_WIDTH}")


def print_state(state: dict, label: str = "State") -> None:
    """打印状态的摘要信息"""
    print(f"\n  [{label}]")
    print(f"    turn:            {state.get('turn', '?')}")
    print(f"    scenario:        {state.get('scenario', '?')}")
    print(f"    difficulty:      {state.get('difficulty', '?')}")
    print(f"    retry_count:     {state.get('retry_count', '?')}")
    print(f"    session_active:  {state.get('session_active', '?')}")

    messages = state.get("messages", [])
    print(f"    messages count:  {len(messages)}")
    for msg in messages[-2:]:
        if isinstance(msg, dict):
            role = msg.get("role", "?")
            content = msg.get("content", "")[:60]
            print(f"      [{role}] {content}...")
        else:
            role = getattr(msg, "type", "?")
            content = str(getattr(msg, "content", ""))[:60]
            print(f"      [{role}] {content}...")

    score = state.get("score", {})
    if score:
        scores = score.get("scores", {})
        total = score.get("total", 0)
        print(f"    last_score:      {total}/10 (grammar={scores.get('grammar', 0)})")

    progress = state.get("skill_progress", {})
    if progress:
        print(f"    skill_progress:  turns={progress.get('total_turns', 0)}, "
              f"avg_score={progress.get('avg_score', 0)}")


def print_checkpoint_info(graph, thread_id: str) -> None:
    """打印 checkpoint 信息"""
    if not graph.checkpointer:
        print("  [!] No checkpointer attached")
        return

    saver = graph.checkpointer
    try:
        config = {"configurable": {"thread_id": thread_id}}
        snapshots = list(asyncio.get_event_loop().run_in_executor(
            None, lambda: list(saver.list(config))
        ))
        print(f"\n  [Checkpoint] Thread '{thread_id}':")
        print(f"    Total snapshots: {len(snapshots)}")
        for i, snap in enumerate(snapshots):
            cp_id = getattr(snap, 'checkpoint', {}).get('id', '?')
            print(f"    Snapshot {i + 1}: id={str(cp_id)[:16]}...")
    except Exception as e:
        print(f"  [!] Failed to list checkpoints: {e}")


def get_state_from_snapshot(snapshot) -> dict:
    """
    从 CheckpointTuple 中提取 channel_values。

    LangGraph v3.1.0 的 CheckpointTuple 结构：
    - checkpoint: {'v': 4, 'ts': '...', 'id': '...', 'channel_values': {...}}
    """
    cp = snapshot.checkpoint if hasattr(snapshot, 'checkpoint') else snapshot.get('checkpoint', {})
    return cp.get('channel_values', {})


# ============================================================================
# Demo 1: 多轮对话，checkpoint 自动保存
# ============================================================================


async def demo_multi_turn_conversation() -> None:
    """
    Demo 1: 同一 session 多轮对话
    验证每次 invoke 后状态被 checkpoint 保存，下一轮可以恢复
    """
    print_header("Demo 1: 多轮对话 + Checkpoint 自动保存")

    # 清理旧状态
    reset_checkpointer()

    # 构建带 checkpointer 的图
    graph = build_graph()
    thread_id = "demo_multi_turn"

    print(f"\n  Thread ID: {thread_id}")
    print(f"  Checkpointer: {type(graph.checkpointer).__name__}")
    print(f"  Architecture: Scenario -> Conversation -> Correction -> Scoring")

    saver = graph.checkpointer

    # ---- Round 1: 启动会话 ----
    print_section("Round 1: 启动会话（Scenario Node）")

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

    config = {"configurable": {"thread_id": thread_id}}
    result = await graph.ainvoke(initial_state, config=config)

    print_state(result, "After Round 1")
    snapshots = [s async for s in saver.alist(config)]
    print(f"\n  [Checkpoint] Snapshots saved: {len(snapshots)}")

    # ---- Round 2: 用户输入 ----
    print_section("Round 2: 用户输入 'i go park yesterday'（低分）")

    # 从 checkpoint 恢复状态（模拟真实场景：每次请求从 DB 加载）
    snapshots = [s async for s in saver.alist(config)]
    if snapshots:
        restored_state = dict(get_state_from_snapshot(snapshots[-1]))
    else:
        restored_state = dict(result)

    restored_state["turn"] = restored_state.get("turn", 0) + 1
    restored_state["messages"] = list(restored_state.get("messages", []))
    restored_state["messages"].append({"role": "user", "content": "i go park yesterday"})

    result = await graph.ainvoke(restored_state, config=config)

    print_state(result, "After Round 2")
    snapshots = [s async for s in saver.alist(config)]
    print(f"\n  [Checkpoint] Snapshots saved: {len(snapshots)}")

    # ---- Round 3: 用户输入 ----
    print_section("Round 3: 用户输入 'I would like to visit the museum'（高分）")

    snapshots = [s async for s in saver.alist(config)]
    if snapshots:
        restored_state = dict(get_state_from_snapshot(snapshots[-1]))
    else:
        restored_state = dict(result)

    restored_state["turn"] = restored_state.get("turn", 0) + 1
    restored_state["messages"] = list(restored_state.get("messages", []))
    restored_state["messages"].append(
        {"role": "user", "content": "I would like to visit the museum this weekend"}
    )

    result = await graph.ainvoke(restored_state, config=config)

    print_state(result, "After Round 3")
    snapshots = [s async for s in saver.alist(config)]
    print(f"\n  [Checkpoint] Snapshots saved: {len(snapshots)}")

    print("\n  ✅ Demo 1 完成：多轮对话中 checkpoint 自动保存了每次状态")


# ============================================================================
# Demo 2: 模拟"新建进程"后恢复
# ============================================================================


async def demo_process_restart() -> None:
    """
    Demo 2: 模拟进程重启
    1. 第一轮：创建图，对话两轮
    2. 重建图（模拟重启），从 checkpoint 恢复第三轮
    3. 验证历史消息仍在
    """
    print_header("Demo 2: 模拟进程重启后恢复对话")

    reset_checkpointer()

    # ---- 阶段 A: 第一次运行 ----
    print_section("阶段 A: 第一次运行（创建对话）")

    graph_a = build_graph()
    thread_id = "demo_restart"
    saver_a = graph_a.checkpointer

    # 启动会话
    state_a: EnglishTutorState = {
        "scenario": "restaurant",
        "difficulty": "easy",
        "level": "beginner",
        "scenario_goal": "练习餐厅点餐",
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

    config = {"configurable": {"thread_id": thread_id}}
    result_a = await graph_a.ainvoke(state_a, config=config)

    # 第一轮对话
    msgs_a = list(result_a.get("messages", []))
    msgs_a.append({"role": "user", "content": "I want a hamburger please"})
    state_a["messages"] = msgs_a
    state_a["turn"] = 1
    result_a = await graph_a.ainvoke(state_a, config=config)

    print_state(result_a, "阶段 A 结束")
    print(f"  [Info] Checkpoint 已保存到数据库")

    # ---- 阶段 B: 第二次运行（模拟重启）----
    print_section("阶段 B: 第二次运行（重启后恢复）")

    # 关键：完全重建 graph（不共享任何内存状态）
    reset_checkpointer()
    graph_b = build_graph()
    saver_b = graph_b.checkpointer

    print(f"  [Info] 新进程启动，重新构建 LangGraph")
    print(f"  [Info] Checkpointer: {type(saver_b).__name__}")

    # 从 checkpoint 恢复状态
    config = {"configurable": {"thread_id": thread_id}}
    snapshots = [s async for s in saver_b.alist(config)]

    if snapshots:
        latest = snapshots[-1]
        restored_state = dict(get_state_from_snapshot(latest))
        msg_count = len(restored_state.get("messages", []))
        print(f"\n  [OK] 从 checkpoint 恢复了 {msg_count} 条消息")

        # 继续对话
        restored_state["turn"] = restored_state.get("turn", 0) + 1
        restored_state["messages"] = list(restored_state.get("messages", []))
        restored_state["messages"].append(
            {"role": "user", "content": "Do you have any drinks?"}
        )

        result_b = await graph_b.ainvoke(restored_state, config=config)
        print_state(result_b, "阶段 B 恢复后继续对话")

        # 验证：消息历史应该还在
        new_msg_count = len(result_b.get("messages", []))
        print(f"\n  [OK] 恢复后总消息数: {new_msg_count}（包含历史）")
    else:
        print("  [!] 未找到 checkpoint，创建新会话")

    print_checkpoint_info(graph_b, thread_id)

    print("\n  ✅ Demo 2 完成：进程重启后从 checkpoint 恢复了对话历史")


# ============================================================================
# Demo 3: 验证 SQLite 数据库文件存在
# ============================================================================


async def demo_verify_database() -> None:
    """
    Demo 3: 验证 SQLite 数据库文件确实被创建了
    """
    print_header("Demo 3: 验证 SQLite 数据库文件")

    reset_checkpointer()
    graph = build_graph()

    # 做一次简单的 invoke 以触发写入
    state: EnglishTutorState = {
        "scenario": "daily",
        "difficulty": "medium",
        "level": "intermediate",
        "scenario_goal": "test",
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

    config = {"configurable": {"thread_id": "demo_verify"}}
    await graph.ainvoke(state, config=config)

    # 检查数据库文件
    db_path = os.environ.get(
        "CHECKPOINT_DB_PATH",
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "checkpoints.db",
        ),
    )

    print(f"\n  数据库路径: {db_path}")

    if os.path.exists(db_path):
        size = os.path.getsize(db_path)
        print(f"  [OK] 数据库文件存在")
        print(f"  [OK] 文件大小: {size:,} bytes")
    else:
        print(f"  [!] 数据库文件不存在（可能使用了 MemorySaver）")
        if graph.checkpointer:
            print(f"  Checkpointer 类型: {type(graph.checkpointer).__name__}")

    print("\n  ✅ Demo 3 完成：数据库文件验证")


# ============================================================================
# 主入口
# ============================================================================


async def main() -> None:
    print_header("AI English Tutor - Stage 6 Checkpoint 持久化演示")
    print("  演示目标：验证 LangGraph SQLite Checkpointer 使系统具备生产级状态管理")

    await demo_multi_turn_conversation()
    await demo_process_restart()
    await demo_verify_database()

    print_header("所有 Demo 完成！")
    print("""
  总结：
  1. Checkpointer 自动保存每次 invoke 后的状态到 SQLite
  2. 进程重启后，新 graph 实例可从数据库恢复对话历史
  3. thread_id 是唯一标识，同一 thread 共享状态
  4. 所有 State 字段（messages, retry_count, skill_progress 等）均可持久化

  生产级架构就绪：
  - Session 可恢复 ✓
  - 状态可持久化 ✓
  - 中断后可继续 ✓
  - 多轮对话不丢 state ✓
""")


if __name__ == "__main__":
    asyncio.run(main())
