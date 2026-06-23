"""
端到端测试 - 验证完整的多Agent协作流程

测试场景：餐厅点餐 (Restaurant Ordering)
流程：
    1. 启动会话 → Scenario Agent 生成开场白
    2. 用户说 "I want a burger" → Conversation Agent 回复
                                → Correction Agent 纠错
                                → Scoring Agent 评分
    3. 用户说 "That sounds delicious, how much is it?" → 同上流程
    4. 输出完整结果汇总

用法:
    cd ai-english-tutor
    PYTHONPATH=. python tests/test_full_pipeline.py
"""

import asyncio
import json
from agents.orchestrator import Orchestrator


async def test_full_pipeline():
    """测试完整的多Agent协作流程"""
    print("=" * 70)
    print("  AI英语口语陪练系统 - 端到端完整流程测试 (v2)")
    print("=" * 70)

    # 创建 Orchestrator
    orch = Orchestrator()

    # ---- 第1步：启动场景 ----
    print("\n" + "─" * 70)
    print("  Step 1: 初始化场景 (Scenario Agent)")
    print("─" * 70)

    ctx = await orch.start_session(
        scenario="restaurant",
        difficulty="medium",
        level="intermediate",
    )
    print(f"  场景: {ctx.scenario} ({ctx.metadata.get('scenario_name')})")
    print(f"  难度: {ctx.difficulty} ({ctx.metadata.get('difficulty_description')})")
    print(f"  目标: {ctx.scenario_goal}")
    print(f"  开场白: {ctx.metadata.get('opening_line', '')[:100]}")

    # ---- 第2步：第一轮对话 ----
    print("\n" + "─" * 70)
    print("  Step 2: 第一轮对话 (Conversation + Correction + Scoring)")
    print("─" * 70)

    user_input_1 = "I want a burger"
    print(f"\n  👤 用户: {user_input_1}")

    result_1 = await orch.chat(user_input_1)

    print(f"\n  🤖 AI回复: {result_1.ai_reply[:120]}...")

    print(f"\n  ✏️  纠错结果:")
    if result_1.correction:
        corr = result_1.correction
        print(f"    原句: {corr.get('original', '')}")
        print(f"    修正: {corr.get('corrected', '')}")
        print(f"    建议: {corr.get('suggestion', '')}")
        print(f"    解释: {corr.get('explanation', '')}")
        if corr.get("errors"):
            for err in corr["errors"]:
                print(f"    🔍 [{err['type']}] {err['issue']}")
    else:
        print("    (无纠错结果)")

    print(f"\n  📊 评分结果:")
    if result_1.score:
        sc = result_1.score
        scores = sc.get("scores", {})
        print(f"    总分: {sc.get('total', 0)}/10")
        print(f"    流利度:   {scores.get('fluency', 0):.1f}/10")
        print(f"    语法:     {scores.get('grammar', 0):.1f}/10")
        print(f"    词汇:     {scores.get('vocabulary', 0):.1f}/10")
        print(f"    自然度:   {scores.get('naturalness', 0):.1f}/10")
        print(f"    英文反馈: {sc.get('feedback_en', '')}")
        print(f"    中文建议: {sc.get('feedback_zh', '')}")
        if sc.get("strengths"):
            print(f"    优点: {', '.join(sc['strengths'])}")
        if sc.get("improvements"):
            print(f"    改进: {', '.join(sc['improvements'])}")
    else:
        print("    (无评分结果)")

    # ---- 第3步：第二轮对话 ----
    print("\n" + "─" * 70)
    print("  Step 3: 第二轮对话")
    print("─" * 70)

    user_input_2 = "Yes please, I'd like a large fries too"
    print(f"\n  👤 用户: {user_input_2}")

    result_2 = await orch.chat(user_input_2)

    print(f"\n  🤖 AI回复: {result_2.ai_reply[:120]}...")

    print(f"\n  ✏️  纠错结果:")
    if result_2.correction:
        corr = result_2.correction
        print(f"    原句: {corr.get('original', '')}")
        print(f"    修正: {corr.get('corrected', '')}")
        print(f"    建议: {corr.get('suggestion', '')}")

    print(f"\n  📊 评分结果:")
    if result_2.score:
        sc = result_2.score
        scores = sc.get("scores", {})
        print(f"    总分: {sc.get('total', 0)}/10")
        print(f"    流利度: {scores.get('fluency', 0):.1f} | "
              f"语法: {scores.get('grammar', 0):.1f} | "
              f"词汇: {scores.get('vocabulary', 0):.1f} | "
              f"自然度: {scores.get('naturalness', 0):.1f}")

    # ---- 第4步：第三轮对话 ----
    print("\n" + "─" * 70)
    print("  Step 4: 第三轮对话")
    print("─" * 70)

    user_input_3 = "How much is the total bill?"
    print(f"\n  👤 用户: {user_input_3}")

    result_3 = await orch.chat(user_input_3)

    print(f"\n  🤖 AI回复: {result_3.ai_reply[:120]}...")

    print(f"\n  ✏️  纠错结果:")
    if result_3.correction:
        corr = result_3.correction
        print(f"    原句: {corr.get('original', '')}")
        print(f"    修正: {corr.get('corrected', '')}")

    print(f"\n  📊 评分结果:")
    if result_3.score:
        sc = result_3.score
        scores = sc.get("scores", {})
        print(f"    总分: {sc.get('total', 0)}/10")
        print(f"    流利度: {scores.get('fluency', 0):.1f} | "
              f"语法: {scores.get('grammar', 0):.1f} | "
              f"词汇: {scores.get('vocabulary', 0):.1f} | "
              f"自然度: {scores.get('naturalness', 0):.1f}")

    # ---- 第5步：会话摘要 ----
    print("\n" + "=" * 70)
    print("  会话摘要")
    print("=" * 70)

    summary = orch.get_session_summary()
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # 结束会话
    orch.end_session()
    print("\n" + "=" * 70)
    print("  ✅ 完整流程测试通过！")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_full_pipeline())
