"""
WebSocket Handler — 实时 LangGraph 执行处理器

提供逐 Node 事件推送，前端可实时看到：
开场白 → AI 回复 → 纠错卡片 → 评分面板

协议设计：
    Client → Server: {"type": "chat", "message": "..."}
    Server → Client:
        - {"type": "node_complete", "node": "conversation", "ai_reply": "..."}
        - {"type": "correction", "data": {...}}
        - {"type": "score", "data": {...}, "skill_progress": {...}}
        - {"type": "retry", "reason": "..."}
        - {"type": "chat_complete", "session_id": "...", "turn": N, ...}
        - {"type": "error", "detail": "..."}
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import WebSocket

from agents.graph_builder import get_graph
from api.sessions import get_or_create_session, make_initial_state

logger = logging.getLogger(__name__)


async def handle_chat_websocket(websocket: WebSocket, payload: dict[str, Any]) -> None:
    """
    处理 WebSocket 聊天消息。

    流程：
    1. 解析客户端消息
    2. 获取或创建 session/thread
    3. 逐个调用 LangGraph Node，每步推送事件
    4. 推送最终结果
    """
    message = payload.get("message", "").strip()
    if not message:
        await websocket.send_json({"type": "error", "detail": "消息不能为空"})
        return

    # 获取/创建会话
    session_id, thread_id = get_or_create_session()
    config = {"configurable": {"thread_id": thread_id}}

    # 恢复或创建状态
    graph = get_graph()
    restored = await _do_restore(graph, thread_id)
    state = restored if restored else make_initial_state()

    # 注入用户消息并递增轮次
    state["turn"] = state.get("turn", 0) + 1
    state["messages"].append({"role": "user", "content": message})

    # ===== 逐个 Node 执行 + 事件推送 =====

    # 1. Conversation Node（场景开场白在 start_session 时已生成）
    state = await graph.nodes["conversation"].ainvoke(state, config=config)
    ai_reply = state.get("ai_reply", "")

    await websocket.send_json({
        "type": "node_complete",
        "node": "conversation",
        "ai_reply": ai_reply,
    })

    # 2. Correction Node
    state = await graph.nodes["correction"].ainvoke(state, config=config)
    correction_data = state.get("correction", {})

    await websocket.send_json({
        "type": "correction",
        "data": correction_data,
    })

    # 3. Scoring Node（可能返回 Command 对象用于条件路由）
    result = await graph.nodes["scoring"].ainvoke(state, config=config)

    if isinstance(result, dict):
        score_data = result.get("score", {})
        skill_progress = result.get("skill_progress", {})
        retry_count = result.get("retry_count", 0)
    else:
        # Command 对象 — 路由回 conversation，暂不推送评分
        logger.info("[WebSocket] Scoring returned Command, routing back to conversation")
        await websocket.send_json({
            "type": "retry",
            "reason": "评分较低，正在重新练习...",
        })
        # 重新运行 conversation + correction
        state = await graph.nodes["conversation"].ainvoke(state, config=config)
        ai_reply = state.get("ai_reply", "")
        await websocket.send_json({
            "type": "node_complete",
            "node": "conversation",
            "ai_reply": ai_reply,
        })
        state = await graph.nodes["correction"].ainvoke(state, config=config)
        await websocket.send_json({
            "type": "correction",
            "data": state.get("correction", {}),
        })
        # 重新 scoring
        result = await graph.nodes["scoring"].ainvoke(state, config=config)
        if isinstance(result, dict):
            score_data = result.get("score", {})
            skill_progress = result.get("skill_progress", {})
            retry_count = result.get("retry_count", 0)
        else:
            score_data = {}
            skill_progress = {}
            retry_count = 0

    await websocket.send_json({
        "type": "score",
        "data": score_data,
        "skill_progress": skill_progress,
        "retry_count": retry_count,
    })

    # 4. 保存完整状态到 checkpoint
    if isinstance(result, dict):
        final_state = {**state, **result}
    else:
        final_state = state

    try:
        await graph.ainvoke(final_state, config=config)
    except Exception as e:
        logger.warning(f"[WebSocket] Failed to save checkpoint: {e}")

    # 5. 最终完成事件
    await websocket.send_json({
        "type": "chat_complete",
        "session_id": session_id,
        "turn": final_state.get("turn", 0),
        "full_response": {
            "ai_reply": ai_reply,
            "correction": correction_data,
            "score": score_data,
            "skill_progress": skill_progress,
            "retry_count": retry_count,
        },
    })


async def _do_restore(graph: Any, thread_id: str) -> dict[str, Any] | None:
    """从 checkpoint 异步恢复状态"""
    try:
        snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
        if snapshot and snapshot.values:
            return dict(snapshot.values)
    except Exception as e:
        logger.warning(f"[WebSocket] Failed to restore state: {e}")
    return None
