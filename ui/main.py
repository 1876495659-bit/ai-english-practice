"""
AI English Tutor - 主界面 (Streamlit MVP)

三栏布局：
  左栏 (25%)：场景选择 + 难度 + 会话控制
  中栏 (50%)：聊天窗口（对话历史）
  右栏 (25%)：评分面板 + 学习进度

启动：
    cd ai-english-tutor
    streamlit run ui/main.py --server.port 8501
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from typing import Any, Optional

# 确保项目根目录在 Python path 中（Streamlit 运行时可能不包含）
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from ui.client import APIClient

# ============================================================================
# 页面配置
# ============================================================================

st.set_page_config(
    page_title="AI English Coach",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================================
# 常量
# ============================================================================

SCENARIOS = {
    "daily": {"name": "日常对话", "icon": "💬"},
    "interview": {"name": "英语面试", "icon": "💼"},
    "restaurant": {"name": "餐厅点餐", "icon": "🍽️"},
    "travel": {"name": "旅行出行", "icon": "✈️"},
    "meeting": {"name": "商务会议", "icon": "🤝"},
}

DIFFICULTIES = ["easy", "medium", "hard"]
DIFFICULTY_LABELS = {"easy": "Beginner", "medium": "Intermediate", "hard": "Advanced"}

LEVELS = ["beginner", "intermediate", "advanced"]

DIMENSION_LABELS = {
    "fluency": ("Fluency", "流利度"),
    "grammar": ("Grammar", "语法"),
    "vocabulary": ("Vocabulary", "词汇"),
    "naturalness": ("Naturalness", "自然度"),
}

# ============================================================================
# CSS 样式
# ============================================================================

st.markdown("""
<style>
    /* 全局 */
    [data-testid="stAppViewContainer"] { background: #0f172a; }
    [data-testid="stHeader"] { display: none; }
    [data-testid="stToolbar"] { display: none; }

    /* 滚动条 */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }

    /* 聊天消息气泡 */
    .chat-user {
        background: #1e40af;
        color: white;
        padding: 10px 16px;
        border-radius: 16px 16px 4px 16px;
        margin: 4px 0;
        max-width: 85%;
        font-size: 14px;
        line-height: 1.5;
    }
    .chat-ai {
        background: #1e293b;
        color: #e2e8f0;
        padding: 10px 16px;
        border-radius: 16px 16px 16px 4px;
        margin: 4px 0;
        max-width: 85%;
        font-size: 14px;
        line-height: 1.5;
    }
    .chat-correction {
        background: #1a1a2e;
        border-left: 3px solid #f59e0b;
        color: #cbd5e1;
        padding: 8px 14px;
        border-radius: 0 8px 8px 0;
        margin: 6px 0;
        font-size: 13px;
        line-height: 1.5;
    }

    /* 侧边栏 */
    section[data-testid="stSidebar"] { background: #0f172a; }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    .stSelectbox label, .stSlider label { color: #94a3b8 !important; }
    .stSelectbox > div { color: #e2e8f0 !important; }

    /* 评分条 */
    .score-bar-bg {
        background: #1e293b;
        border-radius: 6px;
        height: 10px;
        overflow: hidden;
        margin: 4px 0;
    }
    .score-bar-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.5s ease;
    }

    /* 按钮 */
    .stButton > button {
        background: #2563eb;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 600;
    }
    .stButton > button:hover { background: #1d4ed8; }

    /* 输入框 */
    .stTextInput input {
        background: #1e293b;
        color: #e2e8f0;
        border: 1px solid #334155;
        border-radius: 10px;
    }
    .stTextInput input:focus { border-color: #2563eb; }
    .stTextInput > div > div > div { color: #94a3b8 !important; }

    /* 标签 */
    .metric-label { color: #94a3b8; font-size: 12px; }
    .metric-value { color: #f1f5f9; font-size: 18px; font-weight: 700; }

    /* 分割线 */
    hr { border-color: #1e293b; margin: 12px 0; }

    /* 会话状态指示器 */
    .status-dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .status-active { background: #22c55e; box-shadow: 0 0 6px #22c55e88; }
    .status-idle { background: #64748b; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 会话状态初始化
# ============================================================================


def _init_session_state() -> None:
    """初始化 Streamlit 会话状态"""
    if "client" not in st.session_state:
        api_url = st.session_state.get("api_url", "http://localhost:8000")
        st.session_state.client = APIClient(api_url)
    if "messages" not in st.session_state:
        st.session_state.messages: list[dict[str, Any]] = []
    if "session_id" not in st.session_state:
        st.session_state.session_id: Optional[str] = None
    if "thread_id" not in st.session_state:
        st.session_state.thread_id: Optional[str] = None
    if "scenario" not in st.session_state:
        st.session_state.scenario = "daily"
    if "difficulty" not in st.session_state:
        st.session_state.difficulty = "medium"
    if "level" not in st.session_state:
        st.session_state.level = "intermediate"
    if "session_active" not in st.session_state:
        st.session_state.session_active = False
    if "last_score" not in st.session_state:
        st.session_state.last_score: Optional[dict] = None
    if "last_correction" not in st.session_state:
        st.session_state.last_correction: Optional[dict] = None
    if "skill_progress" not in st.session_state:
        st.session_state.skill_progress: Optional[dict] = None
    if "turn" not in st.session_state:
        st.session_state.turn = 0
    if "retry_count" not in st.session_state:
        st.session_state.retry_count = 0
    if "scenario_name" not in st.session_state:
        st.session_state.scenario_name = ""
    if "api_connected" not in st.session_state:
        st.session_state.api_connected = False


_init_session_state()

# ============================================================================
# 连接检查
# ============================================================================


def _check_connection() -> bool:
    """检查后端 API 连接"""
    try:
        st.session_state.client.health_check()
        st.session_state.api_connected = True
        return True
    except Exception:
        st.session_state.api_connected = False
        return False


# ============================================================================
# 场景选择侧边栏
# ============================================================================

with st.sidebar:
    st.markdown("### 🎓 AI English Coach")
    st.markdown("---")

    # API 地址
    api_url = st.text_input(
        "API Address",
        value="http://localhost:8000",
        key="api_url_input",
        help="后端 FastAPI 服务地址",
    )
    if api_url != st.session_state.get("_last_api_url", ""):
        st.session_state._last_api_url = api_url
        st.session_state.client = APIClient(api_url)
        st.session_state.api_connected = False

    # 连接状态
    connected = _check_connection()
    if connected:
        st.success("✅ Connected")
    else:
        st.error("❌ Backend unreachable")

    st.markdown("---")

    # 场景选择
    st.markdown("### 📋 Scenario")
    scenario_keys = list(SCENARIOS.keys())
    scenario_labels = [f"{SCENARIOS[k]['icon']} {SCENARIOS[k]['name']}" for k in scenario_keys]
    selected_idx = st.selectbox(
        "Choose scenario",
        options=range(len(scenario_keys)),
        format_func=lambda i: scenario_labels[i],
        key="scenario_selector",
        disabled=st.session_state.session_active,
    )
    if selected_idx is not None:
        st.session_state.scenario = scenario_keys[selected_idx]
        st.session_state.scenario_name = SCENARIOS[st.session_state.scenario]["name"]

    # 难度选择
    st.markdown("### 📊 Difficulty")
    diff_idx = st.selectbox(
        "Difficulty level",
        options=range(len(DIFFICULTIES)),
        format_func=lambda i: f"{DIFFICULTIES[i].capitalize()} → {DIFFICULTY_LABELS[DIFFICULTIES[i]]}",
        key="difficulty_selector",
        disabled=st.session_state.session_active,
    )
    if diff_idx is not None:
        st.session_state.difficulty = DIFFICULTIES[diff_idx]

    # 用户水平
    st.markdown("### 👤 Your Level")
    level_idx = st.selectbox(
        "English level",
        options=range(len(LEVELS)),
        format_func=lambda i: LEVELS[i].capitalize(),
        key="level_selector",
        disabled=st.session_state.session_active,
    )
    if level_idx is not None:
        st.session_state.level = LEVELS[level_idx]

    st.markdown("---")

    # 会话控制
    st.markdown("### 🎮 Controls")
    col_start, col_new = st.columns(2)
    with col_start:
        if not st.session_state.session_active:
            if st.button("▶ Start", use_container_width=True):
                try:
                    resp = st.session_state.client.start_session(
                        scenario=st.session_state.scenario,
                        difficulty=st.session_state.difficulty,
                        level=st.session_state.level,
                    )
                    st.session_state.session_id = resp["session_id"]
                    st.session_state.thread_id = resp["thread_id"]
                    st.session_state.session_active = True
                    st.session_state.turn = resp.get("turn", 0)
                    st.session_state.retry_count = resp.get("retry_count", 0)
                    st.session_state.last_score = None
                    st.session_state.last_correction = None
                    st.session_state.skill_progress = None
                    # 清空聊天历史（保留开场白）
                    opening = resp.get("opening_line", "")
                    st.session_state.messages = [
                        {"role": "ai", "content": opening, "correction": None, "score": None}
                    ]
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to start: {e}")
    with col_new:
        if st.session_state.session_active:
            if st.button("🔄 New", use_container_width=True):
                st.session_state.session_active = False
                st.session_state.session_id = None
                st.session_state.thread_id = None
                st.session_state.messages = []
                st.session_state.last_score = None
                st.session_state.last_correction = None
                st.session_state.skill_progress = None
                st.session_state.turn = 0
                st.session_state.retry_count = 0
                st.rerun()

    # 当前状态
    if st.session_state.session_active:
        st.markdown(f'<span class="status-dot status-active"></span><b>Active</b>', unsafe_allow_html=True)
        st.caption(f"Turn {st.session_state.turn} · {st.session_state.retry_count} retries")
    else:
        st.markdown('<span class="status-dot status-idle"></span><b>Idle</b>', unsafe_allow_html=True)

# ============================================================================
# 主区域 - 聊天窗口
# ============================================================================

st.markdown("## 💬 Chat")

# 消息历史
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-user">🙋 You: {msg["content"]}</div>', unsafe_allow_html=True)
    elif msg["role"] == "ai":
        st.markdown(f'<div class="chat-ai">🤖 AI: {msg["content"]}</div>', unsafe_allow_html=True)
        # 纠错信息
        if msg.get("correction") and msg["correction"].get("has_errors"):
            corr = msg["correction"]
            st.markdown(
                f'<div class="chat-correction">'
                f"<b>📝 Correction:</b> {corr.get('explanation', '')}<br>"
                f"<b>Corrected:</b> {corr.get('corrected', '')}<br>"
                f"<b>Suggestion:</b> {corr.get('suggestion', '')}"
                f"</div>",
                unsafe_allow_html=True,
            )
        # 评分信息
        if msg.get("score"):
            score = msg["score"]
            total = score.get("total", 0)
            st.caption(f"**Score: {total}/10** — {score.get('feedback_zh', '')}")

# 输入框
if st.session_state.session_active:
    user_input = st.chat_input("Type your English message...", key="chat_input")
else:
    st.info("Click **Start** to begin a practice session.")
    user_input = None

# 处理发送
if user_input and user_input.strip():
    try:
        resp = st.session_state.client.chat(message=user_input.strip())

        # 提取数据
        ai_reply = resp.get("ai_reply", "")
        correction = resp.get("correction")
        score = resp.get("score")
        skill_progress = resp.get("skill_progress")
        turn = resp.get("turn", 0)
        retry_count = resp.get("retry_count", 0)
        difficulty = resp.get("difficulty", "medium")

        # 更新会话状态
        st.session_state.turn = turn
        st.session_state.retry_count = retry_count
        st.session_state.last_score = score
        st.session_state.last_correction = correction
        st.session_state.skill_progress = skill_progress
        st.session_state.difficulty = difficulty

        # 添加消息到历史
        st.session_state.messages.append({
            "role": "user",
            "content": user_input.strip(),
            "correction": correction,
            "score": score,
        })
        st.session_state.messages.append({
            "role": "ai",
            "content": ai_reply,
            "correction": correction,
            "score": score,
        })

        st.rerun()

    except Exception as e:
        st.error(f"Failed to send message: {e}")

# ============================================================================
# 底部 - 评分面板 + 学习进度（三栏布局的右栏）
# ============================================================================

st.markdown("---")

# 三栏：评分维度 | 学习进度 | 会话统计
col_score, col_progress, col_stats = st.columns(3, gap="large")

# --- 左：四维评分 ---
with col_score:
    st.markdown("### 📊 Scores")
    score = st.session_state.last_score
    if score:
        scores = score.get("scores", {})
        for dim, (label_en, label_zh) in DIMENSION_LABELS.items():
            val = scores.get(dim, 0)
            pct = val / 10 * 100
            # 颜色
            if val >= 7:
                color = "#22c55e"
            elif val >= 5:
                color = "#f59e0b"
            else:
                color = "#ef4444"
            st.markdown(
                f'<div class="metric-label">{label_en} ({label_zh})</div>'
                f'<div class="score-bar-bg"><div class="score-bar-fill" style="width:{pct}%;background:{color};"></div></div>'
                f'<div class="metric-value" style="color:{color};">{val:.1f}/10</div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("No scores yet — send a message to get evaluated.")

# --- 中：学习进度 ---
with col_progress:
    st.markdown("### 📈 Progress")
    progress = st.session_state.skill_progress
    if progress:
        traj = progress.get("improvement_trajectory", [])
        st.caption(f"Total turns: **{progress.get('total_turns', 0)}**")
        st.caption(f"Avg score: **{progress.get('avg_score', 0)}**")
        st.caption(f"Weakest: **{progress.get('weakest_dimension', '-')}**")
        st.caption(f"Strongest: **{progress.get('strongest_dimension', '-')}**")
        if traj:
            # 简易趋势条
            max_val = max(traj) if traj else 1
            bars = "".join("▓" for _ in range(int(v / max_val * 15))) if traj else ""
            st.markdown(f"**History:** {bars}")
    else:
        st.caption("No progress data yet.")

# --- 右：会话统计 ---
with col_stats:
    st.markdown("### 📋 Session")
    st.caption(f"Scenario: **{st.session_state.scenario_name}**")
    st.caption(f"Difficulty: **{DIFFICULTY_LABELS.get(st.session_state.difficulty, st.session_state.difficulty)}**")
    st.caption(f"Turn: **{st.session_state.turn}**")
    st.caption(f"Retries: **{st.session_state.retry_count}**")
    if st.session_state.last_score:
        total = st.session_state.last_score.get("total", 0)
        st.caption(f"Last score: **{total}/10**")
    if st.session_state.last_correction:
        corr = st.session_state.last_correction
        if corr.get("polished"):
            st.caption(f"**Polished:** {corr['polished'][:60]}...")

# ============================================================================
# 页脚
# ============================================================================

st.markdown("---")
st.caption(
    "AI English Coach · LangGraph StateGraph Architecture · "
    "Scenario → Conversation → Correction → Scoring"
)
