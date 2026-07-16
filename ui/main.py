"""
AI 英语口语陪练系统 - 主界面 (Streamlit)

三栏布局：
  左侧 (22%)：场景选择 + 难度 + 会话控制
  中间 (56%)：聊天窗口（对话历史 + 纠错 + 评分）
  右侧 (22%)：评分面板 + 学习进度

启动：
    cd ai-english-tutor
    streamlit run ui/main.py --server.port 8501
"""

from __future__ import annotations

import sys
import urllib.parse
from pathlib import Path

import streamlit as st
from typing import Any, Optional

# 确保项目根目录在 Python path 中
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from ui.client import APIClient

# ============================================================================
# 页面配置
# ============================================================================

st.set_page_config(
    page_title="AI英语口语陪练",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================================
# 常量定义
# ============================================================================

SCENARIOS = {
    "daily": {"name": "日常对话", "icon": "💬", "desc": "自由闲聊，提升流利度"},
    "interview": {"name": "英语面试", "icon": "💼", "desc": "模拟面试，展示能力"},
    "restaurant": {"name": "餐厅点餐", "icon": "🍽️", "desc": "点餐、询问、付款"},
    "travel": {"name": "旅行出行", "icon": "✈️", "desc": "问路、住宿、交通"},
    "meeting": {"name": "商务会议", "icon": "🤝", "desc": "会议讨论、发表观点"},
}

DIFFICULTIES = {
    "easy": {"label": "入门", "emoji": "🌱"},
    "medium": {"label": "进阶", "emoji": "🌿"},
    "hard": {"label": "精通", "emoji": "🌳"},
}

LEVELS = {
    "beginner": {"label": "初级", "desc": "基础词汇，简单句型"},
    "intermediate": {"label": "中级", "desc": "常用表达，复合句型"},
    "advanced": {"label": "高级", "desc": "地道表达，复杂句式"},
}

DIMENSION_LABELS = {
    "fluency": ("流利度", "🗣️"),
    "grammar": ("语法", "📝"),
    "vocabulary": ("词汇", "📚"),
    "naturalness": ("自然度", "🎭"),
}

DIMENSION_COLORS = {
    "fluency": "#3b82f6",
    "grammar": "#8b5cf6",
    "vocabulary": "#f59e0b",
    "naturalness": "#10b981",
}

# ============================================================================
# CSS 样式 — 专业商务风格
# ============================================================================

st.markdown("""
<style>
    /* ========== 全局 ========== */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        min-height: 100vh;
    }
    [data-testid="stHeader"] { display: none; }
    [data-testid="stToolbar"] { display: none; }

    /* 滚动条 */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #475569; border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: #64748b; }

    /* ========== 聊天消息 ========== */
    .msg-user {
        background: linear-gradient(135deg, #1e40af, #2563eb);
        color: #fff;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 6px 0;
        max-width: 88%;
        font-size: 14px;
        line-height: 1.6;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3);
        margin-left: auto;
    }
    .msg-ai {
        background: linear-gradient(135deg, #1e293b, #334155);
        color: #e2e8f0;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 6px 0;
        max-width: 88%;
        font-size: 14px;
        line-height: 1.6;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }
    .msg-opening {
        background: linear-gradient(135deg, #1a2332, #243447);
        color: #93c5fd;
        padding: 14px 20px;
        border-radius: 16px;
        margin: 8px 0;
        max-width: 90%;
        font-size: 14px;
        line-height: 1.6;
        border-left: 3px solid #3b82f6;
    }

    /* 纠错卡片 */
    .corr-card {
        background: #1a1a2e;
        border-left: 3px solid #f59e0b;
        border-radius: 0 10px 10px 0;
        padding: 10px 16px;
        margin: 8px 0 8px 20px;
        font-size: 13px;
        line-height: 1.6;
        color: #cbd5e1;
        max-width: 85%;
    }
    .corr-card .corr-title {
        color: #f59e0b;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .corr-card .corr-corrected {
        color: #86efac;
        font-weight: 500;
    }

    /* ========== 侧边栏 ========== */
    section[data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid #1e293b;
    }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    .stSidebar .css-1e5gk1n { background: #0f172a; }

    /* 侧边栏标题 */
    .sidebar-title {
        font-size: 20px;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 4px;
    }
    .sidebar-subtitle {
        font-size: 12px;
        color: #64748b;
        margin-bottom: 16px;
    }

    /* 场景卡片 */
    .scenario-card {
        background: #1e293b;
        border: 2px solid #334155;
        border-radius: 12px;
        padding: 12px 14px;
        margin: 6px 0;
        cursor: pointer;
        transition: all 0.2s;
    }
    .scenario-card:hover {
        border-color: #3b82f6;
        background: #1e3a5f;
    }
    .scenario-card.active {
        border-color: #3b82f6;
        background: linear-gradient(135deg, #1e3a5f, #1e40af);
        box-shadow: 0 0 12px rgba(59, 130, 246, 0.3);
    }
    .scenario-card .sc-icon { font-size: 20px; }
    .scenario-card .sc-name { font-weight: 600; color: #f1f5f9; font-size: 14px; }
    .scenario-card .sc-desc { font-size: 11px; color: #94a3b8; margin-top: 2px; }

    /* 难度选择 */
    .diff-btn {
        background: #1e293b;
        border: 2px solid #334155;
        border-radius: 10px;
        padding: 8px 12px;
        margin: 3px;
        cursor: pointer;
        font-size: 13px;
        color: #e2e8f0;
        transition: all 0.2s;
        text-align: center;
        display: inline-block;
    }
    .diff-btn:hover { border-color: #3b82f6; }
    .diff-btn.active {
        border-color: #3b82f6;
        background: #1e3a5f;
        color: #fff;
        font-weight: 600;
    }

    /* 按钮 */
    .stButton > button {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
        font-size: 14px;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8, #1e40af);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
    }
    .btn-danger {
        background: linear-gradient(135deg, #dc2626, #b91c1c) !important;
    }
    .btn-danger:hover {
        background: linear-gradient(135deg, #b91c1c, #991b1b) !important;
    }

    /* 输入框 */
    .stTextInput input {
        background: #1e293b;
        color: #e2e8f0;
        border: 2px solid #334155;
        border-radius: 12px;
        padding: 12px 16px;
        font-size: 14px;
    }
    .stTextInput input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15);
    }
    .stTextInput > div > div > div { color: #94a3b8 !important; }

    /* 聊天输入区 */
    .chat-input-area {
        background: #1e293b;
        border-radius: 16px;
        padding: 16px;
        border: 2px solid #334155;
    }

    /* ========== 评分条 ========== */
    .score-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
    }
    .score-bar-bg {
        background: #0f172a;
        border-radius: 8px;
        height: 8px;
        overflow: hidden;
        margin: 6px 0;
    }
    .score-bar-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* ========== 面板卡片 ========== */
    .panel-card {
        background: #1e293b;
        border-radius: 14px;
        padding: 20px;
        margin: 8px 0;
        border: 1px solid #334155;
    }
    .panel-title {
        font-size: 15px;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .panel-stat {
        display: flex;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px solid #1e293b;
        font-size: 13px;
    }
    .panel-stat:last-child { border-bottom: none; }
    .panel-stat .label { color: #64748b; }
    .panel-stat .value { color: #e2e8f0; font-weight: 600; }

    /* ========== 头部 ========== */
    .main-header {
        text-align: center;
        padding: 20px 0 10px;
    }
    .main-header h1 {
        font-size: 28px;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .main-header .subtitle {
        font-size: 13px;
        color: #64748b;
        margin-top: 4px;
    }

    /* ========== 状态指示 ========== */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    .status-active {
        background: rgba(34, 197, 94, 0.15);
        color: #22c55e;
    }
    .status-idle {
        background: rgba(100, 116, 139, 0.15);
        color: #64748b;
    }
    .status-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        display: inline-block;
    }
    .dot-active { background: #22c55e; box-shadow: 0 0 6px #22c55e88; }
    .dot-idle { background: #64748b; }

    /* ========== 欢迎区 ========== */
    .welcome-card {
        background: linear-gradient(135deg, #1e293b, #1e3a5f);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 40px 30px;
        text-align: center;
        margin: 20px 0;
    }
    .welcome-card .welcome-icon { font-size: 48px; margin-bottom: 12px; }
    .welcome-card .welcome-title {
        font-size: 20px;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 8px;
    }
    .welcome-card .welcome-desc {
        font-size: 13px;
        color: #94a3b8;
        line-height: 1.6;
    }

    /* 分割线 */
    hr { border-color: #1e293b; margin: 8px 0; }

    /* 侧边栏选择器样式 */
    .stSelectbox label, .stSlider label { color: #94a3b8 !important; font-size: 12px; }
    .stSelectbox > div { color: #e2e8f0 !important; }

    /* 页脚 */
    .footer {
        text-align: center;
        padding: 20px 0 10px;
        font-size: 11px;
        color: #475569;
    }
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
    # 语音相关状态
    if "is_recording" not in st.session_state:
        st.session_state.is_recording = False
    if "tts_audio_url" not in st.session_state:
        st.session_state.tts_audio_url = ""
    if "asr_pending_text" not in st.session_state:
        st.session_state.asr_pending_text = ""
    if "asr_pending_confirmed" not in st.session_state:
        st.session_state.asr_pending_confirmed = False
    # 语音设置
    if "tts_voice" not in st.session_state:
        st.session_state.tts_voice = "alloy"
    if "tts_speed" not in st.session_state:
        st.session_state.tts_speed = 1.0


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
# 回调函数
# ============================================================================


def _select_scenario(scenario_key: str, scenario_name: str) -> None:
    """场景选择回调"""
    st.session_state.scenario = scenario_key
    st.session_state.scenario_name = scenario_name


def _select_difficulty(diff: str) -> None:
    """难度选择回调"""
    st.session_state.difficulty = diff


def _select_level(lvl: str) -> None:
    """水平选择回调"""
    st.session_state.level = lvl


# ============================================================================
# 侧边栏 — 场景选择 + 会话控制
# ============================================================================

with st.sidebar:
    # Logo + 标题
    st.markdown('<div class="sidebar-title">🎓 AI口语陪练</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">LangGraph 多Agent协作架构</div>', unsafe_allow_html=True)
    st.markdown("---")

    # API 地址
    api_url = st.text_input(
        "API 地址",
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
        st.success("✅ 已连接")
    else:
        st.error("❌ 后端未连接")

    st.markdown("---")

    # 场景选择（卡片式）
    st.markdown("**📋 选择场景**")
    scenario_keys = list(SCENARIOS.keys())
    for i, key in enumerate(scenario_keys):
        sc = SCENARIOS[key]
        is_active = st.session_state.scenario == key and not st.session_state.session_active
        btn_type = "primary" if is_active else "secondary"
        col_left, col_right = st.columns([3, 1])
        with col_left:
            st.markdown(
                f'<div class="scenario-card {"active" if is_active else ""}">'
                f'<span class="sc-icon">{sc["icon"]}</span> '
                f'<span class="sc-name">{sc["name"]}</span><br>'
                f'<span class="sc-desc">{sc["desc"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col_right:
            st.button("选择", key=f"sc_{key}", use_container_width=True, type=btn_type,
                     on_click=_select_scenario, args=(key, sc["name"]))

    # 难度选择
    st.markdown("**📊 难度等级**")
    diff_cols = st.columns(3)
    for i, diff in enumerate(DIFFICULTIES):
        d = DIFFICULTIES[diff]
        is_active = st.session_state.difficulty == diff
        btn_class = "diff-btn active" if is_active else "diff-btn"
        with diff_cols[i]:
            st.markdown(
                f'<div class="{btn_class}">'
                f'{d["emoji"]} {d["label"]}</div>',
                unsafe_allow_html=True,
            )
            st.button("选择", key=f"diff_{diff}", use_container_width=True,
                     type="primary" if is_active else "secondary",
                     on_click=_select_difficulty, args=(diff,))

    # 用户水平
    st.markdown("**👤 您的水平**")
    level_cols = st.columns(3)
    for i, lvl in enumerate(LEVELS):
        lv = LEVELS[lvl]
        is_active = st.session_state.level == lvl
        btn_class = "diff-btn active" if is_active else "diff-btn"
        with level_cols[i]:
            st.markdown(
                f'<div class="{btn_class}">{lv["label"]}</div>',
                unsafe_allow_html=True,
            )
            st.button("选择", key=f"lvl_{lvl}", use_container_width=True,
                     type="primary" if is_active else "secondary",
                     on_click=_select_level, args=(lvl,))

    st.markdown("---")

    # 语音设置
    st.markdown("**🔊 语音设置**")

    # TTS 声音选择
    voices_resp = {}
    try:
        voices_resp = st.session_state.client.get_voices()
    except Exception:
        pass

    voice_options = voices_resp.get("voices", [
        {"name": "alloy", "description": "中性"},
        {"name": "nova", "description": "女声"},
        {"name": "echo", "description": "男声"},
        {"name": "fable", "description": "英式"},
        {"name": "onyx", "description": "沉稳"},
        {"name": "shimmer", "description": "轻快"},
    ])
    voice_labels = [f"{v['name']} ({v.get('description', '')})" for v in voice_options]
    voice_names = [v["name"] for v in voice_options]

    current_voice_idx = voice_names.index(st.session_state.tts_voice) if st.session_state.tts_voice in voice_names else 0

    selected_voice = st.selectbox(
        "TTS 声音",
        options=voice_labels,
        index=current_voice_idx,
        key="tts_voice_select",
    )
    # 同步选择结果
    sel_idx = voice_labels.index(selected_voice) if selected_voice in voice_labels else 0
    st.session_state.tts_voice = voice_names[sel_idx]

    # TTS 语速滑块
    st.slider(
        "语速",
        min_value=0.5,
        max_value=2.0,
        value=st.session_state.tts_speed,
        step=0.1,
        key="tts_speed_slider",
        help="0.5 = 慢速, 1.0 = 正常, 2.0 = 快速",
    )
    st.session_state.tts_speed = st.session_state.tts_speed_slider

    st.markdown("---")

    # 会话控制
    st.markdown("**🎮 会话控制**")
    col_start, col_new = st.columns(2)
    with col_start:
        if not st.session_state.session_active:
            if st.button("▶ 开始练习", use_container_width=True):
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
                    opening = resp.get("opening_line", "")
                    st.session_state.messages = [
                        {"role": "opening", "content": opening}
                    ]
                    st.toast("🎉 会话已开启！", icon="🎉")
                    st.rerun()
                except Exception as e:
                    st.error(f"启动失败：{e}")
    with col_new:
        if st.session_state.session_active:
            if st.button("🔄 新会话", use_container_width=True, type="secondary"):
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

    # 状态指示
    st.markdown("---")
    if st.session_state.session_active:
        st.markdown(
            '<div class="status-badge status-active">'
            '<span class="status-dot dot-active"></span>练习中</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"第 {st.session_state.turn} 轮 · 重试 {st.session_state.retry_count} 次")
    else:
        st.markdown(
            '<div class="status-badge status-idle">'
            '<span class="status-dot dot-idle"></span>未开始</div>',
            unsafe_allow_html=True,
        )

# ============================================================================
# 主区域 — 头部
# ============================================================================

st.markdown("""
<div class="main-header">
    <h1>🎓 AI英语口语陪练</h1>
    <div class="subtitle">LangGraph 多Agent协作 · 场景 → 对话 → 纠错 → 评分</div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# 主区域 — 聊天窗口
# ============================================================================

if not st.session_state.session_active:
    # 欢迎区
    st.markdown("""
    <div class="welcome-card">
        <div class="welcome-icon">👋</div>
        <div class="welcome-title">欢迎使用 AI 英语口语陪练</div>
        <div class="welcome-desc">
            在左侧选择练习场景和难度，点击「开始练习」即可开始。<br>
            系统将为您生成场景开场白，并进行智能纠错与四维评分。
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # 消息历史
    for idx, msg in enumerate(st.session_state.messages):
        if msg["role"] == "opening":
            st.markdown(
                f'<div class="msg-opening">🤖 <b>{st.session_state.scenario_name}</b>：{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        elif msg["role"] == "user":
            st.markdown(f'<div class="msg-user">🙋 {msg["content"]}</div>', unsafe_allow_html=True)
        elif msg["role"] == "ai":
            # 生成唯一 ID 用于 TTS 播放控制
            msg_id = f"msg_{idx}"
            st.markdown(
                f'<div class="msg-ai" id="{msg_id}">'
                f'<span id="{msg_id}_text">🤖 {msg["content"]}</span>'
                f'<button id="{msg_id}_play" onclick="playTTS(this, \'{msg_id}\')" '
                f'style="float:right;background:none;border:none;color:#64748b;cursor:pointer;font-size:16px;" '
                f'title="播放语音">🔊</button>'
                f'</div>',
                unsafe_allow_html=True,
            )
            # 纠错信息
            corr = msg.get("correction")
            if corr and corr.get("has_errors"):
                st.markdown(
                    f'<div class="corr-card">'
                    f'<div class="corr-title">📝 纠错</div>'
                    f'{corr.get("explanation", "")}<br>'
                    f'<span class="corr-corrected">✅ 修正：{corr.get("corrected", "")}</span><br>'
                    f'💡 建议：{corr.get("suggestion", "")}',
                    unsafe_allow_html=True,
                )
            # 评分信息
            score = msg.get("score")
            if score:
                total = score.get("total", 0)
                fb_zh = score.get("feedback_zh", "")
                st.caption(f"📊 综合评分：<b>{total}/10</b> — {fb_zh}")

    # 输入框
    st.markdown('<div class="chat-input-area">', unsafe_allow_html=True)

    # 显示 ASR 识别结果（如果有待确认的文本）
    asr_pending = st.session_state.get("asr_pending_text", "")
    if asr_pending:
        st.markdown(
            f'<div style="background:#1e3a5f;border:1px solid #3b82f6;border-radius:10px;padding:10px 14px;margin-bottom:10px;font-size:13px;">'
            f'🎤 <b style="color:#93c5fd;">语音识别：</b><span style="color:#e2e8f0;">{asr_pending}</span></div>',
            unsafe_allow_html=True,
        )
        # 发送/清除按钮
        col_asr_send, col_asr_clear = st.columns(2)
        with col_asr_send:
            if st.button("✓ 发送", use_container_width=True, type="primary"):
                st.session_state.asr_pending_confirmed = True
                st.session_state._asr_pending_text = asr_pending
                st.session_state.asr_pending_text = ""
                st.rerun()
        with col_asr_clear:
            if st.button("✗ 清除", use_container_width=True, type="secondary"):
                st.session_state.asr_pending_text = ""
                st.rerun()

    # 语音输入按钮
    col_input, col_voice = st.columns([4, 1])
    with col_voice:
        if not st.session_state.is_recording:
            if st.button("🎤", use_container_width=True, type="primary"):
                try:
                    st.session_state.is_recording = True
                    st.toast("🎤 正在录音，请说话...", icon="🎤")
                    # 将 API base URL 注入 JS 上下文，避免相对路径解析到错误端口
                    api_base = st.session_state.client.base_url
                    st.components.v1.html(f"""
<script>
let mediaRecorder = null;
let audioChunks = [];
const _apiBase = "{api_base}";

async function startRecording() {{
    try {{
        const stream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        mediaRecorder.ondataavailable = (e) => {{ audioChunks.push(e.data); }};
        mediaRecorder.onstop = async () => {{
            const audioBlob = new Blob(audioChunks, {{ type: 'audio/webm' }});
            const formData = new FormData();
            formData.append('file', audioBlob, 'audio.webm');
            formData.append('language', 'en');
            try {{
                const response = await fetch(_apiBase + '/api/asr/transcribe', {{ method: 'POST', body: formData }});
                const result = await response.json();
                if (result.status === 'success' && result.text) {{
                    // 通过 window.parent 刷新 Streamlit iframe 的 URL
                    const url = new URL(window.parent.location.href);
                    url.searchParams.set('_asr_result', result.text);
                    window.parent.location.href = url.toString();
                }} else {{
                    alert('识别失败: ' + (result.detail || '未知错误'));
                }}
            }} catch (err) {{
                alert('ASR 请求失败: ' + err.message);
            }}
            stream.getTracks().forEach(track => track.stop());
        }};
        mediaRecorder.start();
    }} catch (err) {{
        alert('无法访问麦克风: ' + err.message);
    }}
}}

function stopRecording() {{
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {{
        mediaRecorder.stop();
    }}
}}

window.startRecording = startRecording;
window.stopRecording = stopRecording;
startRecording();
</script>
""", height=0)
                except Exception as e:
                    st.error(f"录音启动失败：{e}")
        else:
            if st.button("⏹️ 停止", use_container_width=True, type="secondary"):
                try:
                    st.session_state.is_recording = False
                    st.toast("⏹️ 录音已停止，正在识别...", icon="⏹️")
                    st.components.v1.html("""
<script>
window.stopRecording();
</script>
""", height=0)
                except Exception as e:
                    st.error(f"录音停止失败：{e}")

    user_input = col_input.chat_input(placeholder="输入英语或语音识别后发送", key="chat_input")
    st.markdown("</div>", unsafe_allow_html=True)

    # 处理 ASR 结果 — 通过 query_params 接收（JS 刷新页面后 Streamlit 读取）
    asr_param = st.query_params.get("_asr_result", "")
    if asr_param:
        st.session_state.asr_pending_text = urllib.parse.unquote(asr_param)
        st.rerun()

    # 处理 ASR 确认发送
    if st.session_state.get("asr_pending_confirmed", False):
        asr_text = st.session_state.get("_asr_pending_text", "")
        st.session_state.asr_pending_confirmed = False
        st.session_state._asr_pending_text = ""
        if asr_text and asr_text.strip():
            try:
                resp = st.session_state.client.chat(message=asr_text.strip())
                ai_reply = resp.get("ai_reply", "")
                correction = resp.get("correction")
                score = resp.get("score")
                skill_progress = resp.get("skill_progress")
                turn = resp.get("turn", 0)
                retry_count = resp.get("retry_count", 0)
                difficulty = resp.get("difficulty", "medium")

                st.session_state.turn = turn
                st.session_state.retry_count = retry_count
                st.session_state.last_score = score
                st.session_state.last_correction = correction
                st.session_state.skill_progress = skill_progress
                st.session_state.difficulty = difficulty

                st.session_state.messages.append({
                    "role": "user",
                    "content": asr_text.strip(),
                    "correction": correction,
                    "score": score,
                })
                st.session_state.messages.append({
                    "role": "ai",
                    "content": ai_reply,
                    "correction": correction,
                    "score": score,
                })
                st.toast("✅ 语音识别成功并已发送", icon="🎤")
                st.rerun()
            except Exception as e:
                st.error(f"语音发送失败：{e}")

    # 处理文本输入发送
    if user_input and user_input.strip():
        try:
            resp = st.session_state.client.chat(message=user_input.strip())

            ai_reply = resp.get("ai_reply", "")
            correction = resp.get("correction")
            score = resp.get("score")
            skill_progress = resp.get("skill_progress")
            turn = resp.get("turn", 0)
            retry_count = resp.get("retry_count", 0)
            difficulty = resp.get("difficulty", "medium")

            st.session_state.turn = turn
            st.session_state.retry_count = retry_count
            st.session_state.last_score = score
            st.session_state.last_correction = correction
            st.session_state.skill_progress = skill_progress
            st.session_state.difficulty = difficulty

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
            st.error(f"发送失败：{e}")

    # TTS 播放 JavaScript — 调用后端合成音频并播放
    api_base = st.session_state.client.base_url
    st.components.v1.html(f"""
<script>
const _apiBase = "{api_base}";
const _ttsVoice = "{st.session_state.tts_voice}";
const _ttsSpeed = {st.session_state.tts_speed};
let currentAudio = null;

async function playTTS(btn, msgId) {{
    // 查找该消息的纯文本内容
    const msgEl = document.getElementById(msgId + '_text');
    if (!msgEl) return;
    let text = msgEl.textContent.replace(/^🤖\\s*/, '').trim();
    if (!text) return;

    // 如果正在播放同一条，则停止
    if (currentAudio && currentAudio.dataset.msgId === msgId) {{
        currentAudio.pause();
        currentAudio = null;
        btn.innerHTML = '🔊';
        return;
    }}

    // 停止之前的播放
    if (currentAudio) {{
        currentAudio.pause();
        currentAudio = null;
    }}

    btn.innerHTML = '⏳';
    btn.disabled = true;

    try {{
        const resp = await fetch(_apiBase + '/api/tts/synthesize', {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify({{
                text: text,
                voice: _ttsVoice,
                speed: _ttsSpeed,
            }}),
        }});
        const data = await resp.json();
        if (data.status === 'success' && data.audio_base64) {{
            currentAudio = new Audio('data:audio/wav;base64,' + data.audio_base64);
            currentAudio.dataset.msgId = msgId;
            currentAudio.onended = () => {{
                btn.innerHTML = '🔊';
                btn.disabled = false;
                currentAudio = null;
            }};
            await currentAudio.play();
        }} else {{
            btn.innerHTML = '🔊';
            btn.disabled = false;
            alert('TTS 合成失败: ' + (data.detail || '未知错误'));
        }}
    }} catch (err) {{
        btn.innerHTML = '🔊';
        btn.disabled = false;
        console.error('TTS 请求失败:', err);
    }}
}}
</script>
""", height=0)

# ============================================================================
# 底部 — 三栏面板
# ============================================================================

st.markdown("---")

col_score, col_progress, col_stats = st.columns(3, gap="medium")

# --- 左：四维评分 ---
with col_score:
    st.markdown('<div class="panel-title">📊 四维评分</div>', unsafe_allow_html=True)
    score = st.session_state.last_score
    if score:
        scores = score.get("scores", {})
        for dim, (label, emoji) in DIMENSION_LABELS.items():
            val = scores.get(dim, 0)
            pct = val / 10 * 100
            color = DIMENSION_COLORS.get(dim, "#3b82f6")
            bar_color = "#22c55e" if val >= 7 else "#f59e0b" if val >= 5 else "#ef4444"
            st.markdown(
                f'<div class="score-card">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<span style="font-size:13px;font-weight:600;color:#e2e8f0;">{emoji} {label}</span>'
                f'<span style="font-size:16px;font-weight:700;color:{bar_color};">{val:.1f}</span>'
                f'</div>'
                f'<div class="score-bar-bg"><div class="score-bar-fill" style="width:{pct}%;background:{bar_color};"></div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("发送消息后显示评分")

# --- 中：学习进度 ---
with col_progress:
    st.markdown('<div class="panel-title">📈 学习进度</div>', unsafe_allow_html=True)
    progress = st.session_state.skill_progress
    if progress:
        traj = progress.get("improvement_trajectory", [])
        st.markdown(
            f'<div class="panel-card">'
            f'<div class="panel-stat"><span class="label">总轮次</span><span class="value">{progress.get("total_turns", 0)}</span></div>'
            f'<div class="panel-stat"><span class="label">平均评分</span><span class="value">{progress.get("avg_score", 0)}</span></div>'
            f'<div class="panel-stat"><span class="label">最弱项</span><span class="value">{progress.get("weakest_dimension", "-")}</span></div>'
            f'<div class="panel-stat"><span class="label">最强项</span><span class="value">{progress.get("strongest_dimension", "-")}</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if traj:
            max_val = max(traj) if traj else 1
            bars = "".join("▓" for _ in range(int(v / max_val * 15))) if traj else ""
            st.markdown(f'**📉 成绩趋势：** {bars}')
    else:
        st.caption("练习后将显示学习进度")

# --- 右：会话统计 ---
with col_stats:
    st.markdown('<div class="panel-title">📋 会话信息</div>', unsafe_allow_html=True)
    diff_label = DIFFICULTIES.get(st.session_state.difficulty, {}).get("label", st.session_state.difficulty)
    st.markdown(
        f'<div class="panel-card">'
        f'<div class="panel-stat"><span class="label">场景</span><span class="value">{st.session_state.scenario_name}</span></div>'
        f'<div class="panel-stat"><span class="label">难度</span><span class="value">{diff_label}</span></div>'
        f'<div class="panel-stat"><span class="label">轮次</span><span class="value">{st.session_state.turn}</span></div>'
        f'<div class="panel-stat"><span class="label">重试</span><span class="value">{st.session_state.retry_count}</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if st.session_state.last_correction:
        corr = st.session_state.last_correction
        if corr.get("polished"):
            st.markdown(
                f'<div class="panel-card">'
                f'<div class="panel-stat"><span class="label">✨ 地道表达</span></div>'
                f'<div style="color:#86efac;font-size:12px;padding:4px 0;">{corr["polished"][:80]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

# ============================================================================
# 页脚
# ============================================================================

st.markdown("""
<div class="footer">
    AI英语口语陪练 · LangGraph StateGraph 架构 · 场景 → 对话 → 纠错 → 评分
</div>
""", unsafe_allow_html=True)
