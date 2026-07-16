# AI English Coach - Project Memory

## 项目目标
本项目是一个基于 **LangGraph StateGraph** 多Agent协作的 AI 英语口语陪练系统，支持：
- 多场景口语训练（面试 / 点餐 / 旅行 / 会议 / 日常）
- 语音输入与输出（ASR + TTS）
- 智能纠错与表达优化
- 口语能力四维评分与反馈

---

## 核心架构原则

### 1. LangGraph StateGraph 唯一调度
**StateGraph 是整个系统的唯一调度引擎。**

```
User Input → State → [scenario → conversation → correction → scoring] → END → Response
```

- 所有 Agent 变为 **Node**（异步函数）
- **State** 是唯一数据载体（TypedDict）
- Node 之间 **零直接调用**，仅通过 State 读写数据
- 图的拓扑结构在 `graph_builder.py` 中声明式定义

### 2. Node 隔离原则
- 每个 Node 是独立的 async 函数，接收 `state` 返回 `updates dict`
- Node 只修改 State 中属于自己的字段
- 新增 Node 只需：① 在 State 中添加字段 ② 在 graph 中注册
- 禁止 Node 之间互相 import 或调用

### 3. 输出必须结构化
所有 Agent Node 的输出必须为结构化数据（dict/JSON），不允许自由文本。

### 4. Prompt 与代码分离
- 所有 prompt 模板存放在 `/prompts` 目录
- 场景配置数据存放在 `agents/scenarios.py`
- 代码中禁止写长 prompt 或硬编码场景数据
- Node 通过 `agents/prompts_loader.load_prompt()` 加载模板

### 5. 可扩展性原则
系统支持：
- 新增 Node（只需在 State + graph 中注册）
- 新增场景（只需在 `scenarios.py` 添加配置）
- 替换 LLM provider（config/providers.py 工厂模式）

### 6. 上传
修改代码时候用中文上传commits并同步到github上

### 7. SQLite Checkpointer 生产化（Stage 9c）
- 使用 AsyncSqliteSaver 实现真正的会话持久化
- 数据库文件：data/checkpoints.db
- 连接生命周期由 FastAPI lifespan 管理
- 测试环境通过 CHECKPOINT_DB_PATH=:memory: 回退到 MemorySaver

---

## Node 职责定义

| Node | 文件 | 职责 | 写入 State 字段 |
|------|------|------|----------------|
| scenario | `scenario_node.py` | 场景初始化、生成开场白 | `messages`, `metadata`, `scenario_goal` |
| conversation | `conversation_node.py` | 生成 AI 对话回复 | `messages`, `ai_reply` |
| correction | `correction_node.py` | 语法纠错、表达优化 | `correction` |
| scoring | `scoring_node.py` | 四维评分（fluency/grammar/vocabulary/naturalness） | `score`, `skill_progress`, `retry_count` |

---

## 禁止事项

- 禁止 UI 逻辑进入后端 Node
- 禁止 Node 之间直接调用
- 禁止非结构化输出
- 禁止破坏 StateGraph 流程
- 禁止简化为单体模型调用
- 禁止保留 Orchestrator 模式（已废弃）

---

## 当前开发阶段（实时更新此阶段）

**Stage 9e**：Web UI 完整语音集成 + Docker 部署支持
- ✅ Stage 9c：SQLite Checkpointer 生产化（AsyncSqliteSaver + FastAPI lifespan）
- ✅ Stage 9d：ASR/TTS 语音集成 + API 增强
  - `agents/asr.py`：OpenAI Whisper 语音转文本服务（含 mock 回退、语言检测）
  - `agents/tts.py`：OpenAI TTS 文本转语音服务（支持 6 种声音、语速调节、批量合成）
  - `api/main.py`：新增 `/api/asr/transcribe`、`/api/tts/synthesize`、`/api/tts/voices` 端点
  - `api/main.py`：全局异常处理器（RequestValidationError + 通用 Exception）
  - `api/main.py`：请求模型增强验证（场景/难度/水平校验、消息长度限制 500 字符、extra forbid）
  - `config/settings.py`：新增 asr_enabled、asr_language、tts_enabled、tts_voice、tts_speed 配置项
  - `requirements.txt`：添加 python-multipart 依赖（文件上传支持）
  - `ui/main.py`：聊天输入区增加 🎤 语音输入按钮
  - `setup.py`：交互式 .env 配置向导（选择 Provider → 填入 API Key → 生成配置）
  - `conversation_node.py`：mock 回复从 3 轮扩展至 10 轮（5 场景 × 10 轮 = 50 条）
- ✅ 单元测试 82/82 全部通过（test_api / test_rule_engine / test_scoring_node / test_graph_builder / test_langgraph_flow）
- ✅ Stage 9e：Web UI 完整语音集成
  - `ui/client.py`：新增 `transcribe()`（ASR 上传音频）、`synthesize()`（TTS 合成）、`get_voices()`（获取声音列表）
  - `ui/main.py`：🎤 语音输入 — 点击录音 → MediaRecorder 采集 → POST `/api/asr/transcribe` → iframe 刷新父页面（URL 带 `_asr_result`）→ Streamlit 读取 query_params → 存入 session_state → 显示蓝色卡片 → 用户点击"发送"调用 client.chat() → 完整 conversation→correction→scoring 流程
  - `ui/main.py`：🔊 TTS 播放 — 每条 AI 消息右侧显示播放按钮，点击调用 `/api/tts/synthesize` → base64 WAV → Audio API 播放
  - `ui/main.py`：侧边栏新增语音设置 — TTS 声音选择（6 种）、语速滑块（0.5~2.0）
- ✅ Bug 8 修复：ASR 结果回传（两步交互 + URL 刷新）
  - 录音完成后 iframe 通过 `window.top.location.href` 刷新父页面携带 `_asr_result` query param
  - Streamlit 读取后存入 session_state，在输入框上方显示蓝色卡片
  - 用户点击 "✓ 发送" 或 "✗ 清除" 确认/丢弃
- ✅ Bug 9 修复：会话持久化（`AsyncSqliteSaver` 异步调用修复）
  - 将同步 `list()` 改为异步 `alist()`，解决 `Synchronous calls to AsyncSqliteSaver` 错误
  - 使用 `graph.aget_state(config)` 替代手动解析 CheckpointTuple
  - 修复 `_is_session_active()` 避免同步/异步混用
  - 验证：三轮对话 turn=1→2→3，AI 回复逐轮变化，skill_progress 正常累积
- ✅ Bug 10 修复：Q&A 不智能 + 语音无法识别（2026-07-15）
  - **问题 1**：LLM 关闭时 `_mock_reply()` 是纯静态字典查找，完全不理解用户输入
    - 修复：`conversation_node.py` — 新增 `_smart_mock_reply()` 基于关键词分类的动态回复引擎
    - 支持 12+ 意图类别（问候/近况/喜好/活动/工作/食物/天气/感谢/道歉/告别/疑问等）
    - 根据用户输入长度自适应：短句鼓励多说，长文给予详细回应
    - 结合场景给出引导性回复（如面试场景引导专业表达）
    - 原始静态字典保留为 `_fallback_static_reply()` 兜底
  - **问题 2**：JS `fetch('/api/asr/transcribe')` 在 Streamlit iframe 内解析到 8501 端口而非 8000
    - 修复：`ui/main.py` — 将 `st.session_state.client.base_url` 注入 ASR/TTS JS 上下文
    - ASR: `fetch(_apiBase + '/api/asr/transcribe')`
    - TTS: `fetch(_apiBase + '/api/tts/synthesize')`
  - **问题 3**：FastAPI 缺少 CORS 中间件，跨域请求被浏览器拦截
    - 修复：`api/main.py` — 添加 `CORSMiddleware` 允许所有来源（开发环境）

---

## Agent 功能分析与测试状态

### Scenario Node (`scenario_node.py`)
- **功能**: 场景初始化、生成开场白、更新metadata
- **结构**: 简单async函数，依赖`agents.scenarios` JSON配置
- **测试状态**: ✅ 导入正常，逻辑完整
- **关键代码**: turn==0时从`opening_lines`选开场白注入messages

### Conversation Node (`conversation_node.py`)
- **功能**: 根据场景和历史生成AI英语回复
- **结构**: LLM调用 + mock回退（llm_enabled=False时使用）
- **测试状态**: ✅ 双通道设计完整，mock回复覆盖5个场景×10轮次
- **关键代码**: `_build_conversation_history`提取最近6轮对话上下文

### Correction Node (`correction_node.py`)
- **功能**: 语法纠错+表达优化（4层正则检测）
- **结构**: 规则引擎（默认）+ LLM通道（可选）
- **测试状态**: ✅ 规则引擎实现完整，包含：
  - 基础语法（大小写、标点、句号缺失）
  - 语法结构（主谓一致、冠词错误、不规则动词时态）
  - 中式英语（25种常见模式）
  - 表达升级（按难度层级替换词汇）
- **关键数据**: `_IRREGULAR_VERBS`(50+)、`_CHINGLISH_PATTERNS`(25)、`_POLISH_UPGRADES`(50+)

### Scoring Node (`scoring_node.py`)
- **功能**: 四维评分（流利度/语法/词汇/自然度）+ 自适应学习
- **结构**: 规则引擎启发式评分 + LLM评分（可选）
- **测试状态**: ✅ 核心逻辑完整：
  - 评分算法：基于词数、高级词汇、复杂短语、句子长度
  - Skill Progress追踪：total_turns、error_frequency、improvement_trajectory
  - 自适应难度：连续高分→提升，连续低分→降低
  - Loop Training：低分(Command)路由回conversation重练
- **关键常量**: `DIFFICULTY_UP_THRESHOLD=7.5`, `DIFFICULTY_DOWN_THRESHOLD=4.5`

### LLM Client (`llm_client.py`)
- **功能**: 统一OpenAI/Anthropic/Groq调用接口
- **结构**: provider分发 + 结构化JSON输出 + safe_llm_call回退
- **测试状态**: ✅ 代码结构完整，需API Key才能实际调用
- **关键函数**: `call_llm()`文本, `call_llm_json()`结构化, `safe_llm_call()`带fallback

### Graph Builder (`graph_builder.py`)
- **功能**: 组装4个Node成StateGraph有向图 + Checkpointer管理
- **结构**: StateGraph注册 + 边定义 + SQLite/MemorySaver双模式
- **测试状态**: ✅ 图构建逻辑正确，lifespan管理SQLite连接生命周期
- **关键流程**: scenario → conversation → correction → scoring → END

### ASR Service (`asr.py`)
- **功能**: 语音转文本（Speech to Text）
- **结构**: OpenAI Whisper API + mock 回退
- **测试状态**: ✅ 代码结构完整，需API Key才能实际调用
- **关键函数**: `transcribe_audio()`, `detect_language()`

### TTS Service (`tts.py`)
- **功能**: 文本转语音（Text to Speech）
- **结构**: OpenAI TTS API + mock 回退
- **测试状态**: ✅ 代码结构完整，支持 6 种声音、语速调节
- **关键函数**: `synthesize_speech()`, `synthesize_batch()`, `synthesize_speech_to_url()`

### 支撑模块
- **state.py**: `EnglishTutorState` TypedDict定义，自定义`_append_messages` reducer
- **scenarios.py**: 5场景×3难度JSON配置（面试/点餐/旅行/会议/日常）
- **prompts_loader.py**: 从`prompts/*.txt`加载模板，{{variable}}替换
- **utils.py**: `extract_latest_user_input`兼容LangGraph 1.x BaseMessage

### 整体评估
| 维度 | 状态 | 说明 |
|------|------|------|
| 代码结构 | ✅ 优秀 | Node零耦合，State唯一数据载体 |
| 容错机制 | ✅ 完善 | LLM失败→mock回退，SQLite失败→MemorySaver |
| 可维护性 | ✅ 良好 | Prompt与代码分离，配置JSON驱动 |
| 测试覆盖 | ✅ 良好 | 82个单元测试全部通过 |
| 生产就绪 | 🟡 基本就绪 | 需配置.env API Key后完整测试 |

---

## 已知问题记录
遇到问题请记录在development_issues中并解决
详见 [DEVELOPMENT_ISSUES.md](DEVELOPMENT_ISSUES.md)

### 已知问题（已修复）

| 问题 | 状态 | 说明 |
|------|------|------|
| LLM 评分/纠错通道 | ✅ 已修复 | `_extract_latest_user_input` 已统一为 `agents/utils.py` 公共函数，兼容 LangGraph 1.x BaseMessage |
| Pydantic Config 弃用 | ✅ 已修复 | 升级为 `SettingsConfigDict` |
| SQLite Checkpointer | ✅ 已修复 | Stage 9c 手动管理 aiosqlite 连接生命周期 |
| LangGraph 1.x add_messages reducer | ✅ 已修复 | scoring_node 优先从 `correction.original` 获取用户输入 |

---

## 设计理念

本项目目标不是 demo，而是：

**可扩展 AI 教学系统架构（Production-ready design）**

基于 LangGraph 的状态机架构天然支持：
- 条件分支（如：纠错后决定是否跳过评分）
- 循环迭代（如：用户要求重新生成）
- 并行执行（如：纠错和评分同时运行）
- 持久化（Checkpoint 支持会话恢复）
