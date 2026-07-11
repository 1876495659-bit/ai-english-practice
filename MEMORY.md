# AI English Coach - Project Memory

## 项目目标
本项目是一个基于 **LangGraph StateGraph** 多Agent协作的 AI 英语口语陪练系统，支持：
- 多场景口语训练（面试 / 点餐 / 旅行 / 会议 / 日常）
- 语音输入与输出（ASR + TTS，预留）
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
  上传commits并同步到github上

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

**Stage 9c**：SQLite Checkpointer 生产化（AsyncSqliteSaver + FastAPI lifespan）
- ✅ correction_node.py L249-263 孤立 return 死代码块删除
- ✅ graph_builder.py 返回值类型修正：`StateGraph` → `CompiledStateGraph`
- ✅ SQLite Checkpointer 生产化：手动管理 aiosqlite 连接生命周期
  - `graph_builder.py`：新增 `_create_sqlite_checkpointer()`（async）、`close_sqlite_checkpointer()`、`get_sqlite_checkpointer()`
  - `api/main.py`：用 `@asynccontextmanager lifespan` 替代已弃用的 `@app.on_event("startup")`
  - 原理：`AsyncSqliteSaver.from_conn_string()` 是 context manager（退出关连接），改为手动 `aiosqlite.connect()` + `await saver.setup()`，连接在 lifespan startup 时创建、shutdown 时关闭
  - 数据库文件：`data/checkpoints.db`（自动创建目录）
  - 测试环境通过 `CHECKPOINT_DB_PATH=:memory:` 环境变量回退到 MemorySaver
- ✅ 补充单元测试 52 个（conversation/scenario/prompts/llm_client），总计 149/149 通过
- ✅ Stage 4：LLM 真实接入（统一 LLM 调用层 + mock 回退 + 双通道设计 + 细粒度开关 + 错误隔离）
- ✅ Stage 5：自适应学习（skill_progress 能力追踪 + 难度自适应调整 + Command 条件路由 Loop Training）
- ✅ Stage 6：SQLite Checkpointer 持久化（session 可恢复 + 进程重启恢复 + 中断续练）+ FastAPI RESTful API
- ✅ Stage 7：MVP Web UI（Streamlit 三栏布局：场景选择 + 聊天窗口 + 评分面板）
- ✅ Stage 8：Bug Fixes & Test Hardening
  - `extract_latest_user_input` 抽取为公共函数 `agents/utils.py`，修复 LangGraph 1.x BaseMessage 兼容
  - `correction_node.py` 死代码清理（重复 return ""）
  - `config/settings.py` 升级为 pydantic v2 `SettingsConfigDict`
  - `graph_builder.py` SQLite Checkpointer 回退策略完善（InMemorySaver 默认）
  - **LangGraph 1.x add_messages reducer 行为变化修复**：scoring_node 优先从 `correction.original` 获取用户输入（correction 字段在 checkpoint 中保存正常），再 fallback 到 messages 字段
  - 单元测试 98/98 通过（test_rule_engine / test_scoring_node / test_graph_builder / test_api / test_utils）
  - pytest.ini 配置 + pytest-asyncio 依赖
  - 自定义 `_append_messages` reducer 替代 LangGraph 内置 `add_messages`
- ✅ Python 3.14 兼容性修复
- ✅ LangGraph 1.x 全面兼容（InMemorySaver、CompiledStateGraph、BaseMessage 格式、messages reducer 修复）

---

## 已知问题记录

详见 [DEVELOPMENT_ISSUES.md](DEVELOPMENT_ISSUES.md)

### 已知问题（已降级）

| 问题 | 状态 | 说明 |
|------|------|------|
| LLM 评分/纠错通道 | 🟡 已修复 | `_extract_latest_user_input` 已统一为 `agents/utils.py` 公共函数，兼容 LangGraph 1.x BaseMessage |
| Pydantic Config 弃用 | 🟢 已修复 | 升级为 `SettingsConfigDict` |
| SQLite Checkpointer | 🟡 降级 | LangGraph 1.x 中 `SqliteSaver.from_conn_string()` 返回 context manager，默认回退到 `InMemorySaver`。如需 SQLite 持久化请使用 `demo_checkpoint.py` 手动管理 |

---

## 设计理念

本项目目标不是 demo，而是：

**可扩展 AI 教学系统架构（Production-ready design）**

基于 LangGraph 的状态机架构天然支持：
- 条件分支（如：纠错后决定是否跳过评分）
- 循环迭代（如：用户要求重新生成）
- 并行执行（如：纠错和评分同时运行）
- 持久化（Checkpoint 支持会话恢复）
