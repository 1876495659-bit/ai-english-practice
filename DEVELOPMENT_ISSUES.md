# Stage 5+ 开发问题记录

## 问题 1：retry_count 累加失效导致无限循环 ✅ 已修复

### 修复方案
- 使用 `langgraph.types.Command` 控制条件路由（备选方案 C）
- `scoring_node` 在评分低时返回 `Command(update={..., retry_count: new_count}, goto="conversation")`
- 评分良好或超过 max_retries 时返回普通 dict，路由到 END
- 不再依赖 `operator.add` reducer，避免了 LangGraph 版本兼容性问题

### 验证
- `tests/test_langgraph_flow.py` 包含 Command 路由验证测试
- 低分场景 → retry_count 递增 → 路由回 conversation
- 高分场景 → 直接 END
- 超过 max_retries → 强制 END

---

## 问题 2：LLM 调用失败时的日志噪音 ✅ 已修复

### 修复方案
- `llm_client.py` 模块级 logger 默认 `setLevel(WARNING)`
- 抑制了 LLM 失败时的 debug 级别日志
- `conversation_node` 在 `llm_enabled=False` 时直接走 mock，不调用 LLM
- 其他 Node 通过 `llm_mode_*` 细粒度开关控制

---

## 问题 3：providers.py 依赖缺失 ✅ 已修复

### 修复方案
- 改为惰性加载：只在调用对应 provider 时才 import SDK
- 未安装的 SDK 会给出明确的 pip install 提示

---

## 问题 4：Python 3.14 兼容性 ✅ 已修复

### 修复方案
- `requirements.txt` 改用 `>=` 灵活版本约束，避免 pydantic-core 编译失败
- `starlette` 降级到 0.40.x 兼容 streamlit 1.44
- streamlit 锁定 `>=1.39,<1.45` 避免与新版 starlette 冲突

---

## 问题 5：Prompt 文件与实际使用不一致 ✅ 已修复

### 修复方案
- 创建 `agents/prompts_loader.py` 统一模板加载器
- 更新 `prompts/conversation.txt`、`prompts/correction.txt`、`prompts/scoring.txt` 为真实使用的模板
- `conversation_node`、`correction_node`、`scoring_node` 改用 `load_prompt()` 从文件加载 prompt
- 符合 MEMORY.md 规则 #4：Prompt 与代码分离

---

## 待办事项

### ~~1. 评分归零问题（LangGraph 1.x messages reducer 兼容性）~~ ✅ 已修复
- **修复方案**：`_extract_latest_user_input` 抽取为 `agents/utils.py` 公共函数，统一处理 dict 和 BaseMessage 两种格式
- **影响范围**：correction_node、scoring_node 均使用公共函数

### ~~2. Pydantic config 类写法弃用警告~~ ✅ 已修复
- **修复方案**：升级为 `model_config = SettingsConfigDict(env_file=".env")`

### ~~3. SQLite Checkpointer 初始化失败~~ ✅ 已修复（Stage 9c）
- **现状**：`langgraph-checkpoint-sqlite` 3.x 的 `from_conn_string` 返回 async context manager
- **处理**：手动管理 aiosqlite 连接生命周期
  - 应用启动时：`aiosqlite.connect()` → `AsyncSqliteSaver(conn)` → `await saver.setup()`
  - 应用关闭时：`await conn.close()`
  - FastAPI lifespan 事件替代已弃用的 `@app.on_event("startup")`
- **数据库文件**：`data/checkpoints.db`（自动创建目录）
- **测试兼容**：`CHECKPOINT_DB_PATH=:memory:` 环境变量回退到 MemorySaver

### ~~4. LangGraph 1.x add_messages reducer 行为变化~~ ✅ 已修复
- **现象**：完整图管道流程中，scoring/correction 节点拿不到 user 消息（messages 被 reducer 吞掉）
- **原因**：LangGraph 1.x 的 `add_messages` reducer 在处理增量更新时有行为变化，checkpoint 不保留大部分字段
- **修复方案**：scoring_node 优先从 `correction.original` 获取用户输入（correction 字段在 checkpoint 中保存正常），再 fallback 到 messages 字段
- **影响范围**：规则引擎评分、API 调用、完整图管道流程均正常工作
- **验证**：98 个单元测试全部通过，test_langgraph_flow.py 完整流程测试通过

---

## Stage 9e 修复记录（2026-07-13）

### Bug 7: chat_input 重复 placeholder 参数导致 TypeError ✅ 已修复（第 2 次确认）
- **现象**：Streamlit 报错 `TypeError: ChatMixin.chat_input() got multiple values for argument 'placeholder'`
- **原因**：`col_input.chat_input("", placeholder="...", key="chat_input")` 第一个位置参数就是 placeholder，又传了同名关键字参数，导致重复
- **修复**：去掉位置参数空字符串，只保留 `col_input.chat_input(placeholder="输入英语或语音识别后发送", key="chat_input")`
- **影响文件**：`ui/main.py` L848
- **注意**：Streamlit 有 `.pyc` 缓存，修改后需要刷新页面才能生效

---

## Stage 9e 已知问题（待修复）

### Bug 8: 语音识别（ASR）结果无法回传到 Streamlit 输入框 ✅ 已修复（第 2 次修复）
- **现象**：用户点击 🎤 录音 → 说话 → 点击 ⏹️ 停止 → ASR 返回文本 → 但文本**不会出现在输入框中也不会自动发送**
- **根因分析**：
  1. 录音 JS 在 `st.components.v1.html` iframe 中执行，ASR 结果通过 `window.history.replaceState` 修改 URL query params
  2. Streamlit 的 `st.query_params.get("asr")` **只能读取页面初始加载时的 URL**，JS 修改 URL 后不会触发 Streamlit 重新读取
  3. 核心问题：**Streamlit 的 query_params 是服务端读取的，JS 修改前端 URL 不会被服务端感知**
  4. （第 2 次）`window.top.location.href` 在 Streamlit 嵌套 iframe 环境中可能被跨域阻止
- **修复方案**：改为两步交互 + URL 刷新机制
  - 录音完成后 iframe 内通过 `window.parent.location.href = url` **刷新父页面**（携带 ASR 结果作为 query param）
  - Streamlit 服务端在页面刷新后读取 `st.query_params.get("_asr_result")`
  - 将结果存入 `st.session_state.asr_pending_text` 并在输入框上方显示蓝色卡片
  - 用户点击 "✓ 发送" 或 "✗ 清除" 按钮确认/丢弃
  - 发送时调用 `client.chat()` 走完整的 conversation → correction → scoring 流程
- **状态**：✅ 已修复（待浏览器端测试）
- **优先级**：高

### Bug 9: AI 回复不走 LangGraph 自适应学习（评分为空） ✅ 已修复
- **现象**：每轮 chat 请求 turn 始终为 1，AI 回复始终是固定开场白，skill_progress 为空
- **根因分析**：
  1. `AsyncSqliteSaver` 的 `list()` 是异步操作，原代码用同步 `list()` 调用导致静默失败
  2. 恢复失败后走 `_make_initial_state()` 创建全新状态，turn 从 0 开始
  3. 每次请求都创建新状态 → 没有对话上下文
- **修复**：
  1. 将 `graph.checkpointer.list()` 改为 `graph.checkpointer.alist()`（异步）
  2. 使用 `graph.aget_state(config)` 获取当前状态（LangGraph 内置异步方法）
  3. 从 checkpoint 恢复状态后更新 turn/messages，再传给 `ainvoke()`
  4. 简化 `_is_session_active()` 为检查 `_sessions` 字典
- **验证**：三轮对话 turn=1 → 2 → 3，AI 回复逐轮变化

### Bug 10: AI 回复不根据用户回答内容（Mock 回复按 turn 索引）
- **现象**：用户说 "It was raining"，AI 回复 "That sounds fun! What else do you like to do in your free time?" — 回复与用户输入**完全无关**
- **根因分析**：
  1. `conversation_node.py` 的 `_mock_reply()` 函数按 `turn` 数字索引返回预设回复
  2. Mock 回复是**线性剧本**，不是基于对话上下文的
  3. 当 `llm_enabled=False` 时，`_generate_reply()` 直接调用 `_mock_reply(scenario, turn)`，完全忽略 `user_history`
  4. 这不是 bug，是设计行为 — 但用户体验极差
- **影响范围**：`agents/conversation_node.py` L146-213
- **修复方案**：
  1. **短期**：改进 mock 回复使其更具通用性（如随机选择、基于关键词匹配）
  2. **长期**：接入真实 LLM（配置 OPENAI_API_KEY），或实现轻量级本地对话引擎
- **优先级**：中（LLM 未配置时的降级行为，配置 API Key 后自动解决）

---

## Stage 9 修复记录（2026-07-10）

### Bug 5: correction_node.py 孤立 return 死代码块（L249-263）✅ 已修复
- **现象**：`correction_node()` 函数结束后有一段不属于任何函数的孤立 `return { ... }` 语句
- **原因**：之前重构留下的残留代码
- **修复**：删除 L249-263 的死代码块
- **验证**：97/97 测试通过

### Bug 6: graph_builder.py 返回值类型标注错误 ✅ 已修复
- **现象**：`build_graph()` 返回 `graph.compile(...)` → `CompiledStateGraph`，但标注为 `StateGraph`
- **影响**：类型检查工具会报错，IDE 自动补全不正确
- **修复**：
  - 导入 `CompiledStateGraph`
  - `build_graph()` 返回类型改为 `CompiledStateGraph`
  - `_app_instance` 类型改为 `Optional[CompiledStateGraph]`
  - `get_graph()` 返回类型改为 `CompiledStateGraph`
- **验证**：97/97 测试通过
