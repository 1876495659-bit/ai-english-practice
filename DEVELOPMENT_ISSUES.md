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
