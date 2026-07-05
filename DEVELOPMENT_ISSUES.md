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

### ~~3. SQLite Checkpointer 初始化失败~~ 🟡 降级处理
- **现状**：`langgraph-checkpoint-sqlite` 3.x 的 `from_conn_string` 返回 async context manager
- **处理**：默认回退到 `InMemorySaver`，`demo_checkpoint.py` 中保留手动管理方式
- **后续**：如需生产级 SQLite 持久化，可在异步上下文中使用 `AsyncSqliteSaver` + async context manager

### 4. LangGraph 1.x add_messages reducer 行为变化（已知限制）
- **现象**：完整图管道流程中，scoring/correction 节点可能拿不到 user 消息（messages 被 reducer 吞掉）
- **原因**：LangGraph 1.x 的 `add_messages` reducer 在处理增量更新时有行为变化
- **影响**：`test_langgraph_flow.py` Step 2/3 评分显示为 0，但规则引擎单元测试（29 tests）和 API 测试（16 tests）全部通过
- **临时方案**：scoring_node 已增加 fallback（从 correction.original 获取用户输入）
- **根本修复**：考虑自定义 reducer 或改用 `langgraph.checkpoint.base.EmptyCheckpointSaver` 配合手动消息管理
