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

## 待办事项

### 1. Prompt 文件与实际使用不一致
- `prompts/*.txt` 文件定义了结构化 prompt 模板，但各 Node 实际使用内嵌字符串
- **建议**：将 Node 中的内嵌 prompt 抽取到 `prompts/` 目录的文件中，通过 `open().read()` 加载
- **优先级**：中（符合 MEMORY.md 规则 #4）

### 2. 缺少 .env 示例文件
- 新用户不知道如何配置 API Key
- **已添加** `.env.example` 文件

### 3. README.md 过时
- README 仍显示 Orchestrator 架构和旧开发计划
- **建议**：重写为 LangGraph StateGraph 架构文档

### 4. 测试覆盖率不足
- 只有 `test_langgraph_flow.py` 一个集成测试
- **建议**：添加单元测试（规则引擎、评分算法、场景配置等）
