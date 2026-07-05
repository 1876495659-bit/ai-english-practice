# AI英语口语陪练系统

基于 LangGraph StateGraph 多 Agent 协作架构的 AI 英语口语练习平台。

## 架构设计

```
用户输入 → FastAPI → LangGraph StateGraph → [scenario → conversation → correction → scoring] → 结构化输出
                                                    ↓
                                               LLM (OpenAI/Claude/Groq)
                                                    ↓
                                             Checkpoint (SQLite)
```

### 核心特性

- **LangGraph StateGraph**：唯一调度引擎，Node 之间零耦合
- **多场景训练**：面试 / 点餐 / 旅行 / 会议 / 日常对话
- **智能纠错**：规则引擎 + LLM 双通道，4 层语法检测
- **四维评分**：流利度 / 语法 / 词汇 / 自然度
- **自适应学习**：skill_progress 追踪 + 难度自动调整 + Loop Training
- **会话持久化**：SQLite Checkpoint，支持进程重启恢复
- **多 LLM 支持**：OpenAI / Anthropic / Groq，细粒度开关控制

## 项目结构

```
ai-english-tutor/
├── config/                 # 配置中心
│   ├── settings.py         # 全局配置（pydantic-settings）
│   └── providers.py        # LLM Provider 工厂
├── agents/                 # LangGraph Node 模块
│   ├── state.py            # EnglishTutorState（统一数据载体）
│   ├── graph_builder.py    # StateGraph 构建 + Checkpointer
│   ├── scenario_node.py    # 场景初始化 Node
│   ├── conversation_node.py # 对话生成 Node
│   ├── correction_node.py  # 纠错 Node（规则 + LLM 双通道）
│   ├── scoring_node.py     # 评分 Node（四维评分 + Command 路由）
│   ├── llm_client.py       # 统一 LLM 调用层
│   └── scenarios.py        # 场景配置数据（5 场景 × 3 难度）
├── api/                    # FastAPI RESTful 接口
│   └── main.py             # session/chat 端点
├── tests/                  # 测试
│   └── test_langgraph_flow.py  # 完整流程验证
├── demos/                  # 演示程序
│   └── demo_checkpoint.py  # Checkpoint 持久化演示
├── prompts/                # Prompt 模板
│   ├── system_rules.md     # 系统规则
│   ├── scenario.txt
│   ├── conversation.txt
│   ├── correction.txt
│   └── scoring.txt
├── ui/                     # Streamlit MVP Web 前端
│   ├── main.py             # 主界面（三栏布局）
│   └── client.py           # API 客户端封装
├── .env.example            # 环境变量模板
├── requirements.txt        # 依赖清单
└── MEMORY.md               # 项目记忆（架构原则 + 开发阶段）
```

## 快速开始

```bash
# 进入项目目录
cd ai-english-tutor

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env，填入你的 API Key

# 运行测试（验证 LangGraph 流程）
PYTHONPATH=. python tests/test_langgraph_flow.py

# 运行 Demo（验证 Checkpoint 持久化）
PYTHONPATH=. python demos/demo_checkpoint.py

# 启动 API 服务（后台运行）
uvicorn api.main:app --reload --port 8000

# 启动 Web UI（新终端窗口）
streamlit run ui/main.py --server.port 8501

# 访问 API 文档
# http://localhost:8000/docs
```

## API 接口

| 端点 | 方法 | 描述 |
|------|------|------|
| `/` | GET | 健康检查 |
| `/api/session/start` | POST | 开始新会话（指定场景） |
| `/api/chat` | POST | 发送消息获取回复（含纠错+评分） |
| `/api/session` | GET | 获取当前会话状态 |
| `/api/session/{id}` | DELETE | 删除会话 |
| `/api/session/end` | POST | 结束会话 |

## 开发阶段

- [x] 项目初始化 & LangGraph StateGraph 架构
- [x] 四 Node 实现（scenario/conversation/correction/scoring）
- [x] LLM 真实接入（OpenAI/Anthropic/Groq）+ mock 回退
- [x] 规则引擎纠错 + 四维评分
- [x] 自适应学习（skill_progress + 难度调整 + Loop Training）
- [x] Checkpoint 持久化（SQLite）
- [x] FastAPI RESTful API
- [x] MVP Web UI（Streamlit 三栏布局）
- [x] Prompt 文件抽取（`prompts_loader.py` + 3 个模板文件已接入 Node）
- [x] 单元测试覆盖（98/98 通过 — 规则引擎、评分算法、图构建、API 端点、工具函数）
- [x] Python 3.14 兼容性修复
- [x] LangGraph 1.x 兼容修复（`extract_latest_user_input` 处理 BaseMessage + InMemorySaver）
- [x] Pydantic v2 配置升级（`SettingsConfigDict`）
- [x] SQLite Checkpointer 回退策略完善
- [ ] ASR/TTS 集成
