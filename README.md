# AI英语口语陪练系统

基于多Agent协作架构的AI英语口语练习平台。

## 架构设计

```
用户 → FastAPI → Orchestrator → [Scenario / Conversation / Correction / Scoring] Agents
                                      ↓
                                   LLM (OpenAI/Claude/Qwen)
                                      ↓
                                   ASR/TTS (预留)
```

### 核心特性

- **多Agent架构**：每个Agent职责单一、独立模块化
- **统一调度**：Orchestrator 作为唯一通信桥梁
- **可扩展LLM层**：支持 OpenAI / Anthropic / Groq
- **Prompt分离**：所有提示词集中在 `prompts/` 目录
- **集中配置**：所有配置在 `config/` 目录下管理

## 快速开始

```bash
# 进入项目目录
cd ai-english-tutor

# 安装依赖
pip install -r requirements.txt

# 运行测试（验证流程）
python tests/test_flow.py

# 启动API服务
python -m uvicorn api.main:app --reload

# 访问 API 文档
# http://localhost:8000/docs
```

## 项目结构

```
ai-english-tutor/
├── config/             # 配置中心
│   ├── settings.py     # 全局配置
│   └── providers.py    # LLM Provider 管理
├── agents/             # Agent 模块
│   ├── base_agent.py   # Agent 基类
│   ├── orchestrator.py # 调度中心
│   └── conversation_agent.py  # 对话Agent (Demo)
├── services/           # 业务服务 (ASR/TTS 预留)
├── prompts/            # 提示词模板
├── api/                # FastAPI 接口
│   └── main.py
└── tests/              # 测试
```

## 开发计划

- [x] 项目初始化 & 基础架构
- [x] Base Agent 抽象类
- [x] Orchestrator 调度中心
- [x] Conversation Agent Demo
- [ ] 场景控制 Agent
- [ ] 语法纠错 Agent
- [ ] 口语评分 Agent
- [ ] ASR/TTS 集成
- [ ] 真实 LLM 接入
- [ ] 前端界面
