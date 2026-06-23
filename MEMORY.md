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

### 5. 可扩展性原则
系统支持：
- 新增 Node（只需在 State + graph 中注册）
- 新增场景（只需在 `scenarios.py` 添加配置）
- 替换 LLM provider（config/providers.py 工厂模式）

---

## Node 职责定义

| Node | 文件 | 职责 | 写入 State 字段 |
|------|------|------|----------------|
| scenario | `scenario_node.py` | 场景初始化、生成开场白 | `messages`, `metadata`, `scenario_goal` |
| conversation | `conversation_node.py` | 生成 AI 对话回复 | `messages`, `ai_reply` |
| correction | `correction_node.py` | 语法纠错、表达优化 | `correction` |
| scoring | `scoring_node.py` | 四维评分（fluency/grammar/vocabulary/naturalness） | `score` |

---

## 禁止事项

- 禁止 UI 逻辑进入后端 Node
- 禁止 Node 之间直接调用
- 禁止非结构化输出
- 禁止破坏 StateGraph 流程
- 禁止简化为单体模型调用
- 禁止保留 Orchestrator 模式（已废弃）

---

## 当前开发阶段

**Stage 3**：LangGraph 状态图架构迁移完成
- 纯 StateGraph 多Agent系统
- Node 隔离 + State 驱动
- FastAPI RESTful API

---

## 设计理念

本项目目标不是 demo，而是：

**可扩展 AI 教学系统架构（Production-ready design）**

基于 LangGraph 的状态机架构天然支持：
- 条件分支（如：纠错后决定是否跳过评分）
- 循环迭代（如：用户要求重新生成）
- 并行执行（如：纠错和评分同时运行）
- 持久化（Checkpoint 支持会话恢复）
