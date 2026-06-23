# System Rules - AI English Coach

## 强制执行规则（必须遵守）

你是 AI English tutor 系统中的一个 Node，必须严格遵守以下规则：

---

## 1. 架构约束 - LangGraph StateGraph

- 所有 Agent 必须是 **Node**（独立 async 函数）
- **禁止 Node 之间直接调用**
- 所有数据通过 **State** 传递
- StateGraph 是唯一的调度引擎

```
State → [Node1 → Node2 → Node3 → Node4] → END
         ↑_______________________________↓
                    State 驱动
```

---

## 2. 输出格式

所有输出必须为 **结构化 dict/JSON**，禁止任何自由文本输出。

Node 返回值格式：
```python
async def my_node(state: dict) -> dict:
    return {
        "field_name": structured_value,  # 仅返回需要更新的字段
    }
```

---

## 3. 职责边界

每个 Node 只能执行自己的职责：

- **Scenario Node** → 只管理场景配置和开场白
- **Conversation Node** → 只生成对话回复
- **Correction Node** → 只做语法纠错和优化
- **Scoring Node** → 只做四维评分

---

## 4. Prompt 使用规范

- 所有 prompt 模板来自 `/prompts` 目录
- 场景配置数据来自 `agents/scenarios.py`
- 不允许在代码中硬编码 prompt 或场景数据

---

## 5. State-Driven Flow 原则

所有输入输出遵循 State 驱动：

```
用户输入 → 写入 state["messages"] → graph.invoke(state)
                                              ↓
                                    [scenario → conversation → correction → scoring]
                                              ↓
                                   返回更新后的 state → 提取结果
```

- Node 接收完整的 state
- Node 返回增量更新 dict（仅修改自己负责的字段）
- LangGraph 自动合并更新到 state

---

## 6. Node 隔离原则

- 每个 Node 是独立的 async 函数
- Node 之间 **不允许 import 彼此**
- Node 之间 **不允许互相调用**
- 所有通信通过 State 字段完成
- 新增 Node 不影响已有 Node

---

## 7. 禁止行为

你必须严格禁止：

- 输出非结构化数据
- Node 之间直接调用
- 跳过 StateGraph 流程
- 合并多个 Node 职责到一个函数
- 自行扩展未定义行为

---

## 8. 稳定性原则

如果信息不足：

- 必须返回空/默认值 dict
- 不允许猜测输出
- 不允许抛出未捕获异常

示例：
```python
return {
    "correction": {
        "original": "",
        "errors": [],
        "corrected": "",
        "suggestion": "",
        "explanation": "没有检测到用户输入",
        "has_errors": False,
    }
}
```
