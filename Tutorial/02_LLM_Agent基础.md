# 第二章：LLM Agent 基础

## 2.1 什么是 LLM Agent？

LLM Agent 是基于大型语言模型（Large Language Model）的智能系统，能够：
- 理解自然语言指令
- 进行多步推理
- 调用外部工具
- 与环境交互

### Agent 的核心能力

1. **规划能力**：将复杂任务分解为子任务
2. **记忆能力**：保持对话上下文和历史信息
3. **工具使用**：调用 API、搜索引擎、代码执行器
4. **自我反思**：评估和修正自己的行为

## 2.2 Agent 的基本架构

```
┌─────────────────────────────────────────────────────────────┐
│                      LLM Agent 架构                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐                                           │
│  │   用户输入   │                                           │
│  └──────┬──────┘                                           │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                     核心控制器                          ││
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐           ││
│  │  │   规划器   │  │   记忆    │  │   工具库  │           ││
│  │  │ (Planner) │  │ (Memory)  │  │ (Tools)   │           ││
│  │  └───────────┘  └───────────┘  └───────────┘           ││
│  └─────────────────────────────────────────────────────────┘│
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────┐                                           │
│  │   执行动作   │                                           │
│  └─────────────┘                                           │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────┐                                           │
│  │   获取反馈   │ ←────────────── 环境响应                  │
│  └─────────────┘                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 2.3 ReAct 范式

ReAct（Reasoning + Acting）是一种将推理和行动交织的 Agent 范式。

### 核心思想

> "Thought → Action → Observation" 循环

### Search-R1 的实现

```python
# 伪代码展示 Agent 循环
while not done:
    # 1. Thought: LLM 生成思考
    thought = llm.generate(prompt + history)
    
    # 2. Action: 解析动作
    if "<search>" in thought:
        query = extract_query(thought)
        results = search_api(query)  # 执行搜索
    elif "<answer>" in thought:
        answer = extract_answer(thought)
        done = True  # 任务完成
    
    # 3. Observation: 将结果加入上下文
    history += results
```

### 搜索标签格式

```
<search>query to search for</search>
<answer>final answer here</answer>
```

## 2.4 工具使用（Tool Use）

### 为什么要使用工具？

1. **扩展能力边界**：弥补 LLM 知识截止的不足
2. **提高准确性**：获取实时信息和精确计算
3. **执行具体任务**：代码执行、文件操作等

### Search-R1 支持的工具

| 工具 | 功能 | 使用场景 |
|------|------|----------|
| BM25Retriever | 稀疏文本检索 | 快速关键词搜索 |
| DenseRetriever | 稠密向量检索 | 语义相似度搜索 |
| Google Search | 在线网页搜索 | 实时信息获取 |
| SerpAPI | 商业搜索 API | 结构化搜索结果 |

### 工具定义示例

```python
# Search-R1 中的搜索工具定义
SEARCH_TOOL = {
    "name": "search",
    "description": "Search for information on the web",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query"
            }
        },
        "required": ["query"]
    }
}
```

## 2.5 提示工程 (Prompt Engineering)

### 零样本提示

```
Answer the following question:
Question: {question}
Answer: Let me search for this.
```

### 少样本提示

```
Example 1:
Question: What is the capital of France?
Thought: I need to search for this.
Search: capital of France
Observation: Paris is the capital of France.
Answer: Paris

Now answer:
Question: {question}
```

### Search-R1 使用的提示模板

```python
# 典型的 Agent prompt 结构
SYSTEM_PROMPT = """You are a helpful assistant with a unique capability 
to search for information when needed. When you need to verify facts 
or gather information, use the <search>query</search> tag. 
When you have the final answer, use the <answer>...</answer> tag."""

USER_PROMPT = """Question: {question}

Please answer the question. You can use search to gather information 
if needed. Think step by step."""
```

## 2.6 多轮推理循环

### 循环终止条件

1. **达到最大轮次**：`turns >= max_turns`
2. **收到有效答案**：成功解析 `<answer>` 标签
3. **超出 token 限制**：生成的 token 超过 max_length

### 状态管理

```python
class AgentState:
    def __init__(self, question: str):
        self.question = question
        self.history = []  # [(role, content), ...]
        self.search_count = 0
        self.done = False
        
    def add_thought(self, thought: str):
        self.history.append(("assistant", thought))
        
    def add_observation(self, obs: str):
        self.history.append(("user", obs))
    
    def get_input(self) -> str:
        return format_conversation(self.question, self.history)
    
    def to_dict(self):
        return {
            "question": self.question,
            "history": self.history,
            "search_count": self.search_count,
            "done": self.done
        }
```

## 2.7 Agent 的训练方法

### SFT（监督微调）

- 使用人工标注的推理轨迹
- 简单但依赖高质量数据

### RL（强化学习）

- **PPO/GRPO**：通过奖励信号学习
- **RLAIF**：使用 AI 反馈

### Search-R1 的创新

1. **多轮搜索交互**：LLM 和搜索引擎的交织
2. **规则奖励**：基于答案正确性的稀疏奖励
3. **端到端训练**：从零训练搜索推理能力

## 2.8 与传统 RL 的对比

| 方面 | 传统 RL | LLM Agent RL |
|------|---------|--------------|
| 状态空间 | 低维向量 | 文本序列 |
| 动作空间 | 有限离散/连续 | 词表大小 (~100k) |
| 奖励 | 即时密集 | 稀疏延迟 |
| 信用分配 | 短时 | 极长序列 |
| 探索方式 | 随机扰动 | 基于 LLM 的采样 |

## 2.9 工程实现要点

### 批处理

```python
# 一次性处理多个请求
batch_prompts = [format_prompt(q) for q in questions]
batch_outputs = model.generate(batch_prompts)
```

### 缓存机制

```python
# 避免重复搜索相同查询
query_cache = {}
def search_with_cache(query):
    if query not in query_cache:
        query_cache[query] = real_search(query)
    return query_cache[query]
```

### 超时处理

```python
import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)  # 30秒超时

try:
    result = search_api(query)
    signal.alarm(0)
except TimeoutException:
    result = "Search timeout"
```

## 2.10 Chain-of-Thought (CoT) vs ReAct

### CoT（思维链）

```
Question: If a train travels 120 miles in 2 hours, what is its speed?
Thought: I need to calculate speed using distance/time.
Thought: Distance = 120 miles, Time = 2 hours
Thought: Speed = 120 / 2 = 60 miles per hour
Answer: 60 mph
```

### ReAct（推理+行动）

```
Question: What year did the Titanic sink?
Thought: I'm not sure about this fact, I should search for it.
Search: Titanic sinking year
Observation: [Results show April 15, 1912]
Thought: The search confirms it was 1912.
Answer: 1912
```

### 对比

| 特性 | CoT | ReAct |
|------|-----|-------|
| 外部信息 | ✗ | ✓ |
| 实时查询 | ✗ | ✓ |
| 幻觉风险 | 较高 | 较低 |
| 推理速度 | 快 | 较慢 |

## 2.11 本章小结

| 概念 | 说明 |
|------|------|
| LLM Agent | 基于 LLM 的智能系统 |
| ReAct | 推理+行动的交替范式 |
| Tool Use | Agent 调用外部工具的能力 |
| 多轮推理 | 迭代调用 LLM 逐步解决问题 |
| 状态管理 | 维护对话历史和上下文 |

## 2.12 思考题

1. 为什么 LLM Agent 需要多轮交互而不是一次性回答？
2. ReAct 和 CoT（Chain of Thought）有什么区别？
3. 如何防止 Agent 进入无限循环？
4. 如何设计有效的 prompt 让 Agent 学会使用工具？

## 2.13 推荐阅读

- Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (2022)
- Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (2022)
- Schick et al., "Toolformer: Language Models Can Teach Themselves to Use Tools" (2023)
- Park et al., "MetaGPT: Multi-Agent Framework for Collaborative Problem Solving" (2023)