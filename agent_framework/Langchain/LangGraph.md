# LangGraph



## 设计

在人工智能和基于代理的系统领域，LangGraph的出现标志着一个重要的进化。建立在LangChain坚实的基础之上，LangGraph旨在促进循环图的创建——这是开发代理运行时经常需要的一个关键组件。该模块与LangChain生态系统无缝集成，提出了一个创新解决方案，不仅补充了现有功能，还引入了新的实用性维度。

循环或图结构在代理构造中起着关键作用，支撑着决策制定和对话过程的复杂性。这种结构使代理能够基于条件和结果在执行过程中回访先前的状态，模仿现实世界互动的非线性特征。这一能力显著提高了代理管理复杂互动的能力，包括循环依赖、条件逻辑和动态数据流，从而促进了更自然的对话和更精细的任务管理。

LangGraph的一个突破性应用是其支持在循环中运行大型语言模型（LLM）。这种方法为更具适应性的应用程序打开了可能性，这些应用能够处理模糊的用例——那些没有预先定义结果的用例。通过根据新的输入和反馈不断调整响应，循环执行允许模型适应各种情景和需求。这种适应性对于处理不明确或不断变化的情况至关重要，从而增强了模型在导航复杂环境中的有效性。

## 核心组件

LangGraph的核心组成部分是其为状态机创建设计的图形表示组件。这些组件使代理能够基于指定的图执行动作，为开发者提供了定义节点和边的工具。

### StateGraph

用于表示状态图的类；通过传入状态定义（state definition）来初始化；

状态定义代表一个随时间更新的中心状态对象，由图中的节点通过返回对这个状态属性（以键值存储的形式）的操作来更新。

状态属性的更新有两种方式：一是完全覆盖一个属性，适用于希望节点返回属性的新值的场景；二是通过增加其值来更新属性，适用于属性是一个行动列表（或类似）且希望节点返回新的行动并自动添加到属性中的场景。

pseudocode：

```python
from langgraph.graph import StateGraph
from typing import TypedDict, List, Annotated
import Operator


class State(TypedDict):
    input: str
    all_actions: Annotated[List[str], operator.add]


graph = StateGraph(State)
```

### Nodes

在初始化状态图后，可通过`graph.add_node(name, value)`向graph中添加结点。

**`value`**参数是一个函数或LCEL可执行对象，它被调用时应接收一个与State对象结构相同的字典作为输入，并返回一个包含State对象需要更新键值的字典。

```python
graph.add_node("model", model)
graph.add_node("tools", tool_executor)
```

a special `END` node that is used to represent the end of the graph. It is important that your cycles be able to end eventually!



### Edges

有了node之后，通过添加egde，决定结点间的关系。

#### **The Starting Edge**

进入结点，或是第一个执行结点。

```python
graph.set_entry_point("model")

```

#### **Normal Edges**

前一个结点又后一个结点所调用；

tool using：

```python
graph.add_edge("tools", "model")

```

#### **Conditional Edges**

通过LLM的function calling能力，决定下一个结点往哪走（routing）；

创建这种边需要三个要素：

- 上游节点：用其输出来决定下一步动作。

- 函数：用来决定下一个访问的节点，应返回一个字符串。

- 映射：根据函数的输出将其映射到另一个节点，键是函数可能的返回值，值是对应的节点名称。

```python

# 使用LLM function call 调用function：should_continue()
# 决定： 1.end or 2. continue
graph.add_conditional_edge(
    "model",
    should_continue,
    {
        "end": END,
        "continue": "tools"
    }
)

```



### Compile

编译使之runnable

```python
app = graph.compile()

```

## Examples
### Chat Agent Executor

### Plan-and-Execute

### LangGraph Retrieval Agent

### Basic Multi-agent Collaboration

## 参考

<https://python.langchain.com/docs/langgraph#documentation>

<https://github.com/langchain-ai/langgraph>

[Langchain blog: LangGraph](https://blog.langchain.dev/langgraph/)

[Langchaingraph与LCEL解析](https://limafang.github.io/Azula_blogs.github.io/2024/01/13/Langchaingraph%E4%B8%8ELCEL%E8%A7%A3%E6%9E%90.html)