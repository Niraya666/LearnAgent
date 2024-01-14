# MetaGPT 学习笔记

# Chapter 1: Introduction 

<https://docs.deepwisdom.ai/main/zh/guide/get_started/introduction.html>

[《MetaGPT智能体开发入门》教程](https://deepwisdom.feishu.cn/docx/RJmTdvZuPozAxFxEpFxcbiPwnQf)

助教GPTs： <https://chat.openai.com/g/g-F4pnkAK5S-professional-tech-tutorial-assistant>





## 安装

**请确保你的系统已安装Python 3.9+**

```!shell
pip install metagpt
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple metagpt==0.5.2（推荐）
```



### 配置API：

> MetaGPT提供两种种配置OpenAI API key的方法，你可以将自己的OpenAI API key保存为环境变量，这样在你本地网络通畅的情况下（请确保你能够访问到openai）就可以直接使用OpenAI服务

```!shell
import os
os.environ["OPENAI_API_KEY"] = "sk-..."  # 填入你自己的OpenAI API key
os.environ["OPENAI_API_MODEL"] = "intended model" # 选择你要使用的模型，例如：gpt-4, gpt-3.5-turbo
os.environ["OPENAI_API_BASE"] = "https://api.openai-forward.com/v1"

```

同时MetaGPT还提供了利用`config.yaml`文件来配置OpenAI API服务的方法

1. 在当前项目的工作目录下，新建一个文件夹`config`并在该文件夹下添加一个`config.yaml`或`key.yaml`文件

2. 拷贝样例配置 [config.yaml](https://github.com/geekan/MetaGPT/blob/main/config/config.yaml) 中的内容到你的新文件中。

3. 在新文件内设置自己的OPENAI API KEY配置



**MetaGPT将会按照下述优先级来读取你的配置：`config/key.yaml > config/config.yaml > environment variable`**

### 尝试

```!shell
#  software startup example

import asyncio
from metagpt.roles import (
    Architect,
    Engineer,
    ProductManager,
    ProjectManager,
)
from metagpt.team import Team

async def startup(idea: str):
    company = Team()
    company.hire(
        [
            ProductManager(),
            Architect(),
            ProjectManager(),
            Engineer(),
        ]
    )
    company.invest(investment=3.0)
    company.start_project(idea=idea)

    await company.run(n_round=5)


asyncio.run(startup(idea="write a cli blackjack game")) # blackjack: 二十一点

```

在notebook中执行下面命令，运行并得到生成的游戏代码

```python
await startup(idea="write a cli blackjack game") # blackjack: 二十一点
```



## 补充
<details>
  <summary>协程与异步IO</summary>
  **[协程与异步IO](https://www.liujiangblog.com/course/python/83)**
  
   > **协程，又称微线程，英文名`Coroutine`**，是运行在单线程中的“并发”，协程相比多线程的一大优势就是省去了多线程之间的切换开销，获得了更高的运行效率。Python中的异步IO模块asyncio就是基本的协程模块。

   > **进程/线程：操作系统提供的一种并发处理任务的能力。**
   >
   > **协程：程序员通过高超的代码能力，在代码执行流程中人为的实现多任务并发，是单个线程内的任务调度技巧。**

   > **yield的语法规则是：在yield这里暂停函数的执行，并返回yield后面表达式的值（默认为None），直到被next()方法再次调用时，从上次暂停的yield代码处继续往下执行。**当没有可以继续next()的时候，抛出异常，该异常可被for循环处理。

   ```python
   def fib(n):
       a, b = 0, 1
       i = 0
       while i < n:
           yield b
           a, b = b, a+b
           i += 1
   
   f = fib(10)
   for item in f:
       print(item)
   ```

   > **每个生成器都可以执行send()方法，为生成器内部的yield语句发送数据**。此时yield语句不再只是`yield xxxx`的形式，还可以是`var = yield xxxx`的赋值形式。**它同时具备两个功能，一是暂停并返回函数，二是接收外部send()方法发送过来的值，重新激活函数，并将这个值赋值给var变量！**

   ```python
   def simple_coroutine():
       print('-> 启动协程')
       y = 10
       x = yield y
       print('-> 协程接收到了x的值:', x)
   
   my_coro = simple_coroutine()
   ret = next(my_coro)
   print(ret)
   my_coro.send(10)
   
   """
   1. `my_coro = simple_coroutine()` - 这行创建了协程的一个实例。
   
   2. `ret = next(my_coro)` - 使用 `next()` 函数开始协程的执行。协程会运行到第一个 `yield` 表达式，然后暂停，并返回 `y` 的值（10）。这个值被存储在变量 `ret` 中。
   
   3. `print(ret)` - 打印 `ret` 的值，结果应该是10。
   
   4. `my_coro.send(10)` - 这行将值10发送回协程。协程从 `yield` 语句处恢复执行，10被赋值给变量 `x`，然后协程继续执行直到完成。
   """
   ```

   > 因为send()方法的参数会成为暂停的yield表达式的值，所以，仅当协程处于暂停状态时才能调用 send()方法，例如`my_coro.send(10)`。不过，如果协程还没激活（状态是`'GEN_CREATED'`），就立即把None之外的值发给它，会出现TypeError。因此，始终要先调用`next(my_coro)`激活协程（也可以调用`my_coro.send(None)`），这一过程被称作预激活。

   > **@asyncio.coroutine：asyncio模块中的装饰器，用于将一个生成器声明为协程。**
   >
   > **yield from 其实就是等待另外一个协程的返回。**

   ```python
   def func():
       for i in range(10):
           yield i
   
   print(list(func()))
   
   ###########
   
   def func():
       yield from range(10)
   
   print(list(func()))
   ```

   ```python
   import asyncio
   import datetime
   
   @asyncio.coroutine  # 声明一个协程
   def display_date(num, loop):
       end_time = loop.time() + 10.0
       while True:
           print("Loop: {} Time: {}".format(num, datetime.datetime.now()))
           if (loop.time() + 1.0) >= end_time:
               break
           yield from asyncio.sleep(2)  # 阻塞直到协程sleep(2)返回结果
   loop = asyncio.get_event_loop()  # 获取一个event_loop
   tasks = [display_date(1, loop), display_date(2, loop)]
   loop.run_until_complete(asyncio.gather(*tasks))  # "阻塞"直到所有的tasks完成
   loop.close()
   ```

   > Python3.5中对协程提供了更直接的支持，引入了`async/await`关键字。上面的代码可以这样改写：使用`async`代替`@asyncio.coroutine`，使用`await`代替`yield from`，代码变得更加简洁可读。从Python设计的角度来说，`async/await`让协程独立于生成器而存在，不再使用yield语法。

   ```python
   import asyncio
   import datetime
   
   async def display_date(num, loop):      # 注意这一行的写法
       end_time = loop.time() + 10.0
       while True:
           print("Loop: {} Time: {}".format(num, datetime.datetime.now()))
           if (loop.time() + 1.0) >= end_time:
               break
           await asyncio.sleep(2)  # 阻塞直到协程sleep(2)返回结果
   
   loop = asyncio.get_event_loop()  # 获取一个event_loop
   tasks = [display_date(1, loop), display_date(2, loop)]
   loop.run_until_complete(asyncio.gather(*tasks))  # "阻塞"直到所有的tasks完成
   loop.close()
   ```

   > asyncio的使用可分三步走：
   >
   > 1. 创建事件循环
   >
   > 2. 指定循环模式并运行
   >
   > 3. 关闭循环
   >
   > 通常我们使用`asyncio.get_event_loop()`方法创建一个循环。

   > 运行循环有两种方法：一是调用`run_until_complete()`方法，二是调用`run_forever()`方法。`run_until_complete()`内置`add_done_callback`回调函数，`run_forever()`则可以自定义`add_done_callback()`

</details>

# Chapter 2: Agent

[【直播回放】MetaGPT作者深度解析直播回放\_哔哩哔哩\_bilibili](https://www.bilibili.com/video/BV1Ru411V7XL/?spm_id_from=333.337.search-card.all.click)

[基于大语言模型的AI Agents—Part 1 | ](https://www.breezedeus.com/article/ai-agent-part1)[Breezedeus.com](Breezedeus.com)

What is Agent: according to MG

> 智能体 = LLM+观察+思考+行动+记忆
>
> 多智能体 = 智能体+环境+SOP+评审+路由+订阅+经济







# Chapter 3: MetaGPT框架组件介绍

在MetaGPT看来，可以将智能体想象成环境中的数字人，其中

> 智能体 = 大语言模型（LLM） + 观察 + 思考 + 行动 + 记忆



![](https://docs.deepwisdom.ai/main/assets/agent_run_flowchart.6c04f3a2.png)

> **在MetaGPT内 `Role` 类是智能体的逻辑抽象**

## Action
<details>
  <summary>源码</summary>


   <https://github.com/geekan/MetaGPT/blob/main/metagpt/actions/action.py>

   这段代码定义了一个Python类的继承体系，用于创建和注册不同种类的“行动”或任务，这些任务能够与LLM交互。代码中类和功能的作用如下：

   1. `action_subclass_registry`：这是一个字典，用于保存所有从`Action`基类继承的子类。

   2. `Action` 类：这是一个基础类，表示一个通用的行动。它们带有以下属性：

      - `name`: 行动的名称。

      - `llm`: 和LLM的接口。默认情况下，此处使用`LLM`类的实例，但通过设置为`exclude=True`来避免在`dict`输出中包括此字段。

      - `context`: 行动的上下文，可以是各种类型，用于行动的执行。

      - `prefix`: 系统消息的前缀。在函数`set_prefix`中被使用。

      - `desc`: 行动描述，用于技能管理。

      - `node`: `ActionNode` 类的实例，用于具体定义行动的结构和行为。

      - `builtin_class_name`: 类的名称，子类的名称会自动设置为此变量的值。

   3. `__init_with_instruction` 方法：初始化方法，用于根据指令创建`ActionNode`实例。

   4. `__init__` 方法：覆盖基类的初始化方法。除了父类的初始化外，还动态地反序列化子类，并注册其类名。如果提供了`instruction`参数，还会调用`__init_with_instruction`方法。

   5. `__init_subclass__` 方法：当创建`Action`的子类时自动调用，将子类注册到`action_subclass_registry`字典中。

   6. `dict` 方法：覆盖基类的`dict`方法，从返回的字典中移除`llm`属性。

   7. `set_prefix` 方法：用于设置消息前缀，并更新相关的`llm`系统提示和`node`的`llm`属性。

   8. `__str__` 和 `__repr__` 方法：定义类实例的字符串表示，方便打印和调试。

   9. `_aask` 方法：异步发送提示到`llm`，通常用于获取模型生成的文本。这个方法默认附加了`prefix`属性作为系统消息。

   10. `_run_action_node` 方法：异步运行与`ActionNode`相关的逻辑，这通常涉及使用上下文信息和`llm`与具体的节点交互。

   11. `run` 方法：是一个异步方法，通常由子类实现具体的行动。如果`node`被设置，则会使用`_run_action_node`处理相关逻辑，否则会提示未实现错误。

</details>   

   

简单例子：simpleCoder

````python
import re
import asyncio
from metagpt.actions import Action

class SimpleWriteCode(Action):

    PROMPT_TEMPLATE = """
    Write a python function that can {instruction} and provide two runnnable test cases.
    Return ```python your_code_here ``` with NO other texts,
    your code:
    """

    def __init__(self, name="SimpleWriteCode", context=None, llm=None):
        super().__init__(name, context, llm)

    async def run(self, instruction: str):

        prompt = self.PROMPT_TEMPLATE.format(instruction=instruction)

        rsp = await self._aask(prompt)

        code_text = SimpleWriteCode.parse_code(rsp)

        return code_text

    @staticmethod
    def parse_code(rsp):
        pattern = r'```python(.*)```'
        match = re.search(pattern, rsp, re.DOTALL)
        code_text = match.group(1) if match else rsp
        return code_text
````





## Message

![message.png](./image/message.png)
<details>
  <summary>源码</summary>


   <https://github.com/geekan/MetaGPT/blob/main/metagpt/schema.py>

   `Message` 类用于表示消息数据，并提供了多种方法来处理消息内容和属性。
</details>  


## Role

<details>
  <summary>源码</summary>

   <https://github.com/geekan/MetaGPT/blob/main/metagpt/roles/role.py>

   run 方法： 

   > 如果有入参message就将message添加到role的记忆中如果没有入参就观察环境中的新消息

   ```python
   async def run(self, message=None):
       """Observe, and think and act based on the results of the observation
           观察，并根据观察结果进行思考和行动。"""
       if message:
           if isinstance(message, str):
               message = Message(message)
           if isinstance(message, Message):
               self.recv(message)
           if isinstance(message, list):
               self.recv(Message("\n".join(message)))
            '''如果message存在，它会检查message的类型，
               如果是字符串，则将其转换为Message对象；
               如果是Message对象，则直接调用recv方法；
               如果是列表，则将列表中的消息合并成一个新的消息，然后再调用recv方法。
               相当于预处理将入参转化为Message对象并添加到role的记忆中'''
       elif not await self._observe():
           # If there is no new information, suspend and wait
           logger.debug(f"{self._setting}: no news. waiting.")
           return
   
       rsp = await self.react()
       # Publish the reply to the environment, waiting for the next subscriber to process
       self._publish_message(rsp)
       return rsp
   
   ```

   本质：Observe, and think and act based on the results of the observation （即ReAct）

   ```python
   async def react(self) -> Message:
       """Entry to one of three strategies by which Role reacts to the observed Message
           通过观察到的消息，角色对其中一种策略进行反应。"""
       if self._rc.react_mode == RoleReactMoRoleReactMode.REACTde.REACT:
           rsp = await self._react()
       elif self._rc.react_mode == RoleReactMode.BY_ORDER:
           rsp = await self._act_by_order()
       elif self._rc.react_mode == RoleReactMode.PLAN_AND_ACT:
           rsp = await self._plan_and_act()
       self._set_state(state=-1) # current reaction is complete, reset state to -1 and todo back to None
       return rsp
   
   
   async def _react(self) -> Message:
           """Think first, then act, until the Role _think it is time to stop and requires no more todo.
           This is the standard think-act loop in the ReAct paper, which alternates thinking and acting in task solving, i.e. _think -> _act -> _think -> _act -> ... 
           Use llm to select actions in _think dynamically
           """
           actions_taken = 0
           rsp = Message("No actions taken yet") # will be overwritten after Role _act
           while actions_taken < self._rc.max_react_loop:
               # think
               await self._think()
               if self._rc.todo is None:
                   break
               # act
               logger.debug(f"{self._setting}: {self._rc.state=}, will do {self._rc.todo}")
               rsp = await self._act()
               actions_taken += 1
           return rsp # return output from the last action
   ```
</details> 
   

   

例子：  

```python
class SimpleCoder(Role):
    def __init__(
        self,
        name: str = "Alice",
        profile: str = "SimpleCoder",
        **kwargs,
    ):
        super().__init__(name, profile, **kwargs)
        self._init_actions([SimpleWriteCode])

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: ready to {self._rc.todo}")
        todo = self._rc.todo  # todo will be SimpleWriteCode()

        msg = self.get_memories(k=1)[0]  # find the most recent messages

        code_text = await todo.run(msg.content)
        msg = Message(content=code_text, role=self.profile,
                      cause_by=type(todo))

        return msg
```





## 实现：技术文档助手

> 因为token限制的原因，我们先通过 `LLM` 大模型生成教程的目录，再对目录按照二级标题进行分块，对于每块目录按照标题生成详细内容，最后再将标题和内容进行拼接，解决 `LLM` 大模型长文本的限制问题。



![0c9a7853-b605-4337-b167-b7077fc38c7a.png](./image/0c9a7853-b605-4337-b167-b7077fc38c7a.png)

1. Action: `WriteDirectory`

   根据用户需求生成文章大纲

   prompt 参考：

   ```python
   COMMON_PROMPT = """
           You are now a seasoned technical professional in the field of the internet. 
           We need you to write a technical tutorial with the topic "{topic}".
           您现在是互联网领域的经验丰富的技术专业人员。
           我们需要您撰写一个关于"{topic}"的技术教程。
           """
   
   DIRECTORY_PROMPT = COMMON_PROMPT + """
           Please provide the specific table of contents for this tutorial, strictly following the following requirements:
           1. The output must be strictly in the specified language, {language}.
           2. Answer strictly in the dictionary format like {{"title": "xxx", "directory": [{{"dir 1": ["sub dir 1", "sub dir 2"]}}, {{"dir 2": ["sub dir 3", "sub dir 4"]}}]}}.
           3. The directory should be as specific and sufficient as possible, with a primary and secondary directory.The secondary directory is in the array.
           4. Do not have extra spaces or line breaks.
           5. Each directory title has practical significance.
           请按照以下要求提供本教程的具体目录：
           1. 输出必须严格符合指定语言，{language}。
           2. 回答必须严格按照字典格式，如{{"title": "xxx", "directory": [{{"dir 1": ["sub dir 1", "sub dir 2"]}}, {{"dir 2": ["sub dir 3", "sub dir 4"]}}]}}。
           3. 目录应尽可能具体和充分，包括一级和二级目录。二级目录在数组中。
           4. 不要有额外的空格或换行符。
           5. 每个目录标题都具有实际意义。
           """
   ```

   而后，根据LLM的rsp: str, 通过`extract_struct` function extract 相应的数据结构，如：

   ```python
   
   >>> text = 'xxx {"x": 1, "y": {"a": 2, "b": {"c": 3}}} xxx'
   >>> result_dict = OutputParser.extract_struct(text, "dict")
   >>> print(result_dict)
   >>> # Output: {"x": 1, "y": {"a": 2, "b": {"c": 3}}}
   ```

2. Action: `WriteContent`

   根据传入的子标题来生成内容

   参考prompt：

   ```python
   COMMON_PROMPT = """
           You are now a seasoned technical professional in the field of the internet. 
           We need you to write a technical tutorial with the topic "{topic}".
           """
           CONTENT_PROMPT = COMMON_PROMPT + """
           Now I will give you the module directory titles for the topic. 
           Please output the detailed principle content of this title in detail. 
           If there are code examples, please provide them according to standard code specifications. 
           Without a code example, it is not necessary.
   
           The module directory titles for the topic is as follows:
           {directory}
   
           Strictly limit output according to the following requirements:
           1. Follow the Markdown syntax format for layout.
           2. If there are code examples, they must follow standard syntax specifications, have document annotations, and be displayed in code blocks.
           3. The output must be strictly in the specified language, {language}.
           4. Do not have redundant output, including concluding remarks.
           5. Strict requirement not to output the topic "{topic}".
           现在我将为您提供该主题的模块目录标题。
           请详细输出此标题的详细原理内容。
           如果有代码示例，请按照标准代码规范提供。
           没有代码示例则不需要提供。
           
           该主题的模块目录标题如下：
           {directory}
           
           严格按照以下要求限制输出：
           1. 遵循Markdown语法格式进行布局。
           2. 如果有代码示例，必须遵循标准语法规范，具备文档注释，并以代码块形式显示。
           3. 输出必须严格使用指定语言{language}。
           4. 不得有冗余输出，包括总结性陈述。
           5. 严禁输出主题"{topic}"。
           """
   ```

3. Role: `TutorialAssistant`

   ReAct

   流程：

   ![436ebc22-98d4-41eb-a825-9297feb7b429.png](./image/436ebc22-98d4-41eb-a825-9297feb7b429.png)

Before running the code, create a config folder in the path where the code will be executed, and place the config.yaml file containing information such as the OpenAI API key into it.
code: [simple coder](./code/agent101_simple_coder.ipynb)
code: [tutorial Assistant & homework](./code/agent101_TutorialAssistant.ipynb)


# Chapter 4: 订阅智能体