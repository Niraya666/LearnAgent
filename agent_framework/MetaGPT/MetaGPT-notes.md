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

+ **[协程与异步IO](https://www.liujiangblog.com/course/python/83)**

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


