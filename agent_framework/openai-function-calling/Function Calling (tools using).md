# Function Calling (tools using)

> 最近，OpenAI已经废弃了"函数"的使用，转而采用"工具"。现在，在配置Chat Completions API时，使用tools参数来提供模型可能生成JSON输入的函数列表。这一更改还在API的响应对象中引入了一个新的finish_reason，称为"tool_calls"，表明之前的function_call完成原因已被废弃。从使用"函数"转向"工具"的转变，表明OpenAI意图拓宽可以集成到模型中的功能范围，潜在地为更复杂的交互铺平道路，比如创建一个多代理系统，不同的OpenAI托管工具或代理可以相互作用

## What is function calling 

> function calling 允许LLM在执行过程中通过指定的参数来调用并执行一个特定的函数，以实现代码的重用和模块化处理。

函数调用功能能够从模型中更可靠地获取结构化的数据回应。在API调用过程中，你可以描述想要执行的功能，并让模型智能选择输出一个包含所需参数的JSON对象，以便调用一个或多个函数。需要注意的是，Chat Completions  API本身并不直接执行任何函数调用；而是生成了一个JSON，你可以利用这个JSON在你的代码中实现函数的调用。





函数调用的主要用途包括但不限于简化代码结构，实现代码重用，提高开发效率；模块化设计，便于代码管理和维护；实现特定功能，如数据处理、用户输入处理、接口调用等，以及促进团队协作和项目的可扩展性。

函数调用功能让你能够从模型中更可靠地获取结构化的数据回应。例如，你可以：

- 创建能够通过调用外部API（如ChatGPT插件）来回答问题的智能助手。

   - 举例来说，你可以定义一个发送邮件的函数**`send_email(to: string, body: string)`**，或者获取当前天气的函数**`get_current_weather(location: string, unit: 'celsius' | 'fahrenheit')`**。

- 把自然语言指令转换为API调用指令。

   - 比如，将“谁是我的主要客户？”的查询转换为调用你内部API的**`get_customers(min_revenue: int, created_before: string, limit: int)`**。

- 从文本中提取结构化数据。

   - 例如，定义一个提取数据的函数**`extract_data(name: string, birthday: string)`**，或执行SQL查询的**`sql_query(query: string)`**。

函数调用的基本操作步骤包括：

1. 结合用户的查询内容和你在functions 参数中定义的一组函数，调用模型。

2. 模型可能会选择调用一个或多个函数。如果是这样，它会生成一个遵循你自定义模式的JSON对象字符串（注意：模型可能会错误地添加参数）。

3. 在你的代码中解析这个JSON字符串，并根据提供的参数调用相应的函数（如果这些参数存在的话）。

4. 将函数的响应作为一个新的消息加入，再次调用模型，让模型把结果总结反馈给用户。

通过这样的方式，函数调用不仅能让交互更加自动化，还能帮助开发者更方便地集成和利用模型的强大能力，同时确保在操作执行前有一个明确的用户确认环节，保障操作的安全性和准确性。

## quick start: 如何使用

### 实现天气查询



```python
!pip install scipy --quiet
!pip install tenacity --quiet
!pip install tiktoken --quiet
!pip install termcolor --quiet
!pip install openai --quiet

os.environ["OPENAI_API_KEY"] = "..."
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored

client = OpenAI()

```

Define a few utilities for making calls to the Chat Completions API and for maintaining and keeping track of the conversation state.

```python
@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, tools=None, tool_choice=None, model=deployment_name):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e


def pretty_print_conversation(messages):
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "function": "magenta",
    }

    for message in messages:
        if message["role"] == "system":
            print(colored(f"system: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "user":
            print(colored(f"user: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "assistant" and message.get("function_call"):
            print(colored(f"assistant: {message['function_call']}\n", role_to_color[message["role"]]))
        elif message["role"] == "assistant" and not message.get("function_call"):
            print(colored(f"assistant: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "function":
            print(colored(f"function ({message['name']}): {message['content']}\n", role_to_color[message["role"]]))

```



Create some function specifications to interface with a hypothetical weather API

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                    },
                },
                "required": ["location", "format"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_n_day_weather_forecast",
            "description": "Get an N-day weather forecast",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                    },
                    "num_days": {
                        "type": "integer",
                        "description": "The number of days to forecast",
                    }
                },
                "required": ["location", "format", "num_days"]
            },
        }
    },
]
```

If we prompt the model about the current weather, it will respond with some clarifying questions.

```python
messages = []
messages.append({"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})
messages.append({"role": "user", "content": "What's the weather like today"})
chat_response = chat_completion_request(
    messages, tools=tools
)
assistant_message = chat_response.choices[0].message
messages.append(assistant_message)
assistant_message
```

output: 

```shell
ChatCompletionMessage(content='Sure, may I know your current location?', role='assistant', function_call=None, tool_calls=None)
```



Once we provide the missing information, it will generate the appropriate function arguments for us.

```python
messages.append({"role": "user", "content": "I'm in Shanghai, China."})
chat_response = chat_completion_request(
    messages, tools=tools
)
assistant_message = chat_response.choices[0].message
messages.append(assistant_message)
assistant_message
```

output:

```shell
ChatCompletionMessage(content=None, role='assistant', function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_VdqOOMp9pagf5ho39Y2HmYV4', function=Function(arguments='{\n  "location": "San Francisco, CA",\n  "format": "celsius",\n  "num_days": 4\n}', name='get_n_day_weather_forecast'), type='function')])
```



Get it to target the other function we've told it about.

```python
messages = []
messages.append({"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})
messages.append({"role": "user", "content": "what is the weather going to be like in Glasgow, Scotland over the next x days"})
messages.append({"role": "user", "content": "in 5 days"})

chat_response = chat_completion_request(
    messages, tools=tools
)
assistant_message = chat_response.choices[0].message
messages.append(assistant_message)
assistant_message
```

output:

```shell
ChatCompletionMessage(content=None, role='assistant', function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_jYEPycIK4RvjiYr4QX529Lso', function=Function(arguments='{\n  "location": "Glasgow, Scotland",\n  "format": "celsius",\n  "num_days": 5\n}', name='get_n_day_weather_forecast'), type='function')])
```



Force the model to use a specific function:

```python
messages = []
messages.append({"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})
messages.append({"role": "user", "content": "Give me a weather report for Toronto, Canada."})
chat_response = chat_completion_request(
    messages, tools=tools, tool_choice={"type": "function", "function": {"name": "get_n_day_weather_forecast"}}
)
chat_response.choices[0].message
```

output:

```shell
ChatCompletionMessage(content=None, role='assistant', function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_0ldecDpV8Vdq8mGPoUewlue3', function=Function(arguments='{\n  "location": "Toronto, Canada",\n  "format": "celsius",\n  "num_days": 1\n}', name='get_n_day_weather_forecast'), type='function')])
```



Parallel Function Calling: call multiple functions in one turn.

(only support: **`gpt-4-turbo-preview`**, **`gpt-4-0125-preview`**, **`gpt-4-1106-preview`**, **`gpt-3.5-turbo-0125`**, and **`gpt-3.5-turbo-1106`**)

```python
messages = []
messages.append({"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})
messages.append({"role": "user", "content": "what is the weather going to be like in San Francisco and Glasgow over the next 4 days"})
chat_response = chat_completion_request(
    messages, tools=tools, model = "gpt4turbo"
)

assistant_message = chat_response.choices[0].message.tool_calls
assistant_message
```



Try to call function and feedback to model

create some dumpy function waiting for calling.

```python
import json
def get_n_day_weather_forecast(location, format="fahrenheit"):
    """
    This function is for illustrative purposes.
    The location and unit should be used to determine weather
    instead of returning a hardcoded response.
    """
    
    return json.dumps({"temperature": "22", "format": format, "description": "Sunny"})

def get_current_weather(location, format="fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "format": format})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "format": format})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "format": format})
    else:
        return json.dumps({"location": location, "temperature": "unknown", "format": format})

available_functions = {
            "get_current_weather": get_current_weather,
            "get_n_day_weather_forecast": get_n_day_weather_forecast,
        }
```



```python
messages = []
messages.append({"role": "user", "content": "What's the weather like in Tokyo!"})
chat_response = chat_completion_request(messages, tools=tools, tool_choice="auto")
assistant_message = chat_response.choices[0].message
assistant_message = json.loads(assistant_message.model_dump_json())
assistant_message["content"] = str(assistant_message["tool_calls"][0]["function"])

#a temporary patch but this should be handled differently
# remove "function_call" from assistant message
del assistant_message["function_call"]

assistant_message

"""
{'content': '{\'arguments\': \'{\\n  "location": "Tokyo",\\n  "format": "celsius"\\n}\', \'name\': \'get_current_weather\'}',
 'role': 'assistant',
 'tool_calls': [{'id': 'call_Tz8S1HgvnaBzf6CFZP1u4d1J',
   'function': {'arguments': '{\n  "location": "Tokyo",\n  "format": "celsius"\n}',
    'name': 'get_current_weather'},
   'type': 'function'}]}

"""
```

```python
# get the weather information to pass back to the model
weather = get_current_weather(messages[1]["tool_calls"][0]["function"]["arguments"])

messages.append({"role": "tool",
                 "tool_call_id": assistant_message["tool_calls"][0]["id"],
                 "name": assistant_message["tool_calls"][0]["function"]["name"],
                 "content": weather})

messages

"""
[{'role': 'user', 'content': "What's the weather like in Tokyo!"},
 {'content': '{\'arguments\': \'{\\n  "location": "Tokyo",\\n  "format": "celsius"\\n}\', \'name\': \'get_current_weather\'}',
  'role': 'assistant',
  'tool_calls': [{'id': 'call_Tz8S1HgvnaBzf6CFZP1u4d1J',
    'function': {'arguments': '{\n  "location": "Tokyo",\n  "format": "celsius"\n}',
     'name': 'get_current_weather'},
    'type': 'function'}]},
 {'role': 'tool',
  'tool_call_id': 'call_Tz8S1HgvnaBzf6CFZP1u4d1J',
  'name': 'get_current_weather',
  'content': '{"location": "Tokyo", "temperature": "10", "format": "fahrenheit"}'}]

"""
```

```python
final_response = chat_completion_request(messages, tools=tools)
final_response

"""
ChatCompletion(id='chatcmpl-8zG0XOGJm9rLukaKdlea9KnoXUdDP', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='The current weather in Tokyo is 10 degrees Celsius.', role='assistant', function_call=None, tool_calls=None), content_filter_results={'hate': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}})], created=1709610233, model='gpt-35-turbo-16k', object='chat.completion', system_fingerprint=None, usage=CompletionUsage(completion_tokens=12, prompt_tokens=263, total_tokens=275), prompt_filter_results=[{'prompt_index': 0, 'content_filter_results': {'hate': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}}}])
"""
```



### 通过function call 执行sql

download sample db

```shell
wget https://www.sqlitetutorial.net/wp-content/uploads/2018/03/chinook.zip -O chinook.zip
unzip chinook.zip
```

```python
import sqlite3

conn = sqlite3.connect("/content/chinook.db")
print("Opened database successfully")
```

```python
def get_table_names(conn):
    """Return a list of table names."""
    table_names = []
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    for table in tables.fetchall():
        table_names.append(table[0])
    return table_names


def get_column_names(conn, table_name):
    """Return a list of column names."""
    column_names = []
    columns = conn.execute(f"PRAGMA table_info('{table_name}');").fetchall()
    for col in columns:
        column_names.append(col[1])
    return column_names


def get_database_info(conn):
    """Return a list of dicts containing the table name and columns for each table in the database."""
    table_dicts = []
    for table_name in get_table_names(conn):
        columns_names = get_column_names(conn, table_name)
        table_dicts.append({"table_name": table_name, "column_names": columns_names})
    return table_dicts

```

Use these utility functions to extract a representation of the database schema.

```python
database_schema_dict = get_database_info(conn)
database_schema_string = "\n".join(
    [
        f"Table: {table['table_name']}\nColumns: {', '.join(table['column_names'])}"
        for table in database_schema_dict
    ]
)
```

Define a function specification for the function. 

Notice that we are inserting the database schema into the function specification.

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "ask_database",
            "description": "Use this function to answer user questions about music. Input should be a fully formed SQL query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": f"""
                                SQL query extracting info to answer the user's question.
                                SQL should be written using this database schema:
                                {database_schema_string}
                                The query should be returned in plain text, not in JSON.
                                """,
                    }
                },
                "required": ["query"],
            },
        }
    }
]
```



Executing SQL queries

```python
def ask_database(conn, query):
    """Function to query SQLite database with a provided SQL query."""
    try:
        results = str(conn.execute(query).fetchall())
    except Exception as e:
        results = f"query failed with error: {e}"
    return results

def execute_function_call(message):
    if message.tool_calls[0].function.name == "ask_database":
        query = json.loads(message.tool_calls[0].function.arguments)["query"]
        results = ask_database(conn, query)
    else:
        results = f"Error: function {message.tool_calls[0].function.name} does not exist"
    return results
```

```python
messages = []
messages.append({"role": "system", "content": "Answer user questions by generating SQL queries against the Chinook Music Database."})
messages.append({"role": "user", "content": "Hi, who are the top 5 artists by number of tracks?"})
chat_response = chat_completion_request(messages, tools)
assistant_message = chat_response.choices[0].message
assistant_message.content = str(assistant_message.tool_calls[0].function)
messages.append({"role": assistant_message.role, "content": assistant_message.content})
if assistant_message.tool_calls:
    results = execute_function_call(assistant_message)
    messages.append({"role": "function", "tool_call_id": assistant_message.tool_calls[0].id, "name": assistant_message.tool_calls[0].function.name, "content": results})
pretty_print_conversation(messages)

'''
system: Answer user questions by generating SQL queries against the Chinook Music Database.

user: Hi, who are the top 5 artists by number of tracks?

assistant: Function(arguments='{\n  "query": "SELECT artists.Name, COUNT(tracks.TrackId) AS TrackCount FROM artists JOIN albums ON artists.ArtistId = albums.ArtistId JOIN tracks ON albums.AlbumId = tracks.AlbumId GROUP BY artists.Name ORDER BY TrackCount DESC LIMIT 5;"\n}', name='ask_database')

function (ask_database): [('Iron Maiden', 213), ('U2', 135), ('Led Zeppelin', 114), ('Metallica', 112), ('Lost', 92)]

'''
```

```python
messages.append({"role": "user", "content": "What is the name of the album with the most tracks?"})
chat_response = chat_completion_request(messages, tools)
assistant_message = chat_response.choices[0].message
assistant_message.content = str(assistant_message.tool_calls[0].function)
messages.append({"role": assistant_message.role, "content": assistant_message.content})
if assistant_message.tool_calls:
    results = execute_function_call(assistant_message)
    messages.append({"role": "function", "tool_call_id": assistant_message.tool_calls[0].id, "name": assistant_message.tool_calls[0].function.name, "content": results})
pretty_print_conversation(messages)

'''
system: Answer user questions by generating SQL queries against the Chinook Music Database.

user: Hi, who are the top 5 artists by number of tracks?

assistant: Function(arguments='{\n  "query": "SELECT artists.Name, COUNT(tracks.TrackId) AS TrackCount FROM artists JOIN albums ON artists.ArtistId = albums.ArtistId JOIN tracks ON albums.AlbumId = tracks.AlbumId GROUP BY artists.Name ORDER BY TrackCount DESC LIMIT 5;"\n}', name='ask_database')

function (ask_database): [('Iron Maiden', 213), ('U2', 135), ('Led Zeppelin', 114), ('Metallica', 112), ('Lost', 92)]

user: What is the name of the album with the most tracks?

assistant: Function(arguments='{\n  "query": "SELECT albums.Title, COUNT(tracks.TrackId) AS TrackCount FROM albums JOIN tracks ON albums.AlbumId = tracks.AlbumId GROUP BY albums.Title ORDER BY TrackCount DESC LIMIT 1;"\n}', name='ask_database')

function (ask_database): [('Greatest Hits', 57)]


'''
```





### use functions with a knowledge base

We'll create an agent that uses data from arXiv to answer questions about academic subjects. It has two functions at its disposal:

- **get_articles**: A function that gets arXiv articles on a subject and summarizes them for the user with links.

- **read_article_and_summarize**: This function takes one of the previously searched articles, reads it in its entirety and summarizes the core argument, evidence and conclusions.

```python
!pip install scipy --quiet
!pip install tenacity --quiet
!pip install tiktoken==0.3.3 --quiet
!pip install termcolor --quiet
!pip install openai --quiet
!pip install arxiv --quiet
!pip install pandas --quiet
!pip install PyPDF2 --quiet
!pip install tqdm --quiet
```

```python
import os
import arxiv
import ast
import concurrent
import json
import os
import pandas as pd
import tiktoken
from csv import writer
from IPython.display import display, Markdown, Latex
from openai import OpenAI
from PyPDF2 import PdfReader
from scipy import spatial
from tenacity import retry, wait_random_exponential, stop_after_attempt
from tqdm import tqdm
from termcolor import colored

GPT_MODEL = "gpt-3.5-turbo-0613"
EMBEDDING_MODEL = "text-embedding-ada-002"
client = OpenAI()

```

Search utilities: Downloaded papers will be stored in a directory (we use .[/data/papers](https://colab.research.google.com/drive/1gYKbkbF_KJc7YskWtkMrNT4LD2b0tkwy#) here). We create a file arxiv_library.csv to store the embeddings and details for downloaded papers to retrieve against using summarize_text.

```python
directory = './data/papers'

# Check if the directory already exists
if not os.path.exists(directory):
    # If the directory doesn't exist, create it and any necessary intermediate directories
    os.makedirs(directory)
    print(f"Directory '{directory}' created successfully.")
else:
    # If the directory already exists, print a message indicating it
    print(f"Directory '{directory}' already exists.")
```

```python
# Set a directory to store downloaded papers
data_dir = os.path.join(os.curdir, "data", "papers")
paper_dir_filepath = "./data/arxiv_library.csv"

# Generate a blank dataframe where we can store downloaded files
df = pd.DataFrame(list())
df.to_csv(paper_dir_filepath)
```



```python
@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def embedding_request(text):
    response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return response


@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def get_articles(query, library=paper_dir_filepath, top_k=5):
    """This function gets the top_k articles based on a user's query, sorted by relevance.
    It also downloads the files and stores them in arxiv_library.csv to be retrieved by the read_article_and_summarize.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query = query,
        max_results = top_k,
        sort_by = arxiv.SortCriterion.SubmittedDate
    )
    result_list = []
    for result in client.results(search):
        result_dict = {}
        result_dict.update({"title": result.title})
        result_dict.update({"summary": result.summary})

        # Taking the first url provided
        result_dict.update({"article_url": [x.href for x in result.links][0]})
        result_dict.update({"pdf_url": [x.href for x in result.links][1]})
        result_list.append(result_dict)

        # Store references in library file
        response = embedding_request(text=result.title)
        file_reference = [
            result.title,
            result.download_pdf(data_dir),
            response.data[0].embedding,
        ]

        # Write to file
        with open(library, "a") as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(file_reference)
            f_object.close()
    return result_list

def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100,
) -> list[str]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = embedding_request(query)
    query_embedding = query_embedding_response.data[0].embedding
    strings_and_relatednesses = [
        (row["filepath"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n]

def read_pdf(filepath):
    """Takes a filepath to a PDF and returns a string of the PDF's contents"""
    # creating a pdf reader object
    reader = PdfReader(filepath)
    pdf_text = ""
    page_number = 0
    for page in reader.pages:
        page_number += 1
        pdf_text += page.extract_text() + f"\nPage Number: {page_number}"
    return pdf_text


# Split a text into smaller chunks of size n, preferably ending at the end of a sentence
def create_chunks(text, n, tokenizer):
    """Returns successive n-sized chunks from provided text."""
    tokens = tokenizer.encode(text)
    i = 0
    while i < len(tokens):
        # Find the nearest end of sentence within a range of 0.5 * n and 1.5 * n tokens
        j = min(i + int(1.5 * n), len(tokens))
        while j > i + int(0.5 * n):
            # Decode the tokens and check for full stop or newline
            chunk = tokenizer.decode(tokens[i:j])
            if chunk.endswith(".") or chunk.endswith("\n"):
                break
            j -= 1
        # If no end of sentence found, use n tokens as the chunk size
        if j == i + int(0.5 * n):
            j = min(i + n, len(tokens))
        yield tokens[i:j]
        i = j


def extract_chunk(content, template_prompt):
    """This function applies a prompt to some input content. In this case it returns a summarized chunk of text"""
    prompt = template_prompt + content
    response = client.chat.completions.create(
        model=GPT_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0
    )
    return response.choices[0].message.content


def summarize_text(query):
    """This function does the following:
    - Reads in the arxiv_library.csv file in including the embeddings
    - Finds the closest file to the user's query
    - Scrapes the text out of the file and chunks it
    - Summarizes each chunk in parallel
    - Does one final summary and returns this to the user"""

    # A prompt to dictate how the recursive summarizations should approach the input paper
    summary_prompt = """Summarize this text from an academic paper. Extract any key points with reasoning.\n\nContent:"""

    # If the library is empty (no searches have been performed yet), we perform one and download the results
    library_df = pd.read_csv(paper_dir_filepath).reset_index()
    if len(library_df) == 0:
        print("No papers searched yet, downloading first.")
        get_articles(query)
        print("Papers downloaded, continuing")
        library_df = pd.read_csv(paper_dir_filepath).reset_index()
    library_df.columns = ["title", "filepath", "embedding"]
    library_df["embedding"] = library_df["embedding"].apply(ast.literal_eval)
    strings = strings_ranked_by_relatedness(query, library_df, top_n=1)
    print("Chunking text from paper")
    pdf_text = read_pdf(strings[0])

    # Initialise tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")
    results = ""

    # Chunk up the document into 1500 token chunks
    chunks = create_chunks(pdf_text, 1500, tokenizer)
    text_chunks = [tokenizer.decode(chunk) for chunk in chunks]
    print("Summarizing each chunk of text")

    # Parallel process the summaries
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(text_chunks)
    ) as executor:
        futures = [
            executor.submit(extract_chunk, chunk, summary_prompt)
            for chunk in text_chunks
        ]
        with tqdm(total=len(text_chunks)) as pbar:
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)
        for future in futures:
            data = future.result()
            results += data

    # Final summary
    print("Summarizing into overall summary")
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {
                "role": "user",
                "content": f"""Write a summary collated from this collection of key points extracted from an academic paper.
                        The summary should highlight the core argument, conclusions and evidence, and answer the user's query.
                        User query: {query}
                        The summary should be structured in bulleted lists following the headings Core Argument, Evidence, and Conclusions.
                        Key points:\n{results}\nSummary:\n""",
            }
        ],
        temperature=0,
    )
    return response

```



a Conversation class to support multiple turns with the API, and some Python functions to enable interaction between the ChatCompletion API and our knowledge base functions.

```python
@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, functions=None, model=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            functions=functions,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e


class Conversation:
    def __init__(self):
        self.conversation_history = []

    def add_message(self, role, content):
        message = {"role": role, "content": content}
        self.conversation_history.append(message)

    def display_conversation(self, detailed=False):
        role_to_color = {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "function": "magenta",
        }
        for message in self.conversation_history:
            print(
                colored(
                    f"{message['role']}: {message['content']}\n\n",
                    role_to_color[message["role"]],
                )
            )
```

Initiate our get_articles and read_article_and_summarize functions

```python
arxiv_functions = [
    {
        "name": "get_articles",
        "description": """Use this function to get academic papers from arXiv to answer user questions.""",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": f"""
                            User query in JSON. Responses should be summarized and should include the article URL reference
                            """,
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_article_and_summarize",
        "description": """Use this function to read whole papers and provide a summary for users.
        You should NEVER call this function before get_articles has been called in the conversation.""",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": f"""
                            Description of the article in plain text based on the user's query
                            """,
                }
            },
            "required": ["query"],
        },
    }
]
```

```python
def chat_completion_with_function_execution(messages, functions=[None]):
    """This function makes a ChatCompletion API call with the option of adding functions"""
    response = chat_completion_request(messages, functions)
    full_message = response.choices[0]
    if full_message.finish_reason == "function_call":
        print(f"Function generation requested, calling function")
        return call_arxiv_function(messages, full_message)
    else:
        print(f"Function not required, responding to user")
        return response


def call_arxiv_function(messages, full_message):
    """Function calling function which executes function calls when the model believes it is necessary.
    Currently extended by adding clauses to this if statement."""

    if full_message.message.function_call.name == "get_articles":
        try:
            parsed_output = json.loads(
                full_message.message.function_call.arguments
            )
            print("Getting search results")
            results = get_articles(parsed_output["query"])
        except Exception as e:
            print(parsed_output)
            print(f"Function execution failed")
            print(f"Error message: {e}")
        messages.append(
            {
                "role": "function",
                "name": full_message.message.function_call.name,
                "content": str(results),
            }
        )
        try:
            print("Got search results, summarizing content")
            response = chat_completion_request(messages)
            return response
        except Exception as e:
            print(type(e))
            raise Exception("Function chat request failed")

    elif (
        full_message.message.function_call.name == "read_article_and_summarize"
    ):
        parsed_output = json.loads(
            full_message.message.function_call.arguments
        )
        print("Finding and reading paper")
        summary = summarize_text(parsed_output["query"])
        return summary

    else:
        raise Exception("Function does not exist and cannot be called")

```



arXiv conversation, Start with a system message

```python
paper_system_message = """You are arXivGPT, a helpful assistant pulls academic papers to answer user questions.
You summarize the papers clearly so the customer can decide which to read to answer their question.
You always provide the article_url and title so the user can understand the name of the paper and click through to access it.
Begin!"""
paper_conversation = Conversation()
paper_conversation.add_message("system", paper_system_message)

```

```python
# Add a user message
paper_conversation.add_message("user", "Hi, how does PPO reinforcement learning work?")
chat_response = chat_completion_with_function_execution(
    paper_conversation.conversation_history, functions=arxiv_functions
)
assistant_message = chat_response.choices[0].message.content
paper_conversation.add_message("assistant", assistant_message)
display(Markdown(assistant_message))

```

output:

```markdown
Function generation requested, calling function
Getting search results
Got search results, summarizing content
I found several papers related to PPO reinforcement learning. Here are a few summaries:

Title: "Bandit Profit-maximization for Targeted Marketing"

Summary: This paper presents near-optimal algorithms for optimizing profit over multiple demand curves, which are dependent on different ancillary variables while maintaining the same price. It is relevant to PPO reinforcement learning as it tackles a sequential profit-maximization problem.
Article URL: Link
Title: "Inferring potential landscapes: A Schrödinger bridge approach to Maximum Caliber"

Summary: This work extends Schrödinger bridges to account for integral constraints along paths, specifically in the context of Maximum Caliber, a Maximum Entropy principle applied in a dynamic context. While not directly related to PPO reinforcement learning, it can provide insights into stochastic dynamics and inference of time-varying potential landscapes.
Article URL: Link
Title: "a-DCF: an architecture agnostic metric with application to spoofing-robust speaker verification"

Summary: This paper proposes an architecture-agnostic detection cost function (a-DCF) for evaluating spoofing-robust automatic speaker verification (ASV) systems. Although it does not focus on PPO reinforcement learning, it provides a metric for evaluating ASV systems in the presence of spoofing attacks.
Article URL: Link
These papers should provide insights into different aspects of reinforcement learning and related topics.
```

Add another user message to induce our system to use the second tool

```python
paper_conversation.add_message(
    "user",
    "Can you read the PPO sequence generation paper for me and give me a summary",
)
updated_response = chat_completion_with_function_execution(
    paper_conversation.conversation_history, functions=arxiv_functions
)
display(Markdown(updated_response.choices[0].message.content))

```

output:

```markdown
Function generation requested, calling function
Finding and reading paper
Chunking text from paper
Summarizing each chunk of text
100%|██████████| 4/4 [00:04<00:00,  1.11s/it]
Summarizing into overall summary
Core Argument:

The paper discusses the potential of using a general-purpose large language model (LLM) to learn the structural biophysics of DNA.
The authors show that fine-tuning a LLM, specifically chatGPT 3.5-turbo, can enhance its ability to analyze and design DNA sequences and their structures.
The study focuses on the formation of secondary structures in DNA, which are governed by base pairing and stacking bonds.
The authors propose a method that involves chaining together models fine-tuned for subtasks and using a chain-of-thought approach to improve the model's performance.
Evidence:

The authors use the NUPACK software suite to provide data for training and validation.
The expert pipeline approach involves using models that have been fine-tuned for subtasks and feeding their outputs into each other.
The models perform better when they explicitly consider the nearest neighbor window and the reverse complement of the sequences.
The pipeline approach, where a separate model determines the reverse complement and feeds it to another model for secondary structure prediction, enhances the accuracy of the predictions.
The performance of the models improves with larger training sets.
Conclusions:

The study demonstrates the potential of using LLMs to learn DNA structural biophysics.
Integrating experimental data and machine learning is important in scientific research.
The expert pipeline approach and breaking down the problem into smaller subtasks improve the performance of the models in DNA sequence analysis.
The combination of chain-of-thought and model pipeline provides the best results in analysis tasks.
The CoT approach, combined with the reverse complement transformation, yields the highest accuracy in design tasks.
The addition of an error checking layer further improves accuracy in design tasks.
Sequence design is more challenging than analysis, but error correction can compensate for the increased difficulty.
Larger training sets benefit design tasks more.
Future research directions include exploring chaining smaller models for performance improvement and using an LLM architecture involving both an encoder and decoder for direct sequence comparison.
```
In detail: [openai_function_call_quickstart](./code/openai_function_call_quickstart.ipynb)


## function calling via open source LLMs

目前支持tool call 的open source LLMs 主要有

- qwent

- chatGLM-6B

- [NexusRaven-13B](https://github.com/nexusflowai/NexusRaven/)

- [gorilla-openfunctions-v1](https://huggingface.co/gorilla-llm/gorilla-openfunctions-v1)



支持以openai API 形式调用tools call 的推理框架有

- [Xinference](https://inference.readthedocs.io/en/latest/models/model_abilities/tools.html) 



### 快速上手

环境

```python
%pip install -U -q  xinference[transformers] openai langchain
!pip install typing-extensions --upgrade
```

```python
## Start Local Server

!nohup xinference-local  > xinference.log 2>&1 &
```

模型加载

```python
!xinference launch -u my-llm --model-name chatglm3 --size-in-billions 6 --model-format pytorch
```

```python
## Interact with the running model
import openai

messages=[
    {
        "role": "user",
        "content": "Who are you?"
    }
]

client = openai.Client(api_key="empty", base_url=f"http://0.0.0.0:9997/v1")
client.chat.completions.create(
    model="my-llm",
    messages=messages,
)

# ChatCompletion(id='chatda6056ac-da01-11ee-b92e-0242ac1c000c', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content="I am an AI assistant named ChatGLM3-6B, which is developed based on the language model jointly trained by Tsinghua University KEG Lab and Zhipu AI Company in 2023. My job is to provide appropriate answers and support to users' questions and requests.", role='assistant', function_call=None, tool_calls=None))], created=1709541198, model='my-llm', object='chat.completion', system_fingerprint=None, usage=CompletionUsage(completion_tokens=-1, prompt_tokens=-1, total_tokens=-1))
```



without tool using

```python
completion = client.chat.completions.create(
    model="my-llm",
    messages=[{"role": "user", "content": "What is the weather like in London?"}]
)
# print(handle_response(completion))
print(completion.choices[0].message.content)

"""
London has a temperate climate with warm summers and cool winters. The average temperature during the summer months (June to August) is around 18°C, while the winter months (December to February) are around 6°C. The city experiences heavy rainfall throughout the year, with an annual precipitation of around 350 mm. The average precipitation on the weekends is around 40 mm. London's cloudy skies are common throughout the year, but they are especially prevalent in December and January.

"""
```

tool using

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]

def get_completion(messages, model="my-llm", temperature=0, max_tokens=500, tools=None, tool_choice=None):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools,
        tool_choice=tool_choice
    )
    return response.choices[0].message
```

```python
# Defines a dummy function to get the current weather
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    weather = {
        "location": location,
        "temperature": "50",
        "unit": unit,
    }

    return json.dumps(weather)

messages = []
messages.append({"role": "user", "content": "What's the weather like in Boston!"})
assistant_message = get_completion(messages, tools=tools, tool_choice="auto")
assistant_message = json.loads(assistant_message.model_dump_json())
assistant_message["content"] = str(assistant_message["tool_calls"][0]["function"])

#a temporary patch but this should be handled differently
# remove "function_call" from assistant message
del assistant_message["function_call"]

messages.append(assistant_message)


# get the weather information to pass back to the model
weather = get_current_weather(messages[1]["tool_calls"][0]["function"]["arguments"])

messages.append({"role": "tool",
                 "tool_call_id": assistant_message["tool_calls"][0]["id"],
                 "name": assistant_message["tool_calls"][0]["function"]["name"],
                 "content": weather})


final_response = get_completion(messages, tools=tools)

final_response

"""
ChatCompletionMessage(content='The current weather in Boston is 50 degrees Fahrenheit.', role='assistant', function_call=None, tool_calls=[])
"""
```

In detail: [tool-calling by using xinference&qwen](./code/xinference_functioncalling_test.ipynb)



## fine-tuning a LLM with tools using ability 





## Recommended Reading



function calling by openAI API:

**[Function calling](https://platform.openai.com/docs/guides/function-calling/function-calling)**

**[Function Calling with LLMs](https://www.promptingguide.ai/applications/function_calling)**

[How to call functions with chat models](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_call_functions_with_chat_models.ipynb)

[Function-calling with an OpenAPI specification](https://github.com/openai/openai-cookbook/blob/main/examples/Function_calling_with_an_OpenAPI_spec.ipynb)

[Fine tuning on synthetic function-calling data](https://github.com/openai/openai-cookbook/blob/main/examples/Fine_tuning_for_function_calling.ipynb)

**[How to use functions with a knowledge base](https://cookbook.openai.com/examples/how_to_call_functions_for_knowledge_retrieval)**



function calling via open source model 

**[Anyscale Endpoints: JSON Mode and Function calling Features](https://www.anyscale.com/blog/anyscale-endpoints-json-mode-and-function-calling-features)**

[xinference: **FunctionCall**](https://github.com/xorbitsai/inference/blob/main/examples/FunctionCall.ipynb)

**[JSON-based Agents With Ollama & LangChain](https://medium.com/neo4j/json-based-agents-with-ollama-langchain-9cf9ab3c84ef)**