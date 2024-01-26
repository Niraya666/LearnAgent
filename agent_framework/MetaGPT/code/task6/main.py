import datetime
import sys
from typing import Optional
from uuid import uuid4

from aiocron import crontab
from metagpt.actions import UserRequirement
from metagpt.actions.action import Action
from metagpt.actions.action_node import ActionNode
from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.tools.web_browser_engine import WebBrowserEngine
from metagpt.utils.common import CodeParser, any_to_str
from metagpt.utils.parse_html import _get_soup
from pytz import BaseTzInfo
from metagpt.logs import logger

import tiktoken



# 先写NODES
LANGUAGE = ActionNode(
    key="Language",
    expected_type=str,
    instruction="Provide the language used in the project, typically matching the user's requirement language.",
    example="en_us",
)


CRON_EXPRESSION = ActionNode(
    key="Cron Expression",
    expected_type=str,
    instruction="If the user requires scheduled triggering, please provide the corresponding 5-field cron expression. "
    "Otherwise, leave it blank.",
    example="",
)


CRAWLER_URL_LIST = ActionNode(
    key="Crawler URL List",
    expected_type=list[str],
    instruction="List the URLs user want to crawl. Leave it blank if not provided in the User Requirement.",
    example=["https://example1.com", "https://example2.com"],
)


PAGE_CONTENT_EXTRACTION = ActionNode(
    key="Page Content Extraction",
    expected_type=str,
    instruction="Specify the requirements and tips to extract from the crawled web pages based on User Requirement.",
    example="Retrieve the titles and content of articles published today.",
)


CRAWL_POST_PROCESSING = ActionNode(
    key="Crawl Post Processing",
    expected_type=str,
    instruction="Specify the processing to be applied to the crawled content, such as summarizing today's news.",
    example="Generate a summary of today's news articles.",
)


INFORMATION_SUPPLEMENT = ActionNode(
    key="Information Supplement",
    expected_type=str,
    instruction="If unable to obtain the Cron Expression, prompt the user to provide the time to receive subscription "
    "messages. If unable to obtain the URL List Crawler, prompt the user to provide the URLs they want to crawl. Keep it "
    "blank if everything is clear",
    example="",
)

# PROMPT_WRITER = ActionNode(
#     key="Generic Prompt Template for Data Extraction",
#     expected_type=str,
#     instruction=(
#         "Create a prompt template for extracting specific information based on user requirements. "
#         "This template should be designed to handle variable inputs, where the user can specify "
#         "their specific extraction needs (such as types of data to be extracted) and the text to be processed. "
#         "The prompt should be structured to guide the LLM in identifying and extracting the relevant information "
#         "from the given text in a consistent and error-free manner. Define placeholders within the prompt for "
#         "user-defined variables like 'data requirements' and 'text to process'."
#     ),
#     example=(
#         "Template: \"Given the user requirement: {data_requirements}, extract the relevant information from "
#         "the following text: {text_to_process}. "
#         "in a concise and accurate manner.\""
#         "\n\nUsage Example: \n- data_requirements: 'startup names, funding details, and main business focus' "
#         "\n- text_to_process: '[Text from web page containing startup funding information]' "
        
#     ),
# )
PROMPT_WRITER = ActionNode(
    key="Generic Prompt Template for Data Extraction",
    expected_type=str,
    instruction=(
        "Create a prompt template for extracting specific information based on user requirements. "
        "This template should be designed to handle variable inputs, where the user can specify "
        "their specific extraction needs (such as types of data to be extracted) and the text to be processed. "
        "The prompt should be structured to guide the LLM in identifying and extracting the relevant information "
        "from the given text in a consistent and error-free manner. Define placeholders within the prompt for "
        "user-defined variables like 'data requirements' and 'text to process'. Ensure that the model's output "
        "is formatted in YAML for clarity and structure. The model should focus on returning only valid and relevant "
        "information, disregarding any irrelevant or invalid data."
    ),
    example=(
        "Template: \"Given the user requirement: {data_requirements}, extract the relevant information from "
        "the following text: {text_to_process}. Format the extracted data in YAML, ensuring only relevant and "
        "valid information is included. "
        "\n- Output Format: YAML in ```yaml```"
    ),
)


NODES = [
    LANGUAGE,
    CRON_EXPRESSION,
    CRAWLER_URL_LIST,
    PAGE_CONTENT_EXTRACTION,
    PROMPT_WRITER,
    CRAWL_POST_PROCESSING,
    INFORMATION_SUPPLEMENT,
]




PARSE_SUB_REQUIREMENTS_NODE = ActionNode.from_children("ParseSubscriptionReq", NODES)

PARSE_SUB_REQUIREMENT_TEMPLATE = """
### User Requirement
{requirements}
"""

SUB_ACTION_TEMPLATE = """
## Requirements
Answer the question based on the provided context {process}. If the question cannot be answered, please summarize the context.

## context
{data}"
"""

# PROMPT_TEMPLATE = """Please complete the web page crawler parse function to achieve the User Requirement. The parse \
# function should take a BeautifulSoup object as input, which corresponds to the HTML outline provided in the Context.

# ```python
# from bs4 import BeautifulSoup

# # only complete the parse function
# def parse(soup: BeautifulSoup):
#     ...
#     # Return the object that the user wants to retrieve, don't use print
# ```

# ## User Requirement
# {requirement}

# ## Context

# The outline of html page to scrabe is show like below:

# ```tree
# {outline}
# ```
# """

# 辅助函数: 获取html css大纲视图
# def get_outline(page):
#     soup = _get_soup(page.html)
#     outline = []

#     def process_element(element, depth):
#         name = element.name
#         if not name:
#             return
#         if name in ["script", "style"]:
#             return

#         element_info = {"name": element.name, "depth": depth}

#         if name in ["svg"]:
#             element_info["text"] = None
#             outline.append(element_info)
#             return

#         element_info["text"] = element.string
#         # Check if the element has an "id" attribute
#         if "id" in element.attrs:
#             element_info["id"] = element["id"]

#         if "class" in element.attrs:
#             element_info["class"] = element["class"]
#         outline.append(element_info)
#         for child in element.children:
#             process_element(child, depth + 1)

#     for element in soup.body.children:
#         process_element(element, 1)

    # return outline
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """
    Calculate the number of tokens in a given string based on the specified encoding.

    :param string: The text string to be tokenized.
    :param encoding_name: The name of the encoding to use for tokenization.
    :return: The number of tokens in the string.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))


def get_outline(page):
    """
    Extract and format the outline of a web page.

    :param page: Web page object containing HTML to be processed.
    :return: A structured outline of the page elements.
    """
    soup = _get_soup(page.html)
    outline = []

    def process_element(element, depth):
        """
        Recursively process each HTML element and add relevant information to the outline.

        :param element: The HTML element to process.
        :param depth: The depth of the element in the HTML tree.
        """
        name = element.name
        if not name or name in ["script", "style"]:
            return

        element_info = {"name": name, "depth": depth}
        if name == "svg":
            element_info["text"] = None
        else:
            element_info["text"] = element.string
            if "id" in element.attrs:
                element_info["id"] = element["id"]
            if "class" in element.attrs:
                element_info["class"] = element["class"]

        outline.append(element_info)

        for child in element.children:
            process_element(child, depth + 1)

    for element in soup.body.children:
        process_element(element, 1)

    return outline
def filter_none(data):
    """
    Filter out entries from a list that do not have a 'text' field.

    :param data: List of dictionaries to be filtered.
    :return: Filtered list.
    """
    return [entry for entry in data if entry.get('text') is not None]


def remove_non_title_class(data):
    """
    Filter out entries from a list that do not have 'title' in their 'class' attribute.

    :param data: List of dictionaries to be filtered.
    :return: Filtered list.
    """
    return [item for item in data if 'class' not in item or ('class' in item and 'title' in item['class'])]



# 触发器：crontab
class CronTrigger:
    def __init__(self, spec: str, tz: Optional[BaseTzInfo] = None) -> None:
        segs = spec.split(" ")
        if len(segs) == 6:
            spec = " ".join(segs[1:])
        self.crontab = crontab(spec, tz=tz)

    def __aiter__(self):
        return self

    async def __anext__(self):
        await self.crontab.next()
        return Message(datetime.datetime.now().isoformat())

# 网页信息获取的Action
class WebInfoExtracting(Action):
    """
    Class to extract information from web pages.
    """

    async def run(self, requirement):
        """
        Main method to run the action for extracting information from a web page.

        :param requirement: The requirement object containing the details for information extraction.
        :return: Extracted information.
        """
        requirement: Message = requirement[-1]
        data = requirement.instruct_content.dict()
        urls = data["Crawler URL List"]
        query = data["Page Content Extraction"]
        prompt_template = data['Generic Prompt Template for Data Extraction']
        # print(prompt_template)
        extract_info = ""

        outline = await self._get_web_outline(urls[0])
        context_len = num_tokens_from_string(outline, "cl100k_base")

        if context_len >= 4096 * 0.7:
            outlines = self._chunking(outline)
        else:
            outlines = [outline]
        i = 0
        for outline_chunk in outlines:
            print("==========================")
            print("extract chunk ", i)
            info = await self._extract_web_info(prompt_template, query, outline_chunk)
            extract_info+=info
            extract_info+="\n"
            i+=1
        print("total_extraction: ", extract_info)

        return extract_info

    def _chunking(self, text, chunk_size=2800):
        """
        Split a string into chunks of a specific size.

        :param text: The text to be chunked.
        :param chunk_size: The size of each chunk.
        :return: List of text chunks.
        """
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    async def _get_web_outline(self, url):
        """
        Get the outline of a web page given its URL.

        :param url: The URL of the web page.
        :return: The formatted outline of the web page.
        """
        page = await WebBrowserEngine().run(url)
        outline = get_outline(page)
        outline = filter_none(outline)
        # outline = remove_non_title_class(outline)

        formatted_outline = "\n".join(
            f"{' ' * i['depth']}{'.'.join([i['name'], *i.get('class', [])])}: {i['text'] if i['text'] else ''}"
            for i in outline
        )
        return formatted_outline

    async def _extract_web_info(self, prompt_template, data_requirements, text_to_process):
        """
        Extract information from a web page using a given template.

        :param prompt_template: The template to format the extraction query.
        :param data_requirements: Requirements for the data extraction.
        :param text_to_process: The text to be processed for information extraction.
        :return: Extracted information.
        """
        response = await self._aask(prompt_template.format(text_to_process=text_to_process, data_requirements=data_requirements))
        return CodeParser.parse_code(block="", text=response)

# 分析订阅需求的Action
class ParseSubRequirement(Action):
    async def run(self, requirements):
        requirements = "\n".join(i.content for i in requirements)
        context = PARSE_SUB_REQUIREMENT_TEMPLATE.format(requirements=requirements)
        node = await PARSE_SUB_REQUIREMENTS_NODE.fill(context=context, llm=self.llm)
        return node

# # 运行订阅智能体的Action
class RunSubscription(Action):
    async def run(self, msgs):
        from metagpt.roles.role import Role
        from metagpt.subscription import SubscriptionRunner

        code = msgs[-1].content
        req = msgs[-2].instruct_content.dict()
        urls = req["Crawler URL List"]
        process = req["Crawl Post Processing"]
        spec = req["Cron Expression"]

        print("====msg====:\n", [msg.content for msg in msgs])


        print("code: ", code)
        print("req: ", req)

        SubAction = self.create_sub_action_cls(code)
        SubRole = type("SubRole", (Role,), {})
        role = SubRole()
        role._init_actions([SubAction])
        runner = SubscriptionRunner()

        async def callback(msg):
            print(msg)

        await runner.subscribe(role, CronTrigger(spec), callback)
        await runner.run()

    @staticmethod
    def create_sub_action_cls(code):
        # modules = {}
        # for url in urls[::-1]:
        #     code, current = code.rsplit(f"# {url}", maxsplit=1)
        #     name = uuid4().hex
        #     module = type(sys)(name)
        #     exec(current, module.__dict__)
        #     modules[url] = module

        class SubAction(Action):
            async def run(self, *args, **kwargs):
                # pages = await WebBrowserEngine().run(*urls)
                # if len(urls) == 1:
                #     pages = [pages]

                # data = []
                # for url, page in zip(urls, pages):
                #     data.append(getattr(modules[url], "parse")(page.soup))
                # return await self.llm.aask(SUB_ACTION_TEMPLATE.format(process=process, data=data))
                return code

        return SubAction

# 从RunSubscription分离出AddSubscriptionTask的action
class AddSubscriptionTask(Action):
    async def run(self, role, trigger, callback):
        runner = SubscriptionRunner()
        await runner.subscribe(role, trigger, callback)
        await runner.run()
# class RunSubscription(Action):
#     async def run(self, msgs):
#         from metagpt.roles.role import Role
#         from metagpt.subscription import SubscriptionRunner
#         print("====msg====:\n", [msg.content for msg in msgs])
#         code = msgs[-1].content
#         req = msgs[-2].instruct_content.dict()
#         urls = req["Crawler URL List"]
#         process = req["Crawl Post Processing"]
#         spec = req["Cron Expression"]

#         # print("code: ", code)
#         # print("req: ", req)

#         # # 创建 SubAction 和 SubRole
#         # print("创建 SubAction 和 SubRole")
#         # SubAction = self.create_sub_action_cls(urls, code, process)
#         # role = self.create_role(SubAction)

#         # 创建 CronTrigger
#         trigger = self.create_trigger(spec)

#         # 创建并运行 AddSubscriptionTask
#         print("===AddSubscriptionTask===")
#         add_subscription_task = AddSubscriptionTask()
#         await add_subscription_task.run(role, trigger, self.callback)

#     def create_role(self, action_cls):
#         SubRole = type("SubRole", (Role,), {})
#         role = SubRole()
#         role._init_actions([action_cls])
#         return role

#     def create_trigger(self, spec):
#         return CronTrigger(spec)

#     async def callback(self, msg):
#         print(msg)

#     @staticmethod
#     def create_sub_action_cls(urls: list[str], code: str, process: str):
#         modules = {}
#         for url in urls[::-1]:
#             code, current = code.rsplit(f"# {url}", maxsplit=1)
#             name = uuid4().hex
#             module = type(sys)(name)
#             exec(current, module.__dict__)
#             modules[url] = module

#         class SubAction(Action):
#             async def run(self, *args, **kwargs):
#                 pages = await WebBrowserEngine().run(*urls)
#                 if len(urls) == 1:
#                     pages = [pages]

#                 data = []
#                 for url, page in zip(urls, pages):
#                     data.append(getattr(modules[url], "parse")(page.soup))
#                 return await self.llm.aask(SUB_ACTION_TEMPLATE.format(process=process, data=data))

#         return SubAction


# # 定义爬虫工程师角色
# class CrawlerEngineer(Role):
#     name: str = "John"
#     profile: str = "Crawling Engineer"
#     goal: str = "Write elegant, readable, extensible, efficient code"
#     constraints: str = "The code should conform to standards like PEP8 and be modular and maintainable"

#     def __init__(self, **kwargs) -> None:
#         super().__init__(**kwargs)

#         self._init_actions([WriteCrawlerCode])
#         self._watch([ParseSubRequirement])

# 定义信息提取工程师角色
class InfoExtractionEngineer(Role):
    name: str = "Alice"
    profile: str = "Information Extraction Engineer"
    goal: str = "Efficiently extract and process web information while ensuring data accuracy and relevance, DONOT WRITE PYTHON CODE!"
    constraints: str = "Extraction must be precise, relevant, and adhere to legal and ethical standards. Avoid over-extraction and ensure data privacy and security."

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # 初始化动作 WebInfoExtracting
        self._init_actions([WebInfoExtracting])
        # 观察需求的解析过程
        self._watch([ParseSubRequirement])




# 定义订阅助手角色
class SubscriptionAssistant(Role):
    """Analyze user subscription requirements."""

    name: str = "Grace"
    profile: str = "Subscription Assistant"
    goal: str = "analyze user subscription requirements to provide personalized subscription services."
    constraints: str = "utilize the same language as the User Requirement"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self._init_actions([ParseSubRequirement, RunSubscription])
        self._watch([UserRequirement, WebInfoExtracting])

    async def _think(self) -> bool:
        cause_by = self.rc.history[-1].cause_by
        if cause_by == any_to_str(UserRequirement):
            state = 0
        elif cause_by == any_to_str(WebInfoExtracting):
            state = 1

        if self.rc.state == state:
            self.rc.todo = None
            return False
        self._set_state(state)
        return True

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: ready to {self.rc.todo}")
        response = await self.rc.todo.run(self.rc.history)
        msg = Message(
            content=response.content,
            instruct_content=response.instruct_content,
            role=self.profile,
            cause_by=self.rc.todo,
            sent_from=self,
        )
        self.rc.memory.add(msg)
        return msg

if __name__ == "__main__":
    import asyncio
    from metagpt.team import Team
    from datetime import timedelta
    import datetime
    current_time = datetime.datetime.now()
    time_in_three_minutes = current_time + timedelta(minutes=6)
    time_str = time_in_three_minutes.strftime('%H:%M')



    team = Team()
    team.hire([SubscriptionAssistant(), InfoExtractionEngineer()])
    team.run_project("从36kr创投平台https://pitchhub.36kr.com/financing-flash爬取所有初创企业融资的信息，获取标题，链接， 时间，总结今天的融资新闻，然后在{time}发送给我".format(time = time_str))
    # team.run_project("从 https://huggingface.co/papers 获取今天推荐的论文的标题和链接，整理成表格，分析这些论文主要主题，然后在{time}发送给我".format(time = time_str))
    # team.run_project("从 https://www.qbitai.com/category/资讯 获取量子位今天推送的文章，总结今天的主要资讯，然后在每天{time}发送给我".format(time = time_str))
    # team.run_project("从 https://www.jiqizhixin.com 获取机器之心今天推送的文章，总结今天的主要资讯。然后在每天{time}发送给我".format(time = time_str))
    asyncio.run(team.run())