import json
import re
import os

from colorama import Fore
from dotenv import load_dotenv
from deepseek import DeepSeekAPI

from agentic_patterns.tool_pattern.tool import Tool
from agentic_patterns.tool_pattern.tool import validate_arguments
from agentic_patterns.utils.completions import build_prompt_structure
from agentic_patterns.utils.completions import ChatHistory
from agentic_patterns.utils.completions import completions_create
from agentic_patterns.utils.completions import update_chat_history
from agentic_patterns.utils.extraction import extract_tag_content

# 加载环境变量（如 API KEY）
load_dotenv()

BASE_SYSTEM_PROMPT = ""

# ReAct 代理的系统提示，描述了 Thought-Action-Observation 循环和工具调用格式
REACT_SYSTEM_PROMPT = """
You operate by running a loop with the following steps: Thought, Action, Observation.
You are provided with function signatures within <tools></tools> XML tags.
You may call one or more functions to assist with the user query. Don' make assumptions about what values to plug
into functions. Pay special attention to the properties 'types'. You should use those types as in a Python dict.

For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:

<tool_call>
{"name": <function-name>,"arguments": <args-dict>, "id": <monotonically-increasing-id>}
</tool_call>

Here are the available tools / actions:

<tools>
%s
</tools>

Example session:

<question>What's the current temperature in Madrid?</question>
<thought>I need to get the current weather in Madrid</thought>
<tool_call>{"name": "get_current_weather","arguments": {"location": "Madrid", "unit": "celsius"}, "id": 0}</tool_call>

You will be called again with this:

<observation>{0: {"temperature": 25, "unit": "celsius"}}</observation>

You then output:

<response>The current temperature in Madrid is 25 degrees Celsius</response>

Additional constraints:

- If the user asks you something unrelated to any of the tools above, answer freely enclosing your answer with <response></response> tags.
"""


class ReactAgent:
    """
    ReAct 代理类：实现 Thought-Action-Observation 循环，支持工具调用。

    属性：
        client (DeepSeekAPI): DeepSeekAPI 客户端，用于模型推理。
        model (str): 使用的模型名称，默认 deepseek-chat。
        tools (list[Tool]): 可用工具列表。
        tools_dict (dict): 工具名称到 Tool 实例的映射。
    """

    def __init__(
        self,
        tools: Tool | list[Tool],
        model: str = "deepseek-chat",
        system_prompt: str = BASE_SYSTEM_PROMPT,
    ) -> None:
        # 初始化 DeepSeekAPI 客户端，要求环境变量中有 DEEPSEEK_API_KEY
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable is required. Please set it in your .env file or environment.")
        self.client = DeepSeekAPI(api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt
        # 工具可以是单个 Tool 或 Tool 列表
        self.tools = tools if isinstance(tools, list) else [tools]
        self.tools_dict = {tool.name: tool for tool in self.tools}

    def add_tool_signatures(self) -> str:
        """
        收集所有可用工具的函数签名（字符串拼接）。
        返回：所有工具签名的拼接字符串。
        """
        return "".join([tool.fn_signature for tool in self.tools])

    def process_tool_calls(self, tool_calls_content: list) -> dict:
        """
        处理每个工具调用，校验参数并执行工具，收集结果。
        参数：
            tool_calls_content (list): 每个元素为 JSON 字符串，描述一次工具调用。
        返回：
            dict: 工具调用 id 到结果的映射。
        """
        observations = {}
        for tool_call_str in tool_calls_content:
            tool_call = json.loads(tool_call_str)
            tool_name = tool_call["name"]
            tool = self.tools_dict[tool_name]

            print(Fore.GREEN + f"\nUsing Tool: {tool_name}")

            # 校验参数并执行工具
            validated_tool_call = validate_arguments(
                tool_call, json.loads(tool.fn_signature)
            )
            print(Fore.GREEN + f"\nTool call dict: \n{validated_tool_call}")

            result = tool.run(**validated_tool_call["arguments"])
            print(Fore.GREEN + f"\nTool result: \n{result}")

            # 以调用 id 作为 key 存储结果
            observations[validated_tool_call["id"]] = result

        return observations

    def run(
        self,
        user_msg: str,
        max_rounds: int = 10,
    ) -> str:
        """
        运行一次完整的 ReAct 交互循环。
        参数：
            user_msg (str): 用户输入。
            max_rounds (int): 最大循环轮数，防止死循环。
        返回：
            str: 最终模型输出（<response> 标签内容）。
        """
        # 构造用户问题 prompt
        user_prompt = build_prompt_structure(
            prompt=user_msg, role="user", tag="question"
        )
        # 拼接系统提示和工具签名
        if self.tools:
            self.system_prompt += (
                "\n" + REACT_SYSTEM_PROMPT % self.add_tool_signatures()
            )

        # 初始化对话历史
        chat_history = ChatHistory(
            [
                build_prompt_structure(
                    prompt=self.system_prompt,
                    role="system",
                ),
                user_prompt,
            ]
        )

        if self.tools:
            # 进入 Thought-Action-Observation 循环
            for _ in range(max_rounds):
                # 让 LLM 生成下一步内容
                completion = completions_create(self.client, chat_history, self.model)

                # 检查是否直接给出最终答案（<response>）
                response = extract_tag_content(str(completion), "response")
                if response.found:
                    return response.content[0]

                # 提取思考和工具调用内容
                thought = extract_tag_content(str(completion), "thought")
                tool_calls = extract_tag_content(str(completion), "tool_call")

                # 更新对话历史（assistant 角色）
                update_chat_history(chat_history, completion, "assistant")

                print(Fore.MAGENTA + f"\nThought: {thought.content[0]}")

                # 如果有工具调用，则执行工具并将 observation 加入历史
                if tool_calls.found:
                    observations = self.process_tool_calls(tool_calls.content)
                    print(Fore.BLUE + f"\nObservations: {observations}")
                    update_chat_history(chat_history, f"{observations}", "user")

        # 如果循环结束还没得到 <response>，直接返回最后一次模型输出
        return completions_create(self.client, chat_history, self.model)
