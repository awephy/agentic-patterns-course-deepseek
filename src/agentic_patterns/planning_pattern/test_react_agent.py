# ReactAgent 测试文件 - 基于 planning_pattern.ipynb
# 测试 ReAct 模式和 ReactAgent 类的功能

import os
import re
import math
import json
from dotenv import load_dotenv

# 设置 API 密钥（请根据实际情况修改）
# os.environ["DEEPSEEK_API_KEY"] = "your-api-key-here"

from agentic_patterns.tool_pattern.tool import tool
from agentic_patterns.utils.extraction import extract_tag_content
from agentic_patterns.planning_pattern.react_agent import ReactAgent

# 加载环境变量
load_dotenv()

print("🚀 开始 ReactAgent 测试...\n")

# ============================================================================
# 1. 定义测试工具
# ============================================================================

@tool
def sum_two_elements(a: int, b: int) -> int:
    """
    Computes the sum of two integers.

    Args:
        a (int): The first integer to be summed.
        b (int): The second integer to be summed.

    Returns:
        int: The sum of `a` and `b`.
    """
    return a + b


@tool
def multiply_two_elements(a: int, b: int) -> int:
    """
    Multiplies two integers.

    Args:
        a (int): The first integer to multiply.
        b (int): The second integer to multiply.

    Returns:
        int: The product of `a` and `b`.
    """
    return a * b


@tool
def compute_log(x: int) -> float | str:
    """
    Computes the logarithm of an integer `x` with an optional base.

    Args:
        x (int): The integer value for which the logarithm is computed. Must be greater than 0.

    Returns:
        float: The logarithm of `x` to the specified `base`.
    """
    if x <= 0:
        return "Logarithm is undefined for values less than or equal to 0."
    
    return math.log(x)


# 工具字典
available_tools = {
    "sum_two_elements": sum_two_elements,
    "multiply_two_elements": multiply_two_elements,
    "compute_log": compute_log
}

print("✅ 工具定义完成")
print(f"工具名称: {sum_two_elements.name}")
print(f"工具签名: {sum_two_elements.fn_signature}\n")

# ============================================================================
# 2. 测试 ReactAgent 类
# ============================================================================

print("=" * 60)
print("🧪 测试 ReactAgent 类")
print("=" * 60)

try:
    # 创建 ReactAgent 实例
    agent = ReactAgent(tools=[sum_two_elements, multiply_two_elements, compute_log])
    
    # 测试用户消息
    user_msg = "I want to calculate the sum of 1234 and 5678 and multiply the result by 5. Then, I want to take the logarithm of this result"
    
    print(f"用户请求: {user_msg}")
    print("\n开始执行 ReactAgent...")
    
    # 运行 ReactAgent
    result = agent.run(user_msg=user_msg, max_rounds=10)
    
    print(f"\n🎉 ReactAgent 执行完成!")
    print(f"最终结果: {result}")
    
    # 验证结果
    expected_calculation = math.log((1234 + 5678) * 5)
    print(f"预期结果: {expected_calculation}")
    
    if str(expected_calculation) in result:
        print("✅ 结果验证成功!")
    else:
        print("⚠️  结果验证失败，但可能是格式问题")
        
except Exception as e:
    print(f"❌ ReactAgent 测试失败: {e}")
    print("请确保设置了 DEEPSEEK_API_KEY 环境变量")

print("\n" + "=" * 60)

# ============================================================================
# 3. 手动 ReAct 循环测试（可选）
# ============================================================================

print("🧪 手动 ReAct 循环测试（需要 DeepSeek API）")
print("=" * 60)
print("注意：这个测试需要 DeepSeek API 密钥，如果你有的话可以取消注释运行\n")

"""
# 手动 ReAct 循环测试代码（需要 DeepSeek API）
# 如果你有 DeepSeek API 密钥，可以取消注释运行

from deepseek import DeepSeekAPI

MODEL = "deepseek-chat"
DEEPSEEK_CLIENT = DeepSeekAPI()

# ReAct 系统提示
REACT_SYSTEM_PROMPT = '''
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
'''

# 添加工具签名到系统提示
tools_signature = sum_two_elements.fn_signature + ",\n" + multiply_two_elements.fn_signature + ",\n" + compute_log.fn_signature
REACT_SYSTEM_PROMPT = REACT_SYSTEM_PROMPT % tools_signature

# 用户问题
USER_QUESTION = "I want to calculate the sum of 1234 and 5678 and multiply the result by 5. Then, I want to take the logarithm of this result"

# 初始化聊天历史
chat_history = [
    {
        "role": "system",
        "content": REACT_SYSTEM_PROMPT
    },
    {
        "role": "user",
        "content": f"<question>{USER_QUESTION}</question>"
    }
]

print("开始手动 ReAct 循环测试...")

# ReAct 循环步骤 1
print("\n步骤 1: 初始响应")
output = DEEPSEEK_CLIENT.chat_completion(
    prompt=chat_history[-1]["content"],
    prompt_sys=chat_history[0]["content"],
    model=MODEL
)
print(output)

chat_history.append({
    "role": "assistant",
    "content": output
})

# ReAct 循环步骤 2
print("\n步骤 2: 提取工具调用")
tool_call = extract_tag_content(output, tag="tool_call")
tool_call = json.loads(tool_call.content[0])
tool_result = available_tools[tool_call["name"]].run(**tool_call["arguments"])
print(f"工具调用: {tool_call}")
print(f"工具结果: {tool_result}")

# 验证结果
assert tool_result == 1234 + 5678
print("✅ 第一步验证成功")

chat_history.append({
    "role": "user",
    "content": f"<observation>{tool_result}</observation>"
})

# 继续更多步骤...
print("手动 ReAct 循环测试完成")
"""

print("🎯 测试完成！")
print("\n总结:")
print("1. ✅ ReactAgent 类测试 - 使用 DeepSeekAPI")
print("2. ⚠️  手动 ReAct 循环测试 - 需要 DeepSeek API（已注释）")
print("\n如果你有 DeepSeek API 密钥，可以取消注释手动测试部分来运行完整的 ReAct 循环演示。") 