#!/usr/bin/env python3
"""
Tool Pattern 测试文件
从 tool_pattern.ipynb 提取的测试代码
"""

import json
import os
import re
import requests
from dotenv import load_dotenv
from deepseek import DeepSeekAPI

from agentic_patterns.tool_pattern.tool import tool
from agentic_patterns.tool_pattern.tool_agent import ToolAgent

# 加载环境变量
load_dotenv()

# 初始化 DeepSeek 客户端
MODEL = "deepseek-chat"
DEEPSEEK_CLIENT = DeepSeekAPI(api_key=os.getenv("DEEPSEEK_API_KEY"))

# 定义系统提示
TOOL_SYSTEM_PROMPT = """
You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. 
You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug 
into functions. Pay special attention to the properties 'types'. You should use those types as in a Python dict.
For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:

<tool_call>
{"name": <function-name>,"arguments": <args-dict>}
</tool_call>

Here are the available tools:

<tools> {
    "name": "get_current_weather",
    "description": "Get the current weather in a given location location (str): The city and state, e.g. Madrid, Barcelona unit (str): The unit. It can take two values; 'celsius', 'fahrenheit'",
    "parameters": {
        "properties": {
            "location": {
                "type": "str"
            },
            "unit": {
                "type": "str"
            }
        }
    }
}
</tools>
"""


def get_current_weather(location: str, unit: str):
    """
    Get the current weather in a given location

    Args:
        location (str): The city and state, e.g. Madrid, Barcelona
        unit (str): The unit. It can take two values; "celsius", "fahrenheit"
    """
    if location == "Madrid":
        return json.dumps({"temperature": 25, "unit": unit})
    else:
        return json.dumps({"temperature": 58, "unit": unit})


def parse_tool_call_str(tool_call_str: str):
    """解析工具调用字符串，移除XML标签并转换为JSON"""
    pattern = r'</?tool_call>'
    clean_tags = re.sub(pattern, '', tool_call_str)
    
    try:
        tool_call_json = json.loads(clean_tags)
        return tool_call_json
    except json.JSONDecodeError:
        return clean_tags
    except Exception as e:
        print(f"Unexpected error: {e}")
        return "There was some error parsing the Tool's output"


def fetch_top_hacker_news_stories(top_n: int):
    """
    Fetch the top stories from Hacker News.

    This function retrieves the top `top_n` stories from Hacker News using the Hacker News API. 
    Each story contains the title, URL, score, author, and time of submission. The data is fetched 
    from the official Firebase Hacker News API, which returns story details in JSON format.

    Args:
        top_n (int): The number of top stories to retrieve.
    """
    top_stories_url = 'https://hacker-news.firebaseio.com/v0/topstories.json'
    
    try:
        response = requests.get(top_stories_url)
        response.raise_for_status()  # Check for HTTP errors
        
        # Get the top story IDs
        top_story_ids = response.json()[:top_n]
        
        top_stories = []
        
        # For each story ID, fetch the story details
        for story_id in top_story_ids:
            story_url = f'https://hacker-news.firebaseio.com/v0/item/{story_id}.json'
            story_response = requests.get(story_url)
            story_response.raise_for_status()  # Check for HTTP errors
            story_data = story_response.json()
            
            # Append the story title and URL (or other relevant info) to the list
            top_stories.append({
                'title': story_data.get('title', 'No title'),
                'url': story_data.get('url', 'No URL available'),
            })
        
        return json.dumps(top_stories)

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return []


def test_basic_weather_function():
    """测试基本的天气函数"""
    print("=== 测试基本天气函数 ===")
    result = get_current_weather(location="Madrid", unit="celsius")
    print(f"Madrid天气: {result}")
    return result


def test_tool_call_parsing():
    """测试工具调用解析"""
    print("\n=== 测试工具调用解析 ===")
    
    # 模拟LLM输出
    tool_call_str = '<tool_call>{"name": "get_current_weather","arguments": {"location": "Madrid", "unit": "celsius"}}</tool_call>'
    
    parsed_output = parse_tool_call_str(tool_call_str)
    print(f"解析结果: {parsed_output}")
    
    # 执行函数调用
    result = get_current_weather(**parsed_output["arguments"])
    print(f"函数执行结果: {result}")
    
    return result


def test_hacker_news_function():
    """测试Hacker News函数"""
    print("\n=== 测试Hacker News函数 ===")
    stories = fetch_top_hacker_news_stories(top_n=3)
    print(f"前3个故事: {stories}")
    return stories


def test_tool_decorator():
    """测试tool装饰器"""
    print("\n=== 测试tool装饰器 ===")
    
    # 使用装饰器创建工具
    hn_tool = tool(fetch_top_hacker_news_stories)
    
    print(f"工具名称: {hn_tool.name}")
    print(f"工具签名: {json.loads(hn_tool.fn_signature)}")
    
    return hn_tool


def test_tool_agent():
    """测试ToolAgent"""
    print("\n=== 测试ToolAgent ===")
    
    # 创建工具
    hn_tool = tool(fetch_top_hacker_news_stories)
    
    # 创建代理
    tool_agent = ToolAgent(tools=[hn_tool])
    
    # 测试不相关的问题
    print("测试不相关的问题:")
    output = tool_agent.run(user_msg="Tell me your name")
    print(f"回答: {output}")
    
    # 测试相关的问题
    print("\n测试相关的问题:")
    output = tool_agent.run(user_msg="Tell me the top 3 Hacker News stories right now")
    print(f"回答: {output}")
    
    return tool_agent


def test_manual_tool_workflow():
    """测试手动工具工作流程"""
    print("\n=== 测试手动工具工作流程 ===")
    
    # 初始化对话历史
    tool_chat_history = [
        {
            "role": "system",
            "content": TOOL_SYSTEM_PROMPT
        }
    ]
    
    user_msg = {
        "role": "user",
        "content": "What's the current temperature in Madrid, in Celsius?"
    }
    
    tool_chat_history.append(user_msg)
    
    # 获取LLM响应
    output = DEEPSEEK_CLIENT.chat_completion(
        messages=tool_chat_history,
        model=MODEL
    )
    
    print(f"LLM输出: {output}")
    
    # 解析工具调用
    parsed_output = parse_tool_call_str(output)
    print(f"解析结果: {parsed_output}")
    
    # 执行工具
    result = get_current_weather(**parsed_output["arguments"])
    print(f"工具执行结果: {result}")
    
    # 将结果添加到对话历史
    agent_chat_history = [user_msg]
    agent_chat_history.append({
        "role": "user",
        "content": f"Observation: {result}"
    })
    
    # 获取最终回答
    final_response = DEEPSEEK_CLIENT.chat_completion(
        messages=agent_chat_history,
        model=MODEL
    )
    
    print(f"最终回答: {final_response}")
    
    return final_response


def main():
    """主函数：运行所有测试"""
    print("开始运行 Tool Pattern 测试...\n")
    
    try:
        # 测试基本函数
        test_basic_weather_function()
        
        # 测试工具调用解析
        test_tool_call_parsing()
        
        # 测试Hacker News函数
        test_hacker_news_function()
        
        # 测试tool装饰器
        test_tool_decorator()
        
        # 测试ToolAgent
        test_tool_agent()
        
        # 测试手动工作流程
        test_manual_tool_workflow()
        
        print("\n所有测试完成！")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 