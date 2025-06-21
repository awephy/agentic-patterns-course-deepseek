#!/usr/bin/env python3

import os
from dotenv import load_dotenv

from agentic_patterns.multiagent_pattern.agent import Agent
from agentic_patterns.multiagent_pattern.crew import Crew
from agentic_patterns.tool_pattern.tool import tool

# 加载环境变量
load_dotenv()


@tool
def write_str_to_txt(string_data: str, txt_filename: str):
    """
    Writes a string to a txt file.

    This function takes a string and writes it to a text file. If the file already exists, 
    it will be overwritten with the new data.

    Args:
        string_data (str): The string containing the data to be written to the file.
        txt_filename (str): The name of the text file to which the data should be written.
    """
    # Write the string data to the text file
    with open(txt_filename, mode='w', encoding='utf-8') as file:
        file.write(string_data)

    print(f"Data successfully written to {txt_filename}")


def test_basic_agent():
    """测试基本 Agent 功能"""
    print("=== 测试基本 Agent 功能 ===")
    
    agent_example = Agent(
        name="Poet Agent",
        backstory="You are a well-known poet, who enjoys creating high quality poetry.",
        task_description="Write a poem about the meaning of life",
        task_expected_output="Just output the poem, without any title or introductory sentences",
    )
    
    result = agent_example.run()
    print(f"Poet Agent 输出:\n{result}")
    
    return agent_example


def test_agent_with_tools():
    """测试带工具的 Agent"""
    print("\n=== 测试带工具的 Agent ===")
    
    agent_tool_example = Agent(
        name="Writer Agent",
        backstory="You are a language model specialised in writing text into .txt files",
        task_description="Write the string 'This is a Tool Agent' into './tool_agent_example.txt'",
        task_expected_output="A .txt file containing the given string",
        tools=[write_str_to_txt],
    )
    
    result = agent_tool_example.run()
    print(f"Writer Agent 输出:\n{result}")
    
    return agent_tool_example


def test_agent_dependencies():
    """测试 Agent 依赖关系"""
    print("\n=== 测试 Agent 依赖关系 ===")
    
    # 创建两个 Agent
    agent_1 = Agent(
        name="Poet Agent",
        backstory="You are a well-known poet, who enjoys creating high quality poetry.",
        task_description="Write a poem about the meaning of life",
        task_expected_output="Just output the poem, without any title or introductory sentences",
    )

    agent_2 = Agent(
        name="Poem Translator Agent",
        backstory="You are an expert translator especially skilled in Ancient Greek",
        task_description="Translate a poem into Ancient Greek", 
        task_expected_output="Just output the translated poem and nothing else"
    )
    
    # 设置依赖关系：agent_2 依赖于 agent_1
    agent_1 >> agent_2
    
    # 检查依赖关系
    print("Agent 1 dependencies: ", agent_1.dependencies)
    print("Agent 1 dependents: ", agent_1.dependents)
    print("Agent 2 dependencies: ", agent_2.dependencies)
    print("Agent 2 dependents: ", agent_2.dependents)
    
    # 运行第一个 Agent
    print("\n运行 Poet Agent:")
    result_1 = agent_1.run()
    print(f"Poet Agent 输出:\n{result_1}")
    
    # 检查第二个 Agent 的上下文
    print(f"\nPoem Translator Agent 的上下文:\n{agent_2.context}")
    
    # 运行第二个 Agent
    print("\n运行 Poem Translator Agent:")
    result_2 = agent_2.run()
    print(f"Poem Translator Agent 输出:\n{result_2}")
    
    return agent_1, agent_2


def test_crew_workflow():
    """测试 Crew 工作流程"""
    print("\n=== 测试 Crew 工作流程 ===")
    
    with Crew() as crew:
        # 创建三个 Agent
        agent_1 = Agent(
            name="Poet Agent",
            backstory="You are a well-known poet, who enjoys creating high quality poetry.",
            task_description="Write a poem about the meaning of life",
            task_expected_output="Just output the poem, without any title or introductory sentences",
        )

        agent_2 = Agent(
            name="Poem Translator Agent",
            backstory="You are an expert translator especially skilled in Spanish",
            task_description="Translate a poem into Spanish", 
            task_expected_output="Just output the translated poem and nothing else"
        )

        agent_3 = Agent(
            name="Writer Agent",
            backstory="You are an expert transcriber, that loves writing poems into txt files",
            task_description="You'll receive a Spanish poem in your context. You need to write the poem into './poem.txt' file",
            task_expected_output="A txt file containing the spanish poem received from the context",
            tools=[write_str_to_txt],
        )

        # 设置依赖关系链
        agent_1 >> agent_2 >> agent_3
        
        # 显示工作流程图
        print("Crew 工作流程图:")
        crew.plot()
        
        # 运行整个工作流程
        print("\n运行 Crew 工作流程:")
        results = crew.run()
        
        # 检查结果是否为 None
        if results is not None:
            print("Crew 执行结果:")
            for agent_name, result in results.items():
                print(f"\n{agent_name}:")
                print(result)
        else:
            print("Crew 执行完成，但未返回结果字典")
    
    return crew


def test_complex_crew_scenario():
    """测试复杂的 Crew 场景"""
    print("\n=== 测试复杂的 Crew 场景 ===")
    
    with Crew() as crew:
        # 创建多个专业 Agent
        researcher = Agent(
            name="Research Agent",
            backstory="You are an expert researcher specializing in AI and machine learning trends.",
            task_description="Research the latest developments in large language models and their applications",
            task_expected_output="A comprehensive summary of recent LLM developments",
        )
        
        writer = Agent(
            name="Content Writer Agent",
            backstory="You are a skilled technical writer who can explain complex topics clearly.",
            task_description="Write a blog post based on the research findings about LLM developments",
            task_expected_output="A well-structured blog post about LLM developments",
        )
        
        editor = Agent(
            name="Editor Agent",
            backstory="You are an experienced editor who ensures content quality and readability.",
            task_description="Review and improve the blog post for clarity and engagement",
            task_expected_output="An improved version of the blog post with better flow and clarity",
        )
        
        publisher = Agent(
            name="Publisher Agent",
            backstory="You are responsible for formatting and publishing content.",
            task_description="Format the final blog post and save it to a file",
            task_expected_output="A properly formatted blog post saved to 'llm_blog_post.txt'",
            tools=[write_str_to_txt],
        )
        
        # 设置工作流程：研究 -> 写作 -> 编辑 -> 发布
        researcher >> writer >> editor >> publisher
        
        print("复杂 Crew 工作流程图:")
        crew.plot()
        
        print("\n执行复杂 Crew 工作流程:")
        results = crew.run()
        
        # 检查结果是否为 None
        if results is not None:
            print("\n复杂 Crew 执行结果:")
            for agent_name, result in results.items():
                print(f"\n{'='*50}")
                print(f"{agent_name}:")
                print(f"{'='*50}")
                print(result)
        else:
            print("复杂 Crew 执行完成，但未返回结果字典")
    
    return crew


def main():
    """主函数：运行所有测试"""
    print("开始运行 Multiagent Pattern 测试...\n")
    
    try:
        # 测试基本 Agent
        test_basic_agent()
        
        # 测试带工具的 Agent
        test_agent_with_tools()
        
        # 测试 Agent 依赖关系
        test_agent_dependencies()
        
        # 测试 Crew 工作流程
        test_crew_workflow()
        
        # 测试复杂 Crew 场景
        test_complex_crew_scenario()
        
        print("\n所有测试完成！")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 