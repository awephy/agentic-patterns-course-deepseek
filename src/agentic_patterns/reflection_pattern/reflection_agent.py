from colorama import Fore
from dotenv import load_dotenv
from deepseek import DeepSeekAPI

from agentic_patterns.utils.completions import build_prompt_structure
from agentic_patterns.utils.completions import completions_create
from agentic_patterns.utils.completions import FixedFirstChatHistory
from agentic_patterns.utils.completions import update_chat_history
from agentic_patterns.utils.logging import fancy_step_tracker

load_dotenv()


# 基础生成系统提示 - 指导模型生成最佳内容
BASE_GENERATION_SYSTEM_PROMPT = """
Your task is to Generate the best content possible for the user's request.
If the user provides critique, respond with a revised version of your previous attempt.
You must always output the revised content.
"""

# 基础反思系统提示 - 指导模型进行批判性分析和建议
BASE_REFLECTION_SYSTEM_PROMPT = """
You are tasked with generating critique and recommendations to the user's generated content.
If the user content has something wrong or something to be improved, output a list of recommendations
and critiques. If the user content is ok and there's nothing to change, output this: <OK>
"""


class ReflectionAgent:
    """
    反思代理类 - 实现生成-反思-改进的迭代模式
    
    核心原理：通过生成-反思-改进的循环来不断提升输出质量。
    工作流程：
    1. 生成阶段：基于用户请求生成初始响应
    2. 反思阶段：以专家身份对生成内容进行批判性分析
    3. 改进阶段：根据反馈生成改进版本
    4. 重复直到达到满意标准或达到最大迭代次数
    
    Attributes:
        model (str): 用于生成和反思的语言模型名称
        client (DeepSeekAPI): DeepSeekAPI客户端实例，用于与语言模型交互
    """

    def __init__(self, model: str = "deepseek-chat"):
        """
        初始化反思代理
        
        Args:
            model (str): 要使用的语言模型名称，默认为 "deepseek-chat"
        
        Raises:
            ValueError: 当 DEEPSEEK_API_KEY 环境变量未设置时抛出
        """
        import os
        # 从环境变量获取 API 密钥
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable is required. Please set it in your .env file or environment.")
        
        # 创建 DeepSeekAPI 客户端实例
        self.client = DeepSeekAPI(api_key=api_key)
        self.model = model

    def _request_completion(
        self,
        history: list,
        verbose: int = 0,
        log_title: str = "COMPLETION",
        log_color: str = "",
    ):
        """
        私有方法：向 DeepSeekAPI 模型请求补全
        
        这是核心的 LLM 调用方法，负责将消息历史转换为模型可理解的格式，
        并调用 DeepSeekAPI 的 chat_completion 方法获取响应。

        Args:
            history (list): 形成对话或反思历史的消息列表
            verbose (int, optional): 详细程度级别，控制输出。默认为 0（无输出）
            log_title (str, optional): 日志标题，用于标识输出阶段
            log_color (str, optional): 日志颜色，用于区分不同阶段

        Returns:
            str: 模型生成的响应内容
        """
        # 调用 completions_create 函数处理消息格式转换和 API 调用
        output = completions_create(self.client, history, self.model)

        # 如果启用详细输出，打印带颜色的日志
        if verbose > 0:
            print(log_color, f"\n\n{log_title}\n\n", output)

        return output

    def generate(self, generation_history: list, verbose: int = 0) -> str:
        """
        生成阶段：基于提供的生成历史生成响应
        
        这是反思模式的第一步，负责根据用户请求和系统提示生成初始内容。
        使用蓝色输出标识生成阶段。

        Args:
            generation_history (list): 形成对话或生成历史的消息列表
            verbose (int, optional): 详细程度级别，控制打印输出。默认为 0

        Returns:
            str: 生成的响应内容
        """
        return self._request_completion(
            generation_history, verbose, log_title="GENERATION", log_color=Fore.BLUE
        )

    def reflect(self, reflection_history: list, verbose: int = 0) -> str:
        """
        反思阶段：对生成历史进行反思和批判
        
        这是反思模式的第二步，负责以专家身份（如 Andrej Karpathy）对生成的内容
        进行批判性分析，提供改进建议或确认内容质量。
        使用绿色输出标识反思阶段。

        Args:
            reflection_history (list): 形成反思历史的消息列表，通常基于之前的生成或交互
            verbose (int, optional): 详细程度级别，控制打印输出。默认为 0

        Returns:
            str: 来自模型的批判或反思响应
        """
        return self._request_completion(
            reflection_history, verbose, log_title="REFLECTION", log_color=Fore.GREEN
        )

    def run(
        self,
        user_msg: str,
        generation_system_prompt: str = "",
        reflection_system_prompt: str = "",
        n_steps: int = 10,
        verbose: int = 0,
    ) -> str:
        """
        运行反思代理的完整工作流程
        
        这是主要的执行方法，实现了完整的生成-反思-改进循环：
        1. 设置系统提示和历史记录
        2. 执行多轮生成-反思循环
        3. 根据反思结果决定是否继续改进
        4. 返回最终的高质量响应
        
        由于反思模式的迭代性质，可能会耗尽 LLM 上下文或使其变慢。
        因此限制聊天历史为三个消息，使用 FixedFirstChatHistory 类来维护固定的第一消息。

        Args:
            user_msg (str): 启动交互的用户消息或查询
            generation_system_prompt (str, optional): 指导生成过程的系统提示
            reflection_system_prompt (str, optional): 指导反思过程的系统提示
            n_steps (int, optional): 要执行的生成-反思循环次数。默认为 10
            verbose (int, optional): 控制打印输出的详细程度级别。默认为 0

        Returns:
            str: 所有循环完成后生成的最终响应
        """
        # 合并用户提供的系统提示和基础系统提示
        generation_system_prompt += BASE_GENERATION_SYSTEM_PROMPT
        reflection_system_prompt += BASE_REFLECTION_SYSTEM_PROMPT

        # 创建生成历史记录 - 使用 FixedFirstChatHistory 保持第一消息固定
        # 这对于维护系统提示在聊天历史中很有用
        generation_history = FixedFirstChatHistory(
            [
                build_prompt_structure(prompt=generation_system_prompt, role="system"),
                build_prompt_structure(prompt=user_msg, role="user"),
            ],
            total_length=3,  # 限制历史长度为3，避免上下文过长
        )

        # 创建反思历史记录 - 只包含反思系统提示
        reflection_history = FixedFirstChatHistory(
            [build_prompt_structure(prompt=reflection_system_prompt, role="system")],
            total_length=3,  # 限制历史长度为3
        )

        # 执行生成-反思循环
        for step in range(n_steps):
            # 显示当前步骤进度
            if verbose > 0:
                fancy_step_tracker(step, n_steps)

            # 第一步：生成响应
            generation = self.generate(generation_history, verbose=verbose)
            update_chat_history(generation_history, generation, "assistant")
            update_chat_history(reflection_history, generation, "user")

            # 第二步：反思和批判生成内容
            critique = self.reflect(reflection_history, verbose=verbose)

            # 智能停止机制：如果反思代理认为内容已经足够好，停止循环
            if "<OK>" in critique:
                # 如果没有提出额外的建议，停止循环
                print(
                    Fore.RED,
                    "\n\nStop Sequence found. Stopping the reflection loop ... \n\n",
                )
                break

            # 第三步：将批判反馈加入生成历史，用于下一轮改进
            update_chat_history(generation_history, critique, "user")
            update_chat_history(reflection_history, critique, "assistant")

        # 返回最终的生成结果
        return generation
