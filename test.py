# To use this test, you need to set your DeepSeek API key in one of these ways:
# 1. Set environment variable: export DEEPSEEK_API_KEY="your-api-key-here"
# 2. Create a .env file with: DEEPSEEK_API_KEY=your-api-key-here
# 3. Set it programmatically (for testing only):

import os
# Uncomment and replace with your actual API key for testing:
# os.environ["DEEPSEEK_API_KEY"] = "your-api-key-here"

from agentic_patterns import ReflectionAgent

agent = ReflectionAgent()

generation_system_prompt = "You are a Python programmer tasked with generating high quality Python code"

reflection_system_prompt = "You are Andrej Karpathy, an experienced computer scientist"

user_msg = "Generate a Python implementation of the Merge Sort algorithm"


final_response = agent.run(
    user_msg=user_msg,
    generation_system_prompt=generation_system_prompt,
    reflection_system_prompt=reflection_system_prompt,
    n_steps=10,
    verbose=1,
)

print(final_response)