import os

from agentscope.agent import ReActAgent

from agentscope.formatter import OpenAIMultiAgentFormatter

from agentscope.model import OpenAIChatModel

from agentscope.tool import Toolkit

def make_model():

    """
    构造一个指向 DeepSeek OpenAI-兼容接口的模型。
    - MODEL_NAME: DeepSeek 模型名，如 deepseek-chat
    - DEEPSEEK_API_KEY: DeepSeek 提供的 API key
    - OPENAI_API_BASE: DeepSeek 的 OpenAI-like base_url，默认为官方 https://api.deepseek.com
    """

    base_url = os.getenv("OPENAI_API_BASE", "https://api.deepseek.com")

    return OpenAIChatModel(

        model_name=os.getenv("MODEL_NAME", "deepseek-chat"),

        api_key=os.getenv("DEEPSEEK_API_KEY"),

        stream=False,

        client_kwargs={"base_url": base_url},

    )

model = make_model()

toolkit = Toolkit()

formatter = OpenAIMultiAgentFormatter()

def build_agent(name: str):

    return ReActAgent(

        name=name,

        sys_prompt="必须严格输出JSON。禁止输出任何多余文本。",

        model=model,

        formatter=formatter,

    )
