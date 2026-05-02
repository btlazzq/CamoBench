import os

# deepseek-chat
DEEPSEEK = {
    "base_url": "https://api.deepseek.com",
    "api_key": os.getenv("DEEPSEEK_API_KEY"),
    #"model": "deepseek-coder"
    "model": "deepseek-chat"
    # "model": "deepseek-reasoner"
}

# qwen-plus
QWEN = {
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "api_key": os.getenv("QWEN_API_KEY"),
    #"model": "qwen-turbo"
    # "model": "qwen-plus"
    # "model": "qwen-max"
    # "model": "qwen-max-longcontext"
    # "model": "qwen2-72b-instruct"
    "model": "qwen2.5-72b-instruct"
    # "model": "qwen3.5-plus"
}

# kimi-k2-turbo-preview
KIMI = {
    "base_url": "https://api.moonshot.cn/v1",
    "api_key": os.getenv("KIMI_JL_API"),
    # "model": "moonshot-v1-8k"
    # "model": "moonshot-v1-32k"
    # "model": "moonshot-v1-128k"
    "model": "kimi-k2-turbo-preview"
}

# doubao-seed-1-6-lite-251015
DOUBAO = {
    "base_url": "https://ark.cn-beijing.volces.com/api/v3",
    "api_key": os.getenv("DOUBAO_JL_API"),
    # "model": "doubao-1-5-lite-32k-250115"
    # "model": "doubao-1-5-pro-32k-250115"
    # "model": "doubao-lite-32k-character-250228"
    # "model": "doubao-seed-1-6-vision-250815"
    # "model": "doubao-seed-1-6-flash-250828"
    # "model": "doubao-seed-1-6-251015"
    # "model": "doubao-seed-1-6-lite-251015"
    # "model": "doubao-seed-1-8-251228"
    # "model": "doubao-seed-2-0-mini-260215"
    # "model": "doubao-seed-2-0-lite-260215"
    "model": "doubao-seed-2-0-pro-260215"
}

# hunyuan-turbos-latest
HUNYUAN = {
    "base_url": "https://api.hunyuan.cloud.tencent.com/v1",
    "api_key": os.getenv("HUNYUAN_JL_API"),
    # "model": "hunyuan-lite"
    # "model": "hunyuan-standard"
    # "model": "hunyuan-pro"
    # "model": "hunyuan-turbo"
    # "model": "hunyuan-turbos-latest"
    # "model": "hunyuan-large"
    "model": "hunyuan-vision"
}

# ernie-4.5-turbo-128k
WENXINYIYAN = {
    "base_url": "https://qianfan.baidubce.com/v2",
    "api_key": os.getenv("WENXINYIYAN_JL_API"),
    # "model": "ernie-3.5"
    # "model": "ernie-3.5-8k"
    # "model": "ernie-4.0"
    # "model": "ernie-4.0-8k"
    # "model": "ernie-4.0-turbo"
    # "model": "ernie-4.5"
    # "model": "ernie-4.5-turbo"
    # "model": "ernie-4.5-turbo-128k"
}

ZHIPU = {
    "base_url": "https://open.bigmodel.cn/api/paas/v4",
    "api_key": os.getenv("ZHIPU_JL_API"),
    # "model": "glm-3-turbo"
    # "model": "glm-4"
    # "model": "glm-4-air"
    # "model": "glm-4-airx"
    # "model": "glm-4-plus"
    # "model": "glm-4-long"
    "model": "glm-4-flash"
    # "model": "glm-5"
}

# gpt-4o-mini
CHATGPT = {
    "base_url": "https://api.openai.com/v1",
    "api_key": os.getenv("OPENAI_API_KEY"),
    # "api_key": os.getenv("CHATGPT_HP_API"),
    # "model": "gpt-3.5-turbo"
    # "model": "gpt-3.5-turbo-16k"
    # "model": "gpt-4"
    # "model": "gpt-4-turbo"
    # "model": "gpt-4.1"
    "model": "gpt-4o"
    # "model": "gpt-4o-mini"
    # "model": "gpt-5"
    # "model": "gpt-5-mini"
}