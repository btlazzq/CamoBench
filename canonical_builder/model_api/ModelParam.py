import os

DEEPSEEK = {

    "base_url": "https://api.deepseek.com",

    "api_key": os.getenv("DEEPSEEK_API_KEY"),

    "model": "deepseek-chat"

}

QWEN = {

    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",

    "api_key": os.getenv("QWEN_API_KEY"),

    "model": "qwen2.5-72b-instruct"

}

KIMI = {

    "base_url": "https://api.moonshot.cn/v1",

    "api_key": os.getenv("KIMI_JL_API"),

    "model": "kimi-k2-turbo-preview"

}

DOUBAO = {

    "base_url": "https://ark.cn-beijing.volces.com/api/v3",

    "api_key": os.getenv("DOUBAO_JL_API"),

    "model": "doubao-seed-2-0-pro-260215"

}

HUNYUAN = {

    "base_url": "https://api.hunyuan.cloud.tencent.com/v1",

    "api_key": os.getenv("HUNYUAN_JL_API"),

    "model": "hunyuan-vision"

}

WENXINYIYAN = {

    "base_url": "https://qianfan.baidubce.com/v2",

    "api_key": os.getenv("WENXINYIYAN_JL_API"),

}

ZHIPU = {

    "base_url": "https://open.bigmodel.cn/api/paas/v4",

    "api_key": os.getenv("ZHIPU_JL_API"),

    "model": "glm-4-flash"

}

CHATGPT = {

    "base_url": "https://api.openai.com/v1",

    "api_key": os.getenv("OPENAI_API_KEY"),

    "model": "gpt-4o"

}
