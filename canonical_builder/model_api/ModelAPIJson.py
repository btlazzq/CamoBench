import model_api.ModelParam as ModelParam
from openai import OpenAI
import random
import json

from label_text.DrugsText import getText as getDrugs
from label_text.GamblingText import getText as getGambling
from label_text.PornText import getText as getPorn
from label_text.BlackText import getText as getBlack
from label_text.SwindleText import getText as getSwindle
from label_text.CurseText import getText as getCurse
from label_text.NoneText import getText as getNone
from label_word.origin_word import RandomSceneList as getInformation

import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def createClient(base_url, api_key):
    return OpenAI(base_url=base_url, api_key=api_key)


def safe_call(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception:
        return ["内容非法，拒绝生成！"]


def clean_json(text):
    """
    清洗模型返回结果
    把字符串 JSON 转成 dict
    """
    if isinstance(text, dict):
        return text

    if not isinstance(text, str):
        return {"text": str(text)}

    text = text.strip()

    # 去掉 ```json ``` 包裹
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]

    try:
        return json.loads(text)
    except Exception:
        return {"text": text}


def getDrugsText(model_name, prompt_number, information):
    config = getattr(ModelParam, model_name, None)
    client = createClient(config["base_url"], config["api_key"])
    return safe_call(getDrugs, client, config["model"], prompt_number, information)


def getGamblingText(model_name, prompt_number, information):
    config = getattr(ModelParam, model_name, None)
    client = createClient(config["base_url"], config["api_key"])
    return safe_call(getGambling, client, config["model"], prompt_number, information)


def getPornText(model_name, prompt_number, information):
    config = getattr(ModelParam, model_name, None)
    client = createClient(config["base_url"], config["api_key"])
    return safe_call(getPorn, client, config["model"], prompt_number, information)


def getBlackText(model_name, prompt_number, information):
    config = getattr(ModelParam, model_name, None)
    client = createClient(config["base_url"], config["api_key"])
    return safe_call(getBlack, client, config["model"], prompt_number, information)


def getSwindleText(model_name, prompt_number, information):
    config = getattr(ModelParam, model_name, None)
    client = createClient(config["base_url"], config["api_key"])
    return safe_call(getSwindle, client, config["model"], prompt_number, information)


def getCurseText(model_name, prompt_number, information):
    config = getattr(ModelParam, model_name, None)
    client = createClient(config["base_url"], config["api_key"])
    return safe_call(getCurse, client, config["model"], prompt_number, information)


def getNoneText(model_name, prompt_number, information):
    config = getattr(ModelParam, model_name, None)
    client = createClient(config["base_url"], config["api_key"])
    return safe_call(getNone, client, config["model"], prompt_number, information)


def getRandomInformation():
    return {
        "sex": random.choice(getInformation.SEX),
        "age": random.choice(getInformation.AGE),
        "profession": random.choice(getInformation.PROFESSION),
        "education": random.choice(getInformation.EDUCATION),
        "platform": random.choice(getInformation.PLATFORM),
        "angle": random.choice(getInformation.ANGLE),
        "length": random.choice(getInformation.LENGTH)
    }


if __name__ == "__main__":
    model_name = "DEEPSEEK"
    prompt_number = 5

    # 创建时间戳目录
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = os.path.join("result", "test", timestamp)

    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "output.json")

    results = []

    for i in range(5):
        randomInformation = getRandomInformation()

        item = {
            "information": randomInformation,
            "drugs": clean_json(getDrugsText(model_name, prompt_number, randomInformation)[0]),
            "gambling": clean_json(getGamblingText(model_name, prompt_number, randomInformation)[0]),
            "porn": clean_json(getPornText(model_name, prompt_number, randomInformation)[0]),
            "black": clean_json(getBlackText(model_name, prompt_number, randomInformation)[0]),
            "swindle": clean_json(getSwindleText(model_name, prompt_number, randomInformation)[0]),
            "curse": clean_json(getCurseText(model_name, prompt_number, randomInformation)[0]),
            "none": clean_json(getNoneText(model_name, prompt_number, randomInformation)[0]),
        }

        results.append(item)

        # 实时打印
        print(json.dumps(item, ensure_ascii=False, indent=2))

    # 写入 JSON 文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: {output_file}")