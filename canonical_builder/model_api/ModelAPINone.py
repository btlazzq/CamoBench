import model_api.ModelParam as ModelParam
from openai import OpenAI
import random
import json
import sys
import os
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from label_text.NoneText import getText as getNone
from label_word.origin_word import RandomSceneList as getInformation

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def createClient(base_url, api_key):
    return OpenAI(base_url=base_url, api_key=api_key)


def safe_call(func, *args, max_retry=3, sleep_seconds=2, **kwargs):
    """
    安全调用 + 简单重试
    """
    last_error = None
    for attempt in range(max_retry):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_error = e
            if attempt < max_retry - 1:
                time.sleep(sleep_seconds * (attempt + 1))
    return [f"错误信息: {str(last_error)}"]


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
            text = parts[1].strip()
            if text.startswith("json"):
                text = text[4:].strip()

    try:
        return json.loads(text)
    except Exception:
        return {"text": text}


def extract_first(result):
    """
    兼容返回 list/tuple/str 的情况
    """
    if isinstance(result, (list, tuple)) and len(result) > 0:
        return result[0]
    return result


def get_model_config(model_name):
    config = getattr(ModelParam, model_name, None)
    if config is None:
        raise ValueError(f"未找到模型配置: {model_name}")
    return config


def getNoneText(model_name, prompt_number, information):
    config = get_model_config(model_name)
    client = createClient(config["base_url"], config["api_key"])
    return safe_call(getNone, client, config["model"], prompt_number, information)


def getRandomInformation():
    # none 不包含 none_role / none_scene
    return {
        "sex": random.choice(getInformation.SEX),
        "age": random.choice(getInformation.AGE),
        "profession": random.choice(getInformation.PROFESSION),
        "education": random.choice(getInformation.EDUCATION),
        "platform": random.choice(getInformation.PLATFORM),
        "angle": random.choice(getInformation.ANGLE),
        "length": random.choice(getInformation.LENGTH),
        "none_role": random.choice(getInformation.NOT_ILLEGAL_ROLE),
        "none_scene": random.choice(getInformation.NOT_ILLEGAL_SCENE),
    }


def build_one_item(i, model_name, prompt_number):
    """
    生成一条样本，仅包含 information + none（正常）
    """
    randomInformation = getRandomInformation()

    item = {
        "information": randomInformation,
        "none": clean_json(extract_first(getNoneText(model_name, prompt_number, randomInformation))),
    }
    return i, item


if __name__ == "__main__":
    model_name = "DEEPSEEK"
    prompt_number = 5
    num_samples = 100
    max_workers = 5

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_id = f"none_{timestamp}"
    output_dir = os.path.join("result", "test", run_id)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "none_100.json")

    results = [None] * num_samples
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(build_one_item, i, model_name, prompt_number)
            for i in range(num_samples)
        ]

        completed = 0
        for future in as_completed(futures):
            try:
                i, item = future.result()
                results[i] = item
                completed += 1

                print(f"\n===== 已完成 {completed}/{num_samples}，样本索引 {i}（仅正常）=====")
                print(json.dumps(item, ensure_ascii=False, indent=2))
            except Exception as e:
                completed += 1
                print(f"\n===== 已完成 {completed}/{num_samples}，但有任务失败 =====")
                print(f"错误信息: {e}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_time
    print(f"\n结果已保存到: {output_file}")
    print(f"总耗时: {elapsed:.2f} 秒")