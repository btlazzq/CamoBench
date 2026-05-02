import model_api.ModelParam as ModelParam
from openai import OpenAI
import random
import json
import sys
import os
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from label_text.BlackText import getText as getBlack
from label_word.origin_word import RandomSceneList as getInformation

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def createClient(base_url, api_key):
    return OpenAI(base_url=base_url, api_key=api_key)


def safe_call(func, *args, max_retry=3, sleep_seconds=2, **kwargs):
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


def getBlackText(model_name, prompt_number, information):
    config = get_model_config(model_name)
    client = createClient(config["base_url"], config["api_key"])
    return safe_call(getBlack, client, config["model"], prompt_number, information)


def make_balanced_sequence(options, n, seed):
    """
    生成长度为 n 的“尽量均衡”序列：每轮对 options 打散一次并拼接，保证整体分布均衡。
    多线程下按索引取值即可保持稳定。
    """
    if n <= 0:
        return []
    if not options:
        return [None] * n

    rng = random.Random(seed)
    out = []
    while len(out) < n:
        batch = list(options)
        rng.shuffle(batch)
        out.extend(batch)
    return out[:n]


def getRandomInformation(i=None, black_role_seq=None, black_scene_seq=None):
    info = {
        "sex": random.choice(getInformation.SEX),
        "age": random.choice(getInformation.AGE),
        "profession": random.choice(getInformation.PROFESSION),
        "education": random.choice(getInformation.EDUCATION),
        "platform": random.choice(getInformation.PLATFORM),
        "angle": random.choice(getInformation.ANGLE),
        "length": random.choice(getInformation.LENGTH),
        "black_role": random.choice(getInformation.BLACK_ROLE),
        "black_scene": random.choice(getInformation.BLACK_SCENE),
    }

    if i is not None and black_role_seq is not None:
        info["black_role"] = black_role_seq[i]
    if i is not None and black_scene_seq is not None:
        info["black_scene"] = black_scene_seq[i]

    return info


def build_one_item(i, model_name, prompt_number, black_role_seq=None, black_scene_seq=None):
    """
    生成一条样本，仅包含 information + black（黑产）
    """
    randomInformation = getRandomInformation(
        i=i,
        black_role_seq=black_role_seq,
        black_scene_seq=black_scene_seq,
    )

    item = {
        "information": randomInformation,
        "black": clean_json(extract_first(getBlackText(model_name, prompt_number, randomInformation))),
    }
    return i, item


if __name__ == "__main__":
    model_name = "DEEPSEEK"
    prompt_number = 5
    num_samples = 500
    max_workers = 5

    # 创建时间戳目录
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_id = f"black_{timestamp}"
    output_dir = os.path.join("result", "test", run_id)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "black_100.json")

    seed_base = int(timestamp)
    black_role_seq = make_balanced_sequence(getInformation.BLACK_ROLE, num_samples, seed=seed_base + 1)
    black_scene_seq = make_balanced_sequence(getInformation.BLACK_SCENE, num_samples, seed=seed_base + 2)

    results = [None] * num_samples
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                build_one_item,
                i,
                model_name,
                prompt_number,
                black_role_seq,
                black_scene_seq,
            )
            for i in range(num_samples)
        ]

        completed = 0
        for future in as_completed(futures):
            try:
                i, item = future.result()
                results[i] = item
                completed += 1

                print(f"\n===== 已完成 {completed}/{num_samples}，样本索引 {i}（仅黑产）=====")
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