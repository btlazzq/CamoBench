import os

import time

import json

import argparse

import random

import hashlib

import hmac

import base64

import requests

from datetime import datetime

import pandas as pd

from typing import List

AK = os.environ["360_API_KEY"]

SK = os.environ["360_API_SECRET"]

HOST = "ai.zyun.360.cn"

def md5(s: str) -> str:

    return hashlib.md5(s.encode("utf-8")).hexdigest()

def hmac_sha1_base64(key: str, msg: str) -> str:

    digest = hmac.new(key.encode("utf-8"), msg.encode("utf-8"), hashlib.sha1).digest()

    return base64.b64encode(digest).decode("utf-8")

def build_param_sign(params: dict, rand_num: str) -> str:

    param_str = "".join(f"{k}={params[k]}" for k in sorted(params))

    first_md5 = md5(param_str)

    final_md5 = md5(first_md5 + rand_num)

    return final_md5

def build_auth_headers(params: dict) -> dict:

    auth_time = str(int(time.time()))

    rand_num = str(random.randint(1, 100000))

    param_sign = build_param_sign(params, rand_num)

    auth_string = "\n".join([AK, auth_time, rand_num, param_sign])

    authorization = AK + ":" + hmac_sha1_base64(SK, auth_string)

    return {

        "Authorization": authorization,

        "Auth-Time": auth_time,

        "Rand-Num": rand_num,

        "Auth-Ver": "1.0",

        "Content-Type": "application/x-www-form-urlencoded",

    }

def call_360_text_moderation(template_id, scene, text, text_type="0") -> str:

    url = f"http://{HOST}/v1/verify/text"

    params = {

        "template_id": template_id,

        "scene": scene,

        "text": text,

        "text_type": text_type,

    }

    headers = build_auth_headers(params)

    try:

        resp = requests.post(url, headers=headers, data=params, timeout=10)

        return json.dumps(resp.json(), ensure_ascii=False)

    except Exception as e:

        return json.dumps({"error": repr(e)}, ensure_ascii=False)

def read_texts_from_txt(file_path):

    with open(file_path, "r", encoding="utf-8") as f:

        return [line.strip() for line in f if line.strip()]

def read_texts_from_xlsx(path: str, column_name: str = "text") -> List[str]:

    if not os.path.exists(path):

        raise FileNotFoundError(f"输入文件不存在：{path}")

    df = pd.read_excel(path)

    if column_name not in df.columns:

        raise ValueError(f"Excel 中不存在列：{column_name}，当前列为：{list(df.columns)}")

    texts = df[column_name].dropna().astype(str).tolist()

    return texts

def main():

    task_flag = input("⚠️ 请输入任务标识（如 N / A / P / M / S / E / APMSE，回车开始，空则退出）：").strip()

    if not task_flag:

        print("已取消执行。")

        return

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    ap = argparse.ArgumentParser()

    ap.add_argument("--input_txt", default=f"../texts_check_data/真实数据汇总.xlsx")

    ap.add_argument("--output_xlsx", default=f"../text_check_result/excel/{timestamp}_text_check_results_system_360_excel_{task_flag}.xlsx")

    ap.add_argument("--output_json", default=f"../text_check_result/json/{timestamp}_text_check_results_system_360_json_{task_flag}.json")

    args = ap.parse_args()

    texts = read_texts_from_xlsx(args.input_txt, column_name="text")

    rows = []

    total_time_ms = 0.0

    TEMPLATE_ID = "176893691787720904461179"

    SCENE = "live"

    def save_partial():

        with open(args.output_json, "w", encoding="utf-8") as f:

            json.dump(rows, f, ensure_ascii=False, indent=2)

        pd.DataFrame(rows).to_excel(args.output_xlsx, index=False)

    for idx, text in enumerate(texts, start=1):

        print("\n" + "=" * 80)

        print(f"[{idx}/{len(texts)}] 原文：{text}")

        start_t = time.perf_counter()

        system_result = call_360_text_moderation(TEMPLATE_ID, SCENE, text)

        cost_time_ms = (time.perf_counter() - start_t) * 1000

        total_time_ms += cost_time_ms

        row = {

            "id": idx,

            "text": text,

            "system_result": system_result,

            "time_ms": round(cost_time_ms, 2),

            "total_time_ms": round(total_time_ms, 2)

        }

        print("【写入结果 row】")

        print(json.dumps(row, ensure_ascii=False, indent=2))

        rows.append(row)

        save_partial()

        time.sleep(2)

if __name__ == "__main__":

    main()
