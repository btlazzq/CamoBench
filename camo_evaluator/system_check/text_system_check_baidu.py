import os

import time

import json

import argparse

from datetime import datetime

import pandas as pd

import requests

from typing import List

API_KEY = os.environ['BAIDU_API_KEY']

SECRET_KEY = os.environ['BAIDU_API_SECRET']

def get_access_token():

    url = "https://aip.baidubce.com/oauth/2.0/token"

    params = {

        "grant_type": "client_credentials",

        "client_id": API_KEY,

        "client_secret": SECRET_KEY

    }

    return str(requests.post(url, params=params).json().get("access_token"))

ACCESS_TOKEN = get_access_token()

BAIDU_URL = f"https://aip.baidubce.com/rest/2.0/solution/v1/text_censor/v2/user_defined?access_token={ACCESS_TOKEN}"

HEADERS = {

    'Content-Type': 'application/x-www-form-urlencoded',

    'Accept': 'application/json'

}

def call_baidu_text_moderation(text: str) -> str:

    payload = f'text={text}&strategyId=1'

    try:

        response = requests.post(BAIDU_URL, headers=HEADERS, data=payload.encode("utf-8"), timeout=10)

        response.encoding = "utf-8"

        return json.dumps(response.json(), ensure_ascii=False)

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

    ap.add_argument("--output_xlsx", default=f"../text_check_result/excel/{timestamp}_text_check_results_system_baidu_excel_{task_flag}.xlsx")

    ap.add_argument("--output_json", default=f"../text_check_result/json/{timestamp}_text_check_results_system_baidu_json_{task_flag}.json")

    args = ap.parse_args()

    texts = read_texts_from_xlsx(args.input_txt, column_name="text")

    rows = []

    total_time_ms = 0.0

    SLEEP_SECONDS = 5

    def save_partial():

        with open(args.output_json, "w", encoding="utf-8") as f:

            json.dump(rows, f, ensure_ascii=False, indent=2)

        pd.DataFrame(rows).to_excel(args.output_xlsx, index=False)

    for idx, text in enumerate(texts, start=1):

        print("\n" + "=" * 80)

        print(f"[{idx}/{len(texts)}] 原文：{text}")

        start_t = time.perf_counter()

        system_result = call_baidu_text_moderation(text)

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

        time.sleep(SLEEP_SECONDS)

if __name__ == "__main__":

    main()
