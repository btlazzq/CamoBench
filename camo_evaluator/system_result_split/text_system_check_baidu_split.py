import json

import os

import glob

from datetime import datetime

import pandas as pd

from typing import Any, Dict, List

def flatten_dict(d: Any, parent_key: str = "", sep: str = "_") -> Dict[str, Any]:

    """递归展开 dict / list"""

    items = {}

    if isinstance(d, dict):

        for k, v in d.items():

            new_key = f"{parent_key}{sep}{k}" if parent_key else k

            items.update(flatten_dict(v, new_key, sep))

    elif isinstance(d, list):

        for i, v in enumerate(d):

            new_key = f"{parent_key}{sep}{i}"

            items.update(flatten_dict(v, new_key, sep))

    else:

        items[parent_key] = d

    return items

def map_labels_baidu(data_items: List[Dict], conclusion: str):

    """依据百度 JSON 的内容映射标签"""

    if conclusion == "合规":

        return "no", "不违规", "none", "none"

    messages = []

    label_descs = []

    for item in data_items:

        if "msg" in item:

            messages.append(item["msg"])

        if "hits" in item:

            for hit in item["hits"]:

                if "wordHitPositions" in hit:

                    for pos in hit["wordHitPositions"]:

                        if "labelDesc" in pos:

                            label_descs.append(pos["labelDesc"])

    text = " ".join(messages + label_descs)

    risk_level = "high"

    if conclusion == "疑似":

        risk_level = "medium"

    if "毒品" in text or "冰毒" in text or "海洛因" in text or "吗啡" in text or "大麻" in text or "摇头丸" in text or "可卡因" in text or "氯胺酮" in text or "麻古" in text:

        return "yes", "涉毒", "drugs", risk_level

    if "赌博" in text or "博彩" in text or "赌场" in text or "炸金花" in text or "六合彩" in text or "娱乐城" in text or "扑克" in text or "时时彩" in text:

        return "yes", "涉赌", "gambling", risk_level

    if "色情" in text or "成人" in text or "约炮" in text or "操逼" in text or "肉棒" in text or "肉穴" in text or "舔阴" in text or "淫叫" in text or "口交" in text or "肛交" in text or "自慰" in text:

        return "yes", "涉黄", "porn", risk_level

    if "辱骂" in text or "傻逼" in text or "蠢猪" in text or "脑残" in text or "丑逼" in text:

        return "yes", "其他", "other", risk_level

    return "yes", "其他", "other", risk_level

def extract_words_baidu(data_items: List[Dict]):

    """从百度 JSON 提取敏感词"""

    words = []

    for item in data_items:

        if "hits" in item:

            for hit in item["hits"]:

                if "words" in hit and isinstance(hit["words"], list):

                    for word in hit["words"]:

                        if word and word.strip():

                            for w in word.split("&"):

                                if w.strip():

                                    words.append(w.strip())

                if "wordHitPositions" in hit:

                    for pos in hit["wordHitPositions"]:

                        if "keyword" in pos:

                            keyword = pos["keyword"]

                            for w in keyword.split("&"):

                                if w.strip():

                                    words.append(w.strip())

    return sorted(set(words))

def main():

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    input_dir = "../text_check_result/json"

    pattern = os.path.join(

        input_dir, "*_text_check_results_system_baidu_excel_*.json"

    )

    input_files = glob.glob(pattern)

    if not input_files:

        pattern = os.path.join(

            input_dir, "*_text_check_results_system_baidu_*.json"

        )

        input_files = glob.glob(pattern)

    if not input_files:

        raise FileNotFoundError("❌ 未找到 system_baidu 输入文件")

    for input_path in input_files:

        filename = os.path.basename(input_path)

        task_flag = filename.replace(

            "_text_check_results_system_baidu_json_", "__"

        ).split("__", 1)[1].replace(".json", "")

        output_xlsx = (

            f"./result_split/word_to_texts_excel/baidu/"

            f"{timestamp}_text_check_results_system_baidu_detail_excel_{task_flag}.xlsx"

        )

        output_json = (

            f"./result_split/json/baidu/"

            f"{timestamp}_text_check_results_system_baidu_detail_json_{task_flag}.json"

        )

        print(f"📄 正在处理: {filename} | task_flag={task_flag}")

        with open(input_path, "r", encoding="utf-8") as f:

            raw_data = json.load(f)

        rows = []

        for item in raw_data:

            row = {

                "id": item.get("id"),

                "text": item.get("text"),

                "time_ms": item.get("time_ms"),

                "total_time_ms": item.get("total_time_ms"),

                "system_result": item.get("system_result"),

            }

            try:

                sys_res = json.loads(item.get("system_result", "{}"))

            except:

                sys_res = {}

            conclusion = sys_res.get("conclusion", "")

            conclusion_type = sys_res.get("conclusionType", 1)

            data_items = sys_res.get("data", [])

            is_illegal, label_zh, label_en, risk_level = map_labels_baidu(data_items, conclusion)

            words_list = extract_words_baidu(data_items)

            row.update(

                {

                    "is_illegal": is_illegal,

                    "label_zh": label_zh,

                    "label_en": label_en,

                    "risk_level": risk_level,

                    "violations": data_items,

                    "violations_count": len(data_items),

                    "words_list": json.dumps(words_list, ensure_ascii=False),

                }

            )

            flat_sys = flatten_dict(sys_res)

            for k, v in flat_sys.items():

                row[f"system_{k}"] = v

            rows.append(row)

        df = pd.DataFrame(rows)

        base_cols = [

            "id",

            "text",

            "is_illegal",

            "label_zh",

            "label_en",

            "risk_level",

            "words_list",

            "violations",

            "violations_count",

            "time_ms",

            "total_time_ms",

        ]

        system_split_cols = sorted(

            [c for c in df.columns if c.startswith("system_") and c != "system_result"]

        )

        final_cols = base_cols + system_split_cols + ["system_result"]

        final_cols = [c for c in final_cols if c in df.columns]

        df = df[final_cols]

        os.makedirs(os.path.dirname(output_xlsx), exist_ok=True)

        os.makedirs(os.path.dirname(output_json), exist_ok=True)

        df.to_excel(output_xlsx, index=False)

        df.to_json(output_json, orient="records", force_ascii=False, indent=2)

        print("✅ 输出完成")

        print("   Excel:", output_xlsx)

        print("   JSON :", output_json)

        print("-" * 60)

if __name__ == "__main__":

    main()
