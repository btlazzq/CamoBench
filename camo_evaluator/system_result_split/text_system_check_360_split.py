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

def calc_risk_level_by_score(scores: List[float]) -> str:

    """
    网易易盾式 risk 映射：
    >=0.7 high
    0.3-0.7 medium
    0.1-0.3 low
    else none
    """

    if not scores:

        return "none"

    max_score = max(scores)

    if max_score >= 0.7:

        return "high"

    elif max_score >= 0.3:

        return "medium"

    elif max_score >= 0.1:

        return "low"

    else:

        return "none"

def map_labels(

    suggest_summary: str,

    label_desc_list: List[str],

    score_list: List[float],

):

    """
    ✔ 是否违规：只看 suggest_summary
    ✔ PASS 直接短路
    ✔ 标签：只用 label_desc
    ✔ 风险等级：只由 score 决定
    """

    if str(suggest_summary).upper() == "PASS":

        return "no", "不违规", "none", "none"

    risk_level = calc_risk_level_by_score(score_list)

    text = " ".join(label_desc_list)

    if "毒" in text or "药" in text:

        return "yes", "涉毒", "drugs", risk_level

    if "赌" in text or "博" in text:

        return "yes", "涉赌", "gambling", risk_level

    if "黄" in text or "情" in text or "色" in text:

        return "yes", "涉黄", "porn", risk_level

    if text.strip() and not text.startswith("未检测"):

        return "yes", "其他", "other", risk_level

    return "yes", "其他", "other", risk_level

def main():

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    input_dir = "../text_check_result/json"

    pattern = os.path.join(

        input_dir, "*_text_check_results_system_360_json_*.json"

    )

    input_files = glob.glob(pattern)

    if not input_files:

        raise FileNotFoundError("❌ 未找到 system_360_json 输入文件")

    for input_path in input_files:

        filename = os.path.basename(input_path)

        task_flag = filename.replace(

            "_text_check_results_system_360_json_", "__"

        ).split("__", 1)[1].replace(".json", "")

        output_xlsx = (

            f"./result_split/word_to_texts_excel/360/"

            f"{timestamp}_text_check_results_system_360_detail_excel_{task_flag}.xlsx"

        )

        output_json = (

            f"./result_split/json/360/"

            f"{timestamp}_text_check_results_system_360_detail_json_{task_flag}.json"

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

            except Exception:

                sys_res = {}

            body = sys_res.get("data", {}).get("body", {})

            suggest_summary = body.get("suggest_summary", "")

            suggest_summary_desc = body.get("suggest_summary_desc", "")

            items_list = body.get("items", [])

            label_desc_list = []

            score_list = []

            words_list = []

            for item_data in items_list:

                label_desc = item_data.get("label_desc", "")

                if label_desc:

                    label_desc_list.append(label_desc)

                score = item_data.get("score")

                if isinstance(score, (int, float)):

                    score_list.append(float(score))

                words = item_data.get("words", [])

                if isinstance(words, list):

                    words_list.extend(words)

            is_illegal, label_zh, label_en, risk_level = map_labels(

                suggest_summary,

                label_desc_list,

                score_list,

            )

            row.update({

                "is_illegal": is_illegal,

                "label_zh": label_zh,

                "label_en": label_en,

                "risk_level": risk_level,

                "suggest_summary": suggest_summary,

                "suggest_summary_desc": suggest_summary_desc,

                "violations": items_list,

                "violations_count": len(items_list),

            })

            row["words_list"] = json.dumps(

                sorted(set([w for w in words_list if w])), ensure_ascii=False

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

            "suggest_summary",

            "suggest_summary_desc",

        ]

        system_cols = sorted(

            [c for c in df.columns if c.startswith("system_") and c != "system_result"]

        )

        final_cols = base_cols + system_cols + ["system_result"]

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
