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

def map_labels(descriptions: List[str]):

    """依据 Description 映射标签"""

    if descriptions and all(desc == "未检测出风险" for desc in descriptions):

        return "no", "不违规", "none"

    text = " ".join(descriptions)

    if "毒" in text:

        return "yes", "涉毒", "drugs"

    if "赌" in text:

        return "yes", "涉赌", "gambling"

    if "黄" in text or "色情" in text:

        return "yes", "涉黄", "porn"

    return "yes", "其他", "other"

def main():

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    input_dir = "../text_check_result/json"

    pattern = os.path.join(

        input_dir, "*_text_check_results_system_aliyun_json_*.json"

    )

    input_files = glob.glob(pattern)

    if not input_files:

        raise FileNotFoundError("❌ 未找到 system_aliyun_json 输入文件")

    for input_path in input_files:

        filename = os.path.basename(input_path)

        task_flag = filename.replace(

            "_text_check_results_system_aliyun_json_", "__"

        ).split("__", 1)[1].replace(".json", "")

        output_xlsx = (

            f"./result_split/excel/aliyun/"

            f"{timestamp}_text_check_results_system_aliyun_detail_excel_{task_flag}.xlsx"

        )

        output_json = (

            f"./result_split/json/aliyun/"

            f"{timestamp}_text_check_results_system_aliyun_detail_json_{task_flag}.json"

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

            sys_res = json.loads(item.get("system_result", "{}"))

            data = sys_res.get("Data", {})

            result_list = data.get("Result", [])

            descriptions = [r.get("Description", "") for r in result_list]

            is_illegal, label_zh, label_en = map_labels(descriptions)

            row.update(

                {

                    "is_illegal": is_illegal,

                    "label_zh": label_zh,

                    "label_en": label_en,

                    "risk_level": data.get("RiskLevel", "none"),

                    "violations": result_list,

                    "violations_count": len(result_list),

                }

            )

            words = []

            for r in result_list:

                rw = r.get("RiskWords")

                if rw:

                    words.extend(

                        [w.strip() for w in rw.replace("&", ",").split(",")]

                    )

            row["words_list"] = json.dumps(

                sorted(set(words)), ensure_ascii=False

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

        df.to_excel(output_xlsx, index=False)

        df.to_json(output_json, orient="records", force_ascii=False, indent=2)

        print("✅ 输出完成")

        print("   Excel:", output_xlsx)

        print("   JSON :", output_json)

        print("-" * 60)

if __name__ == "__main__":

    main()
