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

def map_labels_wangyiyidun(descriptions: List[str]):

    """依据网易易盾的 riskDescription 映射标签"""

    if not descriptions:

        return "no", "不违规", "none", "none"

    text = " ".join(descriptions)

    max_risk = 0

    for desc in descriptions:

        if "高风险" in desc or "suggestionRiskLevel: 2" in desc:

            max_risk = max(max_risk, 2)

        elif "中风险" in desc or "suggestionRiskLevel: 1" in desc:

            max_risk = max(max_risk, 1)

        elif "高危" in desc or "suggestionRiskLevel: 3" in desc:

            max_risk = 3

    risk_mapping = {0: "none", 1: "low", 2: "medium", 3: "high"}

    risk_level = risk_mapping.get(max_risk, "none")

    if "毒" in text or "药" in text:

        return "yes", "涉毒", "drugs", "high"

    if "赌" in text or "博" in text:

        return "yes", "涉赌", "gambling", "high"

    if "黄" in text or "情" in text or "色" in text:

        return "yes", "涉黄", "porn", "high"

    if text.strip() and not text.startswith("未检测"):

        return "yes", "其他", "other", "medium"

    return "no", "不违规", "none", "none"

def extract_words_from_wangyiyidun(result_list: List[Dict]) -> List[str]:

    """从网易易盾结果中提取敏感词"""

    words = []

    for result in result_list:

        if isinstance(result, dict) and "result" in result:

            antispam = result.get("result", {}).get("antispam", {})

            merge_hints = antispam.get("mergeHints", [])

            for hint in merge_hints:

                if isinstance(hint, str):

                    words.append(hint.strip())

            labels = antispam.get("labels", [])

            for label in labels:

                sub_labels = label.get("subLabels", [])

                for sub in sub_labels:

                    details = sub.get("details", {})

                    hit_infos = details.get("hitInfos", [])

                    for hit in hit_infos:

                        value = hit.get("value", "")

                        if value:

                            words.append(value.strip())

    return sorted(set(words))

def main():

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    input_dir = "../text_check_result/json"

    pattern = os.path.join(

        input_dir, "*_text_check_results_system_wangyiyidun_json_*.json"

    )

    input_files = glob.glob(pattern)

    if not input_files:

        raise FileNotFoundError("❌ 未找到 system_wangyiyidun_json 输入文件")

    for input_path in input_files:

        filename = os.path.basename(input_path)

        task_flag = filename.replace(

            "_text_check_results_system_wangyiyidun_json_", "__"

        ).split("__", 1)[1].replace(".json", "")

        output_xlsx = (

            f"./result_split/excel/wangyiyidun/"

            f"{timestamp}_text_check_results_system_wangyiyidun_detail_excel_{task_flag}.xlsx"

        )

        output_json = (

            f"./result_split/json/wangyiyidun/"

            f"{timestamp}_text_check_results_system_wangyiyidun_detail_json_{task_flag}.json"

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

            sys_res_str = item.get("system_result", "")

            result_list = []

            descriptions = []

            try:

                if isinstance(sys_res_str, str):

                    if sys_res_str.startswith("{"):

                        sys_res = json.loads(sys_res_str)

                        result_list = [sys_res]

                    elif sys_res_str.startswith("["):

                        result_list = json.loads(sys_res_str)

                    else:

                        result_list = []

                else:

                    result_list = sys_res_str if isinstance(sys_res_str, list) else [sys_res_str]

                for result in result_list:

                    if isinstance(result, dict):

                        if result.get("code") != 200:

                            continue

                        antispam = result.get("result", {}).get("antispam", {})

                        suggestion = antispam.get("suggestion", 0)

                        if suggestion > 0:

                            risk_desc = antispam.get("riskDescription", "")

                            if risk_desc:

                                descriptions.append(risk_desc)

                            labels = antispam.get("labels", [])

                            for label in labels:

                                sub_labels = label.get("subLabels", [])

                                for sub in sub_labels:

                                    sub_desc = sub.get("riskDescription", "")

                                    if sub_desc:

                                        descriptions.append(sub_desc)

            except (json.JSONDecodeError, KeyError, AttributeError) as e:

                print(f"⚠️  解析 system_result 出错 (id={item.get('id')}): {e}")

                descriptions = ["解析错误"]

                result_list = []

            is_illegal, label_zh, label_en, risk_level = map_labels_wangyiyidun(descriptions)

            words_list = extract_words_from_wangyiyidun(result_list)

            row.update({

                "is_illegal": is_illegal,

                "label_zh": label_zh,

                "label_en": label_en,

                "risk_level": risk_level,

                "violations": json.dumps(descriptions, ensure_ascii=False),

                "violations_count": len(descriptions),

                "words_list": json.dumps(words_list, ensure_ascii=False),

            })

            flat_results = {}

            for i, result in enumerate(result_list):

                flat_result = flatten_dict(result, f"result_{i}")

                flat_results.update(flat_result)

            for k, v in flat_results.items():

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
