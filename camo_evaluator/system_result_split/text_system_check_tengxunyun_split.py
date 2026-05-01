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

def map_tengxunyun_labels(result_value: str, sub_label: str, score: int):

    """依据腾讯云的结果映射标签"""

    if result_value == "0":

        return "no", "不违规", "none"

    if sub_label:

        sub_label_lower = sub_label.lower()

        if "drug" in sub_label_lower:

            return "yes", "涉毒", "drugs"

        elif "gamble" in sub_label_lower:

            return "yes", "涉赌", "gambling"

        elif "porn" in sub_label_lower:

            return "yes", "涉黄", "porn"

    if int(score or 0) >= 80:

        return "yes", "其他", "other"

    return "yes", "其他", "other"

def get_risk_level(score: str):

    """根据分数确定风险等级"""

    score_int = int(score or 0)

    if score_int >= 90:

        return "high"

    elif score_int >= 70:

        return "medium"

    elif score_int > 0:

        return "low"

    else:

        return "none"

def get_keywords_from_tengxunyun(section_data: List[Dict]) -> List[str]:

    """从腾讯云结构提取敏感词"""

    keywords = []

    if not section_data:

        return keywords

    for section in section_data:

        if isinstance(section, dict):

            illegal_info = section.get("IllegalInfo", {})

            if illegal_info:

                kw_str = illegal_info.get("Keywords")

                if kw_str:

                    kw_list = [k.strip() for k in kw_str.split(",") if k.strip()]

                    keywords.extend(kw_list)

                hit_infos = illegal_info.get("HitInfos")

                if hit_infos:

                    if isinstance(hit_infos, list):

                        for hit in hit_infos:

                            if hit.get("Keyword"):

                                keywords.append(hit["Keyword"])

                    elif isinstance(hit_infos, dict) and hit_infos.get("Keyword"):

                        keywords.append(hit_infos["Keyword"])

            ads_info = section.get("AdsInfo", {})

            if ads_info:

                kw_str = ads_info.get("Keywords")

                if kw_str:

                    kw_list = [k.strip() for k in kw_str.split(",") if k.strip()]

                    keywords.extend(kw_list)

    return sorted(set(keywords))

def main():

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    input_dir = "../text_check_result/json"

    pattern = os.path.join(

        input_dir, "*_text_check_results_system_tengxunyun_json_*.json"

    )

    input_files = glob.glob(pattern)

    if not input_files:

        raise FileNotFoundError("❌ 未找到 system_tengxunyun_json 输入文件")

    for input_path in input_files:

        filename = os.path.basename(input_path)

        task_flag = filename.replace(

            "_text_check_results_system_tengxunyun_json_", "__"

        ).split("__", 1)[1].replace(".json", "")

        output_xlsx = (

            f"./result_split/excel/tengxunyun/"

            f"{timestamp}_text_check_results_system_tengxunyun_detail_excel_{task_flag}.xlsx"

        )

        output_json = (

            f"./result_split/json/tengxunyun/"

            f"{timestamp}_text_check_results_system_tengxunyun_detail_json_{task_flag}.json"

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

            jobs_detail = sys_res.get("JobsDetail", {})

            result_value = jobs_detail.get("Result", "0")

            score = jobs_detail.get("Score", "0")

            sub_label = jobs_detail.get("SubLabel", "")

            section_data = jobs_detail.get("Section", [])

            is_illegal, label_zh, label_en = map_tengxunyun_labels(result_value, sub_label, score)

            risk_level = get_risk_level(score)

            keywords = get_keywords_from_tengxunyun(section_data)

            violations_count = 0

            if section_data:

                for section in section_data:

                    if isinstance(section, dict):

                        for key in ["IllegalInfo", "AdsInfo", "AbuseInfo", "PoliticsInfo", "PornInfo", "TerrorismInfo"]:

                            info = section.get(key, {})

                            if info and info.get("HitFlag") in ["1", "2"]:

                                violations_count += 1

            row.update(

                {

                    "is_illegal": is_illegal,

                    "label_zh": label_zh,

                    "label_en": label_en,

                    "risk_level": risk_level,

                    "words_list": json.dumps(keywords, ensure_ascii=False),

                    "violations": json.dumps(section_data, ensure_ascii=False) if section_data else "[]",

                    "violations_count": violations_count,

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
