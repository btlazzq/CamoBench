import json
import re
import ast
import logging
from typing import Tuple, Dict, Any
from json_repair import repair_json

log = logging.getLogger(__name__)

JSON_BLOCK_PATTERN = re.compile(
    r"```(?:json)?\s*(.*?)```", 
    re.DOTALL | re.IGNORECASE 
)

JSON_OBJECT_PATTERN = re.compile(
    r"\{.*\}", 
    re.DOTALL 
)

def parse_llm_json(content: str) -> Tuple[str, Dict[str, Any]]:
    if not content:
        return "", {}
    raw = content.strip()

    block_match = JSON_BLOCK_PATTERN.search(raw) 
    if block_match:
        raw = block_match.group(1).strip() 

    obj_match = JSON_OBJECT_PATTERN.search(raw) 
    if obj_match:
        raw = obj_match.group(0) 

    try:
        return raw, json.loads(raw)
    except Exception:
        pass


    # 清除常见格式错误
    cleaned = (
        raw.replace("\n", " ") # 去除换行
           .replace("\r", "") # 去除回车
           .replace("\t", " ") # 去除制表符
           .replace("\\n", " ") # 去除转义换行
           .replace("\\", "") # 去除反斜杠
           .strip() # 去除首尾空格
    )

    clean = re.sub(r"'", '"', cleaned)

    try:
        return clean, json.loads(clean)
    except Exception:
        pass

    try:
        repaired = repair_json(clean) # 自动修复JSON

        if isinstance(repaired, str):
            repaired_dict = json.loads(repaired) # 解析为字典
        else:
            repaired_dict = repaired

        return json.dumps(repaired_dict, ensure_ascii=False), repaired_dict
    except Exception:
        pass


    try:
        tree = ast.parse(clean) # 解析为抽象语法树
        result = {}
        # 遍历语法树
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # 遍历关键字参数
                for kw in node.keywords:
                    result[kw.arg] = ast.literal_eval(kw.value) # 提取参数值
        if result:
            return json.dumps(result, ensure_ascii=False), result
    except Exception:
        pass

    log.error("JSON解析失败，原始内容为：%s", content)
    return "", {}