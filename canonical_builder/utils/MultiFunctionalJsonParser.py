import json
import re
import ast
import logging
from typing import Tuple, Dict, Any, List
from json_repair import repair_json

# 创建日志对象
log = logging.getLogger(__name__)

# 匹配```json ```或``` ```包裹的内容
JSON_BLOCK_PATTERN = re.compile(
    r"```(?:json)?\s*(.*?)```", # 匹配代码块内容
    re.DOTALL | re.IGNORECASE # 支持跨行和忽略大小写
)

# 匹配第一个{...}结构
JSON_OBJECT_PATTERN = re.compile(
    r"\{.*?\}", # 提取JSON对象主体
    re.DOTALL # 支持跨行
)

# 匹配JSON列表
JSON_LIST_PATTERN = re.compile(
    r"\[.*?\]", # 提取List对象主体
    re.DOTALL # 支持跨行
)

# 核心解析函数（返回dict 或 list）
def core_parse(content: str):
    if not content:
        return None
    # 去除前后空白字符
    raw = content.strip()

    # 提取markdown代码块
    block_match = JSON_BLOCK_PATTERN.search(raw) # 查找代码块
    if block_match:
        raw = block_match.group(1).strip() # 获取代码块内部内容

    # JSON内容解析（直接尝试解析）【1】
    try:
        return json.loads(raw)
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

    # 将单引号替换为双引号
    clean = re.sub(r"'", '"', cleaned)

    # JSON内容解析（初步处理后尝试解析）【2】
    try:
        return json.loads(clean)
    except Exception:
        pass


    # JSON内容解析（使用json_repair修复）【3】
    try:
        repaired = repair_json(clean) # 自动修复JSON

        if isinstance(repaired, str):
            return json.loads(repaired) # 解析为字典

        return repaired
    except Exception:
        pass


    # JSON内容解析（使用AST解析函数修复）【4】
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
            return result
    except Exception:
        pass

    log.error("JSON解析失败，原始内容为：%s", content)
    return None

# 返回单个JSON对象
def parse_single_json(content: str) -> Tuple[str, Dict[str, Any]]:
    parsed = core_parse(content)

    if isinstance(parsed, dict):
        return json.dumps(parsed, ensure_ascii=False), parsed

    if isinstance(parsed, list) and parsed:
        if isinstance(parsed[0], dict):
            return json.dumps(parsed[0], ensure_ascii=False), parsed[0]

    return "", {}

# 返回列表级JSON
def parse_list_json(content: str) -> Tuple[str, List[Dict[str, Any]]]:
    raw = content.strip()
    results = []

    parsed = core_parse(raw)
    if isinstance(parsed, list):
        results.extend(parsed)
    elif isinstance(parsed, dict):
        results.append(parsed)
    else:
        # 查找所有的JSON字典
        matches = JSON_OBJECT_PATTERN.findall(raw)
        for m in matches:
            obj = core_parse(m)
            if isinstance(obj, dict):
                results.append(obj)

    return json.dumps(results, ensure_ascii=False), results

# 列表去重
def parse_list_deduplicate(content: str) -> Tuple[str, List[Dict[str, Any]]]:
    parsed = core_parse(content)
    if not isinstance(parsed, list):
        return "", []

    result = set()
    unique_list = []
    for item in parsed:
        item_str = json.dumps(item, sort_keys=True, ensure_ascii=False)
        if item_str not in result:
            result.add(item_str)
            unique_list.append(item)

    return json.dumps(unique_list, ensure_ascii=False), unique_list