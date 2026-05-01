import json

import re

import ast

import logging

from typing import Tuple, Dict, Any, List

from json_repair import repair_json

log = logging.getLogger(__name__)

JSON_BLOCK_PATTERN = re.compile(

    r"```(?:json)?\s*(.*?)```",

    re.DOTALL | re.IGNORECASE

)

JSON_OBJECT_PATTERN = re.compile(

    r"\{.*?\}",

    re.DOTALL

)

JSON_LIST_PATTERN = re.compile(

    r"\[.*?\]",

    re.DOTALL

)

def core_parse(content: str):

    if not content:

        return None

    raw = content.strip()

    block_match = JSON_BLOCK_PATTERN.search(raw)

    if block_match:

        raw = block_match.group(1).strip()

    try:

        return json.loads(raw)

    except Exception:

        pass

    cleaned = (

        raw.replace("\n", " ")

           .replace("\r", "")

           .replace("\t", " ")

           .replace("\\n", " ")

           .replace("\\", "")

           .strip()

    )

    clean = re.sub(r"'", '"', cleaned)

    try:

        return json.loads(clean)

    except Exception:

        pass

    try:

        repaired = repair_json(clean)

        if isinstance(repaired, str):

            return json.loads(repaired)

        return repaired

    except Exception:

        pass

    try:

        tree = ast.parse(clean)

        result = {}

        for node in ast.walk(tree):

            if isinstance(node, ast.Call):

                for kw in node.keywords:

                    result[kw.arg] = ast.literal_eval(kw.value)

        if result:

            return result

    except Exception:

        pass

    log.error("JSON解析失败，原始内容为：%s", content)

    return None

def parse_single_json(content: str) -> Tuple[str, Dict[str, Any]]:

    parsed = core_parse(content)

    if isinstance(parsed, dict):

        return json.dumps(parsed, ensure_ascii=False), parsed

    if isinstance(parsed, list) and parsed:

        if isinstance(parsed[0], dict):

            return json.dumps(parsed[0], ensure_ascii=False), parsed[0]

    return "", {}

def parse_list_json(content: str) -> Tuple[str, List[Dict[str, Any]]]:

    raw = content.strip()

    results = []

    parsed = core_parse(raw)

    if isinstance(parsed, list):

        results.extend(parsed)

    elif isinstance(parsed, dict):

        results.append(parsed)

    else:

        matches = JSON_OBJECT_PATTERN.findall(raw)

        for m in matches:

            obj = core_parse(m)

            if isinstance(obj, dict):

                results.append(obj)

    return json.dumps(results, ensure_ascii=False), results

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
