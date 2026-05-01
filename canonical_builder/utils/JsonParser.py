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

        return clean, json.loads(clean)

    except Exception:

        pass

    try:

        repaired = repair_json(clean)

        if isinstance(repaired, str):

            repaired_dict = json.loads(repaired)

        else:

            repaired_dict = repaired

        return json.dumps(repaired_dict, ensure_ascii=False), repaired_dict

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

            return json.dumps(result, ensure_ascii=False), result

    except Exception:

        pass

    log.error("JSON解析失败，原始内容为：%s", content)

    return "", {}
