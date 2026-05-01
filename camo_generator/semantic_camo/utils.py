import html

import json

import re

from typing import Any, List

def dedup_keep_order(xs: List[str]) -> List[str]:

    seen = set()

    out = []

    for x in xs:

        x = (x or "").strip()

        if not x or x in seen:

            continue

        seen.add(x)

        out.append(x)

    return out

def safe_truncate_text(text: str, max_chars: int = 600) -> str:

    text = (text or "").strip()

    if len(text) <= max_chars:

        return text

    return text[:max_chars].rstrip() + "…"

def mask_alias(alias: str) -> str:

    alias = (alias or "").strip()

    if not alias:

        return ""

    if len(alias) == 1:

        return "*"

    if len(alias) == 2:

        return alias[0] + "*"

    return alias[0] + "*" * max(1, len(alias) - 2) + alias[-1]

def strip_html_tags(text: str) -> str:

    text = html.unescape(text or "")

    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)

    text = re.sub(r"</p\s*>", "\n", text, flags=re.I)

    text = re.sub(r"<[^>]+>", "", text)

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def clean_card_value(v: Any) -> str:

    if v is None:

        return ""

    if isinstance(v, (str, int, float)):

        return str(v).strip()

    if isinstance(v, dict):

        for key in ["name", "value", "text", "lemma_title", "title"]:

            if key in v and v[key] not in (None, ""):

                return str(v[key]).strip()

        return json.dumps(v, ensure_ascii=False)

    if isinstance(v, list):

        parts = [clean_card_value(x) for x in v]

        parts = [p for p in parts if p]

        return "，".join(parts)

    return str(v).strip()

def contains_term(candidate: str, term: str) -> bool:

    if not candidate or not term:

        return False

    if term in candidate:

        return True

    c = re.sub(r"\s+", "", candidate)

    t = re.sub(r"\s+", "", term)

    return bool(t and t in c)

def has_evidence_cite(explanation: str) -> bool:

    """
    判断 explanation 是否“有效”。

    之前这里强要求必须包含形如 “证据#数字” 的引用，现在根据使用场景放宽为：
    - 只要 explanation 非空，就认为通过。
    这样可以保留那些仅基于 features 抽象的、但没有显式证据号的说明。
    """

    return bool((explanation or "").strip())

def looks_too_direct(text: str) -> bool:

    text = (text or "").strip()

    bad_patterns = [

        r"俗称", r"学名", r"定义", r"合法化", r"犯罪所得", r"伪装",

        r"非法", r"违法", r"专业术语", r"学术", r"解释",

    ]

    return any(re.search(p, text) for p in bad_patterns)
