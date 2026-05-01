from __future__ import annotations

import html

import re

from typing import Iterable, List, Optional

def dedup_keep_order(items: Iterable[str]) -> List[str]:

    seen = set()

    out: List[str] = []

    for item in items:

        if item is None:

            continue

        s = str(item).strip()

        if not s or s in seen:

            continue

        seen.add(s)

        out.append(s)

    return out

def safe_truncate_text(text: Optional[str], max_len: int = 180) -> str:

    s = (text or '').strip()

    if len(s) <= max_len:

        return s

    return s[: max_len - 1].rstrip() + '…'

def strip_html_tags(text: Optional[str]) -> str:

    s = text or ''

    s = re.sub(r'<script[\s\S]*?</script>', ' ', s, flags=re.I)

    s = re.sub(r'<style[\s\S]*?</style>', ' ', s, flags=re.I)

    s = re.sub(r'<[^>]+>', ' ', s)

    s = html.unescape(s)

    s = re.sub(r'\s+', ' ', s).strip()

    return s

def clean_card_value(value) -> str:

    if value is None:

        return ''

    if isinstance(value, list):

        parts = [clean_card_value(v) for v in value]

        return '；'.join([p for p in parts if p])

    if isinstance(value, dict):

        parts = []

        for k, v in value.items():

            cv = clean_card_value(v)

            if cv:

                parts.append(f'{k}:{cv}')

        return '；'.join(parts)

    s = strip_html_tags(str(value))

    s = re.sub(r'\s+', ' ', s).strip(' ；;，,')

    return s

def looks_like_url(text: str) -> bool:

    return bool(re.match(r'^https?://', (text or '').strip(), flags=re.I))
