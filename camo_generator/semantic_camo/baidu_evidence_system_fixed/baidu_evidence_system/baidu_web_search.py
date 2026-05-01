from __future__ import annotations

import json

import os

import re

from typing import Any, Dict, List

from urllib.parse import urljoin

from bs4 import BeautifulSoup

from uuid import uuid4

from .config import AI_SEARCH_TIMEOUT, APPBUILDER_API_KEY, AI_SEARCH_API_URL, BAIDU_SEARCH_URL, MAX_TEXT_LEN

from .http_utils import fetch_text, fetch_json

from .utils import dedup_keep_order, safe_truncate_text, strip_html_tags

BLACKWORD_PATTERNS = [

    r'(?:又称|别称|俗称|黑话|行话|圈内称|业内称|简称|别名)[:：]?([^；。\n]{1,60})',

    r'([^；。\n]{1,60})(?:是|为)?(?:一种)?(?:黑话|行话|圈内说法)',

]

def _is_normal_scene(scene: str | None) -> bool:

    """
    判断场景是否为「正常无害」语境。
    目前场景类别统一为：涉黄、涉赌、黑灰产、诈骗、辱骂、正常无害。
    只有标记为「正常无害」的场景，才视作正常语境。
    """

    if not scene:

        return False

    s = scene.strip()

    if not s:

        return False

    return '正常无害' in s

def _extract_blackword_candidates(text: str, query: str) -> List[str]:

    out: List[str] = []

    s = strip_html_tags(text)

    for pat in BLACKWORD_PATTERNS:

        for m in re.finditer(pat, s):

            chunk = m.group(1).strip(' ：:，、"“”')

            if not chunk:

                continue

            parts = re.split(r'[、/，,；;]|\s{2,}', chunk)

            for p in parts:

                p = p.strip()

                if 1 < len(p) <= 30 and p != query:

                    out.append(p)

    return dedup_keep_order(out)

def parse_baidu_search_results(html: str, query: str, max_results: int = 8) -> List[Dict[str, Any]]:

    soup = BeautifulSoup(html, 'html.parser')

    items: List[Dict[str, Any]] = []

    selectors = [

        'div.result',

        'div.result-op',

        'div.c-container',

    ]

    seen = set()

    for selector in selectors:

        for block in soup.select(selector):

            key = str(block)[:120]

            if key in seen:

                continue

            seen.add(key)

            title_node = block.select_one('h3') or block.select_one('.c-title')

            title = strip_html_tags(title_node.get_text(' ', strip=True) if title_node else '')

            a = None

            if title_node:

                a = title_node.find('a')

            if a is None:

                a = block.find('a', href=True)

            href = a.get('href', '').strip() if a else ''

            href = urljoin('https://www.baidu.com', href) if href.startswith('/') else href

            snippet_node = (

                block.select_one('.c-abstract')

                or block.select_one('.content-right_8Zs40')

                or block.select_one('.c-span-last')

                or block.select_one('.c-color-text')

            )

            snippet = strip_html_tags(snippet_node.get_text(' ', strip=True) if snippet_node else block.get_text(' ', strip=True))

            snippet = re.sub(r'\s+', ' ', snippet).strip()

            snippet = safe_truncate_text(snippet, MAX_TEXT_LEN)

            if not title and not snippet:

                continue

            aliases = _extract_blackword_candidates(f'{title} {snippet}', query)

            items.append(

                {

                    'title': title,

                    'url': href,

                    'snippet': snippet,

                    'aliases': aliases,

                }

            )

            if len(items) >= max_results:

                return items

    return items

async def search_baidu_web(

    client, query: str, *, max_results: int = 8, scene: str | None = None

) -> Dict[str, Any]:

    if not APPBUILDER_API_KEY:

        return {

            'source': 'baidu_web',

            'evidence': ['请设置环境变量 APPBUILDER_API_KEY 后重试。'],

            'alias_pool': [],

            'raw': [],

        }

    bearer = f'Bearer {APPBUILDER_API_KEY}'

    headers = {

        'X-Appbuilder-Authorization': bearer,

        'Authorization': bearer,

        'Content-Type': 'application/json',

        'X-Appbuilder-Request-Id': str(uuid4()),

    }

    scene_has_value = bool(scene and scene.strip())

    scene_prefix = f'在“{scene.strip()}”场景下，' if scene_has_value else ''

    if scene_has_value and _is_normal_scene(scene):

        user_content = (

            f'{scene_prefix}请搜索“{query}”的字面意思并简要说明其含义或用法。'

        )

    else:

        user_content = (

            f'{scene_prefix}请搜索与“{query}”相关的黑话、别称、俗称或行内说法，'

            f'并简要说明含义或用法。以及简要说明“{query}”是什么意思。'

        )

    payload: Dict[str, Any] = {

        'model': 'ernie-3.5-8k',

        'messages': [

            {

                'role': 'user',

                'content': user_content,

            }

        ],

        'search_source': 'baidu_search_v2',

        'resource_type_filter': [{'type': 'web', 'top_k': max_results}],

    }

    try:

        data = await fetch_json(

            client,

            AI_SEARCH_API_URL,

            headers=headers,

            method='POST',

            json_body=payload,

            timeout=AI_SEARCH_TIMEOUT,

        )

    except Exception as e:

        return {

            'source': 'baidu_web',

            'evidence': [f'[AI网页搜索] 请求失败: {type(e).__name__}: {e}'],

            'alias_pool': [],

            'raw': [],

        }

    if os.getenv('DEBUG_AI_SEARCH', '').strip().lower() in ('1', 'true', 'yes'):

        try:

            with open('ai_search_response.json', 'w', encoding='utf-8') as f:

                json.dump(data, f, ensure_ascii=False, indent=2)

        except Exception:

            pass

    api_code = data.get('code')

    if api_code is not None and api_code not in (0, '0', None):

        msg = data.get('message', '') or str(api_code)

        return {

            'source': 'baidu_web',

            'evidence': [f'[AI网页搜索] 接口返回错误: code={api_code}, message={msg}'],

            'alias_pool': [],

            'raw': [],

        }

    is_safe = data.get('is_safe', True)

    if is_safe is False:

        fallback_evidence: List[str] = []

        fallback_aliases: List[str] = []

        fallback_raw: List[Dict[str, Any]] = []

        try:

            search_wd = f'{query} {scene}'.strip() if scene else query

            html = await fetch_text(client, BAIDU_SEARCH_URL, params={'wd': search_wd})

            items = parse_baidu_search_results(html, query=query, max_results=max_results)

            fallback_raw = items

            for item in items:

                title = item.get('title', '') or ''

                url = item.get('url', '') or ''

                snippet = item.get('snippet', '') or ''

                if title or snippet:

                    fallback_evidence.append(f'[百度网页] 标题={title}；摘要={snippet}')

                if url:

                    fallback_evidence.append(f'[百度网页] 链接={url}')

                fallback_aliases.extend(item.get('aliases', []))

            if fallback_evidence:

                fallback_evidence.insert(0, '[AI网页搜索] 因敏感内容未返回，已降级为网页抓取。')

        except Exception:

            pass

        if not fallback_evidence:

            fallback_evidence.append(

                '[AI网页搜索] 接口判定查询涉及敏感内容，未返回结果；网页抓取也未得到结果。'

            )

        return {

            'source': 'baidu_web',

            'evidence': dedup_keep_order(fallback_evidence),

            'alias_pool': dedup_keep_order(fallback_aliases),

            'raw': fallback_raw,

        }

    refs = data.get('references') or []

    if not isinstance(refs, list):

        refs = []

    evidence: List[str] = []

    alias_pool: List[str] = []

    def _get_summary_content(d: Dict[str, Any]) -> str:

        try:

            for choices in (

                d.get('choices'),

                (d.get('result') or {}).get('choices') if isinstance(d.get('result'), dict) else None,

            ):

                if not choices or not isinstance(choices, list):

                    continue

                first = choices[0] if isinstance(choices[0], dict) else {}

                msg = first.get('message') or first.get('delta') or first

                if not isinstance(msg, dict):

                    continue

                content = (msg.get('content') or msg.get('text') or '').strip()

                if content:

                    return content

            return ''

        except Exception:

            return ''

    if not refs:

        summary = _get_summary_content(data)

        if summary:

            evidence.append(f'[AI网页搜索] 总结：{safe_truncate_text(summary, 500)}')

            alias_pool.extend(_extract_blackword_candidates(summary, query))

        if not evidence:

            evidence.append(

                '[AI网页搜索] 本次未返回网页引用与总结。'

                ' 可设置 DEBUG_AI_SEARCH=1 后重跑，会在当前目录生成 ai_search_response.json 便于排查。'

            )

    for ref in refs[:max_results]:

        if not isinstance(ref, dict):

            continue

        title = str(ref.get('title') or '').strip()

        url = str(ref.get('url') or '').strip()

        content = str(ref.get('content') or '').strip()

        snippet = safe_truncate_text(strip_html_tags(content), MAX_TEXT_LEN)

        if title or snippet:

            evidence.append(f'[百度网页] 标题={title}；摘要={snippet}')

        if url:

            evidence.append(f'[百度网页] 链接={url}')

        aliases = _extract_blackword_candidates(f'{title} {snippet}', query)

        alias_pool.extend(aliases)

    return {

        'source': 'baidu_web',

        'evidence': dedup_keep_order(evidence),

        'alias_pool': dedup_keep_order(alias_pool),

        'raw': refs[:max_results],

    }
