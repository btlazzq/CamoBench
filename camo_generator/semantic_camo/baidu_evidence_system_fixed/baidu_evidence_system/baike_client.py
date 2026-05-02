from __future__ import annotations

import re
from typing import Any, Dict, List
from uuid import uuid4
from urllib.parse import quote

from bs4 import BeautifulSoup

from .config import BAIKE_API_KEY, BAIKE_API_URL
from .http_utils import fetch_json, fetch_text
from .utils import clean_card_value, dedup_keep_order, safe_truncate_text, strip_html_tags


async def baike_get_lemma_by_title(client, title: str) -> Dict[str, Any]:
    """
    第一路资源：百科。直接抓取百度百科网页并解析（词条标题、描述、摘要等），不依赖 API Key。
    """
    try:
        url = f'https://baike.baidu.com/item/{quote(title)}'
        html = await fetch_text(client, url)
    except Exception:
        return {}

    soup = BeautifulSoup(html, 'html.parser')

    # 词条标题
    h1 = soup.find('h1')
    lemma_title = (h1.get_text(strip=True) if h1 else title).strip()

    # 优先用 meta description 作为简要描述
    lemma_desc = ''
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    if meta_desc and meta_desc.get('content'):
        lemma_desc = meta_desc['content'].strip()

    # 摘要 / 概述，尝试抓取常见的摘要容器
    summary = ''
    summary_div = soup.find('div', class_='lemma-summary') or soup.find('div', class_='lemmaWgt-lemmaSummary')
    if summary_div:
        summary = summary_div.get_text(' ', strip=True)

    result: Dict[str, Any] = {
        'lemma_title': lemma_title,
        'lemma_desc': lemma_desc,
        'summary': summary,
        'url': url,
        # 兼容后续处理逻辑，先留空结构
        'card': [],
        'relations': [],
        'classify': [],
        'abstract_plain': '',
        'abstract_html': '',
    }
    return result


async def baike_get_lemma_by_id(client, lemma_id: Any) -> Dict[str, Any]:
    if not lemma_id:
        return {}
    if not BAIKE_API_KEY:
        raise RuntimeError('缺少 APPBUILDER_API_KEY。请先设置环境变量。')

    headers = {
        'Authorization': f'Bearer {BAIKE_API_KEY}',
        'Content-Type': 'application/json',
        'X-Appbuilder-Request-Id': str(uuid4()),
    }
    params = {'search_type': 'lemmaId', 'search_key': str(lemma_id)}
    data = await fetch_json(client, BAIKE_API_URL, params=params, headers=headers)

    code = data.get('code')
    if code not in (0, '0', None):
        return {}

    result = data.get('result') or {}
    return result if isinstance(result, dict) else {}



def extract_baike_aliases(result: Dict[str, Any]) -> List[str]:
    alias_pool: List[str] = []
    title = str(result.get('lemma_title') or '').strip()
    if title:
        alias_pool.append(title)

    desc = str(result.get('lemma_desc') or '').strip()
    if desc:
        patterns = [
            r'(?:又称|别名|简称|俗称|亦称|也称)[:：]?([^；。\n]+)',
            r'([^；。\n]+)(?:，|、)?又称',
        ]
        for pat in patterns:
            for m in re.finditer(pat, desc):
                chunk = m.group(1).strip(' ：:，、')
                if chunk:
                    alias_pool.extend(re.split(r'[、/，,；;]', chunk))

    cards = result.get('card') or []
    for item in cards:
        if not isinstance(item, dict):
            continue
        name = str(item.get('name') or item.get('key') or '').strip()
        value = clean_card_value(item.get('value'))
        if not name or not value:
            continue
        if any(k in name for k in ['别名', '又名', '英文名', '外文名', '简称', '俗称', '其他名称']):
            alias_pool.extend(re.split(r'[、/，,；;（）()\s]+', value))

    relations = result.get('relations') or []
    for rel in relations:
        if not isinstance(rel, dict):
            continue
        rel_name = str(rel.get('relation_name') or '').strip()
        rel_title = str(rel.get('lemma_title') or '').strip()
        if rel_name in {'别名', '又名', '简称', '俗称', '英文名', '外文名'} and rel_title:
            alias_pool.append(rel_title)

    alias_pool = [x.strip() for x in alias_pool if x and len(x.strip()) <= 40]
    alias_pool = [x for x in alias_pool if x not in {title, desc}]
    return dedup_keep_order(alias_pool)



def extract_baike_card_evidence(result: Dict[str, Any], max_items: int = 10) -> List[str]:
    lines: List[str] = []
    cards = result.get('card') or []
    for item in cards:
        if not isinstance(item, dict):
            continue
        name = str(item.get('name') or item.get('key') or '').strip()
        value = clean_card_value(item.get('value'))
        if not name or not value:
            continue
        lines.append(f'[百度百科卡片] {name}={safe_truncate_text(value, 120)}')
        if len(lines) >= max_items:
            break
    return lines



def extract_baike_relations_evidence(result: Dict[str, Any], max_items: int = 8) -> List[str]:
    lines: List[str] = []
    relations = result.get('relations') or []
    for rel in relations:
        if not isinstance(rel, dict):
            continue
        rel_name = str(rel.get('relation_name') or '').strip()
        rel_title = str(rel.get('lemma_title') or '').strip()
        if rel_name and rel_title:
            lines.append(f'[百度百科关系] {rel_name}: {rel_title}')
        if len(lines) >= max_items:
            break
    return lines


async def gather_baike_evidence(
    client,
    term: str,
    *,
    max_card_items: int = 10,
    max_relation_items: int = 8,
    max_relation_hops: int = 2,
) -> Dict[str, Any]:
    evidence: List[str] = []
    alias_pool: List[str] = []

    main = await baike_get_lemma_by_title(client, term)
    if not main:
        return {
            'source': 'baike',
            'evidence': ['[百度百科]未检索到词条内容'],
            'alias_pool': [],
            'raw': {},
        }

    lemma_title = str(main.get('lemma_title') or term).strip()
    lemma_desc = str(main.get('lemma_desc') or '').strip()
    lemma_url = str(main.get('url') or '').strip()
    summary = str(main.get('summary') or main.get('abstract_plain') or '').strip()
    abstract_plain = str(main.get('abstract_plain') or '').strip()
    abstract_html = str(main.get('abstract_html') or '').strip()
    classify = main.get('classify') or []

    if lemma_title or lemma_desc:
        evidence.append(f'[百度百科]词条={lemma_title}；义项描述={lemma_desc}')
    if lemma_url:
        evidence.append(f'[百度百科]词条链接={lemma_url}')
    if summary:
        evidence.append(f'[百度百科]摘要：{safe_truncate_text(summary)}')
    elif abstract_plain:
        evidence.append(f'[百度百科]概述：{safe_truncate_text(abstract_plain)}')
    elif abstract_html:
        evidence.append(f'[百度百科]概述：{safe_truncate_text(strip_html_tags(abstract_html))}')
    if classify:
        evidence.append(f"[百度百科]分类：{'，'.join([str(x) for x in classify if x])}")

    evidence.extend(extract_baike_card_evidence(main, max_items=max_card_items))
    evidence.extend(extract_baike_relations_evidence(main, max_items=max_relation_items))
    alias_pool.extend(extract_baike_aliases(main))

    relation_items = main.get('relations') or []
    hop_cnt = 0
    for rel in relation_items:
        if hop_cnt >= max_relation_hops:
            break
        if not isinstance(rel, dict):
            continue
        rel_name = str(rel.get('relation_name') or '').strip()
        rel_id = rel.get('lemma_id')
        rel_title = str(rel.get('lemma_title') or '').strip()
        if not rel_id or not rel_title:
            continue
        sub = await baike_get_lemma_by_id(client, rel_id)
        if not sub:
            continue
        sub_desc = str(sub.get('lemma_desc') or '').strip()
        sub_summary = str(sub.get('summary') or sub.get('abstract_plain') or '').strip()
        brief = sub_desc or safe_truncate_text(sub_summary, 100)
        if brief:
            evidence.append(f'[百度百科关系扩展] {rel_name}->{rel_title}: {brief}')
        hop_cnt += 1

    return {
        'source': 'baike',
        'evidence': dedup_keep_order(evidence),
        'alias_pool': dedup_keep_order([a for a in alias_pool if a and a.strip() and a.strip() != term]),
        'raw': main,
    }
