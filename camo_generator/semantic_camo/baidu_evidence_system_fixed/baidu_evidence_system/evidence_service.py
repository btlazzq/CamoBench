from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from .baidu_web_search import search_baidu_web
from .baike_client import gather_baike_evidence
from .http_utils import build_async_client
from .utils import dedup_keep_order


async def gather_evidence(
    term: str,
    *,
    scene: str | None = None,
    max_total_evidence: int = 25,
    max_card_items: int = 10,
    max_relation_items: int = 8,
    max_relation_hops: int = 2,
    max_web_results: int = 5,
    include_alias_pool: bool = True,
) -> Dict[str, Any]:
    async with build_async_client() as client:
        baike_task = gather_baike_evidence(
            client,
            term,
            max_card_items=max_card_items,
            max_relation_items=max_relation_items,
            max_relation_hops=max_relation_hops,
        )
        web_task = search_baidu_web(client, term, max_results=max_web_results, scene=scene)
        baike_res, web_res = await asyncio.gather(baike_task, web_task, return_exceptions=True)

    evidence: List[str] = []
    alias_pool: List[str] = []
    errors: List[str] = []
    sources: Dict[str, Any] = {}

    SOURCE_LABELS = {'baike': '百科', 'baidu_web': 'AI网页搜索'}
    for name, res in [('baike', baike_res), ('baidu_web', web_res)]:
        if isinstance(res, Exception):
            errors.append(f'{name}: {type(res).__name__}: {res}')
            continue
        sources[name] = {**(res or {}), 'source_label': SOURCE_LABELS.get(name, name)}
        evidence.extend((res or {}).get('evidence', []))
        alias_pool.extend((res or {}).get('alias_pool', []))

    evidence = dedup_keep_order(evidence)
    alias_pool = dedup_keep_order([x for x in alias_pool if x and x.strip() and x.strip() != term])

    if include_alias_pool and alias_pool:
        evidence.append('[AliasPool]公开别名池：' + '，'.join(alias_pool[:50]))

    evidence = dedup_keep_order(evidence)[:max_total_evidence]
    evidence_lines = [f'证据#{i + 1}: {e}' for i, e in enumerate(evidence)]

    return {
        'term': term,
        'scene': scene,
        'evidence_lines': evidence_lines,
        'alias_pool': alias_pool,
        'errors': errors,
        'sources': sources,
        'stats': {
            'max_total_evidence': max_total_evidence,
            'actual_evidence_count': len(evidence_lines),
            'alias_count': len(alias_pool),
        },
    }
