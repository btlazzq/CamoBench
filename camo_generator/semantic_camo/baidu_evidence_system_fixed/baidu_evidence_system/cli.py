from __future__ import annotations

import argparse

import asyncio

import json

from pathlib import Path

from typing import Any, Dict

from .evidence_service import gather_evidence

async def _run(args) -> Dict[str, Any]:

    return await gather_evidence(

        term=args.term,

        scene=args.scene or None,

        max_total_evidence=args.max_total_evidence,

        max_card_items=args.max_card_items,

        max_relation_items=args.max_relation_items,

        max_relation_hops=args.max_relation_hops,

        max_web_results=args.max_web_results,

        include_alias_pool=not args.no_alias_pool,

    )

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(description='百度百科 + 百度网页联合证据检索工具')

    parser.add_argument('--term', required=True, help='要检索的词')

    parser.add_argument('--scene', default='', help='场景（如：赌博、毒品），用于广搜时限定语境，可选')

    parser.add_argument('--max-total-evidence', type=int, default=25, help='最终最多保留多少条证据')

    parser.add_argument('--max-card-items', type=int, default=10, help='百科卡片最多保留多少条')

    parser.add_argument('--max-relation-items', type=int, default=8, help='百科关系最多保留多少条')

    parser.add_argument('--max-relation-hops', type=int, default=2, help='百科关系扩展最多跳多少个')

    parser.add_argument('--max-web-results', type=int, default=5, help='百度网页搜索最多解析多少个结果')

    parser.add_argument('--no-alias-pool', action='store_true', help='不把 alias pool 追加到最后一条证据')

    parser.add_argument('--output', default='', help='输出 json 文件路径，可选')

    return parser

def main() -> None:

    parser = build_parser()

    args = parser.parse_args()

    result = asyncio.run(_run(args))

    text = json.dumps(result, ensure_ascii=False, indent=2)

    print(text)

    if args.output:

        out_path = Path(args.output)

        out_path.parent.mkdir(parents=True, exist_ok=True)

        out_path.write_text(text, encoding='utf-8')

if __name__ == '__main__':

    main()
