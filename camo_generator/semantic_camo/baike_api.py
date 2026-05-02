from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

_this_file = Path(__file__).resolve()
_baidu_fixed = _this_file.parent / "baidu_evidence_system_fixed"
_baidu_fixed_str = str(_baidu_fixed.resolve())
if not _baidu_fixed.exists():
    raise RuntimeError(
        f"未找到 baidu_evidence_system_fixed，请确保目录存在: {_baidu_fixed}. "
        "应在 semantic_camo 目录下。"
    )
if _baidu_fixed_str not in sys.path:
    sys.path.insert(0, _baidu_fixed_str)

from baidu_evidence_system.evidence_service import gather_evidence as _gather_evidence  # noqa: E402


async def gather_evidence(
    term: str,
    max_relation_hops: int = 2,
    *,
    scene: str | None = None,
    max_total_evidence: int = 25,
    max_card_items: int = 10,
    max_relation_items: int = 8,
    max_web_results: int = 5,
    include_alias_pool: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:

    result = await _gather_evidence(
        term,
        scene=scene,
        max_total_evidence=max_total_evidence,
        max_card_items=max_card_items,
        max_relation_items=max_relation_items,
        max_relation_hops=max_relation_hops,
        max_web_results=max_web_results,
        include_alias_pool=include_alias_pool,
    )
    if result is None:
        result = {}
    result.setdefault("evidence_lines", [])
    result.setdefault("alias_pool", [])
    return result
