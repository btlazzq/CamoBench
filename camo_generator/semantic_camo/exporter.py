import json

import os

import re

from datetime import datetime

from typing import Any, Dict, Optional

import pandas as pd

def _safe_filename(term: str, max_len: int = 120) -> str:

    """将 term 转为可作文件名的字符串，避免 / \\ : * ? \" < > | 等。"""

    s = re.sub(r'[/\\:*?"<>|\n\r\t]+', "_", str(term).strip())

    s = s.strip("._ ") or "term"

    return s[:max_len] if len(s) > max_len else s

def _out_dir_resolved(out_dir: Optional[str] = None) -> str:

    """默认输出到 semantic_camo/output_baike。"""

    base_dir = os.path.dirname(os.path.abspath(__file__))

    return out_dir or os.path.join(base_dir, "output_baike")

def _evidence_features_subdir(out_dir: Optional[str] = None) -> str:

    """长期存档：按词 JSON 目录。"""

    subdir = os.path.join(_out_dir_resolved(out_dir), "evidence_features")

    os.makedirs(subdir, exist_ok=True)

    return subdir

def _rag_subdir(out_dir: Optional[str] = None) -> str:

    """RAG 专用目录，与长期存档分开。"""

    subdir = os.path.join(_out_dir_resolved(out_dir), "rag")

    os.makedirs(subdir, exist_ok=True)

    return subdir

def save_evidence_features_term(

    term: str,

    pack: Dict[str, Any],

    out_dir: Optional[str] = None,

    *,

    rag_path: Optional[str] = None,

) -> str:

    """
    单个 term 处理完就写入：证据+特征 JSON（长期存档），并可选追加到本轮的 RAG 用 JSONL（rag_path 按时间戳命名，永久保留）。
    返回: 写入的 JSON 文件路径。
    """

    subdir = _evidence_features_subdir(out_dir)

    created_at = datetime.now().isoformat()

    evidence = pack.get("evidence", []) or []

    features = pack.get("features", {}) or {}

    alias_pool = pack.get("alias_pool", []) or []

    evidence_sources = pack.get("evidence_sources")

    doc = {

        "term": term,

        "created_at": created_at,

        "evidence_lines": evidence,

        "alias_pool": alias_pool,

        "features": features,

    }

    if evidence_sources is not None:

        doc["evidence_sources"] = evidence_sources

    safe = _safe_filename(term)

    json_path = os.path.join(subdir, f"{safe}.json")

    with open(json_path, "w", encoding="utf-8") as f:

        json.dump(doc, f, ensure_ascii=False, indent=2, default=lambda x: str(x))

    if rag_path:

        _dump = lambda obj: json.dumps(obj, ensure_ascii=False, default=lambda x: str(x))

        with open(rag_path, "a", encoding="utf-8") as f:

            for i, line in enumerate(evidence, 1):

                f.write(_dump({"term": term, "type": "evidence", "index": i, "text": line}) + "\n")

            for slot, values in (features or {}).items():

                for v in (values if isinstance(values, list) else [values]):

                    text = f"{slot}: {v}" if isinstance(v, str) else f"{slot}: {_dump(v)}"

                    f.write(_dump({"term": term, "type": "feature", "slot": slot, "value": v, "text": text}) + "\n")

    return json_path

def export_results(results: dict, out_dir: Optional[str] = None):

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_dir = os.path.dirname(os.path.abspath(__file__))

    out_dir = out_dir or os.path.join(base_dir, "output_baike")

    os.makedirs(out_dir, exist_ok=True)

    jsonl_path = os.path.join(out_dir, f"results_{ts}.jsonl")

    xlsx_path = os.path.join(out_dir, f"results_{ts}.xlsx")

    rows = []

    with open(jsonl_path, "w", encoding="utf-8") as f:

        for term, pack in results.items():

            evidence = pack.get("evidence", []) or []

            items = pack.get("items", []) or []

            features = pack.get("features", {}) or {}

            if not items:

                record = {

                    "term": term,

                    "mechanism": pack.get("mechanism", "safe_semantic_abstraction_baike"),

                    "candidate": "",

                    "mode": "",

                    "source_type": "",

                    "explanation": "",

                    "natural_score": None,

                    "semantic_score": None,

                    "abstraction_score": None,

                    "exposure_score": None,

                    "final_score": None,

                    "features": json.dumps(features, ensure_ascii=False),

                    "evidence": " | ".join(evidence),

                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                rows.append(record)

                continue

            for it in items:

                record = {

                    "term": term,

                    "mechanism": pack.get("mechanism", "safe_semantic_abstraction_baike"),

                    "candidate": it.get("candidate", ""),

                    "mode": it.get("mode", ""),

                    "source_type": it.get("source_type", ""),

                    "explanation": it.get("explanation", ""),

                    "natural_score": it.get("natural_score", None),

                    "semantic_score": it.get("semantic_score", None),

                    "abstraction_score": it.get("abstraction_score", None),

                    "exposure_score": it.get("exposure_score", None),

                    "final_score": it.get("final_score", None),

                    "features": json.dumps(features, ensure_ascii=False),

                    "evidence": " | ".join(evidence),

                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                rows.append(record)

    pd.DataFrame(rows).to_excel(xlsx_path, index=False)

    return jsonl_path, xlsx_path

def ensure_evidence_features_ready(out_dir: Optional[str] = None) -> tuple:

    """
    创建长期存档目录 evidence_features，以及单独的 RAG 目录 rag，生成本轮 RAG 用 JSONL 路径（按时间戳）。
    返回: (evidence_features_dir, rag_chunks_path)，其中 rag_chunks_path 在 rag/ 下，与长期存档分离。
    """

    evidence_dir = _evidence_features_subdir(out_dir)

    rag_dir = _rag_subdir(out_dir)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    rag_path = os.path.join(rag_dir, f"rag_chunks_{ts}.jsonl")

    return evidence_dir, rag_path

def export_evidence_features(results: Dict[str, Any], out_dir: Optional[str] = None) -> str:

    """
    将「百科+网页整合证据」与「SemanticFeatureExtractor 输出的 features」按 term 存为 JSON。
    每个 term 一个文件，便于按词查看、复跑和版本管理。
    目录: out_dir/evidence_features/{term_safe}.json
    """

    subdir = _evidence_features_subdir(out_dir)

    created_at = datetime.now().isoformat()

    for term, pack in results.items():

        evidence = pack.get("evidence", []) or []

        features = pack.get("features", {}) or {}

        alias_pool = pack.get("alias_pool", []) or []

        evidence_sources = pack.get("evidence_sources")

        doc = {

            "term": term,

            "created_at": created_at,

            "evidence_lines": evidence,

            "alias_pool": alias_pool,

            "features": features,

        }

        if evidence_sources is not None:

            doc["evidence_sources"] = evidence_sources

        safe = _safe_filename(term)

        path = os.path.join(subdir, f"{safe}.json")

        with open(path, "w", encoding="utf-8") as f:

            json.dump(doc, f, ensure_ascii=False, indent=2, default=lambda x: str(x))

    return subdir
