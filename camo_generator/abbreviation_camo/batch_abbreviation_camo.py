#!/usr/bin/env python3

"""
批量生成缩略语伪装（abbreviation_camo）候选。
- 两字：拼音首字母+保留一字 等
- 三字：去一字 + 拼音首字母组合
- 四字及以上：规则兜底 + LLM 按语义重点生成缩略
每个 term 的全部候选交给判断 LLM，得到缩略程度、语义保留程度、毒性及综合评分，并输出综合推荐的若干缩略。
"""

import argparse

import json

import sys

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

REPO_ROOT = SCRIPT_DIR.parent

for d in (SCRIPT_DIR, REPO_ROOT):

    if str(d) not in sys.path:

        sys.path.insert(0, str(d))

try:

    from abbreviation_engine import generate_abbreviation_candidates, needs_llm_abbreviation

except ImportError:

    from abbreviation_camo.abbreviation_engine import (

        generate_abbreviation_candidates,

        needs_llm_abbreviation,

    )

try:

    from abbreviation_llm import llm_suggest_abbreviations, llm_judge_abbreviations

except ImportError:

    from abbreviation_camo.abbreviation_llm import (

        llm_suggest_abbreviations,

        llm_judge_abbreviations,

    )

def sanitize_filename(s: str) -> str:

    safe = (s or "").replace("/", "_").replace("\\", "_").strip()

    return safe or "unnamed"

def abbreviation_result_to_entries(term: str, raw: list, score_map: dict = None) -> list:

    """将 candidate+explanation 转为 vocabulary 条目，并合并判断分数。"""

    score_map = score_map or {}

    entries = []

    for item in (raw or []):

        candidate = (item.get("candidate") or "").strip()

        if not candidate or candidate == term:

            continue

        explanation = (item.get("explanation") or "").strip()

        e = {

            "term": term,

            "mechanism": "缩略",

            "candidate": candidate,

            "explanation": explanation,

        }

        if candidate in score_map:

            e["abbreviation_score"] = score_map[candidate].get("abbreviation_score")

            e["semantic_score"] = score_map[candidate].get("semantic_score")

            e["has_toxicity"] = score_map[candidate].get("has_toxicity")

            e["composite"] = score_map[candidate].get("composite")

        entries.append(e)

    return entries

def load_terms(terms_file: str, terms_arg: str) -> list:

    terms = []

    if (terms_arg or "").strip():

        terms.extend(t.strip() for t in terms_arg.split(",") if t.strip())

    if (terms_file or "").strip() and Path(terms_file).exists():

        with open(terms_file, "r", encoding="utf-8") as f:

            for line in f:

                t = line.strip().split("\t")[0].strip()

                if t:

                    terms.append(t)

    seen = set()

    out = []

    for t in terms:

        if t not in seen:

            seen.add(t)

            out.append(t)

    return out

def main():

    parser = argparse.ArgumentParser(

        description="批量生成缩略语伪装（规则+LLM），并经判断LLM打分与推荐"

    )

    parser.add_argument(

        "--terms_file",

        type=str,

        default="",

        help="每行一个 term 的 txt 文件（可为 term\\tscene，只取第一列）",

    )

    parser.add_argument(

        "--terms",

        type=str,

        default="",

        help="逗号分隔的 term 列表",

    )

    parser.add_argument(

        "--out_dir",

        type=str,

        default="",

        help="输出根目录，默认 abbreviation_camo/output_abbreviation",

    )

    parser.add_argument(

        "--no_llm",

        action="store_true",

        help="不调用 LLM 生成四字及以上语义缩略，仅用规则",

    )

    parser.add_argument(

        "--no_judge",

        action="store_true",

        help="不调用判断 LLM，不生成评分与推荐",

    )

    parser.add_argument(

        "--top_k",

        type=int,

        default=5,

        help="每个 term 综合推荐的缩略数量",

    )

    parser.add_argument(

        "--max_llm_suggestions",

        type=int,

        default=5,

        help="四字及以上时 LLM 建议的缩略条数",

    )

    args = parser.parse_args()

    terms = load_terms(args.terms_file, args.terms)

    if not terms:

        for name in ("terms_with_scene.txt", "terms.txt"):

            p = REPO_ROOT / name

            if p.exists():

                terms = load_terms(str(p), "")

                break

        if not terms:

            terms = ["冰毒", "海洛因", "替来他明", "含有替来他明的电子烟油"]

    out_root = Path(args.out_dir) if args.out_dir else SCRIPT_DIR / "output_abbreviation"

    vocab_dir = out_root / "vocabulary"

    vocab_dir.mkdir(parents=True, exist_ok=True)

    all_entries_for_jsonl = []

    for term in terms:

        raw = generate_abbreviation_candidates(term)

        if not args.no_llm and needs_llm_abbreviation(term):

            try:

                llm_cands = llm_suggest_abbreviations(term, max_suggestions=args.max_llm_suggestions)

                raw.extend(llm_cands)

            except Exception as e:

                print(f"  [LLM 缩略] {term} 调用失败: {e}")

        if not raw:

            print(f"  跳过（无候选）: {term}")

            continue

        score_map = {}

        recommended = []

        if not args.no_judge:

            try:

                judge_out = llm_judge_abbreviations(term, raw, top_k=args.top_k)

                for s in judge_out.get("scores") or []:

                    score_map[s["candidate"]] = s

                recommended = judge_out.get("recommended") or []

                print(f"  [判断] {term} -> 推荐: {recommended[:3]}...")

            except Exception as e:

                print(f"  [判断LLM] {term} 失败: {e}")

        entries = abbreviation_result_to_entries(term, raw, score_map)

        safe_name = sanitize_filename(term)

        path = vocab_dir / f"{safe_name}.json"

        with open(path, "w", encoding="utf-8") as f:

            json.dump(entries, f, ensure_ascii=False, indent=2)

        print(f"  {term} -> {path.name} (候选 {len(entries)} 条, 推荐 {len(recommended)} 条)")

        for e in entries:

            all_entries_for_jsonl.append(e)

    if all_entries_for_jsonl:

        jsonl_path = out_root / "results_abbreviation.jsonl"

        with open(jsonl_path, "w", encoding="utf-8") as f:

            for e in all_entries_for_jsonl:

                f.write(json.dumps(e, ensure_ascii=False) + "\n")

        print(f"\n已汇总写入: {jsonl_path}")

    print(f"\n完成：共 {len(terms)} 个 term，结果目录 {vocab_dir}")

if __name__ == "__main__":

    main()
