#!/usr/bin/env python3

"""
批量生成 emoji_camo 候选，输出格式与语义伪装的 vocabulary 一致：
每个 term 一个 JSON 文件，内容为数组，每项为 { term, mechanism, candidate, explanation }。
mechanism 固定为 "emoji"。
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

    from emoji_engine import generate_emoji_candidates

except ImportError:

    from emoji_camo.emoji_engine import generate_emoji_candidates

def sanitize_filename(s: str) -> str:

    """转为安全文件名。"""

    safe = (s or "").replace("/", "_").replace("\\", "_").strip()

    return safe or "unnamed"

def emoji_result_to_entries(term: str, raw: dict) -> list:

    """
    将 EmojiEngine.generate() 的 candidates 转为与 按摩.json 同格式的条目列表。
    每条: { term, mechanism: "emoji", candidate, explanation }。
    """

    entries = []

    seen_text = set()

    sep = "；"

    for item in raw.get("candidates", []):

        text = (item.get("text") or "").strip()

        if not text or text in seen_text:

            continue

        seen_text.add(text)

        explain_list = item.get("explain")

        if isinstance(explain_list, list):

            explanation = sep.join(str(e) for e in explain_list if e)

        else:

            explanation = str(explain_list or "")

        entries.append({

            "term": term,

            "mechanism": "emoji",

            "candidate": text,

            "explanation": explanation,

        })

    return entries

def load_terms(terms_file: str, terms_arg: str) -> list:

    """从文件或命令行参数加载 term 列表。"""

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

    parser = argparse.ArgumentParser(description="批量生成 emoji_camo，输出 vocabulary 格式 JSON")

    parser.add_argument(

        "--terms_file",

        type=str,

        default="",

        help="每行一个 term 的 txt 文件（可为 term\\tscene 格式，只取第一列）",

    )

    parser.add_argument(

        "--terms",

        type=str,

        default="",

        help="逗号分隔的 term 列表，如：卧槽,按摩",

    )

    parser.add_argument(

        "--out_dir",

        type=str,

        default="",

        help="输出根目录，其下创建 vocabulary/ 存放各 term 的 JSON；默认为 emoji_camo/output_emoji",

    )

    parser.add_argument(

        "--max_candidates",

        type=int,

        default=8,

        help="每个 term 最多保留的候选数",

    )

    parser.add_argument(

        "--no_word_level",

        action="store_true",

        help="禁用词级映射，仅用字级组合",

    )

    args = parser.parse_args()

    terms = load_terms(args.terms_file, args.terms)

    if not terms:

        root = SCRIPT_DIR.parent

        for name in ("terms_with_scene.txt", "terms.txt"):

            p = root / name

            if p.exists():

                terms = load_terms(str(p), "")

                break

        if not terms:

            terms = ["卧槽", "按摩"]

    out_root = Path(args.out_dir) if args.out_dir else SCRIPT_DIR / "output_emoji"

    vocab_dir = out_root / "vocabulary"

    vocab_dir.mkdir(parents=True, exist_ok=True)

    all_entries_for_jsonl = []

    for term in terms:

        raw = generate_emoji_candidates(

            term,

            max_candidates=args.max_candidates,

            enable_word_level=not args.no_word_level,

        )

        entries = emoji_result_to_entries(term, raw)

        if not entries:

            print(f"  跳过（无候选）: {term}")

            continue

        safe_name = sanitize_filename(term)

        path = vocab_dir / f"{safe_name}.json"

        with open(path, "w", encoding="utf-8") as f:

            json.dump(entries, f, ensure_ascii=False, indent=2)

        print(f"  {term} -> {path.name} ({len(entries)} 条)")

        for e in entries:

            all_entries_for_jsonl.append(e)

    if all_entries_for_jsonl:

        jsonl_path = out_root / "results_emoji.jsonl"

        with open(jsonl_path, "w", encoding="utf-8") as f:

            for e in all_entries_for_jsonl:

                f.write(json.dumps(e, ensure_ascii=False) + "\n")

        print(f"\n已汇总写入: {jsonl_path}")

    print(f"\n完成：共 {len(terms)} 个 term，结果目录 {vocab_dir}")

if __name__ == "__main__":

    main()
