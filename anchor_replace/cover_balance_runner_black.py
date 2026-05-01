import argparse

import json

import random

from pathlib import Path

from typing import Any, Dict, List, Sequence, Set, Tuple

from black_postprocess import (

    clean_and_validate_items,

    compute_coverage,

    flatten_black_items,

    load_black_list_words,

    select_balanced_samples,

    write_json,

)

def read_json(path: Path) -> Any:

    with path.open("r", encoding="utf-8") as f:

        return json.load(f)

def load_pool(input_path: Path) -> List[Dict[str, Any]]:

    """
    BLACK 示例数据源：从本地 JSON 文件/目录加载样本池（不生成任何内容）。
    - 支持：ModelAPIBlack.py 输出（包含 black 字段）或已展平格式
    """

    raw_items: List[Any] = []

    if input_path.is_dir():

        for p in sorted(input_path.glob("*.json")):

            raw_items.append(read_json(p))

    else:

        raw_items.append(read_json(input_path))

    flat: List[Dict[str, Any]] = []

    for raw in raw_items:

        flat.extend(flatten_black_items(raw))

    return flat

def get_next_batch_from_pool(

    pool: Sequence[Dict[str, Any]],

    need_terms: Set[str],

    batch_size: int,

    rng: random.Random,

) -> List[Dict[str, Any]]:

    """
    示例策略：优先抽取包含 need_terms 的样本，不足则随机补齐。
    """

    hits: List[Dict[str, Any]] = []

    rest: List[Dict[str, Any]] = []

    for item in pool:

        wl = item.get("words_list", [])

        if isinstance(wl, list) and any(w in need_terms for w in wl):

            hits.append(item)

        else:

            rest.append(item)

    rng.shuffle(hits)

    rng.shuffle(rest)

    batch = hits[:batch_size]

    if len(batch) < batch_size:

        batch.extend(rest[: batch_size - len(batch)])

    return batch

def _norm_text_for_dedupe(text: Any) -> str:

    if not isinstance(text, str):

        return ""

    return " ".join(text.strip().split())

def dedupe_items_by_text(

    items: Sequence[Dict[str, Any]],

    *,

    black_set: Set[str],

) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:

    """
    按 text 去重：同一 text 冲突时优先保留覆盖 black_set 词更多的条目。
    返回：(去重后 items, 统计信息)。
    """

    chosen_by_text: Dict[str, Dict[str, Any]] = {}

    stats = {"input": 0, "kept": 0, "dropped": 0, "missing_text": 0}

    def score(it: Dict[str, Any]) -> Tuple[int, int]:

        wl = it.get("words_list", [])

        if not isinstance(wl, list):

            return (0, 0)

        hit = sum(1 for w in wl if isinstance(w, str) and w in black_set)

        return (hit, len(wl))

    for it in items:

        stats["input"] += 1

        key = _norm_text_for_dedupe(it.get("text"))

        if not key:

            stats["missing_text"] += 1

            key = f"__missing_text__::{stats['missing_text']}"

        prev = chosen_by_text.get(key)

        if prev is None:

            chosen_by_text[key] = it

            continue

        if score(it) > score(prev):

            chosen_by_text[key] = it

    deduped = list(chosen_by_text.values())

    stats["kept"] = len(deduped)

    stats["dropped"] = stats["input"] - stats["kept"]

    return deduped, stats

def main() -> None:

    parser = argparse.ArgumentParser(description="迭代从样本池收集黑产样本：覆盖检查 + 清洗 + 最终均衡抽样")

    parser.add_argument("--pool", required=True, help="本地样本池 JSON 文件或目录（*.json）")

    parser.add_argument(

        "--black_list_py",

        default="Word_To_Illegal_Text/label_word/origin_word/LabelWordList.py",

        help="包含 BLACK_LIST 的 LabelWordList.py 路径",

    )

    parser.add_argument("--out_dir", default="cover_balance_black_out", help="输出目录")

    parser.add_argument("--target", type=int, default=1000, help="最终输出条数")

    parser.add_argument("--batch", type=int, default=100, help="每轮处理条数（默认100）")

    parser.add_argument("--max_rounds", type=int, default=200, help="最大轮数，避免无限循环")

    parser.add_argument(

        "--keep_extra_words",

        action="store_true",

        help="不丢弃含额外词的样本；改为从 words_list 里移除非 BLACK_LIST 词后继续处理",

    )

    parser.add_argument("--seed", type=int, default=7, help="随机种子（用于可复现抽取）")

    args = parser.parse_args()

    pool_path = Path(args.pool).expanduser().resolve()

    black_list_py = Path(args.black_list_py).expanduser().resolve()

    out_dir = Path(args.out_dir).expanduser().resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(args.seed))

    black_terms = load_black_list_words(black_list_py)

    black_set = set(black_terms)

    pool = load_pool(pool_path)

    write_json(out_dir / "pool_size.json", {"pool_items": len(pool)})

    all_items: List[Dict[str, Any]] = []

    need_terms: Set[str] = set(black_terms)

    for r in range(int(args.max_rounds)):

        focus_terms = need_terms if need_terms else black_set

        batch_raw = get_next_batch_from_pool(

            pool, need_terms=focus_terms, batch_size=int(args.batch), rng=rng

        )

        cleaned, _stats = clean_and_validate_items(

            batch_raw,

            term_words_list=black_terms,

            strict_drop_extra_words=not args.keep_extra_words,

        )

        all_items.extend(cleaned)

        all_items, _dedupe_stats = dedupe_items_by_text(all_items, black_set=black_set)

        cov = compute_coverage(all_items, black_terms)

        need_terms = set(cov["missing_terms"])

        write_json(

            out_dir / "progress.json",

            {

                "round": r + 1,

                "total_items": len(all_items),

                "covered": cov["covered_count"],

                "term_count": cov["term_count"],

                "missing_terms_count": len(cov["missing_terms"]),

                "missing_terms": cov["missing_terms"],

            },

        )

        if not need_terms and len(all_items) >= int(args.target):

            break

    positive_items: List[Dict[str, Any]] = []

    no_final_items: List[Dict[str, Any]] = []

    for it in all_items:

        label_zh = it.get("label_zh")

        is_illegal = str(it.get("is_illegal", "")).lower()

        if label_zh == "黑产" and is_illegal == "yes":

            positive_items.append(it)

        else:

            no_final_items.append(it)

    if no_final_items:

        write_json(out_dir / "black_no_final.json", no_final_items)

    selected, selected_counts = select_balanced_samples(

        positive_items, black_terms, target_n=int(args.target)

    )

    selected, _dedupe_stats_selected = dedupe_items_by_text(selected, black_set=black_set)

    final_cov = compute_coverage(selected, black_terms)

    write_json(out_dir / "final_coverage.json", final_cov)

    write_json(out_dir / "missing_terms.json", final_cov["missing_terms"])

    write_json(out_dir / "selected_term_counts.json", selected_counts)

    out_name = (

        f"final_{len(selected)}.json"

        if len(selected) != int(args.target)

        else f"final_{int(args.target)}.json"

    )

    write_json(out_dir / out_name, selected)

    print(f"pool: {pool_path} ({len(pool)} items)")

    print(f"BLACK_LIST: {black_list_py} ({len(black_terms)} terms)")

    print(f"collected items (黑产且非法): {len(positive_items)} / {len(all_items)} 总清洗后条数")

    print(f"final items: {len(selected)} -> {out_dir / out_name}")

    print(

        f"final coverage: {final_cov['covered_count']}/{final_cov['term_count']}, missing: {len(final_cov['missing_terms'])}"

    )

if __name__ == "__main__":

    main()
