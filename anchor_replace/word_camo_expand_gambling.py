import argparse

import csv

import json

import random

from pathlib import Path

from typing import Any, Dict, List, Tuple

from word_camo_expand_common import load_candidates, replace_with_mixed_mechanisms

BASE_DIR = Path("/Users/btlazzq/Desktop/agent/semantic_camo_safe_system-v3")

TRAIN_DIR = BASE_DIR / "最终测评数据"

INPUT_DIR = TRAIN_DIR / "最终筛选"

OUTPUT_DIR = TRAIN_DIR

DEFAULT_N = 100

def resolve_final_json(label: str, input_dir: Path, default_n: int = 1000) -> Tuple[int, Path]:

    """
    优先使用 <label>_final_<default_n>.json；不存在则取该目录下最大 n 的 <label>_final_*.json。
    """

    expected = input_dir / f"{label}_final_{default_n}.json"

    if expected.exists():

        return default_n, expected

    candidates: List[Tuple[int, Path]] = []

    for p in input_dir.glob(f"{label}_final_*.json"):

        stem = p.stem

        try:

            n_str = stem.split(f"{label}_final_")[-1]

            n = int(n_str)

        except Exception:

            continue

        candidates.append((n, p))

    if not candidates:

        raise FileNotFoundError(f"未找到 {label}_final_*.json: {input_dir}")

    candidates.sort(key=lambda x: x[0], reverse=True)

    return candidates[0][0], candidates[0][1]

TARGET_N, GAMBLING_JSON_PATH = resolve_final_json("gambling", INPUT_DIR, default_n=DEFAULT_N)

ABBREV_DIR = BASE_DIR / "abbreviation_camo/output_abbreviation_gambling/vocabulary"

EMOJI_DIR = BASE_DIR / "emoji_camo/output_emoji_gambling/vocabulary"

PHONETIC_DIR = BASE_DIR / "phonetic_camo/output_phonetic_gambling/vocabulary"

SEMANTIC_DIR = BASE_DIR / "semantic_camo/output_gambling/vocabulary"

SHAPE_DIR = BASE_DIR / "shape_camo/output_shape_gambling/vocabulary"

AUG_JSON_PATH = OUTPUT_DIR / f"gambling_{TARGET_N}_augmented.json"

AUG_CSV_PATH = OUTPUT_DIR / f"gambling_{TARGET_N}_augmented.csv"

MECHANISM_SPECS: List[Tuple[str, Path]] = [

    ("缩略", ABBREV_DIR),

    ("emoji", EMOJI_DIR),

    ("音", PHONETIC_DIR),

    ("义", SEMANTIC_DIR),

    ("形", SHAPE_DIR),

]

def replace_with_mechanism(

    text: str,

    words_list: List[str],

    vocab_dir: Path,

) -> Tuple[str, List[str]]:

    """
    对给定 text 中的每个敏感词 term，在对应 vocab_dir 下随机选一个 candidate 替换。
    返回：替换后的文本、新的 replace_list（与 words_list 对齐）。
    若某个 term 找不到候选，则保持原词不变，并在 replace_list 中仍记录原词。
    """

    new_text = text

    replace_list: List[str] = []

    for term in words_list:

        if not isinstance(term, str):

            replace_list.append(str(term))

            continue

        candidates = load_candidates(vocab_dir, term)

        if candidates:

            repl = random.choice(candidates)

        else:

            repl = term

        replace_list.append(repl)

        if term and repl:

            new_text = new_text.replace(term, repl)

    return new_text, replace_list

def process(*, enable_mixed: bool = True) -> None:

    with GAMBLING_JSON_PATH.open("r", encoding="utf-8") as f:

        items = json.load(f)

    augmented_json: List[Dict[str, Any]] = []

    csv_rows: List[Dict[str, Any]] = []

    for item in items:

        text = item.get("text", "") or ""

        is_illegal = item.get("is_illegal", "")

        label_zh = item.get("label_zh", "")

        words_list = item.get("words_list", []) or []

        if not isinstance(words_list, list):

            words_list = []

        base_entry = {

            "text": text,

            "is_illegal": is_illegal,

            "label_zh": label_zh,

            "mechanism": "原词",

            "words_list": words_list,

        }

        augmented_json.append(base_entry)

        text_abbrev, words_abbrev = replace_with_mechanism(text, words_list, ABBREV_DIR)

        augmented_json.append(

            {

                "text": text_abbrev,

                "is_illegal": is_illegal,

                "label_zh": label_zh,

                "mechanism": "缩略",

                "words_list": words_list,

                "replace_list": words_abbrev,

            }

        )

        text_emoji, words_emoji = replace_with_mechanism(text, words_list, EMOJI_DIR)

        augmented_json.append(

            {

                "text": text_emoji,

                "is_illegal": is_illegal,

                "label_zh": label_zh,

                "mechanism": "emoji",

                "words_list": words_list,

                "replace_list": words_emoji,

            }

        )

        text_phonetic, words_phonetic = replace_with_mechanism(text, words_list, PHONETIC_DIR)

        augmented_json.append(

            {

                "text": text_phonetic,

                "is_illegal": is_illegal,

                "label_zh": label_zh,

                "mechanism": "音",

                "words_list": words_list,

                "replace_list": words_phonetic,

            }

        )

        text_semantic, words_semantic = replace_with_mechanism(text, words_list, SEMANTIC_DIR)

        augmented_json.append(

            {

                "text": text_semantic,

                "is_illegal": is_illegal,

                "label_zh": label_zh,

                "mechanism": "义",

                "words_list": words_list,

                "replace_list": words_semantic,

            }

        )

        text_shape, words_shape = replace_with_mechanism(text, words_list, SHAPE_DIR)

        augmented_json.append(

            {

                "text": text_shape,

                "is_illegal": is_illegal,

                "label_zh": label_zh,

                "mechanism": "形",

                "words_list": words_list,

                "replace_list": words_shape,

            }

        )

        text_mixed = ""

        words_mixed: List[str] = []

        mech_per_word: List[str] = []

        if enable_mixed and words_list:

            text_mixed, words_mixed, mech_per_word = replace_with_mixed_mechanisms(

                text, words_list, MECHANISM_SPECS

            )

            augmented_json.append(

                {

                    "text": text_mixed,

                    "is_illegal": is_illegal,

                    "label_zh": label_zh,

                    "mechanism": "混合",

                    "words_list": words_list,

                    "replace_list": words_mixed,

                    "mechanism_per_word": mech_per_word,

                }

            )

        csv_rows.append(

            {

                "text": text,

                "is_illegal": is_illegal,

                "label_zh": label_zh,

                "words_list": json.dumps(words_list, ensure_ascii=False),

                "text_abbreviation": text_abbrev,

                "words_list_abbreviation": json.dumps(words_abbrev, ensure_ascii=False),

                "text_emoji": text_emoji,

                "words_list_emoji": json.dumps(words_emoji, ensure_ascii=False),

                "text_phonetic": text_phonetic,

                "words_list_phonetic": json.dumps(words_phonetic, ensure_ascii=False),

                "text_semantic": text_semantic,

                "words_list_semantic": json.dumps(words_semantic, ensure_ascii=False),

                "text_shape": text_shape,

                "words_list_shape": json.dumps(words_shape, ensure_ascii=False),

                "text_mixed": text_mixed,

                "words_list_mixed": json.dumps(words_mixed, ensure_ascii=False),

                "mechanism_per_word": json.dumps(mech_per_word, ensure_ascii=False),

            }

        )

    with AUG_JSON_PATH.open("w", encoding="utf-8") as f:

        json.dump(augmented_json, f, ensure_ascii=False, indent=2)

    fieldnames = [

        "text",

        "is_illegal",

        "label_zh",

        "words_list",

        "text_abbreviation",

        "words_list_abbreviation",

        "text_emoji",

        "words_list_emoji",

        "text_phonetic",

        "words_list_phonetic",

        "text_semantic",

        "words_list_semantic",

        "text_shape",

        "words_list_shape",

        "text_mixed",

        "words_list_mixed",

        "mechanism_per_word",

    ]

    with AUG_CSV_PATH.open("w", encoding="utf-8", newline="") as f:

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()

        for row in csv_rows:

            writer.writerow(row)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--no-mixed", action="store_true", help="不生成混合机制样本")

    args = parser.parse_args()

    process(enable_mixed=not args.no_mixed)
