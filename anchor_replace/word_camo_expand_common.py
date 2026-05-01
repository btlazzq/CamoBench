import json

import random

from pathlib import Path

from typing import Any, List, Tuple

def load_candidates(vocab_dir: Path, term: str) -> List[str]:

    """
    从某一类伪装词库中，按 term 读取候选词列表（只取 candidate 字段）。
    若文件不存在或无有效 candidate，则返回空列表。
    """

    path = vocab_dir / f"{term}.json"

    if not path.exists():

        return []

    try:

        with path.open("r", encoding="utf-8") as f:

            data = json.load(f)

    except Exception:

        return []

    candidates: List[str] = []

    if isinstance(data, list):

        for item in data:

            cand = item.get("candidate")

            if isinstance(cand, str) and cand:

                candidates.append(cand)

    return candidates

def replace_with_mixed_mechanisms(

    text: str,

    words_list: List[Any],

    specs: List[Tuple[str, Path]],

) -> Tuple[str, List[str], List[str]]:

    """
    每个敏感词独立随机一种机制并替换。
    当 len(words_list) >= 2 时，强制句内至少两种不同机制（避免整句同一种伪装）。
    返回 (new_text, replace_list, mechanism_per_word)，与 words_list 逐项对齐。
    """

    names = [s[0] for s in specs]

    name_to_dir = {s[0]: s[1] for s in specs}

    n = len(words_list)

    if n == 0:

        return text, [], []

    if n == 1:

        assignments = [random.choice(names)]

    else:

        while True:

            assignments = [random.choice(names) for _ in range(n)]

            if len(set(assignments)) >= 2:

                break

    new_text = text

    replace_list: List[str] = []

    for term, mech in zip(words_list, assignments):

        if not isinstance(term, str):

            replace_list.append(str(term))

            continue

        vocab_dir = name_to_dir[mech]

        candidates = load_candidates(vocab_dir, term)

        repl = random.choice(candidates) if candidates else term

        replace_list.append(repl)

        if term and repl:

            new_text = new_text.replace(term, repl)

    return new_text, replace_list, assignments
