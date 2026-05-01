"""
Deterministic Emoji Engine — v2
基于本地 emoji_map_fixed.json 的规则生成（可一对多）

emoji_map_fixed.json 示例：
{
  "啊": ["😮","😦"],
  "阿": ["😮","😦"],
  "卧雅": ["🛏️🦷", "..."],   # （可选）词级映射
  ...
}

输出结构（不再有 mode / mixed）：
{
  "type": "emoji",
  "term": "海洛因",
  "word_level": { ... },   # 若命中词级映射则给出（否则为 {"hit": false}）
  "per_char": [            # 每字候选列表+解释
    {"char":"海","candidates":[{"text":"🌊","explain":"emoji：将\"海\"替换为\"🌊\""}, ...]},
    ...
  ],
  "candidates": [          # 最终候选（优先词级，否则字级组合）
    {"text":"🌊🧭😶", "explain":[...]} ,
    ...
  ]
}

规则：
- 优先词级映射（若启用且命中）：直接输出词级候选（最多 max_candidates），并给解释。
- 字级组合：
  - 若某字在映射表里有 emoji 候选，则只使用 emoji 候选（最多 max_per_char）
  - 若某字完全没有映射，才允许保留原字符，并解释“无可用emoji，保持不变”
- 最终候选去重、截断 max_candidates
"""

import json

import itertools

import os

from typing import Any, Dict, List, Optional

def _load_json(path: str) -> Any:

    with open(path, "r", encoding="utf-8") as f:

        data = json.load(f)

    if isinstance(data, list):

        data = data[0]

    return data

class EmojiEngine:

    def __init__(

        self,

        map_path: str = "emoji_map.json",

        max_per_char: int = 3,

    ):

        self.base_dir = os.path.dirname(__file__)

        self.map_path = os.path.join(self.base_dir, map_path)

        self.max_per_char = max_per_char

        self.emoji_map: Dict[str, List[str]] = _load_json(self.map_path)

    def char_candidates(self, ch: str) -> Dict[str, Any]:

        lst = self.emoji_map.get(ch) or []

        lst = [x for x in dict.fromkeys(lst) if x]

        if lst:

            lst = lst[: self.max_per_char]

            return {

                "char": ch,

                "candidates": [

                    {"text": e, "explain": f'emoji：将"{ch}"替换为"{e}"'} for e in lst

                ],

            }

        return {"char": ch, "candidates": []}

    def generate(

        self,

        term: str,

        max_candidates: int = 8,

        enable_word_level: bool = True,

    ) -> Dict[str, Any]:

        per_char = [self.char_candidates(ch) for ch in term]

        if enable_word_level and term in self.emoji_map:

            lst = self.emoji_map.get(term) or []

            lst = [x for x in dict.fromkeys(lst) if x]

            lst = lst[:max_candidates]

            word_level_info = {"hit": True}

            candidates = [

                {

                    "text": e,

                    "explain": [f'emoji：词级映射，将"{term}"替换为"{e}"'],

                }

                for e in lst

            ]

            return {

                "type": "emoji",

                "term": term,

                "word_level": word_level_info,

                "per_char": per_char,

                "candidates": candidates,

            }

        word_level_info = {"hit": False}

        options_per_char: List[List[Dict[str, Any]]] = []

        for idx, ch in enumerate(term):

            cands = per_char[idx]["candidates"]

            if cands:

                options_per_char.append(cands)

            else:

                options_per_char.append(

                    [{"text": ch, "explain": f'emoji：该字"{ch}"无可用emoji，保持不变'}]

                )

        combos = itertools.product(*options_per_char)

        out: List[Dict[str, Any]] = []

        seen = set()

        for combo in combos:

            text = "".join(x["text"] for x in combo)

            if text in seen:

                continue

            seen.add(text)

            out.append({"text": text, "explain": [x["explain"] for x in combo]})

            if len(out) >= max_candidates:

                break

        return {

            "type": "emoji",

            "term": term,

            "word_level": word_level_info,

            "per_char": per_char,

            "candidates": out,

        }

_engine_instance: EmojiEngine | None = None

def generate_emoji_candidates(

    term: str,

    max_candidates: int = 8,

    enable_word_level: bool = True,

) -> Dict[str, Any]:

    global _engine_instance

    if _engine_instance is None:

        _engine_instance = EmojiEngine()

    return _engine_instance.generate(

        term=term,

        max_candidates=max_candidates,

        enable_word_level=enable_word_level,

    )

if __name__ == "__main__":

    term = input("请输入词语：").strip()

    out = generate_emoji_candidates(term, max_candidates=8, enable_word_level=True)

    print(json.dumps(out, ensure_ascii=False, indent=2))
