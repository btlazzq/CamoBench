"""
Deterministic Phonetic Engine — v2
包含三部分：
1) 拼音信息：用 pypinyin 获取每个字的完整拼音（可选带声调）
2) 近音候选：基于本地 yin_map.json 的近音字替换（可一对多）
3) 混合候选：拼音 + 近音 的组合（每个字可以选择“拼音”或“近音”，若两者都没有才保留原字）

yin_map.json 示例：
{
  "阿": ["啊","锕","嗄"],
  "啊": ["阿","锕","嗄"]
}

输出结构：
{
  "type": "音变",
  "term": "海洛因",
  "pinyin": {...},        # 拼音信息（逐字+全词）
  "per_char": [...],      # 近音：每字候选列表+解释
  "candidates": [...],    # 近音：最终候选
  "mixed": {
     "per_char": [...],   # 混合：每字可选项（拼音/近音/必要时keep）
     "candidates": [...], # 混合：最终候选
  }
}

规则：
- 近音：
  - 若某字在 yin_map.json 里有候选，则只使用候选（最多 max_yin_per_char，超出则随机采样）
  - 若某字完全没有候选，才允许保留原字，并解释“无可用近音，保持不变”
- 混合（拼音+近音）：
  - 若某字有“拼音”或“近音”任一可用，则不再提供 keep 选项
  - 只有当该字拼音取不到且近音也没有时，才保留原字
- 最终候选去重、截断 max_candidates
"""

import json

import itertools

import os

import random

from typing import Any, Dict, List

try:

    from pypinyin import pinyin as _pypinyin, Style as _Style

except Exception as e:

    _pypinyin = None

    _Style = None

    _PYPINYIN_IMPORT_ERROR = e

else:

    _PYPINYIN_IMPORT_ERROR = None

def _load_json(path: str) -> Any:

    with open(path, "r", encoding="utf-8") as f:

        data = json.load(f)

    if isinstance(data, list):

        data = data[0]

    return data

class PhoneticEngine:

    def __init__(

        self,

        yin_map_path: str = "yin_map.json",

        max_yin_per_char: int = 3,

        seed: int | None = None,

    ):

        self.base_dir = os.path.dirname(__file__)

        self.yin_map_path = os.path.join(self.base_dir, yin_map_path)

        self.max_yin_per_char = max_yin_per_char

        self.rng = random.Random(seed)

        self.yin_map: Dict[str, List[str]] = _load_json(self.yin_map_path)

    def get_pinyin(self, term: str, style: str = "normal", separator: str = " ") -> Dict[str, Any]:

        if _pypinyin is None:

            raise ImportError('缺少依赖：pypinyin。请先安装：pip install pypinyin') from _PYPINYIN_IMPORT_ERROR

        st = _Style.TONE3 if style == "tone3" else _Style.NORMAL

        arr = _pypinyin(term, style=st, heteronym=False, errors="default")

        per_char = []

        for ch, py_list in zip(term, arr):

            py = py_list[0] if py_list else ""

            per_char.append({"char": ch, "pinyin": py})

        text = separator.join(x["pinyin"] for x in per_char if x["pinyin"] != "")

        return {"style": style, "text": text, "per_char": per_char}

    def _char_pinyin_option(self, ch: str, style: str = "normal") -> List[Dict[str, str]]:

        """单字拼音选项（用于混合组合）"""

        if _pypinyin is None:

            return []

        st = _Style.TONE3 if style == "tone3" else _Style.NORMAL

        arr = _pypinyin(ch, style=st, heteronym=False, errors="default")

        py = arr[0][0] if arr and arr[0] else ""

        if not py:

            return []

        return [{"text": py, "explain": f'拼音：将"{ch}"替换为"{py}"'}]

    def char_candidates(self, ch: str) -> Dict[str, Any]:

        lst = self.yin_map.get(ch) or []

        lst = [x for x in dict.fromkeys(lst) if x and x != ch]

        if not lst:

            return {"char": ch, "candidates": []}

        if len(lst) > self.max_yin_per_char:

            picks = self.rng.sample(lst, self.max_yin_per_char)

        else:

            picks = lst

        return {

            "char": ch,

            "candidates": [{"text": y, "explain": f'近音：将"{ch}"替换为"{y}"'} for y in picks],

        }

    def generate_yin(self, term: str, max_candidates: int = 8) -> Dict[str, Any]:

        per_char = [self.char_candidates(ch) for ch in term]

        options_per_char: List[List[Dict[str, Any]]] = []

        for idx, ch in enumerate(term):

            cands = per_char[idx]["candidates"]

            if cands:

                options_per_char.append(cands)

            else:

                options_per_char.append(

                    [{"text": ch, "explain": f'近音：该字"{ch}"无可用近音，保持不变'}]

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

        return {"per_char": per_char, "candidates": out}

    @staticmethod

    def _infer_mechs(explains: List[str]) -> str:

        mechs = set()

        for e in explains:

            if e.startswith("拼音："):

                mechs.add("pinyin")

            elif e.startswith("近音："):

                mechs.add("yin")

            elif "保持不变" in e:

                mechs.add("keep")

        if len(mechs) > 1 and "keep" in mechs:

            mechs.remove("keep")

        if not mechs:

            return "keep"

        order = ["pinyin", "yin", "keep"]

        return "+".join([m for m in order if m in mechs])

    def generate_mixed(

        self,

        term: str,

        max_candidates: int = 8,

        pinyin_style: str = "normal",

    ) -> Dict[str, Any]:

        per_char: List[Dict[str, Any]] = []

        options_per_char: List[List[Dict[str, Any]]] = []

        for ch in term:

            py_opts = self._char_pinyin_option(ch, style=pinyin_style)

            yin_opts = self.char_candidates(ch)["candidates"]

            any_opts = bool(py_opts or yin_opts)

            per_char.append({

                "char": ch,

                "candidates": (

                    [{"type": "pinyin", **o} for o in py_opts] +

                    [{"type": "yin", **o} for o in yin_opts]

                )

            })

            opts = []

            opts += py_opts

            opts += yin_opts

            if not any_opts:

                opts = [{"text": ch, "explain": f'混合：该字"{ch}"无拼音/近音可用，保持不变'}]

            seen = set()

            uniq = []

            for o in opts:

                if o["text"] in seen:

                    continue

                seen.add(o["text"])

                uniq.append(o)

            options_per_char.append(uniq)

        combos = itertools.product(*options_per_char)

        out: List[Dict[str, Any]] = []

        seen = set()

        for combo in combos:

            text = "".join(x["text"] for x in combo)

            if text in seen:

                continue

            seen.add(text)

            explains = [x["explain"] for x in combo]

            out.append({"text": text, "explain": explains, "mode": self._infer_mechs(explains)})

            if len(out) >= max_candidates:

                break

        return {"per_char": per_char, "candidates": out}

    def generate(

        self,

        term: str,

        max_candidates: int = 8,

        pinyin_style: str = "normal",

        pinyin_separator: str = " ",

    ) -> Dict[str, Any]:

        pinyin_info = self.get_pinyin(term, style=pinyin_style, separator=pinyin_separator)

        yin_info = self.generate_yin(term, max_candidates=max_candidates)

        mixed_info = self.generate_mixed(term, max_candidates=max_candidates, pinyin_style=pinyin_style)

        return {

            "type": "音变",

            "term": term,

            "pinyin": pinyin_info,

            "per_char": yin_info["per_char"],

            "candidates": yin_info["candidates"],

            "mixed": mixed_info,

        }

_engine_instance: PhoneticEngine | None = None

def generate_phonetic_candidates(

    term: str,

    max_candidates: int = 8,

    seed: int | None = None,

    pinyin_style: str = "normal",

    pinyin_separator: str = " ",

) -> Dict[str, Any]:

    global _engine_instance

    if _engine_instance is None or (seed is not None):

        _engine_instance = PhoneticEngine(seed=seed)

    return _engine_instance.generate(

        term=term,

        max_candidates=max_candidates,

        pinyin_style=pinyin_style,

        pinyin_separator=pinyin_separator,

    )

if __name__ == "__main__":

    term = input("请输入词语：").strip()

    out = generate_phonetic_candidates(term, max_candidates=8, seed=42, pinyin_style="normal")

    print(json.dumps(out, ensure_ascii=False, indent=2))
