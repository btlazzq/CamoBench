"""
Deterministic Shape Engine
基于本地chaizi.json的拆字生成器
chaizi.json格式：key为原字，value为拆字后的组成
{
  "卧": "臣卜",
  "项": "工页",
  "功": "工力",
  "攻": "工攵",
  "荆": "茾刂",
  "邪": "牙阝",
  "雅": "牙隹",
  "期": "其月",
  "欺": "其欠"
}
基于本地fanti_map.json的简繁转换器
fanti_map.json格式：key为原字，value为原字的繁体形式
{
  "专": "專",
  "业": "業",
  "丛": "叢",
  "东": "東",
  "丝": "絲",
  "丢": "丟",
  "两": "兩",
  "严": "嚴"
}
基于本地xingjinzi_map_merged.json形近字的转换器
xingjinzi_map_merged.json格式：key为原字,value为原字的形近字,一个原字可能对应多个形近字。
{
  "阿": [
    "厕"
  ],
  "啊": [
    "啃"
  ],
  "哎": [
    "哗",
    "吱"
  ],
  "哀": [
    "衰",
    "蓑",
    "猿"
  ],
  "埃": [
    "梯",
    "弟",
    "涕",
    "递",
    "挨",
    "唉"
  ]
}
业务逻辑如下：
首先，输入一个词，对这个词一个字一个字进行变换。
每个字都进行这三个json的检索。
然后输出：1.这个词（不管几个字）所有的字都进行相同的变换（拆字、简繁转换和形近的），同时给出解释，比如输入“卧雅”的时候，输出“臣卜牙隹”，也要输出解释，将“卧”进行拆字变为“臣卜”，将“雅”进行拆字“变为”牙隹“。
2.返回词里面，不同字可以有不同的变换，比如拆字和繁体等等组合。同样也给解释。
3.这些解释前面都要加一个，统一属于形变。
4.我需要一个候选列表+解释，比如“卧”这个字，遍历三个json，找到是否有value，并给出如何变换的。词的每个字都要有。
4.对于形近字，列表出现一对多的情况，可以随机选择三个作为候选列表。
"""

"""
Deterministic Shape Engine (extended) — v3

形变（Shape Transformation）候选生成器，基于本地规则：
- chaizi.json: { "卧": "臣卜", ... }               # 拆字
- fanti_map.json: { "专": "專", ... }             # 简->繁
- xingjinzi_map_merged.json: { "哀": ["衰","蓑"], ... }  # 形近字（可一对多）

1) 如果某字“本身有任意形变候选”（拆字/繁体/形近至少一个），则在 mixed/uniform 里不再输出
   “该字X保持不变”的解释；只有当该字“三种都没有”时，才允许保留原字并解释“保持不变”。
2) explain 里不再使用「」引号，直接用中文双引号 ""，并且整体前缀仍为 “形变：”。

输出：
- type: 总的变换类型（固定 "形变"）
- mode: 形变里的具体方法（uniform: chaizi/fanti/similar；mixed: 机制组合）
"""

import json

import itertools

import os

import random

from typing import Any, Dict, List, Tuple

def _load_json(path: str) -> Any:

    with open(path, "r", encoding="utf-8") as f:

        return json.load(f)

class ShapeEngine:

    def __init__(

        self,

        chaizi_path: str = "chaizi.json",

        fanti_path: str = "fanti_map.json",

        xingjinzi_path: str = "xingjinzi_map_merged.json",

        max_parts_per_char: int = 3,

        max_similar_pick: int = 3,

        seed: int | None = None,

    ):

        self.base_dir = os.path.dirname(__file__)

        self.chaizi_map = _load_json(os.path.join(self.base_dir, chaizi_path))

        self.fanti_map = _load_json(os.path.join(self.base_dir, fanti_path))

        self.xingjinzi_map = _load_json(os.path.join(self.base_dir, xingjinzi_path))

        self.max_parts_per_char = max_parts_per_char

        self.max_similar_pick = max_similar_pick

        self._rng = random.Random(seed)

    def _decompose_candidates(self, ch: str) -> List[Tuple[str, str]]:

        if ch not in self.chaizi_map:

            return []

        parts_str = self.chaizi_map[ch]

        parts = list(parts_str)

        if len(parts) > self.max_parts_per_char:

            return []

        cand = "".join(parts[: self.max_parts_per_char])

        exp = f'形变：将"{ch}"拆字为"{cand}"'

        return [(cand, exp)]

    def _fanti_candidates(self, ch: str) -> List[Tuple[str, str]]:

        if ch not in self.fanti_map:

            return []

        cand = self.fanti_map[ch]

        exp = f'形变：将"{ch}"简繁转换为"{cand}"'

        return [(cand, exp)]

    def _similar_candidates(self, ch: str) -> List[Tuple[str, str]]:

        lst = self.xingjinzi_map.get(ch)

        if not lst:

            return []

        uniq = [x for x in dict.fromkeys(lst) if x and x != ch]

        if not uniq:

            return []

        if len(uniq) > self.max_similar_pick:

            picks = self._rng.sample(uniq, self.max_similar_pick)

        else:

            picks = uniq

        out: List[Tuple[str, str]] = []

        for s in picks:

            exp = f'形变：将"{ch}"替换为形近字"{s}"'

            out.append((s, exp))

        return out

    def _has_any_candidates(self, ch: str) -> bool:

        return bool(self._decompose_candidates(ch) or self._fanti_candidates(ch) or self._similar_candidates(ch))

    def char_candidates(self, ch: str) -> Dict[str, Any]:

        all_items: List[Dict[str, str]] = []

        for cand, exp in self._decompose_candidates(ch):

            all_items.append({"type": "chaizi", "text": cand, "explain": exp})

        for cand, exp in self._fanti_candidates(ch):

            all_items.append({"type": "fanti", "text": cand, "explain": exp})

        for cand, exp in self._similar_candidates(ch):

            all_items.append({"type": "similar", "text": cand, "explain": exp})

        return {"char": ch, "candidates": all_items}

    def _term_uniform(self, term: str, mode: str) -> List[Dict[str, Any]]:

        """
        mode: "chaizi" | "fanti" | "similar"

        规则：
        - 若某字在该模式下无候选：
            * 若该字在三种机制下“完全无候选”，允许保留原字并解释保持不变
            * 否则（该字其实在其他机制下有候选），这里不输出“保持不变”，直接不保留该字（导致该组合不成立）
              —— 实际效果：uniform 强制“尽量用同一种变换”，不让有候选的字回退成 keep
        - 若整词该 mode 下没有任何实际变换，则不输出该 mode
        """

        per_char: List[List[Tuple[str, str]]] = []

        any_changed = False

        for ch in term:

            if mode == "chaizi":

                cands = self._decompose_candidates(ch)

                mode_cn = "拆字"

            elif mode == "fanti":

                cands = self._fanti_candidates(ch)

                mode_cn = "简繁转换"

            elif mode == "similar":

                cands = self._similar_candidates(ch)

                mode_cn = "形近字"

            else:

                raise ValueError(f"Unknown mode: {mode}")

            if cands:

                any_changed = True

                per_char.append(cands)

            else:

                if not self._has_any_candidates(ch):

                    per_char.append([(ch, f'形变：该字"{ch}"无可用形变，保持不变')])

                else:

                    return []

        if not any_changed:

            return []

        results: List[Dict[str, Any]] = []

        for combo in itertools.product(*per_char):

            text = "".join([x[0] for x in combo])

            explain = [x[1] for x in combo]

            results.append({"type": "形变", "text": text, "explain": explain, "mode": mode})

        return results

    @staticmethod

    def _infer_mechs_from_explains(explains: List[str]) -> List[str]:

        mechs = set()

        for e in explains:

            if "拆字" in e:

                mechs.add("chaizi")

            elif "简繁转换" in e:

                mechs.add("fanti")

            elif "形近字" in e:

                mechs.add("similar")

            elif "保持不变" in e:

                mechs.add("keep")

        if len(mechs) > 1 and "keep" in mechs:

            mechs.remove("keep")

        if not mechs:

            return ["keep"]

        order = ["chaizi", "fanti", "similar", "keep"]

        return [m for m in order if m in mechs]

    def _term_mixed(self, term: str) -> List[Dict[str, Any]]:

        per_char: List[List[Tuple[str, str]]] = []

        for ch in term:

            chaizi = self._decompose_candidates(ch)

            fanti = self._fanti_candidates(ch)

            similar = self._similar_candidates(ch)

            any_cands = bool(chaizi or fanti or similar)

            options: List[Tuple[str, str]] = []

            if not any_cands:

                options.append((ch, f'形变：该字"{ch}"无可用形变，保持不变'))

            options += chaizi

            options += fanti

            options += similar

            seen = set()

            uniq_opts: List[Tuple[str, str]] = []

            for t, e in options:

                if t in seen:

                    continue

                seen.add(t)

                uniq_opts.append((t, e))

            if not uniq_opts:

                uniq_opts = [(ch, f'形变：该字"{ch}"无可用形变，保持不变')]

            per_char.append(uniq_opts)

        results: List[Dict[str, Any]] = []

        for combo in itertools.product(*per_char):

            text = "".join([x[0] for x in combo])

            explain = [x[1] for x in combo]

            mechs = self._infer_mechs_from_explains(explain)

            mode = "+".join(mechs)

            results.append({"type": "形变", "text": text, "explain": explain, "mode": mode})

        return results

    def generate(self, term: str, max_candidates: int = 8) -> Dict[str, Any]:

        per_char_info = [self.char_candidates(ch) for ch in term]

        uniform_all: List[Dict[str, Any]] = []

        for mode in ["chaizi", "fanti", "similar"]:

            uniform_all.extend(self._term_uniform(term, mode))

        mixed_all = self._term_mixed(term)

        def uniq_limit(items: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:

            out: List[Dict[str, Any]] = []

            seen = set()

            for it in items:

                t = it["text"]

                if t in seen:

                    continue

                seen.add(t)

                out.append(it)

                if len(out) >= limit:

                    break

            return out

        return {

            "type": "形变",

            "term": term,

            "per_char": per_char_info,

            "uniform": uniq_limit(uniform_all, max_candidates),

            "mixed": uniq_limit(mixed_all, max_candidates),

        }

_engine_instance: ShapeEngine | None = None

def generate_shape_candidates(term: str, max_candidates: int = 8, seed: int | None = None) -> Dict[str, Any]:

    global _engine_instance

    if _engine_instance is None or (seed is not None):

        _engine_instance = ShapeEngine(seed=seed)

    return _engine_instance.generate(term, max_candidates=max_candidates)

if __name__ == "__main__":

    term = input("请输入词语：").strip()

    out = generate_shape_candidates(term, max_candidates=8, seed=42)

    print(json.dumps(out, ensure_ascii=False, indent=2))
