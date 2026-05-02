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

    # =========================
    # 单字：拆字候选
    # =========================
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

    # =========================
    # 单字：简繁候选（简->繁）
    # =========================
    def _fanti_candidates(self, ch: str) -> List[Tuple[str, str]]:
        if ch not in self.fanti_map:
            return []
        cand = self.fanti_map[ch]
        exp = f'形变：将"{ch}"简繁转换为"{cand}"'
        return [(cand, exp)]

    # =========================
    # 单字：形近字候选（随机取最多3个）
    # =========================
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

    # =========================
    # 单字：是否有任意形变候选
    # =========================
    def _has_any_candidates(self, ch: str) -> bool:
        return bool(self._decompose_candidates(ch) or self._fanti_candidates(ch) or self._similar_candidates(ch))

    # =========================
    # 单字：汇总三类候选（候选列表+解释）
    # =========================
    def char_candidates(self, ch: str) -> Dict[str, Any]:
        all_items: List[Dict[str, str]] = []

        for cand, exp in self._decompose_candidates(ch):
            all_items.append({"type": "chaizi", "text": cand, "explain": exp})

        for cand, exp in self._fanti_candidates(ch):
            all_items.append({"type": "fanti", "text": cand, "explain": exp})

        for cand, exp in self._similar_candidates(ch):
            all_items.append({"type": "similar", "text": cand, "explain": exp})

        return {"char": ch, "candidates": all_items}

    # =========================
    # 词级：全词统一变换
    # =========================
    def _term_uniform(self, term: str, mode: str) -> List[Dict[str, Any]]:
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

    # =========================
    # 词级：混搭变换（每字可不同机制）
    # =========================
    def _term_mixed(self, term: str) -> List[Dict[str, Any]]:
        per_char: List[List[Tuple[str, str]]] = []

        for ch in term:
            # 三种机制候选
            chaizi = self._decompose_candidates(ch)
            fanti = self._fanti_candidates(ch)
            similar = self._similar_candidates(ch)
            any_cands = bool(chaizi or fanti or similar)

            options: List[Tuple[str, str]] = []
            # 只有当该字完全没有任何候选时，才允许 keep
            if not any_cands:
                options.append((ch, f'形变：该字"{ch}"无可用形变，保持不变'))

            options += chaizi
            options += fanti
            options += similar

            # 去重（同一输出只保留一条解释）
            seen = set()
            uniq_opts: List[Tuple[str, str]] = []
            for t, e in options:
                if t in seen:
                    continue
                seen.add(t)
                uniq_opts.append((t, e))

            # 极端情况：如果 options 为空（理论上不会），兜底 keep
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

    # =========================
    # 主接口
    # =========================
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