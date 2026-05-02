import json
import itertools
import os
import random
from typing import Any, Dict, List

try:
    from pypinyin import pinyin as _pypinyin, Style as _Style
except Exception as e:  # pragma: no cover
    _pypinyin = None
    _Style = None
    _PYPINYIN_IMPORT_ERROR = e
else:
    _PYPINYIN_IMPORT_ERROR = None


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):  # 兼容 [ { ... } ]
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

    # -------------------------
    # 拼音：逐字 + 全词
    # -------------------------
    def get_pinyin(self, term: str, style: str = "normal", separator: str = " ") -> Dict[str, Any]:
        if _pypinyin is None:
            raise ImportError('缺少依赖：pypinyin。请先安装：pip install pypinyin') from _PYPINYIN_IMPORT_ERROR

        st = _Style.TONE3 if style == "tone3" else _Style.NORMAL
        arr = _pypinyin(term, style=st, heteronym=False, errors="default")  # [['hai'], ['luo'], ...]
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

    # -------------------------
    # 近音：单字候选列表+解释
    # -------------------------
    def char_candidates(self, ch: str) -> Dict[str, Any]:
        lst = self.yin_map.get(ch) or []
        lst = [x for x in dict.fromkeys(lst) if x and x != ch]  # 去重 & 过滤自身
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

    # -------------------------
    # 近音：字级组合生成
    # -------------------------
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

    # -------------------------
    # 混合：拼音 + 近音
    # -------------------------
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

            # 记录 per_char（展示给你看有哪些可用项）
            per_char.append({
                "char": ch,
                "candidates": (
                    [{"type": "pinyin", **o} for o in py_opts] +
                    [{"type": "yin", **o} for o in yin_opts]
                )
            })

            # 组合选项（只用 text/explain）
            opts = []
            opts += py_opts
            opts += yin_opts

            # 若该字两者都没有，才允许 keep
            if not any_opts:
                opts = [{"text": ch, "explain": f'混合：该字"{ch}"无拼音/近音可用，保持不变'}]

            # 去重（同一输出只留一条解释）
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

    # -------------------------
    # 总接口：拼音信息 + 近音 + 混合
    # -------------------------
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
