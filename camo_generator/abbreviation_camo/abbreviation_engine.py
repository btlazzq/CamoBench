"""
缩略语伪装引擎（abbreviation_camo）

规则：
- 两字词：拼音首字母 + 保留一字，另一字用拼音首字母（如：冰毒→b毒、冰d、bd）
- 三字词：去掉一字 + 含拼音首字母的变体（如：海洛因→海因、洛因、hl因、海ly、hly 等）
- 四字及以上：由 LLM 根据语义重点生成缩略（见 abbreviation_llm）
"""

from typing import List, Dict, Any, Optional

try:

    from pypinyin import pinyin as _pypinyin, Style as _Style

except Exception:

    _pypinyin = None

    _Style = None

def _get_pinyin_initials(s: str) -> List[str]:

    """取每个码点的拼音首字母，小写（逐字查询，避免整串拉丁连写时 pypinyin 合并为一段导致长度与 len(s) 不一致）。"""

    if not s:

        return []

    if _pypinyin is None:

        return [""] * len(s)

    out: List[str] = []

    for ch in s:

        arr = _pypinyin(ch, style=_Style.NORMAL, heteronym=False, errors="default")

        py_list = arr[0] if arr else []

        py = (py_list[0] if py_list else "")[:1].lower()

        out.append(py if py.isalpha() else "")

    return out

def generate_abbreviation_candidates(term: str) -> List[Dict[str, Any]]:

    """
    对 term 生成缩略候选（仅规则：两字、三字）。
    四字及以上不在此生成，由 LLM 生成。
    返回列表，每项为 {"candidate": str, "explanation": str}。
    """

    term = (term or "").strip()

    if not term:

        return []

    n = len(term)

    result: List[Dict[str, Any]] = []

    seen: set = set()

    initials = _get_pinyin_initials(term)

    def add(candidate: str, explanation: str) -> None:

        if not candidate or candidate == term or candidate in seen:

            return

        seen.add(candidate)

        result.append({"candidate": candidate, "explanation": explanation})

    if n == 2:

        i0, i1 = initials[0], initials[1]

        if i1:

            add(term[0] + i1, f'缩略：首字保留+第二字拼音首字母，"{term}"→"{term[0]}{i1}"')

        if i0:

            add(i0 + term[1], f'缩略：首字拼音首字母+第二字保留，"{term}"→"{i0}{term[1]}"')

        if i0 and i1:

            add(i0 + i1, f'缩略：双字拼音首字母，"{term}"→"{i0}{i1}"')

        return result

    if n == 3:

        add(term[0] + term[1], f'缩略：三字去末字，"{term}"→"{term[0]}{term[1]}"')

        add(term[0] + term[2], f'缩略：三字去中间，"{term}"→"{term[0]}{term[2]}"')

        add(term[1] + term[2], f'缩略：三字去首字，"{term}"→"{term[1]}{term[2]}"')

        h, l, y = initials[0], initials[1], initials[2]

        if h:

            add(h + term[1] + term[2], f'缩略：首字首字母+后两字，"{term}"→"{h}{term[1]}{term[2]}"')

            add(h + term[2], f'缩略：首字首字母+末字，"{term}"→"{h}{term[2]}"')

        if l:

            add(term[0] + l + term[2], f'缩略：首字+中间首字母+末字，"{term}"→"{term[0]}{l}{term[2]}"')

            add(term[0] + l + y, f'缩略：首字+中间与末字首字母，"{term}"→"{term[0]}{l}{y}"')

        if y:

            add(term[0] + term[1] + y, f'缩略：前两字+末字首字母，"{term}"→"{term[0]}{term[1]}{y}"')

            add(term[1] + y, f'缩略：第二字+末字首字母，"{term}"→"{term[1]}{y}"')

        if h and l:

            add(h + l + term[2], f'缩略：前两字首字母+末字，"{term}"→"{h}{l}{term[2]}"')

        if h and y:

            add(h + term[1] + y, f'缩略：首字首字母+第二字+末字首字母，"{term}"→"{h}{term[1]}{y}"')

        if l and y:

            add(term[0] + l + y, f'缩略：首字+后两字首字母，"{term}"→"{term[0]}{l}{y}"')

        if h and l and y:

            add(h + l + y, f'缩略：三字拼音首字母，"{term}"→"{h}{l}{y}"')

        return result

    if n == 4:

        add(term[0] + term[1], f'缩略：取前两字，"{term}"→"{term[0]}{term[1]}"')

        add(term[0] + term[-1], f'缩略：取首尾字，"{term}"→"{term[0]}{term[-1]}"')

    return result

def needs_llm_abbreviation(term: str) -> bool:

    """是否应由 LLM 生成语义重点缩略（四字及以上）。"""

    return len((term or "").strip()) >= 4
