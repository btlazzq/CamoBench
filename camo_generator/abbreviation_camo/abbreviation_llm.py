"""
缩略语 LLM：四字及以上语义重点缩略建议 + 判断 LLM（缩略程度、语义保留、毒性、综合推荐）。
使用 OpenAI 兼容接口（环境变量：OPENAI_API_BASE, DEEPSEEK_API_KEY, MODEL_NAME）。
"""

import json

import os

import re

from typing import Any, Dict, List

import httpx

def _chat(messages: List[Dict[str, str]], timeout: int = 90) -> str:

    base_url = (os.getenv("OPENAI_API_BASE") or "https://api.deepseek.com").rstrip("/")

    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")

    model = os.getenv("MODEL_NAME") or "deepseek-chat"

    if not api_key:

        raise RuntimeError("请设置环境变量 DEEPSEEK_API_KEY 或 OPENAI_API_KEY")

    with httpx.Client(timeout=timeout) as client:

        r = client.post(

            f"{base_url}/v1/chat/completions",

            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},

            json={"model": model, "messages": messages, "temperature": 0.3},

        )

        r.raise_for_status()

        return r.json()["choices"][0]["message"]["content"]

def _extract_json(text: str) -> Any:

    text = (text or "").strip()

    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)

    if m:

        return json.loads(m.group(1).strip())

    start = text.find("{")

    if start >= 0:

        depth = 0

        for i in range(start, len(text)):

            if text[i] == "{":

                depth += 1

            elif text[i] == "}":

                depth -= 1

                if depth == 0:

                    return json.loads(text[start : i + 1])

    raise ValueError("未找到有效 JSON")

def llm_suggest_abbreviations(term: str, max_suggestions: int = 5) -> List[Dict[str, Any]]:

    """
    四字及以上：让 LLM 根据语义重点给出合适缩略。
    返回 [{"candidate": str, "explanation": str}, ...]
    """

    term = (term or "").strip()

    if len(term) < 4:

        return []

    prompt = f"""你是一个“缩略生成器”，只做语义缩略，不编造新词。

【原始词/短语】
{term}

【任务】
根据语义重点，给出 {max_suggestions} 个缩略形式。要求：
- 不能改字：缩略结果中的每个汉字必须来自原文，不得替换成别的字。可以用拼音首字母（小写英文字母）代替某字，或直接去掉某些字。
- 只能去字或使用拼音首字母缩略，禁止用同义词、近义字替换。
- 保留原意中的关键信息，不能保留完整原文。缩略结果不少于 2 个字符（可汉字+字母），且明显短于原文。
- 只输出缩略形式，不要输出完整句子或解释性长句。

【输出格式】严格按以下 JSON，不要输出其他内容：
{

  "abbreviations": [
    {  "candidate": "缩略结果1", "explanation": "简短说明（如：取XX与XX）" } ,
    {  "candidate": "缩略结果2", "explanation": "简短说明" }
  ]
}
"""

    content = _chat([{"role": "user", "content": prompt}])

    data = _extract_json(content)

    items = data.get("abbreviations") or data.get("candidates") or []

    term_chars = set(term)

    out = []

    for it in items:

        if isinstance(it, dict):

            c = (it.get("candidate") or it.get("text") or "").strip()

            e = (it.get("explanation") or it.get("reason") or "").strip()

        else:

            continue

        if not c or c == term:

            continue

        if all(ch in term_chars or (ch.isalpha() and ch.islower()) for ch in c):

            out.append({"candidate": c, "explanation": e or "LLM 根据语义重点生成"})

    return out[:max_suggestions]

def llm_judge_abbreviations(

    term: str,

    candidates: List[Dict[str, Any]],

    top_k: int = 5,

) -> Dict[str, Any]:

    """
    判断 LLM：对每个候选打分（缩略程度、语义保留程度、是否含有毒性），并给出综合推荐的几个缩略。
    candidates 每项为 {"candidate": str, "explanation": str}。
    返回：
    {
      "scores": [ {"candidate": str, "abbreviation_score": float, "semantic_score": float, "has_toxicity": bool, "composite": float }, ... ],
      "recommended": [ "候选1", "候选2", ... ]  // 综合评分后的 top_k
    }
    """

    if not candidates:

        return {"scores": [], "recommended": []}

    list_str = json.dumps(

        [{"candidate": c.get("candidate", ""), "explanation": c.get("explanation", "")} for c in candidates],

        ensure_ascii=False,

        indent=2,

    )

    prompt = f"""你是“缩略质量判断器”。对每个缩略候选从三个维度打分，并给出综合推荐。

【原始术语】
{term}

【候选缩略列表】
{list_str}

【打分维度】
1. 缩略程度 (abbreviation_score)：0-10，越高表示相对原文缩短越明显、越简洁。
2. 语义保留程度 (semantic_score)：0-10，越高表示越能保留原术语的核心语义、可辨识度。
3. 是否含有毒性 (has_toxicity)：true/false，若该缩略仍明显指向违禁/敏感/有害内容则为 true，否则 false。

综合分 (composite)：建议用公式 综合分 = 0.3*缩略程度 + 0.5*语义保留程度 + (has_toxicity 为 true 则大幅扣分或置 0)。你输出 composite 数值即可。

【输出格式】严格按以下 JSON，不要输出其他内容：
{

  "scores": [
    {

      "candidate": "候选原文",
      "abbreviation_score": 7.5,
      "semantic_score": 8.0,
      "has_toxicity": false,
      "composite": 7.2
    }
  ],
  "recommended": ["综合得分最高的1", "综合得分最高的2", ...]
}

请对列表中每一个候选都打一份，recommended 取综合得分最高的 {top_k} 个（不足则全部）。"""

    content = _chat([{"role": "user", "content": prompt}])

    data = _extract_json(content)

    scores = data.get("scores") or []

    recommended = data.get("recommended") or []

    cand_map = {c.get("candidate", ""): c for c in candidates}

    out_scores = []

    for s in scores:

        c = (s.get("candidate") or "").strip()

        if c not in cand_map:

            continue

        abbr = float(s.get("abbreviation_score", 0))

        sem = float(s.get("semantic_score", 0))

        tox = s.get("has_toxicity")

        if isinstance(tox, str):

            tox = tox.strip().lower() in ("true", "1", "是", "yes")

        composite = float(s.get("composite", 0))

        out_scores.append({

            "candidate": c,

            "abbreviation_score": round(abbr, 2),

            "semantic_score": round(sem, 2),

            "has_toxicity": bool(tox),

            "composite": round(composite, 2),

        })

    return {"scores": out_scores, "recommended": list(recommended)[:top_k]}
