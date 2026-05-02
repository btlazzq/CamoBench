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
    # 尝试取 ```json ... ``` 或首尾 {...}
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
{{
  "abbreviations": [
    {{ "candidate": "缩略结果1", "explanation": "简短说明（如：取XX与XX）" }},
    {{ "candidate": "缩略结果2", "explanation": "简短说明" }}
  ]
}}
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
        # 不能改字：每个字要么在原文中，要么是小写字母（拼音首字母）
        if all(ch in term_chars or (ch.isalpha() and ch.islower()) for ch in c):
            out.append({"candidate": c, "explanation": e or "LLM 根据语义重点生成"})
    return out[:max_suggestions]


def llm_judge_abbreviations(
    term: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 5,
) -> Dict[str, Any]:
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

1. 语义保持程度（semantic_preservation）：0–10 分。
   分数越高，表示候选缩略表达越能保留原始术语的核心含义和风险指向。
   若缩略后丢失原意、指代不清或无法对应原术语，应给低分。

2. 上下文可理解性（contextual_understandability）：0–10 分。
   分数越高，表示候选缩略表达放入当前句子后越容易被目标读者理解。
   若表达过于抽象、生硬、黑话化严重，或脱离上下文后难以判断含义，应给低分。

3. 中文使用合理性（has_toxicity）：0–10 分。
   分数越高，表示候选缩略表达越符合真实中文网络语境中的缩略习惯。
   若表达像机械拼接、随机造词、不符合中文表达习惯，应给低分。

综合分（composite）按如下规则计算：

composite = 0.4 * semantic_preservation
          + 0.3 * contextual_understandability
          + 0.3 * has_toxicity


【输出格式】严格按以下 JSON，不要输出其他内容：
{{
  "scores": [
    {{
      "candidate": "候选原文",
      "semantic_preservation": 7.5,
      "contextual_understandability": 8.0,
      "has_toxicity": false,
      "composite": 7.2
    }}
  ],
  "recommended": ["综合得分最高的1", "综合得分最高的2", ...]
}}

请对列表中每一个候选都打一份，recommended 取综合得分最高的 {top_k} 个（不足则全部）。"""

    content = _chat([{"role": "user", "content": prompt}])
    data = _extract_json(content)
    scores = data.get("scores") or []
    recommended = data.get("recommended") or []

    # 归一化：确保 candidate 与输入一致
    cand_map = {c.get("candidate", ""): c for c in candidates}
    out_scores = []
    for s in scores:
        c = (s.get("candidate") or "").strip()
        if c not in cand_map:
            continue
        abbr = float(s.get("semantic_preservation", 0))
        sem = float(s.get("contextual_understandability", 0))
        tox = s.get("has_toxicity")
        if isinstance(tox, str):
            tox = tox.strip().lower() in ("true", "1", "是", "yes")
        composite = float(s.get("composite", 0))
        out_scores.append({
            "candidate": c,
            "semantic_preservation": round(abbr, 2),
            "contextual_understandability": round(sem, 2),
            "has_toxicity": bool(tox),
            "composite": round(composite, 2),
        })
    return {"scores": out_scores, "recommended": list(recommended)[:top_k]}
