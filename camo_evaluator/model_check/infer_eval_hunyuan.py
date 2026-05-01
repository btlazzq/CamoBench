from __future__ import annotations

import argparse

import csv

import json

import os

import time

from concurrent.futures import ThreadPoolExecutor, as_completed

from typing import Any, Dict, List, Optional, Tuple

import httpx

from sklearn.metrics import (

    accuracy_score,

    classification_report,

    confusion_matrix,

    f1_score,

)

from tqdm import tqdm

SYSTEM_PROMPT = "You are a helpful assistant."

USER_TASK_PROMPT = """你是中文内容审核专家。

【任务】
对给定文本进行五分类：
涉毒 / 涉赌 / 涉黄 / 涉诈 / 不违规

【原则】
类别取值必须是上述五者之一。
“涉诈”表示不属于涉毒、涉赌、涉黄的其他违规行为，如诈骗、黑灰产及非法牟利等。

【输出】
只输出 JSON，不要多余文本，且只包含一个字段 label：
{
  "label": "涉毒 / 涉赌 / 涉黄 / 涉诈 / 不违规"
}
"""

ALLOWED_LABELS = {"涉毒", "涉赌", "涉黄", "其他", "不违规"}

LABEL_MAP = {

    "涉黄赌毒": "其他",

    "黄": "涉黄",

    "赌": "涉赌",

    "毒": "涉毒",

    "正常": "不违规",

    "无违规": "不违规",

    "未违规": "不违规",

}

EVAL_LABEL_ORDER = ["不违规", "涉黄", "涉赌", "涉毒", "其他"]

EVAL_LABEL2ID = {x: i for i, x in enumerate(EVAL_LABEL_ORDER)}

PARSE_FAIL_LABEL = "__解析失败__"

DEFAULT_API_BASE = "https://api.hunyuan.cloud.tencent.com/v1"

DEFAULT_COMPLETIONS_PATH = "/chat/completions"

def _norm_str(x: Any) -> str:

    if x is None:

        return ""

    return str(x).strip()

def normalize_minimal_output(obj: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:

    if not isinstance(obj, dict):

        return None, "parsed_not_dict"

    if "label" not in obj:

        return None, "missing_label"

    label = _norm_str(obj.get("label", ""))

    if not label:

        return None, "missing_label"

    label = LABEL_MAP.get(label, label)

    if label not in ALLOWED_LABELS:

        return None, f"bad_label:{label}"

    return {"label": label}, None

def load_jsonl(path: str) -> List[Dict[str, Any]]:

    rows: List[Dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as f:

        for lineno, line in enumerate(f, start=1):

            line = line.strip()

            if not line:

                continue

            try:

                obj = json.loads(line)

            except Exception as e:

                raise ValueError(f"[{path}] line {lineno} not valid JSON: {e}\n{line[:200]}") from e

            if not isinstance(obj, dict):

                raise ValueError(f"[{path}] line {lineno} must be a JSON object (dict).")

            rows.append(obj)

    return rows

def dump_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:

        for r in rows:

            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def save_json(obj: Dict[str, Any], path: str) -> None:

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:

        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_confusion_matrix_csv(cm: List[List[int]], labels: List[str], path: str) -> None:

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(["gold\\pred"] + labels)

        for i, lab in enumerate(labels):

            writer.writerow([lab] + cm[i])

def normalize_gold_label(x: Any) -> Optional[str]:

    s = _norm_str(x)

    if not s:

        return None

    s = LABEL_MAP.get(s, s)

    if s in ALLOWED_LABELS:

        return s

    return None

def pred_label_for_eval(out_row: Dict[str, Any]) -> str:

    if out_row.get("ok") and out_row.get("label") in EVAL_LABEL2ID:

        return str(out_row["label"])

    return PARSE_FAIL_LABEL

def run_eval_and_save(

    rows_in: List[Dict[str, Any]],

    outputs: List[Dict[str, Any]],

    label_field: str,

    metrics_path: str,

    cm_csv_path: str,

) -> None:

    if len(rows_in) != len(outputs):

        raise ValueError(f"行数不一致: input={len(rows_in)} outputs={len(outputs)}")

    gold_strs: List[Optional[str]] = []

    for r in rows_in:

        if label_field not in r:

            gold_strs.append(None)

        else:

            gold_strs.append(normalize_gold_label(r.get(label_field)))

    for i, (r, g) in enumerate(zip(rows_in, gold_strs)):

        raw = r.get(label_field, None)

        if raw is None:

            continue

        if _norm_str(raw) == "":

            continue

        if g is None:

            raise ValueError(

                f"行 {i}: {label_field}={raw!r} 无法映射为"

                f"「不违规/涉黄/涉赌/涉毒/其他」之一（与 infer_eval_qwen_cls_lora 一致）"

            )

    gold_available = all(g is not None for g in gold_strs)

    if not gold_available:

        if any(g is not None for g in gold_strs):

            print(

                "[WARN] 部分样本缺少有效 gold label，跳过评估（需每行都有可识别的 label）。"

            )

        else:

            print("[INFO] 输入未提供完整 gold label，已跳过指标评估。")

        return

    y_true: List[int] = []

    y_pred: List[int] = []

    eval_names = EVAL_LABEL_ORDER + [PARSE_FAIL_LABEL]

    fail_id = len(EVAL_LABEL_ORDER)

    for g, out_row in zip(gold_strs, outputs):

        assert g is not None

        y_true.append(EVAL_LABEL2ID[g])

        pl = pred_label_for_eval(out_row)

        y_pred.append(EVAL_LABEL2ID[pl] if pl in EVAL_LABEL2ID else fail_id)

    labels_all = list(range(len(eval_names)))

    acc = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(

        y_true,

        y_pred,

        average="macro",

        labels=list(range(len(EVAL_LABEL_ORDER))),

        zero_division=0,

    )

    report = classification_report(

        y_true,

        y_pred,

        labels=labels_all,

        target_names=eval_names,

        digits=6,

        output_dict=True,

        zero_division=0,

    )

    cm = confusion_matrix(y_true, y_pred, labels=labels_all)

    n_parse_fail = sum(1 for p in y_pred if p == fail_id)

    metrics: Dict[str, Any] = {

        "num_samples": len(y_true),

        "num_parse_fail_pred": n_parse_fail,

        "accuracy": round(float(acc), 6),

        "macro_f1": round(float(macro_f1), 6),

        "labels": eval_names,

        "classification_report": report,

        "confusion_matrix": cm.tolist(),

    }

    save_json(metrics, metrics_path)

    save_confusion_matrix_csv(cm.tolist(), eval_names, cm_csv_path)

    print(f"[OK] Saved metrics to: {metrics_path}")

    print(f"[OK] Saved confusion matrix csv to: {cm_csv_path}")

    print("\n===== Eval Metrics =====")

    print(f"Samples   : {len(y_true)}")

    print(f"解析失败(作独立预测类): {n_parse_fail}")

    print(f"Accuracy  : {acc:.6f}")

    print(f"Macro-F1  : {macro_f1:.6f}")

    print("\n===== Per-class F1 =====")

    for lab in eval_names:

        row = report.get(lab)

        if not isinstance(row, dict):

            continue

        f1v = row["f1-score"]

        p = row["precision"]

        r = row["recall"]

        print(f"{lab}: P={p:.6f} R={r:.6f} F1={f1v:.6f}")

def align_preds_to_rows(

    rows_in: List[Dict[str, Any]],

    pred_rows: List[Dict[str, Any]],

) -> List[Dict[str, Any]]:

    pred_by_id: Dict[Any, Dict[str, Any]] = {}

    for p in pred_rows:

        if "id" not in p:

            raise ValueError("预测 jsonl 每行需含 'id' 字段，便于与输入对齐")

        pred_by_id[p["id"]] = p

    out: List[Dict[str, Any]] = []

    for ig, r in enumerate(rows_in):

        pid = r.get("id", ig)

        if pid not in pred_by_id:

            raise KeyError(

                f"输入第 {ig} 行 id={pid!r} 在预测文件中不存在；"

                f"请确认切分推理时保留了原始 id 且合并时未丢行"

            )

        out.append(pred_by_id[pid])

    return out

def safe_json_parse(s: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:

    if not s:

        return None, "empty_output"

    s = s.strip()

    try:

        obj = json.loads(s)

        if isinstance(obj, dict):

            normed, err = normalize_minimal_output(obj)

            return normed, err

    except Exception:

        pass

    l = s.find("{")

    r = s.rfind("}")

    if l >= 0 and r > l:

        sub = s[l : r + 1]

        try:

            obj = json.loads(sub)

            if isinstance(obj, dict):

                normed, err = normalize_minimal_output(obj)

                return normed, err

        except Exception as e:

            return None, f"json_extract_parse_fail: {e}"

    return None, "json_not_found"

def strip_text_prefix(text: str) -> str:

    s = (text or "").strip()

    if s.startswith("文本："):

        return s[len("文本：") :].strip()

    return s

def build_messages(text: str) -> List[Dict[str, str]]:

    t_clean = strip_text_prefix(text)

    user_prompt = (

        f"{SYSTEM_PROMPT}\n\n"

        f"{USER_TASK_PROMPT}\n\n"

        f"待分类文本：{t_clean}"

    )

    return [

        {"role": "user", "content": user_prompt},

    ]

def _completions_url(api_base: str) -> str:

    base = api_base.rstrip("/")

    if base.endswith("/v1"):

        return base + "/chat/completions"

    return base + DEFAULT_COMPLETIONS_PATH

def hunyuan_chat_once(

    client: httpx.Client,

    url: str,

    api_key: str,

    model: str,

    messages: List[Dict[str, str]],

    max_tokens: int,

    temperature: float,

    top_p: float,

    max_retries: int,

) -> Tuple[str, Optional[str]]:

    headers = {

        "Authorization": f"Bearer {api_key}",

        "Content-Type": "application/json",

    }

    body: Dict[str, Any] = {

        "model": model,

        "messages": messages,

        "max_tokens": max_tokens,

        "stream": False,

    }

    if temperature > 0:

        body["temperature"] = temperature

        body["top_p"] = top_p

    else:

        body["temperature"] = 0.0

    last_err: Optional[str] = None

    for attempt in range(max_retries + 1):

        try:

            resp = client.post(url, headers=headers, json=body)

            if resp.status_code == 200:

                data = resp.json()

                choices = data.get("choices") or []

                if not choices:

                    return "", "api_empty_choices"

                msg = choices[0].get("message") or {}

                content = msg.get("content")

                if content is None:

                    return "", "api_no_content"

                return str(content), None

            if resp.status_code in (429, 500, 502, 503):

                last_err = f"http_{resp.status_code}:{resp.text[:500]}"

                time.sleep(min(2 ** attempt, 30))

                continue

            return "", f"http_{resp.status_code}:{resp.text[:500]}"

        except httpx.TimeoutException as e:

            last_err = f"timeout:{e}"

            time.sleep(min(2 ** attempt, 30))

        except Exception as e:

            return "", f"request_error:{type(e).__name__}:{e}"

    return "", last_err or "api_retries_exhausted"

def infer_one_row(

    client: httpx.Client,

    url: str,

    api_key: str,

    model: str,

    max_tokens: int,

    temperature: float,

    top_p: float,

    max_retries: int,

    row_index: int,

    row: Dict[str, Any],

    batch_offset: int,

) -> Tuple[int, Dict[str, Any]]:

    _id = row.get("id", batch_offset + row_index)

    text = _norm_str(row.get("text", ""))

    messages = build_messages(text)

    t0 = time.time()

    raw, api_err = hunyuan_chat_once(

        client=client,

        url=url,

        api_key=api_key,

        model=model,

        messages=messages,

        max_tokens=max_tokens,

        temperature=temperature,

        top_p=top_p,

        max_retries=max_retries,

    )

    elapsed_ms = (time.time() - t0) * 1000.0

    if api_err:

        out: Dict[str, Any] = {

            "id": _id,

            "text": text,

            "label": None,

            "raw": raw,

            "parsed": None,

            "ok": False,

            "error": f"api:{api_err}",

            "time_ms": round(elapsed_ms, 3),

        }

        return row_index, out

    parsed, err = safe_json_parse(raw)

    out = {

        "id": _id,

        "text": text,

        "label": parsed.get("label") if parsed else None,

        "raw": raw,

        "parsed": parsed,

        "ok": err is None,

        "error": err,

        "time_ms": round(elapsed_ms, 3),

    }

    return row_index, out

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--input_jsonl", required=True)

    ap.add_argument(

        "--output_jsonl",

        default="",

        help="推理输出预测 jsonl；若同时指定 --output_dir，只写文件名即可",

    )

    ap.add_argument(

        "--output_dir",

        default="",

        help="结果目录：与 --output_jsonl 联用时，实际路径为 output_dir/basename(output_jsonl)",

    )

    ap.add_argument(

        "--eval_only",

        action="store_true",

        help="仅评估：需 --pred_jsonl 与带 gold 的 --input_jsonl 按 id 对齐",

    )

    ap.add_argument("--pred_jsonl", default="")

    ap.add_argument(

        "--api_key",

        default="",

        help="默认读取环境变量 HUNYUAN_API_KEY",

    )

    ap.add_argument(

        "--api_base",

        default=os.environ.get("HUNYUAN_API_BASE", DEFAULT_API_BASE).strip() or DEFAULT_API_BASE,

        help=f"例如 {DEFAULT_API_BASE}",

    )

    ap.add_argument(

        "--model",

        default="hunyuan-turbos-latest",

        help="如 hunyuan-turbos-latest、hunyuan-t1-latest（以腾讯云文档为准）",

    )

    ap.add_argument("--batch_size", type=int, default=8, help="并发请求数（每批并行条数）")

    ap.add_argument("--max_tokens", type=int, default=512)

    ap.add_argument("--temperature", type=float, default=0.0)

    ap.add_argument("--top_p", type=float, default=0.9)

    ap.add_argument("--timeout", type=float, default=120.0, help="单条请求超时（秒）")

    ap.add_argument("--max_retries", type=int, default=3, help="429/5xx/超时重试次数")

    ap.add_argument("--label_field", type=str, default="label")

    ap.add_argument("--no_eval", action="store_true")

    args = ap.parse_args()

    def _resolve_under_dir(path: str, out_dir: str) -> str:

        if not out_dir.strip():

            return path

        if os.path.dirname(path):

            return path

        return os.path.join(os.path.abspath(out_dir), path)

    if args.eval_only:

        if not args.pred_jsonl.strip():

            ap.error("--eval_only 需要指定 --pred_jsonl")

        pred_path = _resolve_under_dir(args.pred_jsonl.strip(), args.output_dir.strip())

        rows = load_jsonl(args.input_jsonl)

        pred_rows = load_jsonl(pred_path)

        try:

            outputs = align_preds_to_rows(rows, pred_rows)

        except (KeyError, ValueError) as e:

            raise SystemExit(f"[eval_only] 对齐失败: {e}") from e

        if len(pred_rows) != len(rows):

            print(

                f"[WARN] 预测行数 {len(pred_rows)} != 输入行数 {len(rows)}；"

                f"已按输入顺序取 {len(outputs)} 条对齐评估（请确认无重复/遗漏 id）"

            )

        out_dir = os.path.dirname(os.path.abspath(pred_path)) or "."

        stem, _ = os.path.splitext(os.path.basename(pred_path))

        run_eval_and_save(

            rows_in=rows,

            outputs=outputs,

            label_field=args.label_field,

            metrics_path=os.path.join(out_dir, f"{stem}_metrics.json"),

            cm_csv_path=os.path.join(out_dir, f"{stem}_confusion_matrix.csv"),

        )

        return

    api_key = args.api_key.strip() or os.environ.get("HUNYUAN_API_KEY", "").strip()

    if not api_key:

        ap.error("请设置 HUNYUAN_API_KEY 或传入 --api_key")

    out_dir_opt = args.output_dir.strip()

    if out_dir_opt:

        os.makedirs(out_dir_opt, exist_ok=True)

        od_abs = os.path.abspath(out_dir_opt)

        if args.output_jsonl.strip():

            args.output_jsonl = os.path.join(od_abs, os.path.basename(args.output_jsonl))

        else:

            args.output_jsonl = os.path.join(od_abs, "predictions.jsonl")

    elif not args.output_jsonl.strip():

        ap.error("推理模式需要 --output_jsonl，或同时指定 --output_dir")

    url = _completions_url(args.api_base)

    rows = load_jsonl(args.input_jsonl)

    print(f"[API] url={url} model={args.model}")

    print(f"[Data] loaded {len(rows)} rows from {args.input_jsonl}")

    for i, r in enumerate(rows):

        if "text" not in r:

            raise ValueError(f"Row {i} missing key 'text': {r}")

        if not strip_text_prefix(_norm_str(r.get("text", ""))):

            raise ValueError(f"Row {i} has empty text: {r}")

    outputs: List[Optional[Dict[str, Any]]] = [None] * len(rows)

    bs = max(1, args.batch_size)

    with httpx.Client(timeout=args.timeout) as client:

        pbar = tqdm(range(0, len(rows), bs), desc="Hunyuan API")

        for start in pbar:

            batch = rows[start : start + bs]

            with ThreadPoolExecutor(max_workers=len(batch)) as ex:

                futs = [

                    ex.submit(

                        infer_one_row,

                        client,

                        url,

                        api_key,

                        args.model,

                        args.max_tokens,

                        args.temperature,

                        args.top_p,

                        args.max_retries,

                        k,

                        batch[k],

                        start,

                    )

                    for k in range(len(batch))

                ]

                for fut in as_completed(futs):

                    idx, out_row = fut.result()

                    outputs[start + idx] = out_row

            ok_batch = sum(1 for j in range(len(batch)) if outputs[start + j] and outputs[start + j]["ok"])

            pbar.set_postfix_str(f"本批 ok {ok_batch}/{len(batch)}")

    assert all(x is not None for x in outputs)

    outputs_list = outputs

    dump_jsonl(args.output_jsonl, outputs_list)

    ok_cnt = sum(1 for r in outputs_list if r.get("ok"))

    print(f"[OK] wrote {len(outputs_list)} rows to {args.output_jsonl} | ok={ok_cnt} | bad={len(outputs_list)-ok_cnt}")

    if not args.no_eval:

        out_dir = os.path.dirname(os.path.abspath(args.output_jsonl)) or "."

        stem, _ = os.path.splitext(os.path.basename(args.output_jsonl))

        run_eval_and_save(

            rows_in=rows,

            outputs=outputs_list,

            label_field=args.label_field,

            metrics_path=os.path.join(out_dir, f"{stem}_metrics.json"),

            cm_csv_path=os.path.join(out_dir, f"{stem}_confusion_matrix.csv"),

        )

if __name__ == "__main__":

    main()
