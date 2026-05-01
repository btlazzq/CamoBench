from __future__ import annotations

import argparse

import csv

import json

import os

from typing import Any, Dict, List, Optional, Tuple

ALLOWED_LABELS = {"涉毒", "涉赌", "涉黄", "涉诈", "不违规"}

LABEL_MAP = {

    "涉黄赌毒": "涉诈",

    "黄": "涉黄",

    "赌": "涉赌",

    "毒": "涉毒",

    "正常": "不违规",

    "无违规": "不违规",

    "未违规": "不违规",

}

EVAL_LABEL_ORDER = ["不违规", "涉黄", "涉赌", "涉毒", "涉诈"]

EVAL_LABEL2ID = {x: i for i, x in enumerate(EVAL_LABEL_ORDER)}

PARSE_FAIL_LABEL = "__解析失败__"

NUM_CLASSES = len(EVAL_LABEL_ORDER)

EVERY7_KIND_BY_START: Dict[int, str] = {

    1: "原词",

    2: "缩略",

    3: "emoji",

    4: "音",

    5: "义",

    6: "形",

    7: "混合",

}

def _norm_str(x: Any) -> str:

    if x is None:

        return ""

    return str(x).strip()

def normalize_gold_label(x: Any) -> Optional[str]:

    s = _norm_str(x)

    if not s:

        return None

    s = LABEL_MAP.get(s, s)

    if s in ALLOWED_LABELS:

        return s

    return None

def pred_label_for_eval(out_row: Dict[str, Any]) -> str:

    """
    兼容两类预测格式：
    1) infer_eval_* 输出：含 ok / parsed 字段；
    2) 轻量输出：仅含 label 字段。
    评估时只要存在合法 label 即采纳，不再强依赖 ok 字段。
    """

    lab = _norm_str(out_row.get("label"))

    lab = LABEL_MAP.get(lab, lab)

    if lab in EVAL_LABEL2ID:

        return lab

    parsed = out_row.get("parsed")

    if isinstance(parsed, dict):

        lab = _norm_str(parsed.get("label"))

        lab = LABEL_MAP.get(lab, lab)

        if lab in EVAL_LABEL2ID:

            return lab

    if out_row.get("ok") is False:

        return PARSE_FAIL_LABEL

    return PARSE_FAIL_LABEL

def align_preds_to_gold_rows(

    gold_rows: List[Dict[str, Any]],

    pred_rows: List[Dict[str, Any]],

) -> List[Dict[str, Any]]:

    """按 gold 行顺序，用 id（缺省为行下标）对齐预测行。"""

    pred_by_id: Dict[Any, Dict[str, Any]] = {}

    for p in pred_rows:

        if "id" not in p:

            raise ValueError("预测 jsonl 每行需含 'id' 字段以便按 id 对齐")

        pred_by_id[p["id"]] = p

    out: List[Dict[str, Any]] = []

    for i, g in enumerate(gold_rows):

        pid = g.get("id", i)

        if pid not in pred_by_id:

            raise KeyError(

                f"gold 第 {i} 行 id={pid!r} 在预测文件中不存在，请检查合并结果或关闭 --align_by_id"

            )

        out.append(pred_by_id[pid])

    return out

def load_jsonl(path: str) -> List[Dict[str, Any]]:

    rows: List[Dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as f:

        for lineno, line in enumerate(f, start=1):

            line = line.strip()

            if not line:

                continue

            obj = json.loads(line)

            if not isinstance(obj, dict):

                raise ValueError(f"[{path}] line {lineno} must be a JSON object.")

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

        w = csv.writer(f)

        w.writerow(["gold\\pred"] + labels)

        for i, lab in enumerate(labels):

            w.writerow([lab] + cm[i])

def pred_id_for_eval(out_row: Dict[str, Any]) -> int:

    pl = pred_label_for_eval(out_row)

    if pl in EVAL_LABEL2ID:

        return EVAL_LABEL2ID[pl]

    return NUM_CLASSES

def eval_from_pairs(

    gold_strs: List[str],

    pred_ids: List[int],

) -> Tuple[float, float, List[List[int]], Dict[str, Dict[str, float]], int]:

    """gold_strs: 已规范五类字符串；pred_ids: 0..4 或 NUM_CLASSES 表示解析失败。"""

    n = len(gold_strs)

    fail_id = NUM_CLASSES

    eval_names = EVAL_LABEL_ORDER + [PARSE_FAIL_LABEL]

    nlab = len(eval_names)

    cm = [[0] * nlab for _ in range(nlab)]

    correct = 0

    n_parse_fail = 0

    for g, pid in zip(gold_strs, pred_ids):

        ti = EVAL_LABEL2ID[g]

        if pid == fail_id:

            n_parse_fail += 1

        if pid == ti:

            correct += 1

        cm[ti][pid] += 1

    acc = correct / n if n else 0.0

    f1s: List[float] = []

    report: Dict[str, Dict[str, float]] = {}

    for c in range(NUM_CLASSES):

        tp = cm[c][c]

        fp = sum(cm[j][c] for j in range(nlab) if j != c)

        fn = sum(cm[c][j] for j in range(nlab) if j != c)

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        f1s.append(f1)

        report[EVAL_LABEL_ORDER[c]] = {

            "precision": round(p, 6),

            "recall": round(r, 6),

            "f1-score": round(f1, 6),

            "support": float(sum(cm[c][j] for j in range(nlab))),

        }

    report[PARSE_FAIL_LABEL] = {

        "precision": 0.0,

        "recall": 0.0,

        "f1-score": 0.0,

        "support": 0.0,

    }

    macro_f1 = sum(f1s) / NUM_CLASSES if NUM_CLASSES else 0.0

    return acc, macro_f1, cm, report, n_parse_fail

def run_eval_and_save(

    rows_in: List[Dict[str, Any]],

    outputs: List[Dict[str, Any]],

    label_field: str,

    metrics_path: str,

    cm_csv_path: str,

    subset_meta: Optional[Dict[str, Any]] = None,

) -> Dict[str, Any]:

    if len(rows_in) != len(outputs):

        raise ValueError(f"行数不一致: input={len(rows_in)} outputs={len(outputs)}")

    gold_strs: List[str] = []

    for i, r in enumerate(rows_in):

        g = normalize_gold_label(r.get(label_field))

        if g is None:

            raise ValueError(f"行 {i}: 无效 gold {label_field}={r.get(label_field)!r}")

        gold_strs.append(g)

    pred_ids = [pred_id_for_eval(o) for o in outputs]

    acc, macro_f1, cm, report, n_parse_fail = eval_from_pairs(gold_strs, pred_ids)

    eval_names = EVAL_LABEL_ORDER + [PARSE_FAIL_LABEL]

    metrics: Dict[str, Any] = {

        "num_samples": len(gold_strs),

        "num_parse_fail_pred": n_parse_fail,

        "accuracy": round(float(acc), 6),

        "macro_f1": round(float(macro_f1), 6),

        "labels": eval_names,

        "classification_report": report,

        "confusion_matrix": cm,

    }

    if subset_meta:

        metrics.update(subset_meta)

    save_json(metrics, metrics_path)

    save_confusion_matrix_csv(cm, eval_names, cm_csv_path)

    return metrics

def main() -> None:

    script_dir = os.path.dirname(os.path.abspath(__file__))

    root = os.path.dirname(script_dir)

    default_pred = os.path.join(script_dir, "predictions_merged.jsonl")

    default_gold = os.path.join(root, "3w57test.jsonl")

    default_out = os.path.join(script_dir, "camo_test_3500_7")

    ap = argparse.ArgumentParser(description="步长 7 抽样子集并评估（适配 DeepSeek / Qwen infer 输出）")

    ap.add_argument("--pred_jsonl", default=default_pred, help="预测 jsonl（默认 API_test/predictions_merged.jsonl）")

    ap.add_argument("--gold_jsonl", default=default_gold, help="带 label 的 gold jsonl（默认 仓库根 3w57test.jsonl）")

    ap.add_argument(

        "--out_dir",

        default=default_out,

        help="子集 jsonl 与指标输出目录（默认 API_test/camo_test_3500_7）",

    )

    ap.add_argument(

        "--align_by_id",

        action="store_true",

        help="预测行与 gold 行顺序不一致时，按 id 与 gold 对齐后再抽样",

    )

    ap.add_argument("--label_field", default="label", help="gold 中标签字段名")

    args = ap.parse_args()

    pred_path = os.path.abspath(args.pred_jsonl)

    gold_path = os.path.abspath(args.gold_jsonl)

    out_dir = os.path.abspath(args.out_dir.strip()) if args.out_dir.strip() else os.path.join(script_dir, "camo_test_3500_7")

    os.makedirs(out_dir, exist_ok=True)

    preds = load_jsonl(pred_path)

    golds = load_jsonl(gold_path)

    if args.align_by_id:

        preds = align_preds_to_gold_rows(golds, preds)

    n = len(preds)

    if len(golds) != n:

        raise SystemExit(f"行数不一致: preds={n} gold={len(golds)}（若顺序不同可试 --align_by_id）")

    pred_stem = os.path.splitext(os.path.basename(pred_path))[0]

    summary = []

    for start_1based in range(1, 8):

        kind = EVERY7_KIND_BY_START[start_1based]

        idx = list(range(start_1based - 1, n, 7))

        sub_pred = [preds[i] for i in idx]

        sub_gold = [golds[i] for i in idx]

        stem = f"{pred_stem}_{kind}"

        pred_out = os.path.join(out_dir, f"{stem}.jsonl")

        dump_jsonl(pred_out, sub_pred)

        metrics_path = os.path.join(out_dir, f"{stem}_metrics.json")

        cm_csv_path = os.path.join(out_dir, f"{stem}_confusion_matrix.csv")

        subset_meta = {

            "every7_start_1based": start_1based,

            "subset_kind": kind,

            "pred_source": os.path.basename(pred_path),

            "gold_source": os.path.basename(gold_path),

        }

        m = run_eval_and_save(

            rows_in=sub_gold,

            outputs=sub_pred,

            label_field=args.label_field,

            metrics_path=metrics_path,

            cm_csv_path=cm_csv_path,

            subset_meta=subset_meta,

        )

        print(

            f"[OK] {kind} ({stem}.jsonl) | n={len(idx)} acc={m['accuracy']:.6f} macro_f1={m['macro_f1']:.6f}",

            flush=True,

        )

        summary.append((start_1based, kind, len(idx), m["accuracy"], m["macro_f1"]))

    print("\n===== 七路子集汇总 (步长7：原词/缩略/emoji/音/义/形/混合) =====", flush=True)

    for s, kind, cnt, acc, f1 in summary:

        print(f"{kind} (from{s}): n={cnt}  accuracy={acc:.6f}  macro_f1={f1:.6f}", flush=True)

if __name__ == "__main__":

    main()
