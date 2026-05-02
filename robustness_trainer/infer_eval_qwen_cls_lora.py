import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel


LABELS = ["不违规", "涉黄", "涉赌", "涉毒", "其他"]
LABEL2ID = {x: i for i, x in enumerate(LABELS)}
ID2LABEL = {i: x for x, i in LABEL2ID.items()}


def build_chat_text(
    tokenizer,
    instruction: str,
    user_input: str,
    add_generation_prompt: bool,
    system_role_in_instruction: bool = True,
) -> str:
    instruction = (instruction or "").strip()
    user_input = (user_input or "").strip()

    if system_role_in_instruction:
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_input},
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": (instruction + "\n" + user_input).strip(),
            }
        ]

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    if system_role_in_instruction:
        return f"[SYSTEM]\n{instruction}\n[USER]\n{user_input}"
    return f"{instruction}\n{user_input}"


def normalize_label(x: Any) -> str:
    return str(x).strip()


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def save_json(obj: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_confusion_matrix_csv(cm: np.ndarray, labels: List[str], path: str):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gold\\pred"] + labels)
        for i, lab in enumerate(labels):
            writer.writerow([lab] + cm[i].tolist())


def load_label_mapping(lora_dir: str):
    label2id_path = os.path.join(lora_dir, "label2id.json")
    id2label_path = os.path.join(lora_dir, "id2label.json")

    label2id = LABEL2ID
    id2label = ID2LABEL

    if os.path.exists(label2id_path):
        with open(label2id_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        label2id = {str(k): int(v) for k, v in raw.items()}
        id2label = {v: k for k, v in label2id.items()}

    if os.path.exists(id2label_path):
        with open(id2label_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # json 里 key 可能是字符串
        id2label = {int(k): str(v) for k, v in raw.items()}
        label2id = {v: k for k, v in id2label.items()}

    labels = [id2label[i] for i in range(len(id2label))]
    return label2id, id2label, labels


def batch_iter(lst: List[Any], batch_size: int):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--base_model", required=True)
    ap.add_argument("--lora_dir", required=True)
    ap.add_argument("--input_file", required=True)
    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--instruction_field", type=str, default="instruction")
    ap.add_argument("--input_field", type=str, default="input")
    ap.add_argument("--label_field", type=str, default="label")

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_seq_len", type=int, default=1024)

    ap.add_argument("--add_generation_prompt", action="store_true")
    ap.add_argument("--no_system_role_in_instruction", action="store_true")

    ap.add_argument("--use_bf16", action="store_true")
    ap.add_argument("--use_fp16", action="store_true")
    ap.add_argument("--use_4bit", action="store_true")

    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--local_files_only", action="store_true")

    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.use_bf16 and args.use_fp16:
        raise ValueError("不能同时设置 --use_bf16 和 --use_fp16")

    # ========= 读标签映射 =========
    label2id, id2label, labels_in_order = load_label_mapping(args.lora_dir)

    # ========= tokenizer =========
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ========= model =========
    quant_cfg: Optional[BitsAndBytesConfig] = None
    if args.use_4bit:
        compute_dtype = torch.bfloat16 if args.use_bf16 else torch.float16
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )

    torch_dtype = None
    if args.use_bf16:
        torch_dtype = torch.bfloat16
    elif args.use_fp16:
        torch_dtype = torch.float16

    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(labels_in_order),
        quantization_config=quant_cfg,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.eos_token_id = tokenizer.eos_token_id
    base_model.config.problem_type = "single_label_classification"
    base_model.config.label2id = label2id
    base_model.config.id2label = id2label

    model = PeftModel.from_pretrained(base_model, args.lora_dir)
    model.eval()

    if not args.use_4bit:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
    else:
        device = None  # bitsandbytes + accelerate 自己处理

    # ========= 数据 =========
    ds = load_dataset("json", data_files={"data": args.input_file})["data"]

    records: List[Dict[str, Any]] = []
    texts: List[str] = []
    gold_ids: List[Optional[int]] = []

    for ex in ds:
        instruction = ex.get(args.instruction_field, "")
        user_input = ex.get(args.input_field, "")
        label = ex.get(args.label_field, None)

        text = build_chat_text(
            tokenizer=tokenizer,
            instruction=str(instruction),
            user_input=str(user_input),
            add_generation_prompt=args.add_generation_prompt,
            system_role_in_instruction=not args.no_system_role_in_instruction,
        )

        texts.append(text)
        records.append(dict(ex))

        if label is None:
            gold_ids.append(None)
        else:
            lab = normalize_label(label)
            if lab not in label2id:
                raise ValueError(f"发现未知标签: {lab}，标签映射中不存在")
            gold_ids.append(label2id[lab])

    # ========= 推理 =========
    all_pred_ids: List[int] = []
    all_logits: List[List[float]] = []
    all_probs: List[List[float]] = []

    for batch_texts in batch_iter(texts, args.batch_size):
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=args.max_seq_len,
            return_tensors="pt",
        )

        if device is not None:
            encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            logits = outputs.logits

        logits_np = logits.detach().float().cpu().numpy()
        probs_np = softmax_np(logits_np, axis=-1)
        pred_ids = np.argmax(logits_np, axis=-1).tolist()

        all_pred_ids.extend(pred_ids)
        all_logits.extend(logits_np.tolist())
        all_probs.extend(probs_np.tolist())

    # ========= 写逐条结果 =========
    pred_path = os.path.join(args.output_dir, "predictions.jsonl")
    gold_available = all(x is not None for x in gold_ids)

    with open(pred_path, "w", encoding="utf-8") as f:
        for i, rec in enumerate(records):
            pred_id = int(all_pred_ids[i])
            pred_label = id2label[pred_id]

            obj = {
                **rec,
                "pred_id": pred_id,
                "pred_label": pred_label,
                "logits": [round(float(x), 6) for x in all_logits[i]],
                "probs": {
                    id2label[j]: round(float(all_probs[i][j]), 6)
                    for j in range(len(all_probs[i]))
                },
            }

            if gold_ids[i] is not None:
                gold_id = int(gold_ids[i])
                gold_label = id2label[gold_id]
                obj["gold_id"] = gold_id
                obj["gold_label"] = gold_label
                obj["correct"] = (gold_id == pred_id)

            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[OK] Saved predictions to: {pred_path}")

    # ========= 评估 =========
    if gold_available:
        y_true = [int(x) for x in gold_ids]
        y_pred = [int(x) for x in all_pred_ids]

        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro")

        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(len(labels_in_order))),
            target_names=labels_in_order,
            digits=6,
            output_dict=True,
            zero_division=0,
        )

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(len(labels_in_order))),
        )

        metrics = {
            "num_samples": len(y_true),
            "accuracy": round(float(acc), 6),
            "macro_f1": round(float(macro_f1), 6),
            "labels": labels_in_order,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }

        metrics_path = os.path.join(args.output_dir, "metrics.json")
        save_json(metrics, metrics_path)

        cm_csv_path = os.path.join(args.output_dir, "confusion_matrix.csv")
        save_confusion_matrix_csv(cm, labels_in_order, cm_csv_path)

        print(f"[OK] Saved metrics to: {metrics_path}")
        print(f"[OK] Saved confusion matrix csv to: {cm_csv_path}")

        print("\n===== Eval Metrics =====")
        print(f"Samples   : {len(y_true)}")
        print(f"Accuracy  : {acc:.6f}")
        print(f"Macro-F1  : {macro_f1:.6f}")
        print("\n===== Per-class F1 =====")
        for lab in labels_in_order:
            f1v = report[lab]["f1-score"]
            p = report[lab]["precision"]
            r = report[lab]["recall"]
            print(f"{lab}: P={p:.6f} R={r:.6f} F1={f1v:.6f}")
    else:
        print("[INFO] 输入文件未提供完整 gold label，已跳过指标评估。")


if __name__ == "__main__":
    main()