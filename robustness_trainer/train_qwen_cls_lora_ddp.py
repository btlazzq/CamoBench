import argparse
import json
import os
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)

from peft import (

    LoraConfig,

    get_peft_model,

    prepare_model_for_kbit_training,

)

LABELS = ["不违规", "涉黄", "涉赌", "涉毒", "涉诈"]

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

def formatting_for_batch(

    tokenizer,

    batch: Dict[str, List[Any]],

    instruction_field: str,

    input_field: str,

    add_generation_prompt: bool,

    system_role_in_instruction: bool,

) -> List[str]:

    texts: List[str] = []

    n = len(next(iter(batch.values())))

    instructions = batch.get(instruction_field, [""] * n)

    inputs = batch.get(input_field, [""] * n)

    for inst, inp in zip(instructions, inputs):

        texts.append(

            build_chat_text(

                tokenizer=tokenizer,

                instruction=str(inst),

                user_input=str(inp),

                add_generation_prompt=add_generation_prompt,

                system_role_in_instruction=system_role_in_instruction,

            )

        )

    return texts

def normalize_label(x: Any) -> str:

    return str(x).strip()

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    preds = np.argmax(logits, axis=-1)

    return {

        "accuracy": accuracy_score(labels, preds),

        "macro_f1": f1_score(labels, preds, average="macro"),

    }

def save_label_mapping(output_dir: str):

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "label2id.json"), "w", encoding="utf-8") as f:

        json.dump(LABEL2ID, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, "id2label.json"), "w", encoding="utf-8") as f:

        json.dump(ID2LABEL, f, ensure_ascii=False, indent=2)

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--model_name_or_path", required=True)

    ap.add_argument("--train_file", required=True)

    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--dev_file", default=None)

    ap.add_argument("--report_to", type=str, default="none", help="none|wandb")

    ap.add_argument("--run_name", type=str, default=None)

    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--max_seq_len", type=int, default=1024)

    ap.add_argument("--instruction_field", type=str, default="instruction")

    ap.add_argument("--input_field", type=str, default="input")

    ap.add_argument("--label_field", type=str, default="label")

    ap.add_argument("--add_generation_prompt", action="store_true")

    ap.add_argument("--no_system_role_in_instruction", action="store_true")

    ap.add_argument("--epochs", type=int, default=3)

    ap.add_argument("--lr", type=float, default=2e-4)

    ap.add_argument("--batch_size", type=int, default=4)

    ap.add_argument("--grad_accum", type=int, default=4)

    ap.add_argument("--warmup_ratio", type=float, default=0.03)

    ap.add_argument("--weight_decay", type=float, default=0.0)

    ap.add_argument("--logging_steps", type=int, default=10)

    ap.add_argument("--save_strategy", type=str, default="epoch", choices=["epoch", "steps"])

    ap.add_argument("--save_steps", type=int, default=200)

    ap.add_argument("--eval_strategy", type=str, default=None, choices=[None, "no", "epoch", "steps"])

    ap.add_argument("--eval_steps", type=int, default=200)

    ap.add_argument("--early_stopping_patience", type=int, default=3)

    ap.add_argument("--early_stopping_threshold", type=float, default=0.0)

    ap.add_argument("--lora_r", type=int, default=16)

    ap.add_argument("--lora_alpha", type=int, default=32)

    ap.add_argument("--lora_dropout", type=float, default=0.05)

    ap.add_argument("--use_bf16", action="store_true")

    ap.add_argument("--use_fp16", action="store_true")

    ap.add_argument("--use_4bit", action="store_true")

    ap.add_argument("--gradient_checkpointing", action="store_true")

    ap.add_argument("--local_files_only", action="store_true")

    ap.add_argument("--trust_remote_code", action="store_true")

    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)

    if args.use_bf16 and args.use_fp16:

        raise ValueError("不能同时设置 --use_bf16 和 --use_fp16")

    if args.eval_strategy is None:

        args.eval_strategy = "epoch" if args.dev_file else "no"

    if args.early_stopping_patience > 0 and not args.dev_file:

        raise ValueError("启用早停时必须提供 --dev_file")

    if args.dev_file and args.eval_strategy == "steps":

        args.save_strategy = "steps"

        args.save_steps = args.eval_steps

    tokenizer = AutoTokenizer.from_pretrained(

        args.model_name_or_path,

        use_fast=True,

        trust_remote_code=args.trust_remote_code,

        local_files_only=args.local_files_only,

    )

    if tokenizer.pad_token is None:

        tokenizer.pad_token = tokenizer.eos_token

    data_files = {"train": args.train_file}

    if args.dev_file:

        data_files["dev"] = args.dev_file

    raw_ds = load_dataset("json", data_files=data_files)

    if args.label_field not in raw_ds["train"].column_names:

        raise ValueError(f"训练文件里找不到 label 字段：{args.label_field}")

    def check_and_map_label(ex):

        lab = normalize_label(ex[args.label_field])

        if lab not in LABEL2ID:

            raise ValueError(f"发现非法标签: {lab}，必须属于 {LABELS}")

        ex["label_id"] = LABEL2ID[lab]

        return ex

    raw_train = raw_ds["train"].map(check_and_map_label)

    raw_dev = raw_ds["dev"].map(check_and_map_label) if "dev" in raw_ds else None

    save_label_mapping(args.output_dir)

    def preprocess_function(batch: Dict[str, List[Any]]) -> Dict[str, Any]:

        texts = formatting_for_batch(

            tokenizer=tokenizer,

            batch=batch,

            instruction_field=args.instruction_field,

            input_field=args.input_field,

            add_generation_prompt=args.add_generation_prompt,

            system_role_in_instruction=not args.no_system_role_in_instruction,

        )

        tokenized = tokenizer(

            texts,

            padding=False,

            truncation=True,

            max_length=args.max_seq_len,

        )

        tokenized["labels"] = batch["label_id"]

        return tokenized

    remove_cols_train = raw_train.column_names

    tokenized_train = raw_train.map(

        preprocess_function,

        batched=True,

        remove_columns=remove_cols_train,

        desc="Tokenizing train",

    )

    tokenized_dev = None

    if raw_dev is not None:

        remove_cols_dev = raw_dev.column_names

        tokenized_dev = raw_dev.map(

            preprocess_function,

            batched=True,

            remove_columns=remove_cols_dev,

            desc="Tokenizing dev",

        )

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

    model = AutoModelForSequenceClassification.from_pretrained(

        args.model_name_or_path,

        num_labels=len(LABELS),

        quantization_config=quant_cfg,

        torch_dtype=torch_dtype,

        trust_remote_code=args.trust_remote_code,

        local_files_only=args.local_files_only,

    )

    model.config.pad_token_id = tokenizer.pad_token_id

    model.config.eos_token_id = tokenizer.eos_token_id

    model.config.problem_type = "single_label_classification"

    model.config.label2id = LABEL2ID

    model.config.id2label = ID2LABEL

    if args.use_4bit:

        model = prepare_model_for_kbit_training(model)

    if args.gradient_checkpointing:

        model.gradient_checkpointing_enable()

        model.config.use_cache = False

    modules_to_save = None

    if hasattr(model, "score"):

        modules_to_save = ["score"]

    elif hasattr(model, "classifier"):

        modules_to_save = ["classifier"]

    lora_cfg = LoraConfig(

        r=args.lora_r,

        lora_alpha=args.lora_alpha,

        lora_dropout=args.lora_dropout,

        bias="none",

        task_type="SEQ_CLS",

        target_modules=[

            "q_proj",

            "k_proj",

            "v_proj",

            "o_proj",

            "up_proj",

            "down_proj",

            "gate_proj",

        ],

        modules_to_save=modules_to_save,

    )

    model = get_peft_model(model, lora_cfg)

    model.print_trainable_parameters()

    train_args = TrainingArguments(

        output_dir=args.output_dir,

        num_train_epochs=args.epochs,

        per_device_train_batch_size=args.batch_size,

        per_device_eval_batch_size=args.batch_size,

        gradient_accumulation_steps=args.grad_accum,

        learning_rate=args.lr,

        warmup_ratio=args.warmup_ratio,

        weight_decay=args.weight_decay,

        lr_scheduler_type="cosine",

        logging_steps=args.logging_steps,

        save_strategy=args.save_strategy,

        save_steps=args.save_steps if args.save_strategy == "steps" else None,

        eval_strategy=args.eval_strategy,

        eval_steps=args.eval_steps if args.eval_strategy == "steps" else None,

        save_total_limit=2,

        bf16=args.use_bf16,

        fp16=args.use_fp16,

        optim="adamw_torch",

        report_to=([] if args.report_to == "none" else ["wandb"]),

        run_name=args.run_name,

        remove_unused_columns=True,

        ddp_find_unused_parameters=False,

        load_best_model_at_end=(tokenized_dev is not None and args.eval_strategy != "no"),

        metric_for_best_model="macro_f1" if tokenized_dev is not None else None,

        greater_is_better=True if tokenized_dev is not None else None,

    )

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    callbacks = []

    if tokenized_dev is not None and args.eval_strategy != "no":

        callbacks.append(

            EarlyStoppingCallback(

                early_stopping_patience=args.early_stopping_patience,

                early_stopping_threshold=args.early_stopping_threshold,

            )

        )

    trainer = Trainer(

        model=model,

        args=train_args,

        train_dataset=tokenized_train,

        eval_dataset=tokenized_dev,

        tokenizer=tokenizer,

        data_collator=data_collator,

        compute_metrics=compute_metrics if tokenized_dev is not None else None,

        callbacks=callbacks,

    )

    trainer.train()

    trainer.save_model(args.output_dir)

    tokenizer.save_pretrained(args.output_dir)

    if tokenized_dev is not None:

        metrics = trainer.evaluate()

        with open(os.path.join(args.output_dir, "final_eval_metrics.json"), "w", encoding="utf-8") as f:

            json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved to {args.output_dir}")

if __name__ == "__main__":

    main()
